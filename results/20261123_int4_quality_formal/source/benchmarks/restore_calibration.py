"""Pair real restores and recomputations through the same engine up to the first token."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import traceback

import numpy as np
import torch

from nanovllm import LLM
from nanovllm.engine.restore_policy import CostShape, RestoreCostTable, RestoreDecision, cost_environment
from benchmarks.common import prepare_run, sha256 as file_sha256, write_json, write_jsonl
from benchmarks.cpu_cache_pressure import BASELINE
from benchmarks.phase1 import reset_cache, save_wave, wave


CALIBRATION_SEEDS = (701, 702, 703, 704)
SHAPES = ((512, 1), (512, 4), (2048, 1), (2048, 4), (8192, 1))


class ForcedPolicy:
    """Benchmark intervention. Storage, scheduling and transfer use the production path."""

    def __init__(self, action):
        self.action = action

    def decide(self, shape):
        return RestoreDecision(self.action, "calibration_intervention")


def inputs_for(pool, seed, lengths):
    rng = Random(f"PegaFlow-cost-v1/{pool}/{seed}/{lengths}")

    def tokens(n):
        return [rng.randrange(100, 10000) for _ in range(n)]

    prefixes = [tokens(n) for n in lengths]
    fill = [p + tokens(128) for p in prefixes]
    replay = [p + tokens(128) for p in prefixes]
    # Four nine-page requests hold all 36 GPU pages. Their 32 CPU entries
    # leave room for the largest calibration prefix (8192 tokens / 32 pages).
    interference = [tokens(2176) for _ in range(4)]
    return [("fill", fill), ("interference", interference), ("evicted", replay)]


def state_digest(manager):
    # All sequences have finished. Include CPU slot IDs/LRU order and free GPU page order.
    state = [list(manager.free_block_ids), sorted(manager.hash_to_block_id.items()),
             [(key.hex(), slot) for key, (slot, _) in manager.entries.items()]]
    return sha256(json.dumps(state).encode()).hexdigest()


def make_table(samples, environment):
    buckets = []
    for shape, pairs in samples.items():
        assert {p["seed"] for p in pairs} == set(CALIBRATION_SEEDS) and len(pairs) == len(CALIBRATION_SEEDS)
        r = [p["restore_ms"] for p in pairs]
        c = [p["recompute_ms"] for p in pairs]
        buckets.append(dict(shape=asdict(shape), paired_waves=len(pairs),
                            restore_p50_ms=float(np.median(r)), restore_p90_ms=float(np.percentile(r, 90)),
                            recompute_p50_ms=float(np.median(c)), recompute_p10_ms=float(np.percentile(c, 10))))
    return dict(version=1, environment=environment, calibration_seeds=list(CALIBRATION_SEEDS),
                endpoint="request enqueue to first real token, ms", buckets=buckets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.restore_calibration"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        config = dict(json.loads((BASELINE / "engine.json").read_text())["config"], cpu_prefix_cache_blocks=64)
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        env = cost_environment(llm.config, llm.scheduler.block_manager.namespace)
        assert env["gpu_blocks"] == 36 and llm.model_runner.kv_cache.dtype == torch.bfloat16
        write_json(out / "design.json", dict(config=config, environment=env, shapes=SHAPES,
                   seeds=CALIBRATION_SEEDS, namespace="calibration", output_tokens=2, suffix_tokens=128,
                   margin="restore p90 < recompute p10, exact shape and running count; otherwise recompute",
                   profiler=False, warmup_seed=700, input_state="real fill then 4 disjoint 2176-token requests"))
        samples = defaultdict(list)
        for seed in (700, *CALIBRATION_SEEDS):
            for length, count in SHAPES:
                trace = inputs_for("calibration", seed, [length] * count)
                order = ("restore", "recompute") if seed % 2 else ("recompute", "restore")
                measured = {}
                start_states = []
                for action in order:
                    reset_cache(llm)
                    llm.scheduler.restore_policy = ForcedPolicy(action)
                    for phase, inputs in trace:
                        if phase == "evicted":
                            start_states.append(state_digest(llm.scheduler.block_manager))
                        tag = dict(seed=seed, warmup=seed == 700, length=length, concurrency=count,
                                   arm=action, phase=phase)
                        result = wave(llm, inputs, 2, tag)
                        save_wave(out, result)
                        rows, steps, _ = result
                        assert all(r["preemptions"] == r["cpu_restore_failures"] == r["cpu_save_failures"] == 0
                                   for r in rows)
                        if phase != "evicted":
                            assert all(not r["restore_decisions"] and r["initial_gpu_hit_tokens"] == 0 for r in rows)
                            continue
                        cutoff = min(s["start_s"] for s in steps if not s["is_prefill"])
                        assert max(r["token_s"][0] for r in rows) <= cutoff
                        for i, row in enumerate(rows):
                            decision, = row["restore_decisions"]
                            expected = length if action == "restore" else 0
                            assert (decision["action"], row["initial_gpu_hit_tokens"],
                                    row["initial_cpu_hit_tokens"]) == (action, 0, length)
                            assert row["h2d_blocks"] * 256 == row["cpu_restored_tokens"] == expected
                            assert row["d2h_blocks"] * 256 == length - expected
                            assert row["prompt_computed_tokens"] == length + 128 - expected
                            assert decision["shape"] == asdict(CostShape(0, length, 0, 128, count, i))
                        measured[action] = rows
                assert start_states[0] == start_states[1]
                if seed != 700:
                    for r, c in zip(measured["restore"], measured["recompute"], strict=True):
                        assert r["input_token_ids"] == c["input_token_ids"]
                        shape = CostShape(**r["restore_decisions"][0]["shape"])
                        pair = dict(seed=seed, shape=asdict(shape), restore_ms=r["ttft_ms"],
                                    recompute_ms=c["ttft_ms"], restore_h2d_ms=r["h2d_ms"],
                                    recompute_d2h_ms=c["d2h_ms"], state_sha256=start_states[0],
                                    restore_seq_id=r["seq_id"], recompute_seq_id=c["seq_id"])
                        samples[shape].append(pair)
                        write_jsonl(out / "pairs.jsonl", [pair])
                print(f"seed={seed} prefix={length} concurrency={count} paired", flush=True)
        table = make_table(samples, env)
        table["pairs_sha256"] = file_sha256(out / "pairs.jsonl")
        policy = RestoreCostTable(table, env)
        assert len(policy.buckets) == sum(count for _, count in SHAPES)
        write_json(out / "cost_table.json", table)
        write_json(out / "status.json", dict(passed=True, paired_requests=sum(map(len, samples.values())),
                   measured_wave_pairs=len(SHAPES)*len(CALIBRATION_SEEDS), buckets=len(samples),
                   restore_buckets=sum(v.action == "restore" for v in policy.buckets.values()),
                   table_sha256=file_sha256(out / "cost_table.json")))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
