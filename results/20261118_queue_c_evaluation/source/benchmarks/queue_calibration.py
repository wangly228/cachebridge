"""Calibrate a single request's restore/recompute choice with its peers held at restore."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from shutil import copyfile
import traceback

import numpy as np
import torch

from nanovllm import LLM
from nanovllm.engine.restore_policy import CostShape, RestoreCostTable, RestoreDecision, cost_environment
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.cpu_cache_pressure import BASELINE
from benchmarks.phase1 import reset_cache, save_wave, wave
from benchmarks.restore_calibration import inputs_for, state_digest
from benchmarks.restore_holdout import verify


V1_TABLE = Path("results/20261118_stage2_c_calibration/cost_table.json")
V1_SHA256 = "f66c529cec9aad0f80666f842600c4137ab2837b4407985e8a0a2bbc826ff3c4"
CAL_SEEDS = (1101, 1102, 1103, 1104)
EVAL_SEEDS = (1201, 1202, 1203, 1204, 1205, 1206)
CAL_LAYOUTS = {"single1024": [1024], "uniform1024": [1024]*4,
               "mixed": [512, 1024, 2048, 512], "lighter": [512, 1024, 512, 512]}
EVAL_LAYOUTS = {"mixed": CAL_LAYOUTS["mixed"], "lighter_permuted": [512, 512, 1024, 512]}


class RecomputeOne:
    def __init__(self, lengths, target):
        self.queue_shapes = {CostShape(0, n, 0, 128, len(lengths), i) for i, n in enumerate(lengths)}
        self.target = target
        self.first = {}

    def decide(self, shape):
        self.first.setdefault(shape.running, shape)
        action = "recompute" if shape.running == self.target else "restore"
        return RestoreDecision(action, "calibration_intervention")


def run_cycle(llm, out, trace, policy, output_tokens, tag):
    reset_cache(llm)
    llm.scheduler.restore_policy = policy
    torch.manual_seed(tag["seed"])
    batches = []
    for phase, inputs in trace:
        if phase == "evicted":
            initial_state = state_digest(llm.scheduler.block_manager)
        batch = wave(llm, inputs, output_tokens, dict(tag, phase=phase))
        for row in batch[0]:
            verify(row, "B" if policy is None else "C")
        if phase == "interference":
            assert batch[2]["used_blocks_peak"] == 36
        batches.append(batch)
    for batch in batches:
        save_wave(out, batch)
    return batches, initial_state


def engine_config():
    return dict(json.loads((BASELINE / "engine.json").read_text())["config"], cpu_prefix_cache_blocks=64)


def fit(pairs, env, base, out):
    generic, contextual = defaultdict(list), defaultdict(list)
    for p in pairs:
        shape = CostShape(**p["shape"])
        if p["layout"] in ("single1024", "uniform1024"):
            generic[shape.without_peers()].append(p)
        else:
            contextual[shape].append(p)

    def buckets(groups):
        rows = []
        for shape, group in groups.items():
            assert len(group) == len(CAL_SEEDS) and {p["seed"] for p in group} == set(CAL_SEEDS)
            r = [p["restore_ms"] for p in group]
            c = [p["recompute_ms"] for p in group]
            rows.append(dict(shape=asdict(shape), paired_waves=len(group),
                        restore_p50_ms=float(np.median(r)), restore_p90_ms=float(np.percentile(r, 90)),
                        recompute_p10_ms=float(np.percentile(c, 10)), recompute_p50_ms=float(np.median(c))))
        return rows

    assert len(generic) == 5 and len(contextual) == 8
    length_only = dict(base, buckets=[*base["buckets"], *buckets(generic)],
                       calibration_seeds=[*base["calibration_seeds"], *CAL_SEEDS],
                       base_table_sha256=V1_SHA256, pairs_sha256=sha256(out / "pairs.jsonl"))
    complete = dict(length_only, version=2, buckets=[*length_only["buckets"], *buckets(contextual)])
    for filename, data in (("length_only.json", length_only), ("cost_table.json", complete)):
        RestoreCostTable(data, env)
        write_json(out / filename, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.queue_calibration"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        assert sha256(V1_TABLE) == V1_SHA256
        copyfile(V1_TABLE, out / "v1.json")
        base = json.loads((out / "v1.json").read_text())
        torch.manual_seed(0)
        config = engine_config()
        llm = LLM(args.model, **config)
        env = cost_environment(llm.config, llm.scheduler.block_manager.namespace)
        assert env == base["environment"] and llm.model_runner.kv_cache.dtype == torch.bfloat16
        write_json(out / "design.json", dict(config=config, environment=env, base_sha256=V1_SHA256,
                   calibration_layouts=CAL_LAYOUTS, calibration_seeds=CAL_SEEDS, warmup_seed=1100,
                   evaluation_layouts=EVAL_LAYOUTS, evaluation_seeds=EVAL_SEEDS,
                   pool="queue-cal-v2", suffix_tokens=128, output_tokens=2, evaluation_output_tokens=128,
                   threshold="restore p90 < recompute p10", endpoint="enqueue to first real output token",
                   intervention="restore all peers and vary only the target; match first visible decision context",
                   primary="mixed replay p95 TTFT and uncalibrated recomputes: C2 vs C1; also compare B",
                   evaluation_order="all six permutations of B, C1, C2; one engine and equal starting states",
                   profiler=False))
        pairs = []
        for seed in (1100, *CAL_SEEDS):
            for layout, lengths in CAL_LAYOUTS.items():
                trace = inputs_for("queue-cal-v2", seed, lengths)
                order = [None, *range(len(lengths))]
                if seed % 2 == 0:
                    order.reverse()
                control = None
                alternatives = {}
                states = set()
                for target in order:
                    policy = RecomputeOne(lengths, target)
                    tag = dict(seed=seed, layout=layout, target=target, warmup=seed == 1100)
                    batches, state = run_cycle(llm, out, trace, policy, 2, tag)
                    states.add(state)
                    rows, steps, _ = batches[-1]
                    assert max(r["token_s"][0] for r in rows) <= min(s["start_s"] for s in steps if not s["is_prefill"])
                    for i, row in enumerate(rows):
                        assert row["cpu_restored_tokens"] == (0 if i == target else lengths[i])
                    if target is None:
                        control = (rows, policy.first)
                    else:
                        alternatives[target] = (rows, policy.first[target])
                assert len(states) == 1
                if seed != 1100:
                    reference, first = control
                    for i, length in enumerate(lengths):
                        other, first_recompute = alternatives[i]
                        shape = first[i]
                        assert shape == first_recompute == CostShape(0, length, 0, 128, len(lengths), i,
                            (len(lengths)-1)*128, sum(lengths)-length)
                        assert reference[i]["input_token_ids"] == other[i]["input_token_ids"]
                        row = dict(seed=seed, layout=layout, target=i, shape=asdict(shape),
                            recompute_committed_shape=other[i]["restore_decisions"][0]["shape"],
                            restore_ms=reference[i]["ttft_ms"], recompute_ms=other[i]["ttft_ms"],
                            restore_h2d_ms=reference[i]["h2d_ms"], recompute_d2h_ms=other[i]["d2h_ms"],
                            restore_seq_id=reference[i]["seq_id"], recompute_seq_id=other[i]["seq_id"],
                            peer_ttft_change_ms=[alt["ttft_ms"]-ref["ttft_ms"] for j, (alt, ref)
                                                 in enumerate(zip(other, reference, strict=True)) if j != i],
                            state_sha256=next(iter(states)))
                        pairs.append(row)
                        write_jsonl(out / "pairs.jsonl", [row])
                print(f"seed={seed} layout={layout} complete", flush=True)
        fit(pairs, env, base, out)
        write_json(out / "status.json", dict(passed=True, target_pairs=len(pairs),
                   homogeneous_buckets_added=5, queue_buckets_added=8, frozen_base_buckets=len(base["buckets"]),
                   table_sha256=sha256(out / "cost_table.json"), length_only_sha256=sha256(out / "length_only.json")))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
