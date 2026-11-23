"""Counterbalanced A / write-through B / reuse-admission B on the frozen pressure trace."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter
import traceback

import torch

from nanovllm import LLM
from nanovllm.engine.scheduler import Scheduler
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.compare_cpu_trace import rows
from benchmarks.compare_restore_policy import stats
from benchmarks.cpu_cache_pressure import BASELINE, read_baseline
from benchmarks.phase1 import save_wave, wave


ARMS = ("A", "B", "reuse")
PHASES = ("fill", "hot", "interference", "evicted", "all")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def report(out):
    engine = json.loads((out / "engine.json").read_text())
    arms = tuple(json.loads((out / "design.json").read_text()).get("arms", ARMS))
    _, control, candidate = arms
    requests, waves = (rows(out, name) for name in ("requests.jsonl", "waves.jsonl"))
    cycles = jsonl(out / "cycles.jsonl")
    ref = rows(BASELINE, "requests.jsonl")
    result = {}
    for arm in arms:
        raw = [r for r in requests if r["arm"] == arm]
        by_wave = [w for w in waves if w["arm"] == arm]
        assert len(raw) == 96 and len(by_wave) == 24
        groups = {}
        for phase in PHASES:
            sample = [r for r in raw if phase == "all" or r["phase"] == phase]
            times = [w for w in by_wave if phase == "all" or w["phase"] == phase]
            groups[phase] = stats(sample, times)
            if candidate == "batch":
                for key in ("d2h_batches", "batched_d2h_blocks"):
                    groups[phase][key] = sum(r[key] for r in sample)
        wall = sum(c["wall_s"] for c in cycles if c["arm"] == arm)
        result[arm] = dict(groups=groups, cycle_wall_s=wall,
                           output_tokens_per_wall_second=sum(r["output_tokens"] for r in raw) / wall)
    comparison = {}
    def change(value, base):
        return 100 * (value / base - 1)
    for phase in PHASES:
        a, b, new = (result[arm]["groups"][phase] for arm in arms)
        archive = stats([r for r in ref if phase == "all" or r["phase"] == phase],
                        [w for w in rows(BASELINE, "waves.jsonl") if phase == "all" or w["phase"] == phase])
        latency = new["ttft_ms"]["p95"]
        comparison[phase] = {
            f"{candidate}_vs_A_pct": change(latency, a["ttft_ms"]["p95"]),
            f"{candidate}_vs_{control}_pct": change(latency, b["ttft_ms"]["p95"]),
            f"{candidate}_vs_frozen_A_pct": change(latency, archive["ttft_ms"]["p95"]),
            f"{candidate}_vs_A_pct_by_seed": {s: change(v, a["p95_ms_by_seed"][s])
                                               for s, v in new["p95_ms_by_seed"].items()},
        }
        if candidate == "batch":
            comparison[phase]["batch_vs_reuse_pct_by_seed"] = {
                s: change(v, b["p95_ms_by_seed"][s]) for s, v in new["p95_ms_by_seed"].items()}
    gate = {key: dict(no_reuse_within_5pct=comparison["interference"][key] <= 5,
                      evicted_at_least_20pct_faster=comparison["evicted"][key] <= -20)
            for key in (f"{candidate}_vs_A_pct", f"{candidate}_vs_frozen_A_pct")}
    summary = dict(arms=result, change=comparison, gate=gate, block_bytes=engine["kv_budget"]["block_bytes"],
                   raw_sha256={n: sha256(out / n) for n in ("requests.jsonl", "waves.jsonl", "steps.jsonl",
                                                            "cycles.jsonl", "design.json", "engine.json")})
    if candidate == "batch":
        gate["batch_vs_reuse_pct"] = {f"{p}_p95_lower": comparison[p]["batch_vs_reuse_pct"] < 0
                                      for p in ("fill", "all")}
        controls = [r for r in requests if r["arm"] == control]
        measured = [r for r in requests if r["arm"] == candidate]
        for old, new in zip(controls, measured, strict=True):
            for key in ("seed", "cycle", "phase", "input_token_ids", "output_token_ids", "d2h_blocks",
                        "h2d_blocks", "initial_gpu_hit_tokens", "initial_cpu_hit_tokens", "prompt_computed_tokens"):
                assert new[key] == old[key], key
        early = set()
        for seed, cycle, trace in json.loads((BASELINE / "trace.json").read_text()):
            order = [phase for phase, _ in trace]
            if order.index("evicted") < order.index("hot"):
                early.add((seed, cycle))
        assert len(early) == 3
        def early_replay(r):
            return r["phase"] == "evicted" and (r["seed"], r["cycle"]) in early
        summary["evicted_before_first_reuse"] = {
            arm: stats([r for r in requests if r["arm"] == arm and early_replay(r)],
                       [w for w in waves if w["arm"] == arm and early_replay(w)]) for arm in arms}
    return summary


def select_arm(llm, arm):
    assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
    llm.config.cpu_prefix_cache_blocks = 0 if arm == "A" else llm.model_runner.kv_transfer.capacity
    llm.config.cpu_prefix_admission = "reuse" if arm in ("reuse", "batch") else "always"
    llm.config.cpu_prefix_batch_writes = arm == "batch"
    transfer = None if arm == "A" else llm.model_runner.kv_transfer
    llm.scheduler = Scheduler(llm.config, transfer)


def check_cycle(batches, refs, arm, seed, cycle):
    for requests, _, info in batches:
        phase = info["phase"]
        gpu = 2048 if phase == "hot" else 0
        cpu = 2048 if phase == "evicted" and arm != "A" else 0
        comparison = refs[seed, cycle, "hot" if cpu else phase]
        for row, ref in zip(requests, comparison, strict=True):
            assert row["input_token_ids"] == ref["input_token_ids"] and row["output_token_ids"] == ref["output_token_ids"]
            assert (row["initial_gpu_hit_tokens"], row["initial_cpu_hit_tokens"], row["cpu_restored_tokens"]) == (gpu, cpu, cpu)
            assert row["prompt_computed_tokens"] == row["prefill_tokens"] == 2176 - gpu - cpu
            assert row["h2d_blocks"] * 256 == cpu
            assert row["decode_tokens"] == 127 and row["output_tokens"] == 128
            assert abs(row["ttft_ms"] - 1000 * (row["token_s"][0] - row["enqueue_s"])) < 1e-5
            assert row["preemptions"] == row["cpu_save_failures"] == row["cpu_restore_failures"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", help="new directory for a GPU run")
    action.add_argument("--summarize", help="rebuild the summary on stdout from existing logs")
    parser.add_argument("--batched-fill", action="store_true", help="compare native GPU, serial reuse and batched reuse")
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(report(Path(args.summarize)), indent=2))
        return
    args.module = "benchmarks.write_admission"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        hashes, refs = read_baseline()
        baseline = json.loads((BASELINE / "engine.json").read_text())
        past, now = (json.loads((d / "environment.json").read_text()) for d in (BASELINE, out))
        for field in ("packages", "gpu", "capability", "cuda_runtime", "model_files"):
            assert now[field] == past[field], field
        traces = json.loads((BASELINE / "trace.json").read_text())
        arms = ("A", "reuse", "batch") if args.batched_fill else ARMS
        orders = list(permutations(arms))
        design = dict(trace=str(BASELINE / "trace.json"), frozen_sha256=hashes, orders=[
            dict(seed=s, cycle=c, arms=order) for (s, c, _), order in zip(traces, orders, strict=True)],
            first_use_limit="min(GPU blocks, CPU slots)", recent_keys_limit="CPU slots",
            warmup="pre-touch shared CPU store, then frozen warmup waves per arm",
            initial_state="empty per-arm GPU/CPU indices before each cycle; shared inactive staging in A",
            goals_pct=dict(max_no_reuse_regression=5, min_evicted_gain=20))
        if args.batched_fill:
            design.update(arms=arms, fill_goal="both fill and whole-trace pooled p95 below same-run serial reuse",
                          write_timing="all D2H and pageable copies must finish before publishing CPU keys")
        write_json(out / "design.json", design)
        config = dict(baseline["config"], cpu_prefix_cache_blocks=64, cpu_prefix_admission="reuse")
        if args.batched_fill:
            config["cpu_prefix_batch_writes"] = True
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        runner = llm.model_runner
        assert runner.kv_cache_budget["actual_blocks"] == baseline["kv_budget"]["actual_blocks"]
        assert runner.kv_transfer.block_bytes == baseline["kv_budget"]["block_bytes"]
        assert str(runner.kv_cache.dtype) == str(next(runner.model.parameters()).dtype) == baseline["dtype"]
        assert runner.graph_bs == baseline["graph_batch_sizes"] and 4 in runner.graphs
        write_json(out / "engine.json", dict(baseline, config=config, kv_budget=runner.kv_cache_budget,
                   cpu_slots=runner.kv_transfer.capacity, host_bytes=runner.kv_transfer.slots.nbytes,
                   pinned_staging_bytes=runner.kv_transfer.block_bytes * (2 if args.batched_fill else 1),
                   gpu_staging_bytes=runner.kv_transfer.block_bytes,
                   namespace=llm.scheduler.block_manager.namespace.hex(),
                   arm_admission={arm: ("GPU only" if arm == "A" else "always" if arm == "B" else "reuse")
                                  for arm in arms}))
        runner.kv_transfer.slots.fill(0)
        for arm in arms:
            select_arm(llm, arm)
            for state in ("fill", "hot"):
                inputs = [r["input_token_ids"] for r in refs[None, None, "warmup"] if r["cache"] == state]
                save_wave(out, wave(llm, inputs, 4, dict(arm=arm, phase="warmup", cache=state)))
        for (seed, cycle, trace), order in zip(traces, orders, strict=True):
            for arm in order:
                select_arm(llm, arm)
                torch.manual_seed(seed)
                start = perf_counter()
                batches = [wave(llm, inputs, 128, dict(arm=arm, phase=phase, seed=seed, cycle=cycle))
                           for phase, inputs in trace]
                elapsed = perf_counter() - start
                for batch in batches:
                    save_wave(out, batch)
                write_jsonl(out / "cycles.jsonl", [dict(arm=arm, seed=seed, cycle=cycle, wall_s=elapsed)])
                check_cycle(batches, refs, arm, seed, cycle)
                print(f"seed={seed} cycle={cycle} arm={arm} checked, {elapsed:.2f}s", flush=True)
        summary = report(out)
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(measurement_complete=True, same_shape_equal_requests=288,
                   requests_per_arm=96, cycles_per_arm=6, gate=summary["gate"]))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(measurement_complete=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
