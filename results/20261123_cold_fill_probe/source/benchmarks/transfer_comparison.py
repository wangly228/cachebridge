"""Paired native, synchronous 32-block and candidate CPU-write timing."""
import argparse
from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import torch

import nanovllm
from nanovllm import LLM
from benchmarks.common import ROOT, prepare_run, sha256, write_json, write_jsonl
from benchmarks.first_use_budget import normalized_steps
from benchmarks.first_use_budget_report import summarize, wave_key
from benchmarks.mixed_fill import make_case
from benchmarks.mixed_fill_report import percent, subset
from benchmarks.phase1 import save_wave, wave
from benchmarks.pressure import make_trace
from benchmarks.write_admission import jsonl, select_arm

ARCHIVE = Path("results/20261119_first_use_budget36")
ARMS = ("A", "32", "candidate")
ORDERS = tuple(permutations(ARMS))


def build_trace(stage):
    # The two historical warmups cover every timed prefill batch shape.
    historical = json.loads((ARCHIVE / "trace.json").read_text())
    trace = {"warmup": [dict(c, arms=ARMS) for c in historical["warmup"]], "timed": []}
    seeds = [1899] if stage == "screen" else list(range(1901, 1907))
    for i, seed in enumerate(seeds):
        for cycle in ((1,) if stage == "screen" else ((i % 2), (1 - i % 2))):
            for workload in (("long", "mixed") if i % 2 else ("mixed", "long")):
                if workload == "mixed":
                    phases = make_case(seed, cycle, i)["phases"]
                else:
                    phases = [("pressure1" if p == "interference" else p, inputs)
                              for p, inputs in make_trace(seed, cycle)]
                order = ORDERS[(i + 3 * cycle + int(workload == "long")) % len(ORDERS)]
                trace["timed"].append(dict(seed=seed, cycle=cycle, workload=workload, arms=order, phases=phases))
    return trace


def check_case(case, batches, output):
    for arm in case["arms"]:
        for phase, prompts in case["phases"]:
            rows, steps, w = batches[arm][phase]
            restored = phase == "evicted" and arm != "A"
            ref = batches["A"]["hot" if restored else phase]
            assert normalized_steps((rows, steps, w)) == normalized_steps(ref), (arm, phase)
            assert not w["profiler_enabled"] and w["used_blocks_peak"] <= 36
            assert w["cpu_cache_enabled"] == (arm != "A")
            if arm != "A":
                assert w["cpu_resident_blocks"] <= w["cpu_capacity_blocks"] == 64
                assert not w["cpu_evictions_since_reset"]
            for index, (row, native, prompt) in enumerate(zip(rows, ref[0], prompts, strict=True)):
                key = (case["seed"], case["cycle"], case["workload"], arm, phase, index)
                prefix = len(prompt) - 128
                gpu = prefix if phase == "hot" else 0
                cpu = prefix if restored else 0
                assert row["input_token_ids"] == native["input_token_ids"] == prompt, key
                assert row["output_token_ids"] == native["output_token_ids"], key
                assert len(row["token_s"]) == len(row["output_token_ids"]) == row["output_tokens"] == output, key
                assert row["decode_tokens"] == output - 1 and row["allocations"] == 1, key
                assert row["initial_gpu_hit_tokens"] == row["allocation_gpu_hit_tokens"] == gpu, key
                assert row["initial_cpu_hit_tokens"] == row["cpu_restored_tokens"] == row["h2d_blocks"] * 256 == cpu, key
                assert row["prompt_computed_tokens"] == row["prefill_tokens"] == len(prompt) - gpu - cpu, key
                assert row["preemptions"] == row["cpu_save_failures"] == row["cpu_restore_failures"] == 0, key
                assert not row["restore_decisions"], key
                assert abs(row["ttft_ms"] - 1000 * (row["token_s"][0] - row["enqueue_s"])) < 1e-5, key
                baseline = batches["32"][phase][0][index]
                assert row["d2h_blocks"] == (0 if arm == "A" else baseline["d2h_blocks"]), key
                if phase == "fill" and arm != "A":
                    assert row["d2h_blocks"] * 256 == prefix, key
        written = sum(r["d2h_blocks"] for rows, _, _ in batches[arm].values() for r in rows)
        assert written == (0 if arm == "A" else 32), (arm, written)


def measure_case(llm, out, c, warm, candidate):
    output = 8 if warm else 128
    tag = dict(seed=c["seed"], cycle=c["cycle"], workload=c["workload"], warmup=warm)
    results = {}
    for arm in c["arms"]:
        llm.config.cpu_prefix_transfer_mode = candidate if arm == "candidate" else "sync"
        select_arm(llm, "A" if arm == "A" else "batch")
        manager = llm.scheduler.block_manager
        assert not manager.used_block_ids
        if arm != "A":
            assert manager.first_use_limit == 32 and not manager.entries
        torch.manual_seed(c["seed"])
        start = perf_counter()
        batches = {}
        for phase, prompts in c["phases"]:
            batches[phase] = wave(llm, prompts, output, dict(tag, arm=arm, phase=phase))
            assert getattr(manager, "pending_write", None) is None, (arm, phase, "pending write after wave")
        # wave drives the engine to completion and synchronizes the device.
        # These final checks keep retirement and transfer work inside the cycle.
        assert llm.is_finished() and not manager.used_block_ids
        torch.cuda.synchronize()
        wall = perf_counter() - start
        for batch in batches.values():
            save_wave(out, batch)
        write_jsonl(out / "cycles.jsonl", [dict(tag, arm=arm, wall_s=wall)])
        results[arm] = batches
        print(f"{tag} arm={arm} {wall:.2f}s", flush=True)
    check_case(c, results, output)


def measure(args):
    assert ROOT == Path.cwd() and Path(nanovllm.__file__).resolve().is_relative_to(ROOT)
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        old, now = (json.loads((path / "environment.json").read_text()) for path in (ARCHIVE, out))
        for field in ("gpu", "capability", "packages", "cuda_runtime", "model_files"):
            assert old[field] == now[field], field
        trace = build_trace(args.stage)
        write_json(out / "trace.json", trace)
        write_json(out / "design.json", dict(stage=args.stage, candidate=args.candidate, arms=ARMS,
                   trace_sha256=sha256(out / "trace.json"), paired_seed_bootstrap_draws=4000,
                   bootstrap_rng=1900, mixed_gain_pct=3, max_segment_regression_pct=5,
                   timing="Enqueue to first generated token. Cycle includes all inference, transfer and wait, excludes disk IO.",
                   allocation="Arms share model, GPU pool and preallocated transfer tensors. A uses native GPU manager.",
                   screen="Disjoint seed, both workloads, eviction before hot replay, full 128 output tokens."))
        frozen = json.loads((ARCHIVE / "engine.json").read_text())
        config = dict(frozen["config"], cpu_prefix_first_use_blocks=32, cpu_prefix_transfer_mode="sync")
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        runner = llm.model_runner
        assert llm.config.cpu_prefix_transfer_mode == "sync" and llm.config.kvcache_block_size == 256
        assert runner.kv_cache_budget["actual_blocks"] == 36 and runner.kv_transfer.capacity == 64
        assert len(runner.kv_transfer.write_stages) == 2 and runner.kv_transfer.block_bytes == 36 * 1024**2
        assert runner.kv_cache.dtype == next(runner.model.parameters()).dtype == torch.bfloat16
        assert not runner.enforce_eager and runner.graph_bs == frozen["graph_batch_sizes"] and 4 in runner.graphs
        assert llm.scheduler.block_manager.namespace.hex() == frozen["namespace"]
        runner.kv_transfer.slots.fill(0)
        write_json(out / "engine.json", dict(config=config, kv_budget=runner.kv_cache_budget, cpu_slots=64,
                   engine_module=str(Path(nanovllm.__file__).resolve()), graph_batch_sizes=runner.graph_bs,
                   dtype=str(runner.kv_cache.dtype), namespace=frozen["namespace"],
                   modes={"A": "native GPU", "32": "sync", "candidate": args.candidate}))
        for section, cases in trace.items():
            for case in cases:
                measure_case(llm, out, case, section == "warmup", args.candidate)
        write_json(out / "summary.json", report(out))
        write_json(out / "status.json", dict(complete=True))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(complete=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


def paired(data, seeds):
    groups = {arm: np.array([[r["ttft_ms"] for r in rows if r["seed"] == s] for s in seeds])
              for arm, rows in data.items()}
    draws = np.random.default_rng(1900).integers(len(seeds), size=(4000, len(seeds)))
    p95 = {arm: np.percentile(a, 95) for arm, a in groups.items()}
    by_seed = {arm: np.percentile(a, 95, axis=1) for arm, a in groups.items()}
    boot = {arm: np.percentile(a[draws].reshape(len(draws), -1), 95, axis=1) for arm, a in groups.items()}
    result = {}
    for arm, base in (("32", "A"), ("candidate", "A"), ("candidate", "32")):
        bounds = np.percentile(percent(boot[arm], boot[base]), [2.5, 97.5])
        result[f"{arm}_vs_{base}"] = dict(change_pct=float(percent(p95[arm], p95[base])),
                paired_seed_95pct=bounds.tolist() if len(seeds) > 1 else None,
                by_seed_pct=dict(zip(seeds, percent(by_seed[arm], by_seed[base]).tolist(), strict=True)))
    return result


def report(out):
    design = json.loads((out / "design.json").read_text())
    trace = json.loads((out / "trace.json").read_text())
    assert design["trace_sha256"] == sha256(out / "trace.json")
    assert trace == json.loads(json.dumps(build_trace(design["stage"])))
    rows, steps, waves, cycles = (jsonl(out / f"{name}.jsonl") for name in ("requests", "steps", "waves", "cycles"))
    requests_by_key, steps_by_key = defaultdict(list), defaultdict(list)
    for r in rows:
        requests_by_key[wave_key(r)].append(r)
    for s in steps:
        steps_by_key[wave_key(s)].append(s)
    grouped = {wave_key(w): (requests_by_key[wave_key(w)], steps_by_key[wave_key(w)], w) for w in waves}
    assert set(grouped) == set(requests_by_key) == set(steps_by_key)
    expected_cycles, expected_waves = [], []
    for section, cases in trace.items():
        warm = section == "warmup"
        for c in cases:
            batches = {}
            for arm in c["arms"]:
                key = (c["seed"], c["cycle"], c["workload"], arm)
                expected_cycles.append((*key, warm))
                batches[arm] = {}
                for p, _ in c["phases"]:
                    k = (*key, p)
                    expected_waves.append(k)
                    rr, ss, w = grouped[k]
                    assert len(rr) == 4 and all(r["warmup"] == warm for r in [*rr, *ss, w])
                    batches[arm][p] = grouped[k]
            check_case(c, batches, 8 if warm else 128)
    assert [wave_key(r) for r in waves] == expected_waves
    assert [(r["seed"], r["cycle"], r["workload"], r["arm"], r["warmup"]) for r in cycles] == expected_cycles
    for arm in ARMS:
        def shapes(warm):
            return {(s["scheduled_tokens"], len(s["seq_ids"])) for s in steps
                    if s["is_prefill"] and s["arm"] == arm and s["warmup"] == warm}
        assert shapes(False) <= shapes(True)
    seeds = sorted({r["seed"] for r in rows if not r["warmup"]})
    loads = {}
    for load in ("mixed", "long"):
        def selected(records, arm):
            return [r for r in records if not r["warmup"] and r["arm"] == arm and r["workload"] == load]
        arms, comparisons = {}, {}
        for arm in ARMS:
            raw = selected(rows, arm)
            wall = sum(c["wall_s"] for c in selected(cycles, arm))
            arms[arm] = dict(groups={}, cycle_wall_s=wall, output_tokens_per_wall_s=sum(r["output_tokens"] for r in raw)/wall)
        for group in ("all", "fill", "post_fill", "hot", "no_reuse", "evicted", "evicted_before_hot"):
            data = {arm: subset(selected(rows, arm), group) for arm in ARMS}
            comparisons[group] = paired(data, seeds)
            for arm in ARMS:
                stats = summarize(data[arm], subset(selected(waves, arm), group))
                stats["full_cpu_restore_fraction"] = stats["full_cpu_restores"] / len(data[arm])
                arms[arm]["groups"][group] = stats
        loads[load] = dict(arms=arms, comparisons=comparisons)
    gains = loads["mixed"]["comparisons"]["all"]["candidate_vs_32"]
    formal = design["stage"] == "formal"
    gates = dict(mixed_gain=gains["change_pct"] <= -3,
                 mixed_interval=gains["paired_seed_95pct"][1] < 0 if formal else None)
    for load in loads:
        for group in ("no_reuse", "evicted"):
            gates[f"{load}_{group}"] = loads[load]["comparisons"][group]["candidate_vs_32"]["change_pct"] <= 5
    raw_files = ["cycles.jsonl", "steps.jsonl", "waves.jsonl", "requests.jsonl", "trace.json", "design.json", "engine.json", "environment.json"]
    return dict(workloads=loads, gates=gates, accepted=all(gates.values()) if formal else None,
                timed_requests=sum(not r["warmup"] for r in rows), seeds=seeds,
                native_shape_checked_cpu_requests=sum(not r["warmup"] and r["arm"] != "A" for r in rows),
                raw_sha256={n: sha256(out / n) for n in raw_files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--output")
    actions.add_argument("--summarize", help="audit existing data and print JSON")
    parser.add_argument("--stage", choices=("screen", "formal"), default="screen")
    parser.add_argument("--candidate", default="overlap")
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.transfer_comparison"
    if args.summarize:
        print(json.dumps(report(Path(args.summarize)), indent=2))
    else:
        measure(args)
