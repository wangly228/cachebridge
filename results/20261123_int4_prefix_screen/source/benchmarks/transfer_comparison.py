"""Compare native GPU, a 32-block CPU baseline and a candidate transfer mode."""
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


def build_trace(stage, seed_base=1900):
    # The two historical warmups cover every timed prefill batch shape.
    historical = json.loads((ARCHIVE / "trace.json").read_text())
    trace = {"warmup": [dict(c, arms=ARMS) for c in historical["warmup"]], "timed": []}
    seeds = [seed_base - 1] if stage == "screen" else list(range(seed_base + 1, seed_base + 7))
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
    for arm in batches:
        cpu_arm = arm in ("32", "candidate")
        native_arm = "A_packed" if "A_packed" in batches and arm in ("candidate", "A_packed") else "A"
        for phase, prompts in case["phases"]:
            rows, steps, w = batches[arm][phase]
            restored = phase == "evicted" and cpu_arm
            ref = batches[native_arm]["hot" if restored else phase]
            assert normalized_steps((rows, steps, w)) == normalized_steps(ref), (arm, phase)
            assert not w["profiler_enabled"] and w["used_blocks_peak"] <= 36
            assert w["cpu_cache_enabled"] == cpu_arm
            if cpu_arm:
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
                assert row["d2h_blocks"] == (baseline["d2h_blocks"] if cpu_arm else 0), key
                if phase == "fill" and cpu_arm:
                    assert row["d2h_blocks"] * 256 == prefix, key
        written = sum(r["d2h_blocks"] for rows, _, _ in batches[arm].values() for r in rows)
        assert written == (32 if cpu_arm else 0), (arm, written)


def measure_case(llm, out, c, warm, baseline, candidate):
    output = 8 if warm else 128
    tag = dict(seed=c["seed"], cycle=c["cycle"], workload=c["workload"], warmup=warm)
    results = {}
    arms = list(c["arms"])
    packing = candidate == "cold_pack"
    if packing:
        arms.append("A_packed")
    for arm in arms:
        native = arm in ("A", "A_packed")
        mode = "sync" if native else (candidate if arm == "candidate" and not packing else baseline)
        llm.config.cpu_prefix_transfer_mode = mode
        if packing:
            llm.config.cpu_prefix_pack_cold = arm in ("candidate", "A_packed")
        select_arm(llm, "A" if native else "batch")
        manager = llm.scheduler.block_manager
        tag["transfer_mode"] = "native GPU" if native else manager.transfer_mode
        if packing:
            assert llm.scheduler.pack_cold == llm.config.cpu_prefix_pack_cold
            tag["pack_cold"] = llm.scheduler.pack_cold
        assert not manager.used_block_ids
        if not native:
            assert manager.transfer_mode == mode
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
        destination = out / "reference" if arm == "A_packed" else out
        for batch in batches.values():
            save_wave(destination, batch)
        write_jsonl(destination / "cycles.jsonl", [dict(tag, arm=arm, wall_s=wall)])
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
        packing = args.candidate == "cold_pack"
        if packing:
            (out / "reference").mkdir()
        trace = build_trace(args.stage, args.seed_base)
        write_json(out / "trace.json", trace)
        write_json(out / "design.json", dict(stage=args.stage, baseline=args.baseline, candidate=args.candidate,
                   seed_base=args.seed_base, arms=ARMS,
                   reference="native GPU with packed cold prefill after each three-arm case" if packing else None,
                   trace_sha256=sha256(out / "trace.json"), paired_seed_bootstrap_draws=4000,
                   bootstrap_rng=1900, mixed_gain_pct=3, max_segment_regression_pct=5,
                   timing="Enqueue to first generated token. Cycle includes all inference, transfer and wait, excludes disk IO.",
                   allocation="Arms share model, GPU pool and preallocated transfer tensors. A uses native GPU manager.",
                   screen="Disjoint seed, both workloads, eviction before hot replay, full 128 output tokens."))
        frozen = json.loads((ARCHIVE / "engine.json").read_text())
        config = dict(frozen["config"], cpu_prefix_first_use_blocks=32, cpu_prefix_transfer_mode="sync")
        if packing:
            config["cpu_prefix_pack_cold"] = False
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
                   modes={"A": "native GPU", "32": args.baseline,
                          "candidate": args.baseline if packing else args.candidate},
                   pack_cold_by_arm={"A": False, "32": False, "candidate": True, "A_packed": True} if packing else {}))
        for section, cases in trace.items():
            for case in cases:
                measure_case(llm, out, case, section == "warmup", args.baseline, args.candidate)
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
    assert trace == json.loads(json.dumps(build_trace(design["stage"], design.get("seed_base", 1900))))
    names = ("requests", "steps", "waves", "cycles")
    rows, steps, waves, cycles = (jsonl(out / f"{name}.jsonl") for name in names)
    packing = design["candidate"] == "cold_pack"
    ref_rows, ref_steps, ref_waves, ref_cycles = ([], [], [], [])
    if packing:
        ref_rows, ref_steps, ref_waves, ref_cycles = (jsonl(out / "reference" / f"{name}.jsonl") for name in names)
        for raw in (ref_rows, ref_steps, ref_waves, ref_cycles):
            assert all(r["arm"] == "A_packed" and r["pack_cold"] and r["transfer_mode"] == "native GPU" for r in raw)
    if "baseline" in design:
        modes = json.loads((out / "engine.json").read_text())["modes"]
        assert modes == {"A": "native GPU", "32": design["baseline"],
                         "candidate": design["baseline"] if packing else design["candidate"]}
        for records in (rows, steps, waves, cycles):
            assert all(r["transfer_mode"] == modes[r["arm"]] for r in records)
            if packing:
                assert all(r["pack_cold"] == (r["arm"] == "candidate") for r in records)
    requests_by_key, steps_by_key = defaultdict(list), defaultdict(list)
    for r in rows + ref_rows:
        requests_by_key[wave_key(r)].append(r)
    for s in steps + ref_steps:
        steps_by_key[wave_key(s)].append(s)
    grouped = {wave_key(w): (requests_by_key[wave_key(w)], steps_by_key[wave_key(w)], w) for w in waves + ref_waves}
    assert set(grouped) == set(requests_by_key) == set(steps_by_key)
    expected_cycles, expected_waves = [], []
    expected_ref_cycles, expected_ref_waves = [], []
    cross_shape = defaultdict(lambda: {"requests": 0, "different_greedy_outputs": 0, "different_step_layout": 0})
    for section, cases in trace.items():
        warm = section == "warmup"
        for c in cases:
            batches = {}
            arms = list(c["arms"]) + (["A_packed"] if packing else [])
            for arm in arms:
                key = (c["seed"], c["cycle"], c["workload"], arm)
                if arm == "A_packed":
                    expected_ref_cycles.append((*key, warm))
                else:
                    expected_cycles.append((*key, warm))
                batches[arm] = {}
                for p, _ in c["phases"]:
                    k = (*key, p)
                    if arm == "A_packed":
                        expected_ref_waves.append(k)
                    else:
                        expected_waves.append(k)
                    rr, ss, w = grouped[k]
                    assert len(rr) == 4 and all(r["warmup"] == warm for r in [*rr, *ss, w])
                    batches[arm][p] = grouped[k]
            check_case(c, batches, 8 if warm else 128)
            if packing and not warm:
                for phase, _ in c["phases"]:
                    original = batches["32"][phase]
                    new = batches["candidate"][phase]
                    counts = cross_shape[c["workload"]]
                    counts["requests"] += len(new[0])
                    if normalized_steps(original) != normalized_steps(new):
                        counts["different_step_layout"] += len(new[0])
                    counts["different_greedy_outputs"] += sum(
                        a["output_token_ids"] != b["output_token_ids"] for a, b in zip(original[0], new[0], strict=True))
    assert [wave_key(w) for w in ref_waves] == expected_ref_waves
    assert [(r["seed"], r["cycle"], r["workload"], r["arm"], r["warmup"]) for r in ref_cycles] == expected_ref_cycles
    assert [wave_key(r) for r in waves] == expected_waves
    assert [(r["seed"], r["cycle"], r["workload"], r["arm"], r["warmup"]) for r in cycles] == expected_cycles
    for arm in (*ARMS, "A_packed"):
        arm_steps = ref_steps if arm == "A_packed" else steps
        def shapes(warm):
            return {(s["scheduled_tokens"], len(s["seq_ids"])) for s in arm_steps
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
    extra = {}
    if packing:
        # Native packed runs follow the timed arms, so their timing is descriptive.
        # Their primary purpose is the same-shape greedy and accounting reference.
        extra["native_packed_reference"] = {
            load: summarize([r for r in ref_rows if not r["warmup"] and r["workload"] == load],
                            [w for w in ref_waves if not w["warmup"] and w["workload"] == load])
            for load in ("mixed", "long")
        }
        extra["candidate_vs_current_greedy"] = dict(cross_shape)
        raw_files += [f"reference/{n}.jsonl" for n in names]
    return dict(**extra, workloads=loads, gates=gates, accepted=all(gates.values()) if formal else None,
                timed_requests=sum(not r["warmup"] for r in rows), seeds=seeds,
                native_shape_checked_cpu_requests=sum(not r["warmup"] and r["arm"] != "A" for r in rows),
                raw_sha256={n: sha256(out / n) for n in raw_files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--output")
    actions.add_argument("--summarize", help="audit existing data and print JSON")
    parser.add_argument("--stage", choices=("screen", "formal"), default="screen")
    parser.add_argument("--baseline", choices=("sync", "overlap"), default="sync",
                        help="transfer mode for the current 32-block CPU arm")
    parser.add_argument("--candidate", default="overlap")
    parser.add_argument("--seed-base", type=int, default=1900,
                        help="screen uses base - 1, formal uses base + 1 through base + 6")
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.transfer_comparison"
    if args.summarize:
        print(json.dumps(report(Path(args.summarize)), indent=2))
    else:
        if args.candidate == "cold_pack" and args.baseline != "overlap":
            parser.error("cold_pack requires --baseline overlap")
        if args.baseline == args.candidate:
            parser.error("--candidate and --baseline must identify different modes")
        if args.seed_base < 1 or args.seed_base - 1 == 1800 or 1800 in range(args.seed_base + 1, args.seed_base + 7):
            parser.error("choose a seed base with nonnegative seeds disjoint from warmup seed 1800")
        measure(args)
