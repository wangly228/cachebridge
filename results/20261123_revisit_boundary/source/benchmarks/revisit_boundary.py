"""Measure cold-start whole-sequence p95 against the fraction revisited after GPU eviction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from time import perf_counter
import traceback

import numpy as np
import torch

from nanovllm import LLM
from benchmarks.common import ROOT, prepare_run, sha256, write_json, write_jsonl
from benchmarks.first_use_budget import normalized_steps
from benchmarks.phase1 import save_wave, wave
from benchmarks.write_admission import jsonl, select_arm


ARCHIVE = Path("results/20261119_first_use_budget36")
LEVELS = (0, 3, 6)  # Number of four-request revisit waves in a nine-wave cold run.
ARMS = ("A", "CPU")


def trace_for(case, level):
    phases = dict(case["phases"])
    groups = [phases[p] for p in ("fill", "pressure1", "pressure2")]
    rng = Random(f"cachebridge/revisit-v1/{case['seed']}")
    waves = list(groups)
    for number in range(3, 9):
        revisit = level == 6 or (level == 3 and number % 2 == case["seed"] % 2)
        inputs = []
        for original in groups[number % 3]:
            fresh = [rng.randrange(100, 10000) for _ in range(len(original))]
            inputs.append(original[:-128] + fresh[-128:] if revisit else fresh)
        waves.append(inputs)
    seen = set()
    replay_count = 0
    for inputs in waves:
        assert sorted(map(len, inputs)) == [1152, 1152, 2176, 2176]
        for prompt in inputs:
            key = tuple(prompt[:-128])
            replay_count += key in seen
            seen.add(key)
    assert replay_count == level * 4
    return waves


def make_design(seeds):
    trace = json.loads((ARCHIVE / "trace.json").read_text())
    cases = {c["seed"]: c for c in trace["formal"] if c["workload"] == "mixed" and c["cycle"] == 0}
    runs = []
    # Include the partial-restore shapes (1408,3), (384,3) and (256,2)
    # seen in a metadata-only replay. Offset complete token IDs to keep
    # warmup prefixes disjoint from every formal input; retain FIFO layouts.
    for source, level in ((1801, 6), (1802, 6), (1804, 3)):
        case = cases[source]
        phases = [(p, [[v+12000 for v in prompt] for prompt in inputs])
                  for p, inputs in case["phases"]]
        runs.append((dict(case, seed=source+10000, phases=phases), level, True, ARMS))
    for i, seed in enumerate(seeds):
        for j in range(3):
            level = LEVELS[(i + j) % 3]
            order = ARMS if (i + LEVELS.index(level)) % 2 == 0 else ARMS[::-1]
            runs.append((cases[seed], level, False, order))
    return [dict(seed=case["seed"], revisit_waves=level, warmup=warmup, arms=order,
                 inputs=trace_for(case, level)) for case, level, warmup, order in runs]


def check_pair(run, measured):
    exact = 0
    for i, inputs in enumerate(run["inputs"]):
        a, cpu = (measured[arm][i] for arm in ARMS)
        if normalized_steps(a) == normalized_steps(cpu) and all(
                r["prefill_tokens"] == s["prefill_tokens"] for r, s in zip(a[0], cpu[0], strict=True)):
            assert [r["output_token_ids"] for r in a[0]] == [r["output_token_ids"] for r in cpu[0]]
            exact += len(inputs)
    return exact


def execute(llm, out, run, mode):
    batches = {}
    for arm in run["arms"]:
        select_arm(llm, "A" if arm == "A" else "batch")
        manager = llm.scheduler.block_manager
        if arm == "CPU":
            assert manager.first_use_limit == 32 and not manager.entries
        assert len(manager.free_block_ids) == 36 and not manager.hash_to_block_id
        torch.manual_seed(run["seed"])
        tag = {k: run[k] for k in ("seed", "revisit_waves", "warmup")}
        tag.update(arm=arm, transfer_mode="disabled" if arm == "A" else mode)
        output = 8 if run["warmup"] else 128
        t0 = perf_counter()
        group = []
        for i, inputs in enumerate(run["inputs"]):
            group.append(wave(llm, inputs, output, dict(tag, wave_index=i)))
            assert getattr(manager, "pending_write", None) is None
        elapsed = perf_counter() - t0
        for batch in group:
            save_wave(out, batch)
        write_jsonl(out / "cycles.jsonl", [dict(tag, elapsed_s=elapsed)])
        print(f"{tag}, elapsed={elapsed:.2f}s", flush=True)
        batches[arm] = group
    return check_pair(run, batches)


def records_for(rows, wanted):
    return [row for row in rows if not row["warmup"] and row["revisit_waves"] == wanted]


def report(out):
    requests = jsonl(out / "requests.jsonl")
    waves = jsonl(out / "waves.jsonl")
    prefill_shapes = {True: set(), False: set()}
    with (out / "steps.jsonl").open() as file:
        for line in file:
            step = json.loads(line)
            if step["is_prefill"]:
                prefill_shapes[step["warmup"]].add((step["scheduled_tokens"], len(step["seq_ids"])))
    assert prefill_shapes[False] <= prefill_shapes[True], prefill_shapes[False] - prefill_shapes[True]
    cycles = jsonl(out / "cycles.jsonl")
    design = json.loads((out / "design.json").read_text())
    trace = json.loads((out / "trace.json").read_text())
    assert sha256(out / "trace.json") == design["trace_sha256"]
    expected = {}
    for run in trace:
        seen = set()
        for i, prompts in enumerate(run["inputs"]):
            for position, p in enumerate(prompts):
                prefix = tuple(p[:-128])
                for arm in ARMS:
                    key = run["seed"], run["revisit_waves"], arm, i, position
                    expected[key] = (p, prefix in seen, run["warmup"])
                seen.add(prefix)
    counts = defaultdict(int)
    for r in requests:
        key = tuple(r[k] for k in ("seed", "revisit_waves", "arm", "wave_index"))
        p, revisit, warm = expected.pop((*key, counts[key]))
        counts[key] += 1
        assert p == r["input_token_ids"] and r["warmup"] == warm
        assert r["initial_gpu_hit_tokens"] == 0, (key, r["initial_gpu_hit_tokens"])
        assert r["allocations"] == 1 and r["output_tokens"] == len(r["output_token_ids"]) == (8 if warm else 128)
        assert abs(r["ttft_ms"] - 1000 * (r["token_s"][0] - r["enqueue_s"])) < 1e-5
        assert r["initial_cpu_hit_tokens"] == r["cpu_restored_tokens"] == 256 * r["h2d_blocks"]
        assert r["prefill_tokens"] == r["prompt_computed_tokens"] == len(p) - r["cpu_restored_tokens"]
        assert r["preemptions"] == r["cpu_restore_failures"] == r["cpu_save_failures"] == 0
        if not revisit or r["arm"] == "A":
            assert r["h2d_blocks"] == 0
        if r["arm"] == "A":
            assert r["d2h_blocks"] == 0
        r["evicted_revisit"] = revisit
    assert not expected and all(n == 4 for n in counts.values())
    assert len(waves) == len(counts) and len(cycles) == len(trace) * 2
    seeds = design["seeds"]
    draws = np.random.default_rng(20261123).integers(len(seeds), size=(4000, len(seeds)))
    results = []
    for level in LEVELS:
        rows = records_for(requests, level)
        info = {"revisit_waves": level, "arms": {}}
        boot = {}
        for arm in ARMS:
            sample = [r for r in rows if r["arm"] == arm]
            group = np.array([[r["ttft_ms"] for r in sample if r["seed"] == seed] for seed in seeds])
            assert group.shape == (len(seeds), 36)
            boot[arm] = np.percentile(group[draws].reshape(len(draws), -1), 95, axis=1)
            revisits = [r for r in sample if r["evicted_revisit"]]
            assert len(revisits) == 4 * level * len(seeds)
            wall_s = sum(c["elapsed_s"] for c in records_for(cycles, level) if c["arm"] == arm)
            mem = [w["memory"] for w in records_for(waves, level) if w["arm"] == arm]
            info["arms"][arm] = dict(requests=len(sample), evicted_revisits=len(revisits),
                actual_evicted_revisit_pct=100 * len(revisits) / len(sample),
                whole_p95_ms=float(np.percentile(group, 95)), wall_s=wall_s,
                whole_output_tokens_per_s=sum(r["output_tokens"] for r in sample) / wall_s,
                p95_ms_by_seed=dict(zip(seeds, np.percentile(group, 95, axis=1).tolist(), strict=True)),
                restored_blocks=sum(r["h2d_blocks"] for r in sample), written_blocks=sum(r["d2h_blocks"] for r in sample),
                restored_full_revisits=sum(r["cpu_restored_tokens"] == r["prompt_tokens"]-128 for r in revisits),
                revisited_prefix_blocks=sum((r["prompt_tokens"]-128)//256 for r in revisits),
                first_wave_d2h_blocks=sum(r["d2h_blocks"] for r in sample if r["wave_index"] == 0),
                peak_gpu_allocated_bytes=max(m["gpu_peak_allocated"] for m in mem),
                peak_cpu_rss_bytes=max(m["cpu_peak_rss"] for m in mem))
        a, b = [info["arms"][k]["whole_p95_ms"] for k in ARMS]
        delta = 100 * (b / a - 1)
        ci = np.percentile(100 * (boot["CPU"] / boot["A"] - 1), [2.5, 97.5]).tolist()
        info.update(whole_change_pct=delta, paired_seed_95pct_interval=ci,
                    pooled_improvement=delta < 0, interval_entirely_below_zero=ci[1] < 0)
        results.append(info)
    return dict(levels=results, seeds=seeds, cpu_mode=design["mode"],
                formal_prefill_shapes=sorted(prefill_shapes[False]), warmup_prefill_shapes=sorted(prefill_shapes[True]),
                raw_sha256={p: sha256(out / p) for p in
                ("trace.json", "cycles.jsonl", "requests.jsonl", "steps.jsonl", "waves.jsonl", "engine.json", "design.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--summarize")
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--mode", default="sync", help="selected transfer mode; sync uses the default implementation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1801, 1802, 1803, 1804])
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(report(Path(args.summarize)), indent=2))
        return
    assert len(set(args.seeds)) == len(args.seeds) and set(args.seeds) <= {1801, 1802, 1803, 1804}
    args.module = "benchmarks.revisit_boundary"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        import nanovllm
        assert Path(nanovllm.__file__).resolve().parent == ROOT / "nanovllm"
        reference, current = [json.loads((p / "environment.json").read_text()) for p in (ARCHIVE, out)]
        for field in ("gpu", "capability", "packages", "cuda_runtime"):
            assert reference[field] == current[field], field
        for files in [reference["model_files"], current["model_files"]]:
            for file in files:
                file.pop("download_metadata", None)
        assert reference["model_files"] == current["model_files"]
        runs = make_design(args.seeds)
        write_json(out / "trace.json", runs)
        write_json(out / "design.json", dict(seeds=args.seeds, mode=args.mode, trace_sha256=sha256(out/"trace.json"),
               input_source=str(ARCHIVE/"trace.json"), frozen_input_sha256=sha256(ARCHIVE/"trace.json"),
               endpoint="p95 over every request in complete 9-wave cold sequences, enqueue to first real token",
               layout="4 concurrent, two 1024 and two 2048 prefixes per wave, 128-token tails and 128 outputs",
               levels="0, 3, 6 revisit waves out of 9; three distinct prefix groups cycle with two intervening waves",
               warmup="disjoint warm seeds 11801, 11802, 11804 cover partial-restore shapes; 8 outputs, both arms, fresh indices",
               uncertainty="paired whole-seed percentile bootstrap 4000 draws, RNG 20261123; descriptive with four seeds"))
        prior = json.loads((ARCHIVE / "engine.json").read_text())
        kwargs = dict(prior["config"], cpu_prefix_first_use_blocks=32)
        if args.mode != "sync":
            kwargs["cpu_prefix_transfer_mode"] = args.mode
        torch.manual_seed(0)
        llm = LLM(args.model, **kwargs)
        runner = llm.model_runner
        assert runner.kv_cache_budget["actual_blocks"] == 36 and runner.kv_transfer.capacity == 64
        assert runner.kv_cache.dtype == next(runner.model.parameters()).dtype == torch.bfloat16
        assert not runner.enforce_eager and runner.graph_bs == prior["graph_batch_sizes"]
        assert llm.scheduler.block_manager.first_use_limit == 32 and llm.config.kvcache_block_size == 256
        if args.mode != "sync":
            assert llm.config.cpu_prefix_transfer_mode == args.mode
        write_json(out / "engine.json", dict(config=kwargs, actual_gpu_budget=runner.kv_cache_budget,
                   cpu_slots=runner.kv_transfer.capacity, block_tokens=llm.config.kvcache_block_size,
                   graph_batch_sizes=runner.graph_bs, dtype=str(runner.kv_cache.dtype),
                   engine_import=str(Path(nanovllm.__file__).resolve()), profiler=False))
        runner.kv_transfer.slots.fill(0)
        exact = 0
        for run in runs:
            matches = execute(llm, out, run, args.mode)
            if not run["warmup"]:
                exact += matches
        write_json(out / "summary.json", report(out))
        write_json(out / "status.json", dict(measurement_complete=True, same_shape_greedy_comparisons=exact))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
