"""Paired first-use admission limits on mixed and all-2048 prefix traces."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.mixed_fill import make_case
from benchmarks.phase1 import save_wave, wave
from benchmarks.pressure import make_trace
from benchmarks.write_admission import select_arm


PREVIOUS = Path("results/20261119_mixed_fill_pressure36")
SEEDS = (1801, 1802, 1803, 1804)
ARMS = ("A", "24", "32", "36")
# Four-period Williams design, balanced for position and preceding arm.
ORDERS = (("A", "24", "36", "32"), ("24", "32", "A", "36"),
          ("32", "36", "24", "A"), ("36", "A", "32", "24"))


def case_for(seed, index, workload, cycle):
    if workload == "mixed":
        phases = make_case(seed, cycle, (0, 2, 3, 5)[index])["phases"]
    else:
        phases = [("pressure1" if p == "interference" else p, inputs) for p, inputs in make_trace(seed, cycle)]
    order = ORDERS[(index + 2 * cycle + int(workload == "long")) % 4]
    return dict(seed=seed, cycle=cycle, workload=workload, arms=order, phases=phases)


def cases():
    measured = []
    for i, seed in enumerate(SEEDS):
        for j in range(2):
            cycle = (i + j) % 2
            loads = ("mixed", "long") if cycle == 0 else ("long", "mixed")
            measured.extend(case_for(seed, i, load, cycle) for load in loads)
    return dict(warmup=[case_for(1800, 2, "mixed", 0), case_for(1800, 2, "long", 1)], formal=measured)


def validate_trace(trace):
    past = json.loads((PREVIOUS / "trace.json").read_text())
    frozen = json.loads(Path("results/20261118_pressure36/trace.json").read_text())
    roots = {tuple(p[:256]) for c in [past["warmup"], *past["formal"]] for _, prompts in c["phases"] for p in prompts}
    roots.update(tuple(p[:256]) for _, _, phases in frozen for _, prompts in phases for p in prompts)
    for c in trace["warmup"] + trace["formal"]:
        phases = dict(c["phases"])
        assert phases["hot"] == phases["evicted"]
        for fill, replay in zip(phases["fill"], phases["evicted"], strict=True):
            assert fill[:-128] == replay[:-128] and fill[-128:] != replay[-128:]
        lengths = [1152, 1152, 2176, 2176] if c["workload"] == "mixed" else [2176] * 4
        for p, prompts in c["phases"]:
            assert sorted(map(len, prompts)) == lengths
            if p in ("fill", "pressure1", "pressure2"):
                for prompt in prompts:
                    root = tuple(prompt[:256])
                    assert root not in roots
                    roots.add(root)


def normalized_steps(batch):
    rows, steps, _ = batch
    positions = {r["seq_id"]: i for i, r in enumerate(rows)}
    return [(s["is_prefill"], s["scheduled_tokens"], tuple(positions[i] for i in s["seq_ids"])) for s in steps]


def native_partial_reference(llm, out, c, output, tag):
    # Real GPU-only fill with three target roots and a disjoint fourth root.
    # This yields the 3-hit / 1-miss replay and its subsequent hot wave without
    # editing any GPU page or changing the four-request execution shape.
    select_arm(llm, "A")
    phases = dict(c["phases"])
    fill = phases["fill"][:3] + phases["pressure1"][3:]
    result = {}
    for phase, prompts in (("fill", fill), ("evicted", phases["evicted"]), ("hot", phases["hot"])):
        batch = wave(llm, prompts, 8 if phase == "fill" else output, dict(tag, arm="native_partial", phase=phase))
        save_wave(out, batch)
        result[phase] = batch
    assert [r["initial_gpu_hit_tokens"] for r in result["evicted"][0]] == [2048, 2048, 2048, 0]
    return result


def verify(c, measurements, partial, output):
    for arm in ARMS:
        for phase, prompts in c["phases"]:
            batch = measurements[arm][phase]
            ref = measurements["A"]["hot" if phase == "evicted" and arm != "A" else phase]
            if partial and arm == "24" and phase in ("evicted", "hot"):
                ref = partial[phase]
            assert normalized_steps(batch) == normalized_steps(ref), (c["seed"], arm, phase, "steps")
            for i, (r, native, prompt) in enumerate(zip(batch[0], ref[0], prompts, strict=True)):
                key = c["seed"], c["cycle"], c["workload"], arm, phase, i
                prefix = len(prompt) - 128
                gpu = prefix if phase == "hot" else 0
                cpu = prefix if phase == "evicted" and arm != "A" else 0
                if partial and arm == "24" and i == 3:
                    cpu = 0
                assert r["input_token_ids"] == native["input_token_ids"] == prompt, key
                assert r["output_token_ids"] == native["output_token_ids"], (*key, "greedy")
                assert r["initial_gpu_hit_tokens"] == gpu and r["initial_cpu_hit_tokens"] == cpu, key
                assert r["cpu_restored_tokens"] == r["h2d_blocks"] * 256 == cpu, key
                assert r["prompt_computed_tokens"] == r["prefill_tokens"] == len(prompt) - gpu - cpu, key
                assert r["output_tokens"] == len(r["token_s"]) == len(r["output_token_ids"]) == output, key
                assert r["decode_tokens"] == output - 1 and r["allocations"] == 1, key
                assert r["preemptions"] == r["cpu_save_failures"] == r["cpu_restore_failures"] == 0, key
                assert abs(r["ttft_ms"] - 1000 * (r["token_s"][0] - r["enqueue_s"])) < 1e-5, key
                assert not r["restore_decisions"] and (arm != "A" or r["d2h_blocks"] == 0), key


def run_case(llm, out, c, warmup):
    tag = {k: c[k] for k in ("seed", "cycle", "workload")}
    tag["warmup"] = warmup
    output = 8 if warmup else 128
    measurements = {}
    for arm in c["arms"]:
        select_arm(llm, "A" if arm == "A" else "batch")
        manager = llm.scheduler.block_manager
        if arm != "A":
            assert manager.first_use_limit == 36 and manager.transfer.capacity == 64 and not manager.entries
            manager.first_use_limit = int(arm)
        torch.manual_seed(c["seed"])
        start = perf_counter()
        batches = {p: wave(llm, inputs, output, dict(tag, arm=arm, phase=p)) for p, inputs in c["phases"]}
        elapsed = perf_counter() - start
        for batch in batches.values():
            save_wave(out, batch)
        write_jsonl(out / "cycles.jsonl", [dict(tag, arm=arm, wall_s=elapsed)])
        measurements[arm] = batches
        print(f"{c['seed']} {c['workload']} order={c['cycle']} arm={arm} {elapsed:.2f}s", flush=True)
    partial = None
    if c["workload"] == "long" and c["cycle"] == 1:
        partial = native_partial_reference(llm, out / "reference", c, output, tag)
    verify(c, measurements, partial, output)
    print(f"{c['seed']} {c['workload']} order={c['cycle']} core accounting and native greedy checked", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.module = "benchmarks.first_use_budget"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        previous, current = (json.loads((p / "environment.json").read_text()) for p in (PREVIOUS, out))
        for name in ("gpu", "capability", "packages", "cuda_runtime", "model_files", "git_head"):
            assert previous[name] == current[name], name
        for name, digest in previous["source_sha256"].items():
            if name.startswith("nanovllm/"):
                assert current["source_sha256"][name] == digest, name
        trace = cases()
        validate_trace(trace)
        write_json(out / "trace.json", trace)
        write_json(out / "design.json", dict(seeds=SEEDS, iterations=2, arms=ARMS, orders=ORDERS,
                   trace_sha256=sha256(out / "trace.json"), reference_run=str(PREVIOUS),
                   admission="fresh GPU/CPU indices each arm and cycle; set first_use_limit once before enqueue",
                   no_reuse="all disjoint pressure waves, also report first separately",
                   goals_pct=dict(no_reuse_vs_A_max=5, evicted_vs_A_max=-20),
                   selection="smallest tested budget passing both pooled gates and retaining every early replay in both loads",
                   uncertainty="paired whole-seed bootstrap, 4000 draws with RNG 1800",
                   timing="four queued requests, first real greedy token; reference and file IO outside timing",
                   partial_reference="native three target prefixes and disjoint fourth, then two real replay waves"))
        engine = json.loads((PREVIOUS / "engine.json").read_text())
        torch.manual_seed(0)
        llm = LLM(args.model, **engine["config"])
        runner = llm.model_runner
        assert llm.config.kvcache_block_size == 256 and runner.kv_cache_budget["actual_blocks"] == 36
        assert runner.kv_transfer.capacity == 64 and len(runner.kv_transfer.write_stages) == 2
        assert runner.kv_cache.dtype == next(runner.model.parameters()).dtype == torch.bfloat16
        assert runner.graph_bs == engine["graph_batch_sizes"] and 4 in runner.graphs and not runner.enforce_eager
        assert llm.scheduler.block_manager.namespace.hex() == engine["namespace"]
        write_json(out / "engine.json", dict(engine, kv_budget=runner.kv_cache_budget,
                   first_use_limit_by_arm={"A": None, "24": 24, "32": 32, "36": 36},
                   prefix_tokens_by_workload=dict(mixed=[1024, 2048], long=[2048])))
        (out / "reference").mkdir()
        runner.kv_transfer.slots.fill(0)
        for group, items in trace.items():
            for c in items:
                run_case(llm, out, c, group == "warmup")
        write_json(out / "status.json", dict(measurement_complete=True, workloads=2, seeds=len(SEEDS),
                   formal_requests=1152, formal_cpu_requests_with_native_reference=864))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(measurement_complete=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
