"""Cold-fill variability and synchronized Runner attribution on two frozen waves."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import nanovllm
from nanovllm import LLM
from nanovllm.utils.context import reset_context
from benchmarks.common import ROOT, prepare_run, sha256, write_json, write_jsonl
from benchmarks.first_use_budget import normalized_steps
from benchmarks.phase1 import save_wave, wave
from benchmarks.write_admission import select_arm


ARCHIVE = Path("results/20261119_first_use_budget36")
CASES = ((1803, 0), (1804, 1))
SECTIONS = ("input", "forward_logits", "sampling", "token_return")


def design():
    for comparison, left, right in (("repeat32", "32a", "32b"), ("budget", "32", "36"), ("native", "32", "A")):
        for detailed, repeats in ((False, 4), (True, 2)):
            for repeat in range(repeats):
                for index, (seed, cycle) in enumerate(CASES):
                    order = (left, right) if (repeat + index) % 2 == 0 else (right, left)
                    for arm in order:
                        yield dict(comparison=comparison, detailed=detailed, repeat=repeat, arm=arm,
                                   seed=seed, cycle=cycle, phase="fill", warmup=False)


def collect(llm, inputs, tag):
    runner = llm.model_runner
    llm.config.cpu_prefix_first_use_blocks = 36 if tag["arm"] == "36" else 32
    select_arm(llm, "A" if tag["arm"] == "A" else "batch")
    transfer = runner.kv_transfer
    manager = llm.scheduler.block_manager
    assert not manager.used_block_ids
    if tag["arm"] != "A":
        assert not manager.entries and manager.first_use_limit == llm.config.cpu_prefix_first_use_blocks
    torch.manual_seed(tag["seed"])
    spans = []

    def timed_run(seqs, prefill):
        if not prefill:
            return original(seqs, prefill)
        times = [perf_counter()]
        ids, positions = runner.prepare_prefill(seqs)
        temperatures = runner.prepare_sample(seqs)
        torch.cuda.synchronize()
        times.append(perf_counter())
        logits = runner.run_model(ids, positions, True)
        torch.cuda.synchronize()
        times.append(perf_counter())
        sampled = runner.sampler(logits, temperatures)
        torch.cuda.synchronize()
        times.append(perf_counter())
        output = sampled.tolist()
        times.append(perf_counter())
        reset_context()
        spans.append(dict(kind="prefill", seq_ids=[s.seq_id for s in seqs],
                          blocks=[s.block_table.copy() for s in seqs], times=times))
        return output

    def timed_save(pairs):
        start = perf_counter()
        save(pairs)
        spans.append(dict(kind="save", pairs=list(pairs), times=[start, perf_counter()]))

    if tag["detailed"]:
        original = runner.run
        save = transfer.save_many
        runner.run = timed_run
        transfer.save_many = timed_save
    try:
        batch = wave(llm, inputs, 8 if tag["warmup"] else 128, tag)
    finally:
        if tag["detailed"]:
            del runner.run, transfer.save_many
    if spans:
        origin = llm.step_metrics[0]["start_time"] - batch[1][0]["start_s"]
        for s in spans:
            s["times"] = [t - origin for t in s["times"]]
    return batch, spans


def read_rows(out, filename):
    return [json.loads(line) for line in (out / filename).read_text().splitlines()]


def summarize(out):
    all_rows = read_rows(out, "requests.jsonl")
    rows = [r for r in all_rows if not r["warmup"]]
    waves = [r for r in read_rows(out, "waves.jsonl") if not r["warmup"]]
    spans = read_rows(out, "timeline.jsonl")
    selectors = list(design())
    assert len(rows) == len(selectors) * 4 and len(waves) == len(selectors)
    assert all(all(w[k] == v for k, v in s.items()) for w, s in zip(waves, selectors, strict=True))
    groups = {}
    def match(row, selector):
        return all(row[k] == v for k, v in selector.items())
    def p95(values):
        return float(np.percentile(values, 95))
    comparisons = {}
    for s in selectors:
        key = s["comparison"], s["detailed"], s["arm"]
        groups.setdefault(key, []).extend([r for r in rows if match(r, s)])
    for (comp, detailed, arm), selected in groups.items():
        key = f"{comp}/{'diagnostic' if detailed else 'plain'}"
        target = comparisons.setdefault(key, {})
        target[arm] = dict(n=len(selected), p95_ms=p95([r["ttft_ms"] for r in selected]),
                          mean_write_ms_per_wave=sum(r["d2h_ms"] for r in selected)/(len(selected)/4))
        if detailed:
            events = [v for v in spans if match(v, dict(comparison=comp, detailed=True, arm=arm))]
            nr = len(selected)/4
            prefill = [v for v in events if v["kind"] == "prefill"]
            writes = [v for v in events if v["kind"] == "save"]
            target[arm]["mean_prefill_ms_per_wave"] = {
                part: sum((v["times"][i+1]-v["times"][i])*1000 for v in prefill)/nr
                for i, part in enumerate(SECTIONS)}
            target[arm]["mean_timed_write_ms_per_wave"] = sum(
                (v["times"][-1]-v["times"][0])*1000 for v in writes)/nr
    pair_stats = {}
    for comp, left, right in (("repeat32","32a","32b"), ("budget","32","36"), ("native","32","A")):
        for detailed in (False, True):
            deltas = []
            for seed, cycle in CASES:
                for rep in range(2 if detailed else 4):
                    key = dict(seed=seed, cycle=cycle, repeat=rep, comparison=comp, detailed=detailed)
                    def arm_p95(arm):
                        return p95([r["ttft_ms"] for r in rows if match(r, dict(key, arm=arm))])
                    deltas.append(arm_p95(left)-arm_p95(right))
            pair_stats[f"{comp}/{'diagnostic' if detailed else 'plain'}"] = dict(
                numerator=left, reference=right, paired_wave_p95_delta_ms=deltas,
                median_ms=float(np.median(deltas)), range_ms=[min(deltas),max(deltas)])
    # Compare each real write with the next prefill compute that could proceed
    # while its source pages and destination slots are kept reserved.
    windows = []
    for s in selectors:
        if not s["detailed"] or s["arm"] == "A":
            continue
        events = [v for v in spans if match(v, s)]
        for i, v in enumerate(events):
            if v["kind"] != "save":
                continue
            next_prefill = next((w for w in events[i+1:] if w["kind"]=="prefill"), None)
            if next_prefill:
                windows.append(dict(**s, blocks=len(v["pairs"]),
                                    write_ms=1000*(v["times"][1]-v["times"][0]),
                                    following_forward_ms=1000*(next_prefill["times"][2]-next_prefill["times"][1])))
    return dict(formal_waves=len(waves), formal_requests=len(rows), results=comparisons,
                paired_wave_variability=pair_stats, writes_followed_by_prefill=windows,
                verified_gpu_outputs=True,
                raw_sha256={f:sha256(out/f) for f in ("requests.jsonl","waves.jsonl","steps.jsonl","timeline.jsonl","design.json","engine.json")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--summarize", type=Path)
    args = p.parse_args()
    if args.summarize:
        write_json(args.summarize / "summary.json", summarize(args.summarize))
        return
    assert Path(nanovllm.__file__).resolve().parent.parent == ROOT == Path.cwd()
    args.module = "benchmarks.cold_fill_probe"
    out = prepare_run(args.output, args, args.model)
    current_env = json.loads((out/"environment.json").read_text())
    baseline_env = json.loads((ARCHIVE/"environment.json").read_text())
    for k in ("gpu","capability","packages","model_files","cuda_runtime"):
        assert current_env[k] == baseline_env[k], k
    trace = json.loads((ARCHIVE / "trace.json").read_text())
    inputs = {(c["seed"], c["cycle"]):dict(c["phases"])["fill"] for c in trace["formal"]
              if c["workload"]=="mixed" and (c["seed"],c["cycle"]) in CASES}
    plan = list(design())
    write_json(out/"design.json", dict(trace=str(ARCHIVE/"trace.json"), trace_sha256=sha256(ARCHIVE/"trace.json"),
               source_import=str(Path(nanovllm.__file__).resolve()), schedule=plan, diagnostic_sections=SECTIONS,
               detailed="Synchronize device after preparation, forward/logits and sampling, then materialize token IDs; wrap synchronous save_many. Only prefill steps instrumented.",
               plain="Stock engine instrumentation, fresh cache metadata every wave; write to disk after wave finishes."))
    engine = json.loads((ARCHIVE/"engine.json").read_text())
    torch.manual_seed(0)
    llm = LLM(args.model, **engine["config"])
    try:
        r = llm.model_runner
        assert r.kv_cache_budget["actual_blocks"]==36 and r.kv_transfer.capacity==64
        assert r.kv_cache.dtype == next(r.model.parameters()).dtype == torch.bfloat16
        assert r.graph_bs==engine["graph_batch_sizes"] and not r.enforce_eager
        write_json(out/"engine.json", dict(config=engine["config"], kv_budget=r.kv_cache_budget, namespace=llm.scheduler.block_manager.namespace.hex(), graph=r.graph_bs, dtype=str(r.kv_cache.dtype)))
        r.kv_transfer.slots.fill(0)
        expected = {}
        for index,s in enumerate([dict(comparison="warmup", detailed=False, repeat=0, arm="32", seed=seed,
                                       cycle=cycle, phase="fill", warmup=True) for seed,cycle in CASES] + plan):
            case = s["seed"], s["cycle"]
            batch, spans = collect(llm, inputs[case], s)
            save_wave(out, batch)
            write_jsonl(out/"timeline.jsonl", [dict(**s, **v) for v in spans])
            if not s["warmup"]:
                exact = ([r["output_token_ids"] for r in batch[0]], normalized_steps(batch))
                assert expected.setdefault(case, exact) == exact
                for row, prompt in zip(batch[0], inputs[case], strict=True):
                    assert row["input_token_ids"] == prompt
                    assert abs(row["ttft_ms"] - 1000*(row["token_s"][0] - row["enqueue_s"])) < 1e-6
                    assert row["cpu_save_failures"]==row["cpu_restore_failures"]==row["preemptions"]==0
                    assert row["d2h_blocks"]==(0 if s["arm"]=="A" else (len(prompt)-128)//256)
            print(index, s, round(np.percentile([r["ttft_ms"] for r in batch[0]],95), 2), flush=True)
    finally:
        llm.exit()
    write_json(out/"summary.json", summarize(out))


if __name__ == "__main__":
    main()
