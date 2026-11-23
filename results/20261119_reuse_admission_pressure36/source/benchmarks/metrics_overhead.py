"""Counterbalanced metrics on/off pairs on the archived native-GPU pressure trace."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import torch

from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.phase1 import reset_cache
from benchmarks.pressure import OUTPUT
from nanovllm import LLM, SamplingParams


def timed_wave(llm, inputs, enabled, count):
    llm.clear_metrics()
    llm.config.enable_metrics = enabled
    params = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True)
    torch.cuda.synchronize()
    start = perf_counter()
    ids, enqueued = [], {}
    for p in inputs:
        now = perf_counter()
        seq_id = llm.add_request(p, params)
        ids.append(seq_id)
        enqueued[seq_id] = now
    pending = list(llm.scheduler.waiting)
    assert [seq.seq_id for seq in pending] == ids
    first, finished = {}, {}
    while not llm.is_finished():
        done, _ = llm.step()
        if pending:
            now = perf_counter()
            for seq in pending:
                if seq.num_completion_tokens:
                    first[seq.seq_id] = now
            pending = [s for s in pending if s.seq_id not in first]
        finished.update(done)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    assert all(len(finished[i]) == count for i in ids)
    assert len(llm.request_metrics) == (len(ids) if enabled else 0)
    return dict(metrics=enabled, elapsed_s=elapsed, output_ids=[finished[i] for i in ids],
                observed_first_ms=[(first[i] - enqueued[i]) * 1000 for i in ids],
                append_ttft_ms=[llm.request_metrics[i].as_dict()["ttft_ms"] for i in ids] if enabled else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/20261118_pressure36")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.module = "benchmarks.metrics_overhead"
    source = Path(args.source)
    config = json.loads((source / "engine.json").read_text())["config"]
    model = json.loads((source / "environment.json").read_text())["model_path"]
    out = prepare_run(args.output, args, model)
    trace = json.loads((source / "trace.json").read_text())
    llm = None
    try:
        llm = LLM(model, **config)
        assert llm.config.num_kvcache_blocks == config["max_kvcache_blocks"] and 4 in llm.model_runner.graphs
        write_json(out / "design.json", dict(trace_sha256=sha256(source / "trace.json"), config=config,
                   actual_kv_budget=llm.model_runner.kv_cache_budget,
                   unit="pair of four-wave traces", timed="enqueue through synchronized completion per wave",
                   observer="identical first-visible step-return clock in each arm",
                   estimand="100 * (sum on / sum off - 1)", uncertainty="range of six paired trace deltas"))
        for enabled in [False, True]:
            reset_cache(llm)
            for phase, inputs in trace[0][2][:2]:
                row = timed_wave(llm, inputs, enabled, 4)
                write_jsonl(out / "warmup.jsonl", [dict(phase=phase, **row)])
        pairs = []
        for seed, cycle, waves in trace:
            arms = {}
            for enabled in ([False, True] if (seed + cycle) % 2 else [True, False]):
                reset_cache(llm)
                torch.manual_seed(seed)
                rows = []
                for phase, inputs in waves:
                    rows.append(dict(seed=seed, cycle=cycle, phase=phase,
                                     **timed_wave(llm, inputs, enabled, OUTPUT)))
                write_jsonl(out / "waves.jsonl", rows)
                arms[enabled] = rows
            off, on = arms[False], arms[True]
            assert all(a["output_ids"] == b["output_ids"] for a, b in zip(off, on))
            off_s = sum(r["elapsed_s"] for r in off)
            on_s = sum(r["elapsed_s"] for r in on)
            pair = dict(seed=seed, cycle=cycle, off_s=off_s, on_s=on_s, delta_pct=100*(on_s/off_s-1))
            write_jsonl(out / "pairs.jsonl", [pair])
            pairs.append(pair)
            print(pair, flush=True)
        records = [json.loads(s) for s in (out / "waves.jsonl").read_text().splitlines()]
        proxy = {}
        for phase, _ in trace[0][2]:
            proxy[phase] = {str(enabled): float(np.percentile([t for r in records
                if r["metrics"] == enabled and r["phase"] == phase for t in r["observed_first_ms"]], 95))
                for enabled in (False, True)}
        off = sum(p["off_s"] for p in pairs)
        on = sum(p["on_s"] for p in pairs)
        write_json(out / "summary.json", dict(pairs=len(pairs), off_s=off, on_s=on,
                   delta_pct=100*(on/off-1), paired_range_pct=[min(p["delta_pct"] for p in pairs),
                   max(p["delta_pct"] for p in pairs)], step_return_p95_ms=proxy, all_output_ids_equal=True))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
