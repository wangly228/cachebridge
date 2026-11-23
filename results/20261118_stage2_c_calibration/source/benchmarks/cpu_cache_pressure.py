"""Replay frozen A inputs with synchronous CPU policy B. One cold reset per four-wave cycle."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import distribution, prepare_run, sha256, write_json
from benchmarks.phase1 import reset_cache, save_wave, wave
from benchmarks.pressure import summarize


BASELINE = Path("results/20261118_pressure36")
MANIFEST = Path("results/20261118_stage1_acceptance/SHA256SUMS")


def read_baseline():
    expected = dict(line.split("  ", 1)[::-1] for line in MANIFEST.read_text().splitlines())
    hashes = {}
    for name in ("trace.json", "summary.json", "cycles.json", "engine.json", "requests.jsonl", "environment.json"):
        path = BASELINE / name
        hashes[name] = sha256(path)
        assert hashes[name] == expected[str(path)], path
    refs = defaultdict(list)
    for line in (BASELINE / "requests.jsonl").read_text().splitlines():
        row = json.loads(line)
        refs[row.get("seed"), row.get("cycle"), row["phase"]].append(row)
    return hashes, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.cpu_cache_pressure"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        hashes, refs = read_baseline()
        a_engine = json.loads((BASELINE / "engine.json").read_text())
        traces = json.loads((BASELINE / "trace.json").read_text())
        a_env = json.loads((BASELINE / "environment.json").read_text())
        b_env = json.loads((out / "environment.json").read_text())
        for key in ("packages", "gpu", "cuda_runtime", "capability"):
            assert a_env[key] == b_env[key], key
        assert [(f["name"], f["sha256"]) for f in a_env["model_files"]] == [
            (f["name"], f["sha256"]) for f in b_env["model_files"]]
        config = dict(a_engine["config"], cpu_prefix_cache_blocks=64)
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        runner = llm.model_runner
        transfer = runner.kv_transfer
        block_size = llm.config.kvcache_block_size
        assert runner.kv_cache_budget["actual_blocks"] == a_engine["kv_budget"]["actual_blocks"]
        assert transfer.block_bytes == a_engine["kv_budget"]["block_bytes"]
        assert str(runner.kv_cache.dtype) == str(next(runner.model.parameters()).dtype) == a_engine["dtype"]
        assert runner.graph_bs == a_engine["graph_batch_sizes"] and a_engine["concurrency"] in runner.graphs
        write_json(out / "engine.json", dict(a_engine, config=config, kv_budget=runner.kv_cache_budget,
                   namespace=llm.scheduler.block_manager.namespace.hex(), cpu_pageable_bytes=transfer.slots.nbytes,
                   pinned_staging_bytes=transfer.block_bytes, gpu_staging_bytes=transfer.block_bytes,
                   baseline=str(BASELINE), baseline_sha256=hashes,
                   cpu_write_policy="synchronous write-through on computed full prompt blocks",
                   cold_reset="empty GPU and CPU indices; fixed-size buffers reused"))
        warm = refs[None, None, "warmup"]
        for state in ("fill", "hot"):
            inputs = [r["input_token_ids"] for r in warm if r["cache"] == state]
            assert len(inputs) == a_engine["concurrency"]
            save_wave(out, wave(llm, inputs, 4, dict(phase="warmup", cache=state)))
        records = defaultdict(list)
        cycles, differences = [], []
        prefix, suffix, output = (a_engine[k] for k in ("prefix_tokens", "suffix_tokens", "output_tokens"))
        for seed, cycle, trace in traces:
            reset_cache(llm)
            torch.manual_seed(seed)
            start = perf_counter()
            batches = [wave(llm, inputs, output, dict(phase=phase, seed=seed, cycle=cycle))
                       for phase, inputs in trace]
            elapsed = perf_counter() - start
            cycles.append(dict(seed=seed, cycle=cycle, cold_start_wall_s=elapsed,
                               output_tokens_per_wall_second=sum(b["output_tokens"] for _, _, b in batches) / elapsed))
            for batch in batches:
                save_wave(out, batch)
                rows, _, batch_info = batch
                phase = batch_info["phase"]
                records[phase].append(batch)
                gpu = prefix if phase == "hot" else 0
                cpu = prefix if phase == "evicted" else 0
                compare_phase = "hot" if phase == "evicted" else phase
                for idx, row in enumerate(rows):
                    assert (row["initial_gpu_hit_tokens"], row["initial_cpu_hit_tokens"]) == (gpu, cpu)
                    assert row["prefill_tokens"] == row["prompt_computed_tokens"] == prefix + suffix - gpu - cpu
                    assert row["cpu_restored_tokens"] == row["h2d_blocks"] * block_size == cpu
                    assert row["preemptions"] == row["cpu_restore_failures"] == row["cpu_save_failures"] == 0
                    assert row["decode_tokens"] == output - 1
                    ref = refs[seed, cycle, compare_phase][idx]
                    assert ref["input_token_ids"] == row["input_token_ids"]
                    if ref["output_token_ids"] != row["output_token_ids"]:
                        differences.append(dict(seed=seed, cycle=cycle, phase=phase, request=idx,
                                                compared_to_A=compare_phase))
            write_json(out / "cycles.json", cycles)
            write_json(out / "same_shape_greedy_differences.json", differences)
            print(f"seed={seed} cycle={cycle} finished in {elapsed:.2f}s", flush=True)
        a_summary = json.loads((BASELINE / "summary.json").read_text())
        summary = summarize(records)
        for phase, batches in records.items():
            rows = [row for batch_rows, _, _ in batches for row in batch_rows]
            s = summary[phase]
            s["A_p95_ms"] = a_summary[phase]["ttft_ms"]["p95"]
            s["A_p95_ms_by_seed"] = a_summary[phase]["p95_ms_by_seed"]
            s["p95_change_percent"] = 100 * (s["ttft_ms"]["p95"] / s["A_p95_ms"] - 1)
            s["A_output_tokens_per_engine_second"] = a_summary[phase]["output_tokens_per_engine_second"]
            s["cpu_hit_requests"] = sum(row["initial_cpu_hit_tokens"] > 0 for row in rows)
            for field in ("initial_gpu_hit_tokens", "cpu_restored_tokens", "prompt_computed_tokens",
                          "d2h_blocks", "h2d_blocks", "d2h_ms", "h2d_ms", "cpu_save_failures", "cpu_restore_failures"):
                s[field] = sum(row[field] for row in rows)
            s["d2h_bytes"] = s["d2h_blocks"] * transfer.block_bytes
            s["h2d_bytes"] = s["h2d_blocks"] * transfer.block_bytes
            s["tpot_ms"] = distribution([row["tpot_ms"] for row in rows])
            s["itl_ms"] = distribution([ms for row in rows for ms in row["itl_ms"]])
        a_wall = sum(r["cold_start_wall_s"] for r in json.loads((BASELINE / "cycles.json").read_text()))
        b_wall = sum(r["cold_start_wall_s"] for r in cycles)
        n = sum(s["ttft_ms"]["n"] for s in summary.values())
        write_json(out / "summary.json", dict(phases=summary, A_cold_start_wall_s=a_wall, B_cold_start_wall_s=b_wall,
                   cold_start_wall_change_percent=100 * (b_wall / a_wall - 1),
                   A_total_output_tokens_per_wall_second=n*output/a_wall,
                   B_total_output_tokens_per_wall_second=n*output/b_wall,
                   p95_uncertainty="Per-seed p95 range; 8 correlated requests per phase per seed."))
        write_json(out / "status.json", dict(measurement_complete=True, cache_accounting_passed=True,
                   same_shape_equal_requests=n-len(differences), requests=n, seeds=a_engine["seeds"], cycles=len(cycles)))
        assert n == 96 and not differences, differences
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
