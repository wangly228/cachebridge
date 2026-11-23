"""Paired native-GPU baseline: identical requests immediately after fill or eviction."""
import argparse
from collections import defaultdict
from pathlib import Path
from random import Random
from time import perf_counter
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import distribution, prepare_run, write_json
from benchmarks.phase1 import reset_cache, save_wave, wave


CONCURRENCY = 4
PREFIX = 2048
SUFFIX = 128
OUTPUT = 128
BLOCKS = 36  # Four live sequences of nine blocks exactly fit.


def make_trace(seed, cycle):
    rng = Random(f"PegaFlow-pressure-v1/{seed}/{cycle}")

    def tokens(size):
        return [rng.randrange(100, 10000) for _ in range(size)]

    targets = [tokens(PREFIX) for _ in range(CONCURRENCY)]
    fill = [p + tokens(SUFFIX) for p in targets]
    replay = [p + tokens(SUFFIX) for p in targets]
    interference = [tokens(PREFIX + SUFFIX) for _ in targets]
    # Both orderings begin cold. The immediate successor of an A access is hot.
    order = [("fill", fill), ("hot", replay), ("interference", interference), ("evicted", replay)]
    if cycle % 2:
        order = [order[i] for i in (0, 2, 3, 1)]
    return order


def summarize(records):
    result = {}
    for phase, batches in records.items():
        by_seed = defaultdict(list)
        all_rows = []
        elapsed = 0
        for rows, _, batch in batches:
            all_rows.extend(rows)
            by_seed[batch["seed"]].extend(r["ttft_ms"] for r in rows)
            elapsed += batch["elapsed_s"]
        result[phase] = dict(
            waves=len(batches), ttft_ms=distribution([r["ttft_ms"] for r in all_rows]),
            p95_ms_by_seed={s: distribution(v)["p95"] for s, v in by_seed.items()},
            output_tokens_per_engine_second=sum(r["output_tokens"] for r in all_rows) / elapsed,
            hit_requests=sum(r["initial_gpu_hit_tokens"] > 0 for r in all_rows),
            prefill_tokens=sum(r["prefill_tokens"] for r in all_rows),
            preemptions=sum(r["preemptions"] for r in all_rows),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.pressure"
    out = prepare_run(args.output, args, args.model)
    seeds = [101, 102, 103]
    config = dict(enforce_eager=False, tensor_parallel_size=1, enable_metrics=True,
                  max_num_batched_tokens=2048, max_num_seqs=8, max_model_len=9216,
                  gpu_memory_utilization=0.80, max_kvcache_blocks=BLOCKS)
    traces = [(seed, cycle, make_trace(seed, cycle)) for seed in seeds for cycle in range(2)]
    write_json(out / "trace.json", traces)
    llm = None
    try:
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        runner = llm.model_runner
        assert llm.config.num_kvcache_blocks == BLOCKS and runner.kv_cache.dtype == torch.bfloat16
        assert not runner.enforce_eager and CONCURRENCY in runner.graphs
        write_json(out / "engine.json", dict(config=config, kv_budget=runner.kv_cache_budget,
                   dtype=str(runner.kv_cache.dtype), weight_dtype=str(next(runner.model.parameters()).dtype),
                   graph_batch_sizes=runner.graph_bs, seeds=seeds, cycles_per_seed=2,
                   concurrency=CONCURRENCY, output_tokens=OUTPUT, prefix_tokens=PREFIX,
                   suffix_tokens=SUFFIX, temperature=0, ignore_eos=True, profiler=False))
        for phase, inputs in make_trace(0, 0)[:2]:
            save_wave(out, wave(llm, inputs, 4, dict(phase="warmup", cache=phase)))
        records = defaultdict(list)
        cycles = []
        mismatches = []
        for seed, cycle, trace in traces:
            reset_cache(llm)
            torch.manual_seed(seed)
            batches = []
            start = perf_counter()
            for phase, inputs in trace:
                batches.append(wave(llm, inputs, OUTPUT, dict(phase=phase, seed=seed, cycle=cycle)))
            elapsed = perf_counter() - start
            cycles.append(dict(seed=seed, cycle=cycle, cold_start_wall_s=elapsed,
                               output_tokens_per_wall_second=4 * CONCURRENCY * OUTPUT / elapsed))
            for batch in batches:
                save_wave(out, batch)
            replay_outputs = {}
            for rows, steps, batch in batches:
                phase = batch["phase"]
                records[phase].append((rows, steps, batch))
                hit = PREFIX if phase == "hot" else 0
                for r in rows:
                    assert r["initial_gpu_hit_tokens"] == hit and r["prefill_tokens"] == PREFIX + SUFFIX - hit
                    assert r["preemptions"] == 0 and r["decode_tokens"] == OUTPUT - 1
                replay_outputs[phase] = [r["output_token_ids"] for r in rows]
            for i, (a, b) in enumerate(zip(replay_outputs["hot"], replay_outputs["evicted"])):
                if a != b:
                    mismatches.append(dict(seed=seed, cycle=cycle, request=i,
                                           first_difference=next(j for j in range(OUTPUT) if a[j] != b[j])))
            write_json(out / "cycles.json", cycles)
            print(f"seed={seed} cycle={cycle} finished in {elapsed:.2f}s", flush=True)
        write_json(out / "summary.json", summarize(records))
        write_json(out / "greedy_differences.json", mismatches)
        write_json(out / "status.json", dict(measurement_complete=True, cache_accounting_passed=True,
                   strict_greedy_equal_pairs=CONCURRENCY * len(traces) - len(mismatches),
                   total_replay_pairs=CONCURRENCY * len(traces), stage1_frozen=False))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(measurement_complete=False, stage1_frozen=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
