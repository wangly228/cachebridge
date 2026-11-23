"""Reproducible baseline A, correctness and optional isolated prefill trace.

Run from the repository root with the existing .venv. Each wave enqueues C
requests together and waits for all of them before the next wave (closed loop).
"""
import argparse
from collections import defaultdict
from dataclasses import fields
import gzip
import json
from pathlib import Path
import random
from time import perf_counter
import traceback

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cpu_prefix_cache import TieredBlockManager
from benchmarks.common import distribution, memory, p95_interval, prepare_run, write_json, write_jsonl


def reset_cache(llm):
    """Benchmark-only cold start: invalidate metadata after all requests finish."""
    assert llm.is_finished()
    manager = llm.scheduler.block_manager
    assert not manager.used_block_ids
    if isinstance(manager, TieredBlockManager):
        llm.scheduler.block_manager = TieredBlockManager(len(manager.blocks), manager.block_size,
                                                       manager.transfer, manager.namespace, manager.admission)
    else:
        llm.scheduler.block_manager = BlockManager(len(manager.blocks), manager.block_size)


def prompts(seed, count, length):
    rng = random.Random(seed)
    return [[rng.randrange(100, 10000) for _ in range(length)] for _ in range(count)]


def wave(llm, inputs, output_tokens, tag):
    llm.clear_metrics()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    ids = [llm.add_request(p, SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True)) for p in inputs]
    outputs = {}
    while not llm.is_finished():
        done, _ = llm.step()
        outputs.update(done)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    rows = []
    for i, seq_id in enumerate(ids):
        row = dict(tag, **llm.request_metrics[seq_id].as_dict(start),
                   input_token_ids=inputs[i], output_token_ids=outputs[seq_id])
        assert row["output_tokens"] == output_tokens
        assert row["enqueue_s"] <= row["token_s"][0] <= row["finish_s"] <= elapsed
        rows.append(row)
    steps = [dict(tag, **{k: v for k, v in step.items() if k != "start_time"},
                  start_s=step["start_time"] - start) for step in llm.step_metrics]
    summary = dict(tag, elapsed_s=elapsed, output_tokens=sum(len(v) for v in outputs.values()),
                   output_tokens_per_s=sum(len(v) for v in outputs.values()) / elapsed,
                   memory=memory(), used_blocks_peak=max(s["used_blocks"] for s in steps),
                   profiler_enabled=False, cpu_cache_enabled=llm.config.cpu_prefix_cache_blocks > 0)
    if summary["cpu_cache_enabled"]:
        manager = llm.scheduler.block_manager
        summary.update(cpu_resident_blocks=len(manager.entries), cpu_capacity_blocks=manager.transfer.capacity,
                       cpu_evictions_since_reset=manager.evictions)
    return rows, steps, summary


def save_wave(output, result):
    rows, steps, summary = result
    write_jsonl(output / "requests.jsonl", rows)
    write_jsonl(output / "steps.jsonl", steps)
    write_jsonl(output / "waves.jsonl", [summary])


def correctness(llm, output):
    checks = []
    # Last-block exclusion, partial hits and chunk boundaries.
    lengths = [255, 256, 257, 512, 513, 2049]
    for length in lengths:
        inputs = prompts(90000 + length, 1, length)
        reset_cache(llm)
        cold = wave(llm, inputs, 4, dict(phase="correctness", case=f"cold_{length}"))
        warm = wave(llm, inputs, 4, dict(phase="correctness", case=f"warm_{length}"))
        a, b = cold[0][0], warm[0][0]
        expected_hit = ((length - 1) // 256) * 256
        assert a["output_token_ids"] == b["output_token_ids"], (length, "greedy mismatch")
        assert b["initial_gpu_hit_tokens"] == expected_hit
        assert b["prefill_tokens"] == length - expected_hit
        for result in (cold, warm):
            save_wave(output, result)
        checks.append(dict(length=length, expected_hit=expected_hit, greedy_equal=True))
    # Two requests reuse a complete block and must maintain separate tails.
    base = prompts(91000, 1, 513)[0]
    other = base[:256] + prompts(91001, 1, 257)[0]
    reset_cache(llm)
    references = []
    for p in (base, other):
        reset_cache(llm)
        result = wave(llm, [p], 4, dict(phase="correctness", case="separate_reference"))
        references.append(result[0][0]["output_token_ids"])
        save_wave(output, result)
    reset_cache(llm)
    save_wave(output, wave(llm, [base], 4, dict(phase="correctness", case="shared_prefix_fill")))
    result = wave(llm, [base, other], 4, dict(phase="correctness", case="shared_prefix"))
    assert [r["output_token_ids"] for r in result[0]] == references
    assert result[0][1]["initial_gpu_hit_tokens"] == 256
    save_wave(output, result)
    write_json(output / "correctness.json", dict(passed=True, boundaries=checks, shared_prefix_greedy_equal=True))


def summarize(output):
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "baseline":
            groups[(row["length"], row["concurrency"], row["cache"])].append(row)
    summaries = []
    for (length, concurrency, cache), group in groups.items():
        ttft = [r["ttft_ms"] for r in group]
        summaries.append(dict(length=length, concurrency=concurrency, cache=cache,
                              seeds=sorted({r["seed"] for r in group}), ttft_ms=distribution(ttft),
                              p95_bootstrap_95pct_ms=p95_interval(ttft),
                              uncertainty="Descriptive IID bootstrap; within-wave correlation and small n limit inference.",
                              prefill_tokens=sum(r["prefill_tokens"] for r in group),
                              gpu_hit_tokens=sum(r["initial_gpu_hit_tokens"] for r in group),
                              preemptions=sum(r["preemptions"] for r in group)))
    write_json(output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--requests", type=int, default=4, help="requests per seed/length/C/cache condition")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--kv-blocks", type=int, default=160)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true", help="extra isolated prefill trace, excluded from baseline")
    args = parser.parse_args()
    args.module = "benchmarks.phase1"
    if any(c < 1 or c > 8 or args.requests % c for c in args.concurrency):
        parser.error("C must be in [1, 8] and divide --requests")
    if args.requests < 1 or min(args.lengths) < 1 or max(args.lengths) + args.output_tokens > 9216:
        parser.error("invalid requests/length; prompt plus output must fit max_model_len=9216")
    output = prepare_run(args.output, args, args.model)
    llm = None
    try:
        random.seed(0)
        torch.manual_seed(0)
        llm = LLM(args.model, enforce_eager=args.eager, max_model_len=9216, max_num_seqs=8,
                  max_num_batched_tokens=2048, gpu_memory_utilization=0.8,
                  max_kvcache_blocks=args.kv_blocks, enable_metrics=True)
        runner = llm.model_runner
        assert runner.kv_cache.dtype == torch.bfloat16
        assert next(runner.model.parameters()).dtype == torch.bfloat16
        write_json(output / "engine.json", dict(
            config={f.name: getattr(llm.config, f.name) for f in fields(llm.config) if f.name != "hf_config"},
            hf_config=llm.config.hf_config.to_dict(), kv_budget=runner.kv_cache_budget,
            kv_shape=list(runner.kv_cache.shape), kv_stride=list(runner.kv_cache.stride()),
            kv_dtype=str(runner.kv_cache.dtype), weight_dtype=str(next(runner.model.parameters()).dtype),
            sampling=dict(temperature=0, ignore_eos=True), arrivals="closed-loop waves, ordered simultaneous enqueue",
            timing="enqueue to append of real sampled token; model run materializes IDs via .tolist(); no per-token extra synchronize",
            status="small baseline; no CPU cache; formal freeze requires broader review"))
        # Warm both the full-prefill and paged-attention paths at tested shapes.
        for length in sorted(set(args.lengths + [255, 256, 257, 512, 513, 2049])):
            for c in sorted(set(args.concurrency + [1])):
                inputs = prompts(80000 + length + c, c, length)
                for repeat in range(2):
                    save_wave(output, wave(llm, inputs, 4, dict(phase="warmup", length=length, concurrency=c, repeat=repeat)))
                reset_cache(llm)
        correctness(llm, output)
        for length in args.lengths:
            for c in args.concurrency:
                required = c * ((length + args.output_tokens - 2) // 256 + 1)
                if required > llm.config.num_kvcache_blocks:
                    raise ValueError(f"cold/warm pair requires {required} blocks, pool has {llm.config.num_kvcache_blocks}")
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    inputs = prompts(seed * 1000000 + length * 100 + c, args.requests, length)
                    for offset in range(0, len(inputs), c):
                        reset_cache(llm)
                        pair = []
                        for cache in ("cold", "warm"):
                            tag = dict(phase="baseline", length=length, concurrency=c, seed=seed,
                                       wave=offset // c, cache=cache)
                            result = wave(llm, inputs[offset:offset+c], args.output_tokens, tag)
                            save_wave(output, result)
                            pair.append(result)
                            print(f"{tag}: {result[2]['elapsed_s']:.3f}s", flush=True)
                        assert [r["output_token_ids"] for r in pair[0][0]] == [r["output_token_ids"] for r in pair[1][0]]
        summarize(output)
        if args.profile:
            reset_cache(llm)
            llm.clear_metrics()
            llm.add_request(prompts(99999, 1, 2048)[0], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                llm.step()
                torch.cuda.synchronize()
            trace = output / "prefill_trace.json"
            prof.export_chrome_trace(str(trace))
            with trace.open("rb") as src, gzip.open(output / "prefill_trace.json.gz", "wb") as dst:
                dst.write(src.read())
            trace.unlink()
            (output / "profiler.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        write_json(output / "status.json", dict(success=True, failures=0, cpu_cache_enabled=False))
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        write_json(output / "status.json", dict(success=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
