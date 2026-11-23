"""Isolated first-token costs, without decode, followed by one profiler window."""
import argparse
import gzip
from pathlib import Path
import traceback

import torch

from nanovllm import LLM, SamplingParams
from benchmarks.common import distribution, prepare_run, write_json
from benchmarks.phase1 import prompts, reset_cache, save_wave, wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.module = "benchmarks.prefill_profile"
    if args.repeats < 1 or min(args.lengths) < 1 or max(args.lengths) >= 9216:
        parser.error("repeats >= 1 and 0 < length < 9216 required")
    output = prepare_run(args.output, args, args.model)
    llm = None
    try:
        torch.manual_seed(0)
        llm = LLM(args.model, enforce_eager=True, max_model_len=9216, max_num_seqs=8,
                  max_num_batched_tokens=2048, gpu_memory_utilization=0.8,
                  max_kvcache_blocks=160, enable_metrics=True)
        write_json(output / "engine.json", dict(kv_budget=llm.model_runner.kv_cache_budget,
                   kv_shape=list(llm.model_runner.kv_cache.shape), dtype=str(llm.model_runner.kv_cache.dtype),
                   enforce_eager=True, max_num_batched_tokens=2048, max_model_len=9216,
                   concurrency=1, output_tokens=1, temperature=0, ignore_eos=True,
                   scope="host runner wall cost, includes preparation, sampling and ID materialization; not pure GPU kernel time"))
        costs, matches = [], []
        for length in args.lengths:
            samples = {cache: [] for cache in ("cold", "warm")}
            for repeat in range(args.repeats + 2):
                inputs = prompts(700000 + length * 100 + repeat, 1, length)
                reset_cache(llm)
                pair = []
                for cache in ("cold", "warm"):
                    tag = dict(phase="warmup" if repeat < 2 else "prefill_cost", length=length,
                               repeat=repeat - 2, cache=cache)
                    result = wave(llm, inputs, 1, tag)
                    save_wave(output, result)
                    pair.append(result[0][0]["output_token_ids"])
                    if repeat >= 2:
                        samples[cache].append(dict(ttft_ms=result[0][0]["ttft_ms"],
                            runner_ms=sum(s["run_ms"] for s in result[1]),
                            computed_tokens=result[0][0]["prefill_tokens"],
                            gpu_hit_tokens=result[0][0]["initial_gpu_hit_tokens"]))
                matches.append(dict(length=length, repeat=repeat - 2, greedy_equal=pair[0] == pair[1]))
            for cache, rows in samples.items():
                costs.append(dict(length=length, cache=cache, ttft_ms=distribution([r["ttft_ms"] for r in rows]),
                                  runner_ms=distribution([r["runner_ms"] for r in rows]),
                                  computed_tokens=rows[0]["computed_tokens"], gpu_hit_tokens=rows[0]["gpu_hit_tokens"]))
            print(f"prefill cost complete: {length} tokens", flush=True)
        write_json(output / "costs.json", costs)
        write_json(output / "greedy_comparisons.json", matches)
        # Extra trace is entirely outside the measurements above.
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
        write_json(output / "status.json", dict(success=True, failures=0, greedy_match_all=all(r["greedy_equal"] for r in matches)))
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        write_json(output / "status.json", dict(success=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
