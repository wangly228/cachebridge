"""A/B/C on held-out short, long and heterogeneous waves under genuine GPU pressure."""
import argparse
import json
from multiprocessing import get_context
from pathlib import Path
from shutil import copyfile
import traceback

import torch

from nanovllm import LLM
from nanovllm.engine.cpu_prefix_cache import prefix_key
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.cpu_cache_pressure import BASELINE
from benchmarks.phase1 import reset_cache, save_wave, wave
from benchmarks.restore_calibration import inputs_for, state_digest


LENGTHS = {"short": [512]*4, "long": [8192], "mixed": [512, 1024, 2048, 512]}
SEEDS = (901, 902, 903)


def verify(row, strategy, cpu_miss=False):
    phase = row["phase"]
    prefix = row["prompt_tokens"] - 128
    available = prefix if phase == "evicted" and strategy != "A" and not cpu_miss else 0
    assert row["initial_gpu_hit_tokens"] == 0 and row["initial_cpu_hit_tokens"] == available
    restored = row["cpu_restored_tokens"]
    assert row["h2d_blocks"] * 256 == restored and restored in (0, available)
    assert row["d2h_blocks"] * 256 == (prefix - restored if strategy != "A" else 0)
    assert row["prompt_computed_tokens"] == row["prefill_tokens"] == row["prompt_tokens"] - restored
    assert row["preemptions"] == row["cpu_save_failures"] == row["cpu_restore_failures"] == 0
    if strategy == "C" and available:
        decision, = row["restore_decisions"]
        assert decision["shape"]["cpu_tokens"] == available and decision["restored_tokens"] == restored
        assert (decision["action"] == "restore") == bool(restored)
    else:
        assert not row["restore_decisions"] and restored == available


def oracle(llm, out, seed, trace, c_rows, expected_state, workload="mixed", setup_output_tokens=128):
    """Real B data path. Hide only rejected roots, reproducing C's execution shape independently."""
    reset_cache(llm)
    for phase, inputs in trace:
        if phase == "evicted":
            manager = llm.scheduler.block_manager
            assert state_digest(manager) == expected_state
            for tokens, row in zip(inputs, c_rows, strict=True):
                if row["restore_decisions"][0]["action"] == "recompute":
                    key = prefix_key(manager.namespace, tokens[:manager.block_size])
                    slot, _ = manager.entries.pop(key)
                    manager.free_slots.append(slot)
        result = wave(llm, inputs, 128 if phase == "evicted" else setup_output_tokens,
                      dict(phase=phase, seed=seed, workload=workload))
        save_wave(out, result)
    for i, (ref, measured) in enumerate(zip(result[0], c_rows, strict=True)):
        fields = ("input_token_ids", "output_token_ids", "cpu_restored_tokens", "prompt_computed_tokens")
        exact = all(ref[k] == measured[k] for k in fields)
        write_jsonl(out / "checks.jsonl", [dict(seed=seed, request=i, exact=exact,
                    action=measured["restore_decisions"][0]["action"], output_tokens=ref["output_tokens"])])
        assert exact


def run_strategy(out, model, strategy):
    dest = out / strategy
    dest.mkdir()
    base = json.loads((BASELINE / "engine.json").read_text())["config"]
    config = dict(base, cpu_prefix_cache_blocks=0 if strategy == "A" else 64)
    if strategy == "C":
        config["cpu_restore_cost_table"] = str(out / "cost_table.json")
    torch.manual_seed(0)
    llm = LLM(model, **config)
    try:
        runner = llm.model_runner
        assert llm.config.num_kvcache_blocks == 36 and runner.kv_cache.dtype == torch.bfloat16
        assert 4 in runner.graphs and not runner.enforce_eager
        write_json(dest / "engine.json", dict(config=config, budget=runner.kv_cache_budget,
                   dtype=str(runner.kv_cache.dtype), graph_bs=runner.graph_bs))
        mixed, cycles = [], []
        for seed, name, trace in json.loads((out / "trace.json").read_text()):
            reset_cache(llm)
            torch.manual_seed(seed)
            warmup = seed == 900
            batches = []
            for phase, inputs in trace:
                if strategy == "C" and phase == "evicted" and name == "mixed" and not warmup:
                    before = state_digest(llm.scheduler.block_manager)
                result = wave(llm, inputs, 2 if warmup else 128, dict(
                    strategy=strategy, phase=phase, seed=seed, workload=name, warmup=warmup))
                batches.append(result)
                for row in result[0]:
                    verify(row, strategy)
            for batch in batches:
                save_wave(dest, batch)
            if not warmup:
                cycles.append(dict(seed=seed, workload=name, cold_start_wall_s=sum(
                    batch[2]["elapsed_s"] for batch in batches)))
                if strategy == "C" and name == "mixed":
                    mixed.append((seed, trace, batches[-1][0], before))
                write_json(dest / "cycles.json", cycles)
            print(f"{strategy} seed={seed} {name} complete", flush=True)
        if strategy == "C":
            oracle_dir = out / "mixed_reference"
            oracle_dir.mkdir()
            llm.scheduler.restore_policy = None
            for case in mixed:
                oracle(llm, oracle_dir, *case)
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cost-table", required=True)
    parser.add_argument("--strategy", choices=("A", "B", "C"), help="only run this arm")
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.restore_holdout"
    out = prepare_run(args.output, args, args.model)
    try:
        table = out / "cost_table.json"
        copyfile(args.cost_table, table)
        assert set(SEEDS).isdisjoint(json.loads(table.read_text())["calibration_seeds"])
        traces = [(seed, name, inputs_for("holdout", seed, lengths))
                  for seed in (900, *SEEDS) for name, lengths in LENGTHS.items()]
        write_json(out / "trace.json", traces)
        write_json(out / "design.json", dict(lengths=LENGTHS, seeds=SEEDS, warmup_seed=900,
                   output_tokens=128, suffix_tokens=128, temperature=0, ignore_eos=True,
                   table_sha256=sha256(table), trace_sha256=sha256(out / "trace.json"),
                   profiler=False, process_per_strategy=True,
                   cycle_timing="sum of enqueue-to-wave-finish wall times, fill+interference+replay"))
        strategies = [args.strategy] if args.strategy else list("ABC")
        for strategy in strategies:
            process = get_context("spawn").Process(target=run_strategy, args=(out, args.model, strategy))
            process.start()
            process.join()
            if process.exitcode:
                raise RuntimeError(f"strategy {strategy} worker exited with {process.exitcode}")
        write_json(out / "status.json", dict(passed=True, strategies=strategies,
                   formal_requests_per_strategy=len(SEEDS)*sum(2*len(v)+4 for v in LENGTHS.values()),
                   mixed_oracle_exact_requests=len(SEEDS)*len(LENGTHS["mixed"]) if "C" in strategies else 0))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise


if __name__ == "__main__":
    main()
