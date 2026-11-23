"""Compare serial and double-buffered CPU prefix writes on independent mixed waves."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from random import Random
from time import perf_counter
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.cpu_cache_pressure import BASELINE
from benchmarks.phase1 import save_wave, wave
from benchmarks.write_admission import select_arm


PREVIOUS = Path("results/20261119_fill_batch_pressure36")
ARMS = ("A", "reuse", "batch")
SEEDS = tuple(range(1701, 1707))
LAYOUTS = sorted(set(permutations((1024, 1024, 2048, 2048))))


def make_case(seed, cycle, index):
    rng = Random(f"PegaFlow-mixed-fill-v1/{seed}/{cycle}")

    def tokens(n):
        return [rng.randrange(100, 10000) for _ in range(n)]

    # Complement the FIFO positions in the paired cycle. Across six seeds each
    # of the six layouts occurs once in each access order and pressure pass.
    layout = index if cycle == 0 else 5 - index
    prefixes = [tokens(n) for n in LAYOUTS[layout]]
    fill = [p + tokens(128) for p in prefixes]
    replay = [p + tokens(128) for p in prefixes]
    pressure = [[tokens(n + 128) for n in LAYOUTS[(layout + shift) % 6]] for shift in (1, 3)]
    phases = [("fill", fill), ("hot", replay), ("pressure1", pressure[0]),
              ("pressure2", pressure[1]), ("evicted", replay)]
    if cycle == 1:
        phases = [phases[i] for i in (0, 2, 3, 4, 1)]
    order = list(permutations(ARMS))[(index + 3 * cycle) % 6]
    return dict(seed=seed, cycle=cycle, arms=order, phases=phases)


def check_inputs(cases):
    # Compare actual first full blocks as well as seed labels against the frozen trace.
    historical = json.loads((BASELINE / "trace.json").read_text())
    assert set(SEEDS).isdisjoint(s for s, _, _ in historical)
    roots = {tuple(p[:256]) for _, _, trace in historical for _, inputs in trace for p in inputs}
    for case in cases:
        phases = dict(case["phases"])
        assert phases["hot"] == phases["evicted"]
        for p, q in zip(phases["fill"], phases["hot"], strict=True):
            assert p[:-128] == q[:-128] and p[-128:] != q[-128:]
        for name in ("fill", "pressure1", "pressure2"):
            inputs = phases[name]
            assert sorted(len(p) for p in inputs) == [1152, 1152, 2176, 2176]
            for p in inputs:
                root = tuple(p[:256])
                assert root not in roots
                roots.add(root)


def check_case(case, by_arm, output):
    for arm, by_phase in by_arm.items():
        for phase, inputs in case["phases"]:
            rows = by_phase[phase]
            # In B, restored blocks feed the same 128-token query as native
            # GPU hits. Native cold replay has a different BF16 execution shape.
            restored = arm != "A" and phase == "evicted"
            reference = by_arm["A"]["hot" if restored else phase]
            for pos, (row, prompt, ref, serial) in enumerate(zip(
                    rows, inputs, reference, by_arm["reuse"][phase], strict=True)):
                key = case["seed"], case["cycle"], arm, phase, pos
                prefix = len(prompt) - 128
                gpu = prefix if phase == "hot" else 0
                cpu = prefix if restored else 0
                assert row["input_token_ids"] == ref["input_token_ids"] == prompt, key
                assert row["output_token_ids"] == ref["output_token_ids"], (*key, "greedy")
                assert len(row["output_token_ids"]) == len(row["token_s"]) == row["output_tokens"] == output, key
                assert row["decode_tokens"] == output - 1, key
                assert row["initial_gpu_hit_tokens"] == gpu and row["initial_cpu_hit_tokens"] == cpu, key
                assert row["h2d_blocks"] * 256 == row["cpu_restored_tokens"] == cpu, key
                assert row["prefill_tokens"] == row["prompt_computed_tokens"] == len(prompt) - gpu - cpu, key
                assert abs(row["ttft_ms"] - 1000 * (row["token_s"][0] - row["enqueue_s"])) < 1e-5, key
                assert row["preemptions"] == row["cpu_save_failures"] == row["cpu_restore_failures"] == 0, key
                if arm == "batch":
                    assert row["output_token_ids"] == serial["output_token_ids"], key
                    assert row["d2h_blocks"] == serial["d2h_blocks"] and row["h2d_blocks"] == serial["h2d_blocks"], key
                if phase == "fill":
                    assert row["d2h_blocks"] == (0 if arm == "A" else prefix // 256), key
                if arm != "batch":
                    assert row["d2h_batches"] == row["batched_d2h_blocks"] == 0, key


def execute(llm, out, case, warmup):
    by_arm = {}
    output = 8 if warmup else 128
    for arm in case["arms"]:
        select_arm(llm, arm)
        torch.manual_seed(case["seed"])
        tag = dict(arm=arm, seed=case["seed"], cycle=case["cycle"], warmup=warmup)
        start = perf_counter()
        batches = [wave(llm, inputs, output, dict(tag, phase=phase)) for phase, inputs in case["phases"]]
        elapsed = perf_counter() - start
        for batch in batches:
            save_wave(out, batch)
        write_jsonl(out / "cycles.jsonl", [dict(tag, wall_s=elapsed)])
        by_arm[arm] = {info["phase"]: rows for rows, _, info in batches}
        print(f"seed={case['seed']} cycle={case['cycle']} arm={arm} wall={elapsed:.2f}s", flush=True)
    check_case(case, by_arm, output)
    print(f"seed={case['seed']} cycle={case['cycle']} paths and greedy checked", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.mixed_fill"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        previous = json.loads((PREVIOUS / "engine.json").read_text())
        old, now = [json.loads((folder / "environment.json").read_text()) for folder in (PREVIOUS, out)]
        for key in ("model_files", "packages", "gpu", "capability", "cuda_runtime", "git_head"):
            assert old[key] == now[key], key
        def engine_sources(env):
            return {p: v for p, v in env["source_sha256"].items() if p.startswith("nanovllm/")}
        assert engine_sources(old) == engine_sources(now)
        pilot = make_case(1700, 1, 2)
        cases = [make_case(seed, c, i) for i, seed in enumerate(SEEDS) for c in (i % 2, 1 - i % 2)]
        check_inputs([pilot, *cases])
        write_json(out / "trace.json", dict(warmup=pilot, formal=cases))
        write_json(out / "design.json", dict(arms=ARMS, seeds=SEEDS, cycles_per_seed=2,
                   previous_engine=str(PREVIOUS), same_engine_sources=True, trace_sha256=sha256(out / "trace.json"),
                   cycles=[{k: c[k] for k in ("seed", "cycle", "arms")} for c in cases],
                   initial_state="fresh GPU/CPU indices per arm per cycle; shared pre-touched host allocation",
                   warmup="disjoint seed 1700, all three arms, all five waves, 8 outputs, excluded",
                   no_reuse="pool both consecutive disjoint mixed waves; additionally report each pass",
                   endpoint="per request enqueue to first materialized greedy token; whole includes all five waves",
                   gates=dict(fill_vs_serial_pct_lt=0, whole_vs_serial_pct_lt=0,
                              pooled_no_reuse_vs_native_pct_le=5, evicted_vs_native_pct_le=-20),
                   uncertainty="paired whole-seed bootstrap 95% percentile interval, 4000 resamples, RNG 1700"))
        config = previous["config"]
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        runner, block = llm.model_runner, llm.config.kvcache_block_size
        transfer = runner.kv_transfer
        assert runner.kv_cache_budget["actual_blocks"] == 36 and block == 256 and transfer.capacity == 64
        assert transfer.block_bytes == previous["kv_budget"]["block_bytes"] and len(transfer.write_stages) == 2
        assert runner.kv_cache.dtype == next(runner.model.parameters()).dtype == torch.bfloat16
        assert runner.graph_bs == previous["graph_batch_sizes"] and 4 in runner.graphs and not runner.enforce_eager
        assert llm.scheduler.block_manager.namespace.hex() == previous["namespace"]
        write_json(out / "engine.json", dict(config=config, kv_budget=runner.kv_cache_budget,
                   namespace=previous["namespace"], dtype=str(runner.kv_cache.dtype),
                   weight_dtype=str(next(runner.model.parameters()).dtype), graph_batch_sizes=runner.graph_bs,
                   block_tokens=block, cpu_slots=transfer.capacity, host_bytes=transfer.slots.nbytes,
                   pinned_staging_bytes=sum(t.nbytes for t, _, _ in transfer.write_stages),
                   gpu_staging_bytes=transfer.device_stage.nbytes, prefix_tokens=[1024, 2048], suffix_tokens=128,
                   concurrency=4, output_tokens=128, temperature=0, ignore_eos=True, profiler=False))
        transfer.slots.fill(0)
        execute(llm, out, pilot, True)
        for case in cases:
            execute(llm, out, case, False)
        write_json(out / "status.json", dict(measurement_complete=True, arms=3, seeds=6, formal_cycles_per_arm=12,
                   requests_per_arm=240, timed_requests=720, cpu_requests_checked_against_native=480))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(measurement_complete=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
