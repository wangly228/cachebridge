"""Counterbalance B/C on the same engine and identical resident KV before each measured wave."""
import argparse
import json
from pathlib import Path
from shutil import copyfile
from statistics import median
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import distribution, prepare_run, sha256, write_json, write_jsonl
from benchmarks.cpu_cache_pressure import BASELINE
from benchmarks.phase1 import reset_cache, save_wave, wave
from benchmarks.restore_calibration import inputs_for, state_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cost-table", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.restore_bc_control"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        table = out / "cost_table.json"
        copyfile(args.cost_table, table)
        config = dict(json.loads((BASELINE / "engine.json").read_text())["config"], cpu_prefix_cache_blocks=64,
                      cpu_restore_cost_table=str(table))
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        policy = llm.scheduler.restore_policy
        seeds = range(911, 917)
        write_json(out / "design.json", dict(config=config, table_sha256=sha256(table), seeds=list(seeds),
                   prefix=2048, suffix=128, concurrency=4, warmup_seed=910, output_tokens=2,
                   endpoint="enqueue to first token before decode", profiler=False))
        pairs = []
        for seed in (910, *seeds):
            trace = inputs_for("bc-control", seed, [2048]*4)
            order = "BC" if seed % 2 else "CB"
            states, replay = [], {}
            for arm in order:
                reset_cache(llm)
                llm.scheduler.restore_policy = None if arm == "B" else policy
                for phase, inputs in trace:
                    if phase == "evicted":
                        states.append(state_digest(llm.scheduler.block_manager))
                    batch = wave(llm, inputs, 2, dict(seed=seed, arm=arm, phase=phase, warmup=seed == 910))
                    save_wave(out, batch)
                replay[arm] = batch[0]
                assert all(r["cpu_restored_tokens"] == 2048 and r["prompt_computed_tokens"] == 128
                           and r["h2d_blocks"] == 8 and r["d2h_blocks"] == r["preemptions"] == 0 for r in batch[0])
                assert max(r["token_s"][0] for r in batch[0]) <= min(
                    s["start_s"] for s in batch[1] if not s["is_prefill"])
            assert states[0] == states[1]
            for b, c in zip(replay["B"], replay["C"], strict=True):
                assert c["restore_decisions"][0]["reason"] == "measured_restore" and not b["restore_decisions"]
                assert b["input_token_ids"] == c["input_token_ids"] and b["output_token_ids"] == c["output_token_ids"]
            if seed != 910:
                times = {arm: distribution([r["ttft_ms"] for r in rows])["p95"] for arm, rows in replay.items()}
                pair = dict(seed=seed, order=order, B_p95_ms=times["B"], C_p95_ms=times["C"],
                            C_minus_B_ms=times["C"]-times["B"], state_sha256=states[0])
                pairs.append(pair)
                write_jsonl(out / "pairs.jsonl", [pair])
        deltas = [p["C_minus_B_ms"] for p in pairs]
        write_json(out / "status.json", dict(passed=True, pairs=len(pairs), requests_per_arm=4*len(pairs),
                   C_minus_B_paired_wave_p95_ms=dict(median=median(deltas), min=min(deltas), max=max(deltas)),
                   C_faster_pairs=sum(d < 0 for d in deltas)))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
