"""Vary one cold peer's query length while holding the target, other KV and ready state fixed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from shutil import copyfile
from statistics import median
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.queue_calibration import RecomputeOne, engine_config, run_cycle
from benchmarks.restore_calibration import inputs_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.queue_peer_probe"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        table = out / "cost_table.json"
        source = Path(args.calibration)
        copyfile(source / table.name, table)
        assert sha256(table) == json.loads((source/"status.json").read_text())["table_sha256"]
        seeds = (1301, 1302, 1303, 1304)
        config = dict(engine_config(), cpu_restore_cost_table=str(table))
        write_json(out/"design.json", dict(config=config, table_sha256=sha256(table), warmup_seed=1300,
                   seeds=seeds, target_rank=1, target_prefix=1024, target_query=128, cold_peer_rank=3,
                   cold_peer_prompt_lengths=[640, 2176], pool="queue-peer-probe-v2", output_tokens=2,
                   endpoint="enqueue to first token before decode", profiler=False))
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        frozen = llm.scheduler.restore_policy
        pairs = []
        for seed in (1300, *seeds):
            lengths = [512, 1024, 512, 512]
            base = inputs_for("queue-peer-probe-v2", seed, lengths)
            long_cold = inputs_for("queue-peer-cold-v2", seed, [2048])[-1][1][0]
            states = set()
            for size in ((640, 2176) if seed % 2 else (2176, 640)):
                replay = [*base[-1][1][:3], long_cold[:size]]
                trace = [*base[:2], ("evicted", replay)]
                observations = {}
                for target in ((None, 1) if seed % 2 else (1, None)):
                    policy = RecomputeOne(lengths, target)
                    tag = dict(seed=seed, warmup=seed == 1300, cold_peer_tokens=size, reject_index=target)
                    batches, state = run_cycle(llm, out, trace, policy, 2, tag, cold_peers=(3,))
                    states.add(state)
                    rows, steps, _ = batches[-1]
                    row = rows[1]
                    assert [r["cpu_restored_tokens"] for r in rows] == [512, 1024 if target is None else 0, 512, 0]
                    assert rows[3]["prompt_computed_tokens"] == size and row["initial_cpu_hit_tokens"] == 1024
                    shape = policy.first[1]
                    assert shape.peer_cpu_tokens == 1024 and shape.peer_query_tokens == size + 256
                    target_steps = [s for s in steps if s["is_prefill"] and row["seq_id"] in s["seq_ids"]]
                    step, = target_steps
                    assert max(r["token_s"][0] for r in rows) <= min(s["start_s"] for s in steps if not s["is_prefill"])
                    pred = frozen.decide(shape)
                    action = "restore" if target is None else "recompute"
                    observations[action] = dict(ms=row["ttft_ms"], step_query_tokens=step["scheduled_tokens"],
                        same_step_as_cold=rows[3]["seq_id"] in step["seq_ids"], shape=asdict(shape),
                        forecast_ms=pred.restore_ms if target is None else pred.recompute_ms, basis=pred.cost_basis)
                assert observations["restore"]["shape"] == observations["recompute"]["shape"]
                if seed != 1300:
                    pair = dict(seed=seed, cold_peer_tokens=size, state_sha256=state, **observations)
                    pairs.append(pair)
                    write_jsonl(out/"pairs.jsonl", [pair])
            assert len(states) == 1
        summary = {}
        for size in (640, 2176):
            matched = [p for p in pairs if p["cold_peer_tokens"] == size]
            summary[size] = {action: dict(n=len(matched), ttft_p50_ms=median(p[action]["ms"] for p in matched),
                forecast_ms=matched[0][action]["forecast_ms"], cost_basis=matched[0][action]["basis"],
                cold_in_same_step=sum(p[action]["same_step_as_cold"] for p in matched),
                step_query_tokens=sorted({p[action]["step_query_tokens"] for p in matched}))
                for action in ("restore", "recompute")}
        write_json(out/"status.json", dict(passed=True, target_pairs=len(pairs), summary=summary))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
