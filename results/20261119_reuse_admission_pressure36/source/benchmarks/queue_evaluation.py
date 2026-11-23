"""Counterbalanced mixed-queue comparison of B, frozen C1 and queue-calibrated C2."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from shutil import copyfile
import traceback

import torch

from nanovllm import LLM
from nanovllm.engine.restore_policy import CostShape, RestoreCostTable, cost_environment
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.queue_calibration import CAL_SEEDS, EVAL_LAYOUTS, EVAL_SEEDS, V1_SHA256, engine_config, run_cycle
from benchmarks.restore_calibration import inputs_for
from benchmarks.restore_holdout import oracle


def prefix_roots(path):
    with path.open() as f:
        return {tuple(row["input_token_ids"][:256]) for line in f if (row := json.loads(line))["phase"] == "evicted"}


def audit_cycle(out, seed, layout, batches, state, length_policy):
    branches = {}
    for phase_index, (b_rows, _, _) in enumerate(batches["B"]):
        for arm in ("C1", "C2"):
            rows = batches[arm][phase_index][0]
            for i, (ref, row) in enumerate(zip(b_rows, rows, strict=True)):
                assert ref["input_token_ids"] == row["input_token_ids"]
                if row["phase"] != "evicted":
                    assert ref["output_token_ids"] == row["output_token_ids"]
                    continue
                decision, = row["restore_decisions"]
                forecast = decision[decision["action"] + "_ms"]
                record = dict(seed=seed, layout=layout, request=i, arm=arm, seq_id=row["seq_id"], **decision,
                              observed_ttft_ms=row["ttft_ms"], computed_tokens=row["prompt_computed_tokens"],
                              h2d_blocks=row["h2d_blocks"], d2h_blocks=row["d2h_blocks"],
                              h2d_ms=row["h2d_ms"], d2h_ms=row["d2h_ms"],
                              error_ms=None if forecast is None else row["ttft_ms"] - forecast)
                if arm == "C2":
                    shadow = length_policy.decide(CostShape(**decision["shape"]).without_peers())
                    predicted = shadow.restore_ms if shadow.action == "restore" else shadow.recompute_ms
                    record.update(length_only_action=shadow.action, length_only_reason=shadow.reason,
                                  length_only_prediction_ms=predicted,
                                  length_only_error_ms=row["ttft_ms"]-predicted
                                  if shadow.action == decision["action"] and predicted is not None else None)
                write_jsonl(out / "decisions.jsonl", [record])
            if rows[0]["phase"] == "evicted":
                mask = tuple(i for i, r in enumerate(rows) if not r["cpu_restored_tokens"])
                if mask and mask not in branches:
                    branches[mask] = (arm, rows)
                else:
                    reference = branches[mask][1] if mask else b_rows
                    assert [r["output_token_ids"] for r in reference] == [r["output_token_ids"] for r in rows]
    write_jsonl(out / "matched_states.jsonl", [dict(seed=seed, layout=layout, state_sha256=state)])
    return list(branches.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.queue_evaluation"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        calibration = Path(args.calibration)
        status = json.loads((calibration / "status.json").read_text())
        assert status["passed"]
        for name, digest in (("v1.json", V1_SHA256), ("length_only.json", status["length_only_sha256"]),
                             ("cost_table.json", status["table_sha256"])):
            copyfile(calibration / name, out / name)
            assert sha256(out / name) == digest
        config = dict(engine_config(), cpu_restore_cost_table=str(out / "cost_table.json"))
        seeds = (1200, *EVAL_SEEDS)
        traces = {(seed, layout): inputs_for("queue-eval-v2", seed, lengths)
                  for seed in seeds for layout, lengths in EVAL_LAYOUTS.items()}
        write_json(out / "trace.json", [(s, w, tr) for (s, w), tr in traces.items()])
        roots = {tuple(p[:256]) for tr in traces.values() for phase, prompts in tr if phase == "evicted" for p in prompts}
        assert roots.isdisjoint(prefix_roots(calibration / "requests.jsonl"))
        assert roots.isdisjoint(prefix_roots(Path("results/20261118_stage2_c_holdout/C/requests.jsonl")))
        assert roots.isdisjoint(prefix_roots(Path("results/20261118_stage2_c_calibration/requests.jsonl")))
        assert not set(CAL_SEEDS).intersection(seeds)
        orders = list(permutations(("B", "C1", "C2")))
        write_json(out / "design.json", dict(config=config, pool="queue-eval-v2", layouts=EVAL_LAYOUTS,
                   seeds=EVAL_SEEDS, warmup_seed=1200, output_tokens=128, warmup_output_tokens=2,
                   profiler=False, orders=orders, alternating_layout_order=True, base_sha256=V1_SHA256,
                   table_sha256=status["table_sha256"], length_only_sha256=status["length_only_sha256"],
                   trace_sha256=sha256(out / "trace.json"), disjoint_first_blocks=len(roots),
                   whole_trace="sum of enqueue-to-completion wave times across fill, interference and evicted replay"))
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        env = cost_environment(llm.config, llm.scheduler.block_manager.namespace)
        assert llm.model_runner.kv_cache.dtype == torch.bfloat16 and 4 in llm.model_runner.graphs
        policies = dict(B=None, C1=RestoreCostTable.load(out / "v1.json", env), C2=llm.scheduler.restore_policy)
        length_policy = RestoreCostTable.load(out / "length_only.json", env)
        assert len(policies["C1"].buckets) == 11 and len(policies["C2"].buckets) == 24
        write_json(out / "engine.json", dict(environment=env, budget=llm.model_runner.kv_cache_budget,
                   graph_bs=llm.model_runner.graph_bs, dtype=str(llm.model_runner.kv_cache.dtype)))
        to_check = []
        for seed in seeds:
            warmup = seed == 1200
            layouts = list(EVAL_LAYOUTS)
            if seed % 2 == 0:
                layouts.reverse()
            for layout in layouts:
                trace = traces[seed, layout]
                offset = 0 if layout == "mixed" else 3
                order = orders[(seed - EVAL_SEEDS[0] + offset) % len(orders)]
                batches, states = {}, set()
                for position, arm in enumerate(order):
                    tag = dict(seed=seed, layout=layout, arm=arm, order_position=position, warmup=warmup)
                    result, state = run_cycle(llm, out, trace, policies[arm], 2 if warmup else 128, tag)
                    batches[arm] = result
                    states.add(state)
                assert len(states) == 1
                if not warmup:
                    state, = states
                    for arm, rows in audit_cycle(out, seed, layout, batches, state, length_policy):
                        to_check.append((arm, layout, seed, trace, rows, state))
                print(f"seed={seed} layout={layout} order={','.join(order)} complete", flush=True)
        # Different execution shapes have distinct BF16 references. These checks are untimed.
        llm.scheduler.restore_policy = None
        for arm, layout, seed, trace, rows, state in to_check:
            dest = out / "reference" / arm / layout
            dest.mkdir(parents=True, exist_ok=True)
            oracle(llm, dest, seed, trace, rows, state, workload=layout, setup_output_tokens=2)
        replay_n = len(EVAL_SEEDS)*sum(map(len, EVAL_LAYOUTS.values()))
        write_json(out / "status.json", dict(passed=True, replay_requests_per_arm=replay_n,
                   all_requests_per_arm=len(EVAL_SEEDS)*sum(2*len(v)+4 for v in EVAL_LAYOUTS.values()),
                   verified_C1_C2_greedy_requests=2*len(EVAL_SEEDS)*sum(2*len(v)+4 for v in EVAL_LAYOUTS.values()),
                   independent_reference_replays=sum(len(rows) for _, _, _, _, rows, _ in to_check)))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
