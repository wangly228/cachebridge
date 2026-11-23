"""Measure a target's two paths at both prefill admission boundaries, reusing the prior probe."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from shutil import copyfile
from statistics import median
import traceback

import numpy as np
import torch

from nanovllm import LLM
from nanovllm.engine.restore_policy import CostShape
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.queue_calibration import RecomputeOne, engine_config, run_cycle
from benchmarks.restore_calibration import inputs_for


BASE_TABLE = Path("results/20261118_queue_c_calibration/cost_table.json")
BASE_SHA = "1d859e7823dd9f8de8a3323e3283858f17f74e380772e3bbfd8a7516069dc9c9"
OLD_PROBE = Path("results/20261118_queue_c_peer_probe")
LENGTHS = [512, 1024, 512, 512]
TARGET = 1
SHAPE = CostShape(0, 1024, 0, 128, 4, 1)
CAL_SIZES = (512, 639, 641, 1663, 1664, 1665)
CAL_SEEDS = (1401, 1402, 1403, 1404)
EVAL_SIZES = (608, 704, 1600, 1728)
EVAL_SEEDS = (1501, 1502, 1503, 1504, 1505, 1506)


def trace_for(pool, seed, size):
    base = inputs_for(pool, seed, LENGTHS)
    cold = inputs_for(pool + "-cold", seed, [2048])[-1][1][0]
    return [*base[:2], ("evicted", [*base[-1][1][:3], cold[:size]])]


def observe(batch, size):
    rows, steps, _ = batch
    target = rows[TARGET]
    action = "restore" if target["cpu_restored_tokens"] else "recompute"
    expected = [512, 1024 if action == "restore" else 0, 512, 0]
    assert [r["cpu_restored_tokens"] for r in rows] == expected
    assert target["initial_cpu_hit_tokens"] == 1024 and rows[3]["prompt_computed_tokens"] == size
    step, = [s for s in steps if s["is_prefill"] and target["seq_id"] in s["seq_ids"]]
    assert max(r["token_s"][0] for r in rows) <= min(s["start_s"] for s in steps if not s["is_prefill"])
    admitted = [r for r in rows if r["seq_id"] in step["seq_ids"]]
    same_step = rows[3] in admitted
    assert same_step == (size <= (1664 if action == "restore" else 640))
    assert [r["seq_id"] for r in admitted] == [r["seq_id"] for r in rows[:3 + same_step]]
    work = dict(seqs=len(admitted), query_tokens=step["scheduled_tokens"],
                cpu_tokens=sum(r["cpu_restored_tokens"] for r in admitted))
    assert work["query_tokens"] == sum(r["prompt_computed_tokens"] for r in admitted)
    return dict(ms=target["ttft_ms"], action=action, work=work, target_seq_id=target["seq_id"],
                cold_same_step=same_step, wave_p95_ms=float(np.percentile([r["ttft_ms"] for r in rows], 95)))


def old_pairs():
    status = json.loads((OLD_PROBE/"status.json").read_text())
    assert status["passed"] and sha256(OLD_PROBE/"cost_table.json") == BASE_SHA
    archived = {}
    for line in (OLD_PROBE/"pairs.jsonl").read_text().splitlines():
        p = json.loads(line)
        archived[p["seed"], p["cold_peer_tokens"]] = p
    replay = {}
    for name in ("requests", "steps"):
        grouped = defaultdict(list)
        for line in (OLD_PROBE/f"{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if not row["warmup"] and row["phase"] == "evicted":
                grouped[row["seed"], row["cold_peer_tokens"], row["reject_index"]].append(row)
        replay[name] = grouped
    assert len(archived) == status["target_pairs"] == 8
    for (seed, size), pair in archived.items():
        item = dict(seed=seed, cold_peer_tokens=size, state_sha256=pair["state_sha256"], source="archived")
        for rejected in (None, TARGET):
            key = seed, size, rejected
            obs = observe((replay["requests"][key], replay["steps"][key], None), size)
            assert obs["ms"] == pair[obs["action"]]["ms"]
            item[obs["action"]] = obs
        yield item


def fit(pairs, base):
    # Several deferred cold lengths give the same step. Each seed gets one vote per anchor.
    groups = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        for action in ("restore", "recompute"):
            obs = pair[action]
            w = obs["work"]
            groups[action, w["seqs"], w["cpu_tokens"], w["query_tokens"]][pair["seed"]].append(obs["ms"])
    anchors = []
    for (action, seqs, cpu, query), seeds in sorted(groups.items()):
        data = [median(samples) for samples in seeds.values()]
        assert len(data) >= 4
        anchors.append(dict(shape=asdict(SHAPE), action=action, seqs=seqs, cpu_tokens=cpu, query_tokens=query,
                      seed_count=len(data), p10_ms=float(np.percentile(data, 10)), p50_ms=median(data),
                      p90_ms=float(np.percentile(data, 90))))
    return dict(base, version=3, step_anchors=anchors, parent_sha256=BASE_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.step_calibration"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        assert sha256(BASE_TABLE) == BASE_SHA
        copyfile(BASE_TABLE, out/"v2.json")
        base = json.loads((out/"v2.json").read_text())
        pairs = list(old_pairs())
        write_jsonl(out/"pairs.jsonl", pairs)
        config = dict(engine_config(), cpu_restore_cost_table=str(out/"v2.json"))
        write_json(out/"design.json", dict(config=config, base_sha256=BASE_SHA,
                   archived_pairs_sha256=sha256(OLD_PROBE/"pairs.jsonl"), archived_seeds=[1301,1302,1303,1304],
                   new_seeds=CAL_SEEDS, warmup_seed=1400, new_sizes=CAL_SIZES, archived_sizes=[640,2176],
                   evaluation_seeds=EVAL_SEEDS, evaluation_sizes=EVAL_SIZES, output_tokens=2, profiler=False,
                   pool="step-cal-v3", thresholds="restore p90 < recompute p10; interpolate within measured step",
                   endpoints="target TTFT, replay p95, all-phase p95; separate path accuracy on holdout",
                   evaluation="B, fixed C2 and C3; six orders per size; new inputs, 128 greedy outputs"))
        torch.manual_seed(0)
        llm = LLM(args.model, **config)
        assert llm.model_runner.kv_cache.dtype == torch.bfloat16 and llm.config.num_kvcache_blocks == 36
        for seed in (1400, *CAL_SEEDS):
            sizes = CAL_SIZES if seed % 2 else CAL_SIZES[::-1]
            if seed == 1400:
                sizes = (639, 1663, 1665)
            states = set()
            for size in sizes:
                trace = trace_for("step-cal-v3", seed, size)
                pair = dict(seed=seed, cold_peer_tokens=size, source="new")
                for rejected in ((None, TARGET) if (seed + CAL_SIZES.index(size)) % 2 else (TARGET, None)):
                    tag = dict(seed=seed, cold_peer_tokens=size, reject_index=rejected, warmup=seed==1400)
                    policy = RecomputeOne(LENGTHS, rejected)
                    batches, state = run_cycle(llm, out, trace, policy, 2, tag, cold_peers=(3,))
                    states.add(state)
                    obs = observe(batches[-1], size)
                    assert policy.first[TARGET].without_peers() == SHAPE
                    assert (obs["action"] == "restore") == (rejected is None)
                    pair[obs["action"]] = obs
                pair["state_sha256"] = state
                assert len(states) == 1
                if seed != 1400:
                    pairs.append(pair)
                    write_jsonl(out/"pairs.jsonl", [pair])
            print(f"seed={seed} sizes={len(sizes)} complete", flush=True)
        table = fit(pairs, base)
        write_json(out/"cost_table.json", table)
        write_json(out/"status.json", dict(passed=True, archived_pairs=8, new_pairs=len(pairs)-8,
                   anchors=len(table["step_anchors"]), table_sha256=sha256(out/"cost_table.json"),
                   pairs_sha256=sha256(out/"pairs.jsonl")))
    except BaseException:
        (out/"failure.txt").write_text(traceback.format_exc())
        write_json(out/"status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
