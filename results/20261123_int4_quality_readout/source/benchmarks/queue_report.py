"""Recompute the matched mixed-queue experiment results from archived request and wave records."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from benchmarks.common import sha256, write_json
from benchmarks.compare_restore_policy import formal, read, stats
from benchmarks.queue_calibration import CAL_SEEDS, EVAL_LAYOUTS, EVAL_SEEDS, V1_SHA256


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def error_stats(errors):
    if not errors:
        return dict(n=0)
    return dict(n=len(errors), mae_ms=mean(abs(e) for e in errors), signed_mean_ms=mean(errors),
                absolute_p95_ms=float(np.percentile(np.abs(errors), 95)))


def predictions(records):
    by_arm = {arm: [r for r in records if r["arm"] == arm] for arm in ("C1", "C2")}
    summary = {}
    for arm, rows in by_arm.items():
        summary[arm] = dict(reasons=dict(Counter(r["reason"] for r in rows)),
            cost_basis=dict(Counter(r["cost_basis"] or "unmeasured" for r in rows)),
            actions=dict(Counter(r["action"] for r in rows)),
            uncalibrated_recomputed_tokens=sum(r["recompute_candidate_tokens"] for r in rows
                                               if r["reason"] == "uncalibrated_shape"),
            error=error_stats([r["error_ms"] for r in rows if r["error_ms"] is not None]))
    def key(r):
        return r["seed"], r["layout"], r["request"]
    old = {key(r): r for r in by_arm["C1"]}
    new = {key(r): r for r in by_arm["C2"]}
    assert old.keys() == new.keys()
    common = [k for k in old if old[k]["error_ms"] is not None and new[k]["error_ms"] is not None]
    summary["common_prediction_support"] = dict(C1=error_stats([old[k]["error_ms"] for k in common]),
                                                C2=error_stats([new[k]["error_ms"] for k in common]))
    # Score the untimed length-only table against C2 wall times only when every
    # action matches in that wave. Changing a peer's path can change everyone's TTFT.
    waves = defaultdict(list)
    for r in new.values():
        waves[r["seed"], r["layout"]].append(r)
    matched = [rows for rows in waves.values() if all(r["action"] == r["length_only_action"] for r in rows)]
    scored = [r for rows in matched for r in rows if r["length_only_error_ms"] is not None and r["error_ms"] is not None]
    summary["length_only_on_C2_waves"] = dict(total_waves=len(waves), same_action_waves=len(matched),
        same_action_requests=sum(map(len, matched)), length_only=error_stats([r["length_only_error_ms"] for r in scored]),
        C2=error_stats([r["error_ms"] for r in scored]))
    return summary


def calibration_report(folder):
    groups = defaultdict(list)
    for row in lines(folder / "pairs.jsonl"):
        groups[row["layout"], row["target"]].append(row)
    summary = []
    for (layout, rank), items in groups.items():
        assert len(items) == len(CAL_SEEDS) and {r["seed"] for r in items} == set(CAL_SEEDS)
        assert all(r["shape"] == items[0]["shape"] for r in items)
        peers = [ms for r in items for ms in r["peer_ttft_change_ms"]]
        summary.append(dict(layout=layout, rank=rank, n=len(items), shape=items[0]["shape"],
            restore_p50_ms=median(r["restore_ms"] for r in items),
            recompute_p50_ms=median(r["recompute_ms"] for r in items),
            peer_TTFT_delta_p50_ms=median(peers) if peers else None))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cal, run = args.calibration, args.evaluation
    assert read(cal, "status.json")["passed"] and read(run, "status.json")["passed"]
    design = read(run, "design.json")
    assert design["table_sha256"] == sha256(run/"cost_table.json") == sha256(cal/"cost_table.json")
    assert design["length_only_sha256"] == sha256(run/"length_only.json") == sha256(cal/"length_only.json")
    assert sha256(run/"v1.json") == V1_SHA256 and sha256(run/"trace.json") == design["trace_sha256"]
    requests = formal(run, "requests.jsonl")
    waves = formal(run, "waves.jsonl")
    decisions = lines(run/"decisions.jsonl")
    traces = {(seed, layout, phase): inputs for seed, layout, trace in read(run, "trace.json")
              for phase, inputs in trace}
    matched_states = lines(run/"matched_states.jsonl")
    expected_waves = {(s, w) for s in EVAL_SEEDS for w in EVAL_LAYOUTS}
    assert len(matched_states) == len(expected_waves) and {(s["seed"], s["layout"]) for s in matched_states} == expected_waves
    arms = ("B", "C1", "C2")
    for layout in EVAL_LAYOUTS:
        orders = []
        for seed in EVAL_SEEDS:
            replay = [w for w in waves if (w["layout"], w["seed"], w["phase"]) == (layout, seed, "evicted")]
            orders.append(tuple(w["arm"] for w in sorted(replay, key=lambda w: w["order_position"])))
        assert set(orders) == set(permutations(arms))
    raw = {r["seq_id"]: r for r in requests}
    assert len(raw) == len(requests)
    for d in decisions:
        r = raw[d["seq_id"]]
        assert r["ttft_ms"] == d["observed_ttft_ms"] and r["cpu_restored_tokens"] == d["restored_tokens"]
    groups = {}
    for arm in arms:
        arm_rows = [r for r in requests if r["arm"] == arm]
        arm_waves = [w for w in waves if w["arm"] == arm]
        assert len(arm_rows) == read(run, "status.json")["all_requests_per_arm"]
        counts = Counter()
        for row in arm_rows:
            k = row["seed"], row["layout"], row["phase"]
            assert row["input_token_ids"] == traces[k][counts[k]] and len(row["output_token_ids"]) == 128
            counts[k] += 1
        for layout in (None, *EVAL_LAYOUTS):
            for phase in (None, "fill", "interference", "evicted"):
                def in_group(row):
                    return (layout is None or row["layout"] == layout) and (phase is None or row["phase"] == phase)
                rows, selected_waves = list(filter(in_group, arm_rows)), list(filter(in_group, arm_waves))
                name = f"{layout or 'pooled'}/{phase or 'all'}"
                s = stats(rows, selected_waves)
                s["wave_wall_s"] = sum(w["elapsed_s"] for w in selected_waves)
                groups.setdefault(name, {})[arm] = s
    paired = {}
    for name, group in groups.items():
        paired[name] = {}
        for reference in ("C1", "B"):
            a, b = group["C2"], group[reference]
            delta = {s: a["p95_ms_by_seed"][s]-b["p95_ms_by_seed"][s] for s in EVAL_SEEDS}
            paired[name][reference] = dict(pooled_change_percent=100*(a["ttft_ms"]["p95"]/b["ttft_ms"]["p95"]-1),
                seed_deltas_ms=delta, median_delta_ms=median(delta.values()), min_delta_ms=min(delta.values()),
                max_delta_ms=max(delta.values()), C2_faster_seeds=sum(d < 0 for d in delta.values()))
    summary = dict(groups=groups, paired_C2_minus_reference=paired,
        predictions={name: predictions([r for r in decisions if name == "pooled" or r["layout"] == name])
                     for name in ("pooled", *EVAL_LAYOUTS)},
        calibration=calibration_report(cal), status=read(run, "status.json"),
        raw_sha256={f: sha256(run/f) for f in ("requests.jsonl", "steps.jsonl", "waves.jsonl", "decisions.jsonl",
                                                "matched_states.jsonl", "trace.json", "v1.json", "cost_table.json")})
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"summary.json", summary)


if __name__ == "__main__":
    main()
