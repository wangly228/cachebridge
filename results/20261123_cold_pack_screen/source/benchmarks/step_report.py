"""Audit boundary calibration, holdout policy comparisons and both observed target paths offline."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from statistics import median

from benchmarks.common import distribution, sha256, write_json, write_jsonl
from benchmarks.compare_restore_policy import stats
from benchmarks.queue_report import error_stats
from benchmarks.step_calibration import BASE_SHA, CAL_SEEDS, CAL_SIZES, EVAL_SEEDS, EVAL_SIZES, TARGET, observe
from benchmarks.step_evaluation import ARMS, PHASES, lines, load_cycle, read_json


def paired(differences):
    return dict(n=len(differences), median_ms=median(differences), min_ms=min(differences),
                max_ms=max(differences), negative=sum(d < 0 for d in differences))


def calibration_summary(cal):
    pairs = lines(cal/"pairs.jsonl")
    assert len([p for p in pairs if p["source"] == "archived"]) == 8
    assert {(p["seed"], p["cold_peer_tokens"]) for p in pairs if p["source"] == "new"} == {
        (s, z) for s in CAL_SEEDS for z in CAL_SIZES}
    groups = {}
    for size in sorted({p["cold_peer_tokens"] for p in pairs}):
        cases = [p for p in pairs if p["cold_peer_tokens"] == size]
        group = {}
        for action in ("restore", "recompute"):
            target = [p[action] for p in cases]
            assert all(o["work"] == target[0]["work"] for o in target)
            group[action] = dict(ttft_ms=distribution([o["ms"] for o in target]), work=target[0]["work"],
                                 wave_p95_ms=distribution([o["wave_p95_ms"] for o in target]))
        groups[size] = group
    return groups


def case_summary(cases):
    summary = {}
    for arm in ARMS:
        observed = [p["arms"][arm] for p in cases]
        summary[arm] = dict(target_ms=distribution([o["ms"] for o in observed]),
                            cold_peer_ms=distribution([o["cold_peer_ms"] for o in observed]),
                            within_wave_p95_ms=distribution([o["wave_p95_ms"] for o in observed]),
                            target_actions=dict(Counter(o["action"] for o in observed)))
    for ref in ("B", "C2"):
        summary[f"C3_minus_{ref}"] = {k: paired([p["arms"]["C3"][k]-p["arms"][ref][k] for p in cases])
                                       for k in ("ms", "wave_p95_ms", "cold_peer_ms")}
    for arm in ("C2", "C3"):
        errors = {a: [p["path_observations"][a]["ms"] - p["forecast"][arm][a+"_ms"] for p in cases]
                  for a in ("restore", "recompute")}
        summary[arm]["common_path_error"] = {k: error_stats(v) for k,v in errors.items()}
        summary[arm]["common_path_error"]["both"] = error_stats(errors["restore"] + errors["recompute"])
        summary[arm]["own_selected_path_error"] = error_stats([p["arms"][arm]["ms"]
            - p["forecast"][arm][p["arms"][arm]["action"] + "_ms"] for p in cases])
    switches = [p for p in cases if p["arms"]["C3"]["action"] != p["arms"]["B"]["action"]]
    summary["switches_vs_B"] = dict(n=len(switches),
        target_faster=sum(p["arms"]["C3"]["ms"] < p["arms"]["B"]["ms"] for p in switches),
        wave_p95_faster=sum(p["arms"]["C3"]["wave_p95_ms"] < p["arms"]["B"]["wave_p95_ms"] for p in switches))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cal, run = args.calibration, args.evaluation
    cs, status, design = read_json(cal/"status.json"), read_json(run/"status.json"), read_json(run/"design.json")
    assert cs["passed"] and status["passed"] and sha256(cal/"pairs.jsonl") == cs["pairs_sha256"]
    assert cs["table_sha256"] == sha256(cal/"cost_table.json") == sha256(run/"cost_table.json") == design["table_sha256"]
    assert sha256(run/"v2.json") == sha256(cal/"v2.json") == BASE_SHA
    assert read_json(run/"cost_table.json")["buckets"] == read_json(run/"v2.json")["buckets"]
    assert sha256(run/"trace.json") == design["trace_sha256"]
    trace = {(s,z): t for s,z,t in read_json(run/"trace.json")}
    requests, waves = defaultdict(list), defaultdict(list)
    orders, cases, path_checks, controls = defaultdict(set), [], 0, 0
    manifest = {}
    for seed in EVAL_SEEDS:
        for size in EVAL_SIZES:
            name = f"{seed}-{size}"
            folder = run/"measurements"/name
            observations, forecasts, states, positions = {}, {}, set(), {}
            for arm in ARMS:
                data, state = load_cycle(folder/arm)
                states.add(state)
                for (phase, inputs), (rows, steps, w) in zip(trace[seed,size], data, strict=True):
                    assert [r["input_token_ids"] for r in rows] == inputs and [len(r["output_token_ids"]) for r in rows] == [128]*4
                    assert all((r["phase"], r["seed"], r["cold_peer_tokens"], r["arm"]) == (phase,seed,size,arm) for r in rows)
                    assert all(r["preemptions"] == r["cpu_restore_failures"] == r["cpu_save_failures"] == 0 for r in rows)
                    assert w["output_tokens"] == 512 and not w["profiler_enabled"]
                    assert not w["warmup"] and w["phase"] == phase
                    positions[arm] = w["order_position"]
                    requests[arm].extend(rows)
                    waves[arm].append(w)
                for filename in ("requests.jsonl", "steps.jsonl", "waves.jsonl", "state.json"):
                    path = folder/arm/filename
                    manifest[str(path.relative_to(run))] = sha256(path)
                observation = observe(data[-1], size)
                observation["cold_peer_ms"] = data[-1][0][3]["ttft_ms"]
                observations[arm] = observation
                if arm != "B":
                    forecasts[arm], = data[-1][0][TARGET]["restore_decisions"]
                    assert forecasts[arm]["action"] == observation["action"]
            assert len(states) == 1 and set(positions.values()) == {0,1,2}
            order = tuple(sorted(positions, key=positions.get))
            orders[size].add(order)
            assert observations["B"]["action"] == "restore"
            both = {"restore": dict(observations["B"], source="B", output_tokens=128)}
            for arm in ("C3", "C2"):
                if observations[arm]["action"] == "recompute":
                    both["recompute"] = dict(observations[arm], source=arm, output_tokens=128)
                    break
            else:
                control_dir = run/"controls"/name
                control, state = load_cycle(control_dir)
                assert state in states and [r["input_token_ids"] for r in control[-1][0]] == trace[seed,size][-1][1]
                assert all(r["output_tokens"] == 2 for r in control[-1][0])
                measured = observe(control[-1], size)
                assert measured["action"] == "recompute"
                both["recompute"] = dict(measured, source="force_recompute", output_tokens=2)
                controls += 1
                for file in control_dir.iterdir():
                    manifest[str(file.relative_to(run))] = sha256(file)
            for action, obs in both.items():
                assert forecasts["C3"]["step_work"][action] == obs["work"]
                path_checks += 1
            cases.append(dict(seed=seed, cold_peer_tokens=size, state_sha256=states.pop(), order=order,
                         arms=observations, forecast=forecasts, path_observations=both))
    assert all(v == set(permutations(ARMS)) for v in orders.values())
    assert controls == status["two_output_counterfactuals"] and len(cases) == status["matched_cases"]
    exact = []
    for path in (run/"reference").glob("*/*/checks.jsonl"):
        exact.extend(lines(path))
        manifest[str(path.relative_to(run))] = sha256(path)
    assert len(exact) == status["extra_independent_reference_replays"] and all(r["exact"] for r in exact)
    groups = {}
    targets = {}
    for size in (None, *EVAL_SIZES):
        group = "pooled" if size is None else str(size)
        targets[group] = case_summary([p for p in cases if size is None or p["cold_peer_tokens"] == size])
        for phase in (None, *PHASES):
            def selected(r):
                return (size is None or r["cold_peer_tokens"] == size) and (phase is None or r["phase"] == phase)
            group_metrics = {}
            for arm in ARMS:
                selected_waves = list(filter(selected, waves[arm]))
                group_metrics[arm] = stats(list(filter(selected, requests[arm])), selected_waves)
                group_metrics[arm]["wall_s"] = sum(w["elapsed_s"] for w in selected_waves)
            for reference in ("B", "C2"):
                a, b = group_metrics["C3"], group_metrics[reference]
                deltas = [a["p95_ms_by_seed"][s] - b["p95_ms_by_seed"][s] for s in EVAL_SEEDS]
                group_metrics[f"C3_minus_{reference}"] = dict(
                    relative_p95_percent=100*(a["ttft_ms"]["p95"]/b["ttft_ms"]["p95"]-1),
                    paired_seed_p95_ms=paired(deltas), seed_deltas_ms=dict(zip(EVAL_SEEDS,deltas,strict=True)))
            groups[f"{group}/{phase or 'all'}"] = group_metrics
    for arm in ARMS:
        assert len(requests[arm]) == status["per_arm_requests"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output/"cases.jsonl", cases)
    write_json(args.output/"raw_sha256.json", manifest)
    write_json(args.output/"summary.json", dict(groups=groups, targets=targets, calibration=calibration_summary(cal),
               verified_step_projections=path_checks, status=status, cal_status=cs,
               cases_sha256=sha256(args.output/"cases.jsonl"), raw_manifest_sha256=sha256(args.output/"raw_sha256.json")))


if __name__ == "__main__":
    main()
