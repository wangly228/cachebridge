"""Audit common inputs and report end-to-end A/B/C latency, decisions and prediction errors."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

from benchmarks.common import distribution, sha256, write_json, write_jsonl
from benchmarks.compare_cpu_trace import latency
from benchmarks.cpu_cache_pressure import BASELINE


def read(folder, name):
    return json.loads((folder / name).read_text())


def formal(folder, name):
    with (folder / name).open() as file:
        rows = (json.loads(line) for line in file)
        return [r for r in rows if r["phase"] != "warmup" and not r.get("warmup")]


def stats(rows, waves):
    result = latency(rows)
    seed_p95 = result["p95_ms_by_seed"].values()
    result["seed_p95_range_ms"] = [min(seed_p95), max(seed_p95)]
    result["output_tokens_per_engine_second"] = sum(r["output_tokens"] for r in rows) / sum(w["elapsed_s"] for w in waves)
    result["tpot_ms"] = distribution([r["tpot_ms"] for r in rows])
    result["itl_ms"] = distribution([v for r in rows for v in r["itl_ms"]])
    result["gpu_hit_requests"] = sum(r["initial_gpu_hit_tokens"] > 0 for r in rows)
    result["cpu_hit_requests"] = sum(r.get("initial_cpu_hit_tokens", 0) > 0 for r in rows)
    for k in ("initial_gpu_hit_tokens", "initial_cpu_hit_tokens", "cpu_restored_tokens", "prompt_computed_tokens",
              "prefill_tokens", "preemptions", "d2h_blocks", "h2d_blocks", "d2h_ms", "h2d_ms",
              "cpu_save_failures", "cpu_restore_failures"):
        result[k] = sum(r.get(k, 0) for r in rows)
    result["peak_memory_bytes"] = {k: max(w["memory"][k] for w in waves) for k in waves[0]["memory"]}
    return result


def group_names(row):
    names = ["all", row["phase"]]
    if "workload" in row and row["phase"] == "evicted":
        names.append("evicted_" + row["workload"])
    return names


def report(suite, folders, out):
    runs, summary, decisions = {}, {}, []
    raw_hashes = {}
    for strategy, folder in folders.items():
        requests = formal(folder, "requests.jsonl")
        runs[strategy] = requests
        waves = formal(folder, "waves.jsonl")
        groups, group_waves = defaultdict(list), defaultdict(list)
        for row in requests:
            for name in group_names(row):
                groups[name].append(row)
            for d in row.get("restore_decisions", []):
                assert strategy == "C" and d["allocation"] == 0
                prediction = d[d["action"] + "_ms"]
                error = None if prediction is None else row["ttft_ms"] - prediction
                decisions.append(dict(suite=suite, workload=row.get("workload", "frozen"), seed=row["seed"],
                    cycle=row.get("cycle", 0), seq_id=row["seq_id"], phase=row["phase"], **d,
                    actual_ttft_ms=row["ttft_ms"], actual_h2d_ms=row["h2d_ms"], actual_d2h_ms=row["d2h_ms"],
                    prompt_computed_tokens=row["prompt_computed_tokens"], actual_minus_predicted_ms=error))
        for w in waves:
            for name in group_names(w):
                group_waves[name].append(w)
        groups_summary = {name: stats(items, group_waves[name]) for name, items in groups.items()}
        wall = sum(c["cold_start_wall_s"] for c in read(folder, "cycles.json"))
        summary[strategy] = dict(groups=groups_summary, total_wall_s=wall,
                                 output_tokens_per_wall_second=sum(r["output_tokens"] for r in requests)/wall)
        raw_hashes[strategy] = {f: sha256(folder / f) for f in ("requests.jsonl", "waves.jsonl", "cycles.json", "engine.json")}
    expected = 96 if suite == "frozen" else 90
    assert all(len(rows) == expected for rows in runs.values())
    native_equal = mirror_equal = oracle_equal = 0
    mixed_checks = {}
    if suite == "holdout":
        path = folders["C"].parent / "mixed_reference/checks.jsonl"
        for line in path.read_text().splitlines():
            c = json.loads(line)
            mixed_checks[c["seed"], c["request"]] = c["exact"]
    index = Counter()
    for a, b, c in zip(runs["A"], runs["B"], runs["C"], strict=True):
        for key in ("seed", "cycle", "workload", "phase", "input_token_ids", "output_tokens"):
            assert a.get(key) == b.get(key) == c.get(key)
        native_equal += c["output_token_ids"] == a["output_token_ids"]
        mirror_equal += c["output_token_ids"] == b["output_token_ids"]
        group = (c["seed"], c.get("cycle"), c["phase"], c.get("workload"))
        if c.get("workload") == "mixed" and c["phase"] == "evicted":
            assert mixed_checks[c["seed"], index[group]]
        else:
            ref = b if c["cpu_restored_tokens"] or c["phase"] != "evicted" else a
            assert ref["output_token_ids"] == c["output_token_ids"]
        index[group] += 1
        oracle_equal += 1
    errors = [abs(d["actual_minus_predicted_ms"]) for d in decisions if d["actual_minus_predicted_ms"] is not None]
    write_jsonl(out / "decisions.jsonl", decisions)
    return dict(strategies=summary, source_runs={s: str(p) for s, p in folders.items()}, raw_sha256=raw_hashes,
                decision_counts=dict(Counter(d["reason"] for d in decisions)),
                predicted_requests=len(errors), mean_absolute_prediction_error_ms=mean(errors) if errors else None,
                same_phase_equal_to_A=native_equal, same_phase_equal_to_B=mirror_equal,
                same_shape_exact_requests=oracle_equal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--frozen-c", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--holdout-b", type=Path, help="use an independently repeated B arm with identical inputs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = sha256(args.calibration / "cost_table.json")
    assert digest == read(args.calibration, "status.json")["table_sha256"]
    assert digest == sha256(args.frozen_c / "cost_table.json") == sha256(args.holdout / "cost_table.json")
    assert read(args.holdout, "status.json")["passed"] and read(args.frozen_c, "status.json")["measurement_complete"]
    b_dir = Path("results/20261118_stage2_b_pressure36")
    assert read(args.frozen_c, "engine.json")["baseline_sha256"] == read(b_dir, "engine.json")["baseline_sha256"]
    holdout = {s: args.holdout / s for s in "ABC"}
    if args.holdout_b:
        for filename in ("trace.json", "cost_table.json"):
            assert sha256(args.holdout / filename) == sha256(args.holdout_b / filename)
        original, replacement = (read(p, "environment.json") for p in (args.holdout, args.holdout_b))
        for field in ("gpu", "packages", "cuda_runtime", "model_files"):
            assert original[field] == replacement[field]
        for name, source_digest in original["source_sha256"].items():
            if name.startswith("nanovllm/"):
                assert replacement["source_sha256"][name] == source_digest
        assert "B" in read(args.holdout_b, "status.json")["strategies"] and read(args.holdout_b, "status.json")["passed"]
        holdout["B"] = args.holdout_b / "B"
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(table_sha256=digest,
                  frozen=report("frozen", dict(A=BASELINE, B=b_dir, C=args.frozen_c), args.output),
                  holdout=report("holdout", holdout, args.output))
    write_json(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
