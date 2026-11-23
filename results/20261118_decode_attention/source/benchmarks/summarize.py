"""Read existing artifacts without rerunning GPU experiments; preserve failures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from benchmarks.common import distribution, p95_interval, write_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path, nargs="+")
    parser.add_argument("--prefill", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = dict(status="preliminary; stage 1 not frozen", artifacts={}, baseline=[], transfer=[])
    for path in [args.baseline, args.prefill, args.probe, *args.transfer]:
        report["artifacts"][str(path)] = json.loads((path / "status.json").read_text())
    groups, waves, paired = defaultdict(list), defaultdict(list), defaultdict(dict)
    for row in read_jsonl(args.baseline / "requests.jsonl"):
        if row["phase"] != "baseline":
            continue
        key = (row["length"], row["concurrency"], row["cache"])
        groups[key].append(row)
        paired[(row["length"], row["concurrency"], row["seed"], row["wave"], tuple(row["input_token_ids"]))][row["cache"]] = row["output_token_ids"]
    for row in read_jsonl(args.baseline / "waves.jsonl"):
        if row["phase"] == "baseline":
            waves[(row["length"], row["concurrency"], row["cache"])].append(row)
    for (length, concurrency, cache), rows in groups.items():
        runs = waves[(length, concurrency, cache)]
        values = [r["ttft_ms"] for r in rows]
        report["baseline"].append(dict(length=length, concurrency=concurrency, cache=cache,
            ttft_ms=distribution(values), p95_iid_bootstrap_95pct_ms=p95_interval(values),
            tpot_ms=distribution([r["tpot_ms"] for r in rows]),
            itl_ms=distribution([t for r in rows for t in r["itl_ms"]]),
            measured_waves_output_tokens_per_s=sum(r["output_tokens"] for r in runs) / sum(r["elapsed_s"] for r in runs),
            elapsed_s=sum(r["elapsed_s"] for r in runs),
            prefill_tokens=sum(r["prefill_tokens"] for r in rows),
            initial_gpu_hit_tokens=sum(r["initial_gpu_hit_tokens"] for r in rows),
            preemptions=sum(r["preemptions"] for r in rows),
            gpu_peak_allocated=max(r["memory"]["gpu_peak_allocated"] for r in runs),
            gpu_peak_reserved=max(r["memory"]["gpu_peak_reserved"] for r in runs),
            cpu_rss_max=max(r["memory"]["cpu_rss"] for r in runs),
            paired_greedy_mismatches=sum(v.get("cold") != v.get("warm") for k, v in paired.items() if k[:2] == (length, concurrency))))
    for path in args.transfer:
        groups = defaultdict(list)
        for row in read_jsonl(path / "transfer.jsonl"):
            groups[(row["strategy"], row["blocks"])].append(row)
        for (strategy, blocks), rows in groups.items():
            report["transfer"].append(dict(run=str(path), strategy=strategy, blocks=blocks,
                bytes_per_direction=rows[0]["bytes_per_direction"],
                d2h_wall_ms=distribution([r["d2h"]["wall_ms"] for r in rows]),
                h2d_wall_ms=distribution([r["h2d"]["wall_ms"] for r in rows])))
    report["prefill"] = json.loads((args.prefill / "costs.json").read_text())
    report["probe"] = json.loads((args.probe / "comparison.json").read_text())
    report["limitations"] = [
        "Baseline run failed strict greedy equality in one C=4/L=2048 request; do not mark it accepted.",
        "Baseline n=4 and prefill n=5 per condition; p95/bootstrap are descriptive only, IID assumption ignores within-wave correlation.",
        "Cold/warm waves are separate timing windows; cold population cost is disclosed separately from warm replay.",
        "Initial transfer run failed at 32-block pinned allocation; successful earlier points are retained, not a successful whole run.",
        "Bounded 32-block path differs from fully pinned small transfers and includes pageable CPU copies.",
        "No CPU cache, CUDA Graph baseline, multi-seed formal run, or GPU-capacity-pressure trace has been accepted.",
    ]
    write_json(args.output, report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
