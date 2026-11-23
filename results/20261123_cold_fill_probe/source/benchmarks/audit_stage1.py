"""Audit the dated stage-1 acceptance evidence offline and hash its frozen inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.common import sha256, write_json


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archived-engine", action="store_true",
                        help="audit the frozen source snapshot after development has moved on")
    args = parser.parse_args()
    root = Path("results")
    pressure = root / "20261118_pressure36"
    fixed = root / "20261118_fixed_path_probe"
    metrics = root / "20261118_accept_metrics"
    probe = root / "20261118_accept_prefill"
    upstream = root / "20261118_accept_upstream"
    src = load(pressure / "environment.json")["source_sha256"]
    for rel, digest in src.items():
        assert sha256(pressure / "source" / rel) == digest, rel
        if rel.startswith("nanovllm/") and not args.archived_engine:
            assert sha256(rel) == digest, rel
    assert load(metrics / "design.json")["trace_sha256"] == sha256(pressure / "trace.json")
    originals = [r for r in rows(pressure / "requests.jsonl") if r["phase"] != "warmup"]
    indexed = {}
    for r in originals:
        indexed.setdefault((r["seed"], r["cycle"], r["phase"]), []).append(r["output_token_ids"])
    assert len(indexed) == 24 and all(len(batch) == 4 for batch in indexed.values())
    wave_rows = rows(metrics / "waves.jsonl")
    assert len({(r["seed"], r["cycle"], r["phase"], r["metrics"]) for r in wave_rows}) == len(wave_rows) == 48
    assert all(r["output_ids"] == indexed[r["seed"], r["cycle"], r["phase"]] for r in wave_rows)
    ref_outputs = rows(upstream / "pressure.jsonl")
    assert len(ref_outputs) == len(originals) == 96
    assert all(r["exact"] and r["output_ids"] == indexed[r["seed"], r["cycle"], r["phase"]][r["request"]]
               for r in ref_outputs)
    ref_rows = rows(upstream / "fixed_path.jsonl")
    assert len({(r["group"], r["cache"], r["request"], r["position"]) for r in ref_rows}) == len(ref_rows) == 288
    assert all(r["exact"] for r in ref_rows)
    outliers = {(r["group"], r["request"], r["position"]) for r in rows(fixed / "comparisons.jsonl") if not r["passed"]}
    comparisons = rows(probe / "logit_comparisons.jsonl")
    assert all(r["equal"] for r in comparisons)
    for mode in ("cold", "warm", "warm_remap", "cold_tail"):
        subset = [r for r in comparisons if r["mode"] == mode]
        assert len({(r["group"], r["request"], r["position"]) for r in subset}) == len(subset) == 144
        assert {(r["group"], r["request"], r["position"]) for r in subset if r["v1_outlier"]} == outliers
    decode = [r for r in comparisons if r["mode"] == "cold_tail" and r["position"] > 0]
    assert len(decode) == 135 and all(r["expected"] == "cold" for r in decode)
    pairs = rows(metrics / "pairs.jsonl")
    off = sum(r["elapsed_s"] for r in wave_rows if not r["metrics"])
    on = sum(r["elapsed_s"] for r in wave_rows if r["metrics"])
    assert len(pairs) == 6 and np.isclose(on/off-1, load(metrics / "summary.json")["delta_pct"]/100)
    assert not load(fixed / "status.json")["success"]
    files = set()
    for folder in (pressure, fixed, probe, metrics, upstream, root / "20261117_phase1_smoke_eager",
                   root / "20261117_kv_transfer", root / "20261117_kv_transfer_bounded32",
                   root / "20261117_prefill_profile", root / "20261118_decode_attention"):
        files.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        files.add(folder.with_suffix(".console.txt"))
    files.update(root / f for f in ("20261117_validation.json", "20261118_validation.json",
                                    "20261117_unit_tests.txt", "20261118_protocol_analysis_tests.txt"))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "SHA256SUMS"
    manifest.write_text("".join(f"{sha256(p)}  {p}\n" for p in sorted(files)))
    write_json(args.output / "decision.json", dict(
        baseline_A="accepted and frozen for the recorded 36-block configuration",
        cross_shape_v1="failed" if outliers else "passed",
        all_stage1_checks_passed=not outliers,
        completed_review=True, archived_pressure_requests=len(originals), upstream_exact_logit_rows=len(ref_rows),
        upstream_exact_greedy_requests=len(ref_outputs), original_screen_failures=len(outliers),
        cold_tail_exact_decode_positions=len(decode), metrics_on_off_wave_pairs=len(wave_rows)//2,
        metrics_on_vs_off_percent=100*(on/off-1), evidence_files=len(files),
        base_commit=load(upstream / "reference.json")["base"], manifest_sha256=sha256(manifest)))


if __name__ == "__main__":
    main()
