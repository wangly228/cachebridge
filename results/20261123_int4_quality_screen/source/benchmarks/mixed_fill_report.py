"""Rebuild the mixed-write audit and paired-seed latency comparison from raw logs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from benchmarks.common import distribution, sha256
from benchmarks.compare_restore_policy import stats
from benchmarks.mixed_fill import ARMS, SEEDS, check_case, check_inputs
from benchmarks.write_admission import jsonl


def audit(out, trace, requests, waves, cycles):
    cases = [trace["warmup"], *trace["formal"]]
    check_inputs(cases)
    request_groups, step_groups = defaultdict(list), defaultdict(list)

    def wave_key(row):
        return row["seed"], row["cycle"], row["arm"], row["phase"]

    for row in requests:
        request_groups[wave_key(row)].append(row)
    prefill_shapes = {True: set(), False: set()}
    for step in jsonl(out / "steps.jsonl"):
        step_groups[wave_key(step)].append(step)
        if step["is_prefill"]:
            prefill_shapes[step["warmup"]].add((step["scheduled_tokens"], len(step["seq_ids"])))
    assert prefill_shapes[False] <= prefill_shapes[True]
    expected_waves = [(c["seed"], c["cycle"], a, p) for c in cases for a in c["arms"] for p, _ in c["phases"]]
    assert [wave_key(w) for w in waves] == expected_waves
    assert set(request_groups) == set(step_groups) == set(expected_waves)
    assert [(c["seed"], c["cycle"], c["arm"]) for c in cycles] == [k[:3] for k in expected_waves[::5]]
    for row in [*requests, *waves, *cycles]:
        assert row["warmup"] == (row["seed"] == trace["warmup"]["seed"])
    assert sorted((c["seed"], c["cycle"]) for c in trace["formal"]) == [(s, c) for s in SEEDS for c in (0, 1)]
    for cycle in (0, 1):
        paired = [c for c in trace["formal"] if c["cycle"] == cycle]
        assert len({tuple(c["arms"]) for c in paired}) == 6
        for phase in ("fill", "pressure1", "pressure2"):
            assert len({tuple(map(len, dict(c["phases"])[phase])) for c in paired}) == 6
    for w in waves:
        assert not w["profiler_enabled"] and w["used_blocks_peak"] <= 36
        assert w["cpu_cache_enabled"] == (w["arm"] != "A")
        if w["cpu_cache_enabled"]:
            assert w["cpu_resident_blocks"] <= w["cpu_capacity_blocks"] == 64
    matched_waves = 0
    for c in cases:
        root = c["seed"], c["cycle"]
        warmup = c["seed"] == trace["warmup"]["seed"]
        by_arm = {a: {p: request_groups[(*root, a, p)] for p, _ in c["phases"]} for a in ARMS}
        check_case(c, by_arm, 8 if warmup else 128)

        def model_steps(arm, phase):
            queue = {r["seq_id"]: i for i, r in enumerate(by_arm[arm][phase])}
            return [(s["is_prefill"], s["scheduled_tokens"], tuple(queue[i] for i in s["seq_ids"]))
                    for s in step_groups[(*root, arm, phase)]]

        for arm in ("reuse", "batch"):
            for phase, _ in c["phases"]:
                native = "hot" if phase == "evicted" else phase
                assert model_steps(arm, phase) == model_steps("A", native), (*root, arm, phase)
                if not warmup:
                    matched_waves += 1
    return dict(formal_requests=sum(not r["warmup"] for r in requests),
                excluded_warmup_requests=sum(r["warmup"] for r in requests),
                formal_prefill_shapes_covered_in_warmup=sorted(prefill_shapes[False]),
                cpu_waves_with_matching_native_steps=matched_waves,
                cpu_requests_with_matching_native_greedy=matched_waves * 4)


def subset(rows, group):
    if group == "all":
        return rows
    if group == "post_fill":
        return [r for r in rows if r["phase"] != "fill"]
    if group == "no_reuse":
        return [r for r in rows if r["phase"] in ("pressure1", "pressure2")]
    if group == "evicted_before_hot":
        return [r for r in rows if r["phase"] == "evicted" and r["cycle"] == 1]
    return [r for r in rows if r["phase"] == group]


def percent(new, base):
    return 100 * (new / base - 1)


def comparison(samples, draws):
    p95, boot, seed_p95 = {}, {}, {}
    for arm, rows in samples.items():
        clusters = np.asarray([[r["ttft_ms"] for r in rows if r["seed"] == s] for s in SEEDS])
        p95[arm] = np.percentile(clusters, 95)
        seed_p95[arm] = np.percentile(clusters, 95, axis=1)
        boot[arm] = np.percentile(clusters[draws].reshape(len(draws), -1), 95, axis=1)
    result = {}
    for base in ("A", "reuse"):
        paired = percent(boot["batch"], boot[base])
        result[f"batch_vs_{base}"] = dict(p95_change_pct=float(percent(p95["batch"], p95[base])),
            paired_seed_interval_pct=np.percentile(paired, [2.5, 97.5]).tolist(),
            by_seed_pct=dict(zip(SEEDS, percent(seed_p95["batch"], seed_p95[base]).tolist(), strict=True)))
    return result


def report(out):
    files = ("requests.jsonl", "waves.jsonl", "steps.jsonl", "cycles.jsonl", "trace.json",
             "design.json", "engine.json", "environment.json", "status.json")
    hashes = {name: sha256(out / name) for name in files}
    design = json.loads((out / "design.json").read_text())
    assert hashes["trace.json"] == design["trace_sha256"]
    assert json.loads((out / "status.json").read_text())["measurement_complete"]
    trace = json.loads((out / "trace.json").read_text())
    requests, waves, cycles = (jsonl(out / name) for name in ("requests.jsonl", "waves.jsonl", "cycles.jsonl"))
    checks = audit(out, trace, requests, waves, cycles)
    formal = [r for r in requests if not r["warmup"]]
    by_arm = {a: [r for r in formal if r["arm"] == a] for a in ARMS}
    groups = ("fill", "hot", "pressure1", "pressure2", "no_reuse", "evicted", "evicted_before_hot", "post_fill", "all")
    draws = np.random.default_rng(1700).integers(len(SEEDS), size=(4000, len(SEEDS)))
    arms, changes = {}, {}
    for arm, raw in by_arm.items():
        times = [w for w in waves if w["arm"] == arm and not w["warmup"]]
        wall = sum(c["wall_s"] for c in cycles if c["arm"] == arm and not c["warmup"])
        phases = {}
        for group in groups:
            rows = subset(raw, group)
            group_waves = subset(times, group)
            data = stats(rows, group_waves)
            data["waves"] = len(group_waves)
            data["wave_wall_s"] = sum(w["elapsed_s"] for w in group_waves)
            data["peak_active_gpu_blocks"] = max(w["used_blocks_peak"] for w in group_waves)
            if arm != "A":
                data["end_wave_cpu_resident_blocks"] = sorted({w["cpu_resident_blocks"] for w in group_waves})
                data["max_cpu_evictions_since_reset"] = max(w["cpu_evictions_since_reset"] for w in group_waves)
            for key in ("d2h_batches", "batched_d2h_blocks"):
                data[key] = sum(r[key] for r in rows)
            data["by_prefix_length"] = {}
            for length in (1024, 2048):
                sample = [r for r in rows if r["prompt_tokens"] == length + 128]
                data["by_prefix_length"][length] = dict(ttft_ms=distribution([r["ttft_ms"] for r in sample]),
                    d2h_ms=sum(r["d2h_ms"] for r in sample), d2h_blocks=sum(r["d2h_blocks"] for r in sample),
                    d2h_batches=sum(r["d2h_batches"] for r in sample),
                    batched_d2h_blocks=sum(r["batched_d2h_blocks"] for r in sample),
                    one_full_write_group_requests=sum(r["d2h_batches"] == 1 and r["batched_d2h_blocks"] == length // 256
                                                      for r in sample))
            phases[group] = data
        worst = sorted(raw, key=lambda r: r["ttft_ms"], reverse=True)[:len(raw) // 20]
        arms[arm] = dict(groups=phases, wall_s=wall, output_tokens_per_wall_s=sum(r["output_tokens"] for r in raw) / wall,
                         slowest_5pct_by_phase=Counter(r["phase"] for r in worst))
    for group in groups:
        changes[group] = comparison({a: subset(raw, group) for a, raw in by_arm.items()}, draws)
    paired_cycles = []
    for c in trace["formal"]:
        item = dict(seed=c["seed"], cycle=c["cycle"])
        for group in ("fill", "all"):
            pair = [distribution([r["ttft_ms"] for r in subset(by_arm[a], group)
                                  if (r["seed"], r["cycle"]) == (c["seed"], c["cycle"])])["p95"]
                    for a in ("reuse", "batch")]
            item[group + "_batch_vs_reuse_pct"] = percent(pair[1], pair[0])
        paired_cycles.append(item)
    limits = design["gates"]
    gates = dict(fill_lower=changes["fill"]["batch_vs_reuse"]["p95_change_pct"] < limits["fill_vs_serial_pct_lt"],
                 whole_lower=changes["all"]["batch_vs_reuse"]["p95_change_pct"] < limits["whole_vs_serial_pct_lt"],
                 no_reuse_within_limit=changes["no_reuse"]["batch_vs_A"]["p95_change_pct"] <= limits["pooled_no_reuse_vs_native_pct_le"],
                 evicted_gain_reached=changes["evicted"]["batch_vs_A"]["p95_change_pct"] <= limits["evicted_vs_native_pct_le"])
    return dict(audit=checks, gates=gates, arms=arms, comparison=changes, report_sha256=sha256(__file__),
                paired_cycles=paired_cycles, raw_sha256=hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(report(parser.parse_args().run), indent=2))
