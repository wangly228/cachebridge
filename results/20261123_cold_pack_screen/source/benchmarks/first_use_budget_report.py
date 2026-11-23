"""Audit the budget experiment and summarize latency, saved writes and lost restores."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from benchmarks.common import sha256
from benchmarks.compare_restore_policy import stats
from benchmarks.first_use_budget import ARMS, SEEDS, cases, validate_trace, verify
from benchmarks.mixed_fill_report import percent, subset
from benchmarks.write_admission import jsonl


def wave_key(row):
    return row["seed"], row["cycle"], row["workload"], row["arm"], row["phase"]


def load_batches(folder):
    requests, steps, waves = [jsonl(folder / n) for n in ("requests.jsonl", "steps.jsonl", "waves.jsonl")]
    grouped_requests, grouped_steps = defaultdict(list), defaultdict(list)
    for r in requests:
        grouped_requests[wave_key(r)].append(r)
    for s in steps:
        grouped_steps[wave_key(s)].append(s)
    shapes = {True: set(), False: set()}
    for s in steps:
        if s["is_prefill"]:
            shapes[s["warmup"]].add((s["scheduled_tokens"], len(s["seq_ids"])))
    assert shapes[False] <= shapes[True]
    batches = {}
    for w in waves:
        k = wave_key(w)
        assert k not in batches and len(grouped_requests[k]) == 4 and grouped_steps[k]
        assert not w["profiler_enabled"] and w["used_blocks_peak"] <= 36
        assert w["cpu_cache_enabled"] == (w["arm"] in ("24", "32", "36"))
        if w["cpu_cache_enabled"]:
            assert w["cpu_resident_blocks"] <= w["cpu_capacity_blocks"] == 64
        batches[k] = (grouped_requests[k], grouped_steps[k], w)
    assert set(batches) == set(grouped_requests) == set(grouped_steps)
    return requests, waves, batches


def audit(out, trace, batches, refs, cycles):
    validate_trace(trace)
    assert trace == json.loads(json.dumps(cases()))
    expected_cycles, expected_waves, expected_refs = [], [], []
    for part, items in trace.items():
        warmup = part == "warmup"
        for c in items:
            base = c["seed"], c["cycle"], c["workload"]
            lookup = {}
            for arm in c["arms"]:
                expected_cycles.append((*base, arm))
                lookup[arm] = {}
                for phase, _ in c["phases"]:
                    key = (*base, arm, phase)
                    expected_waves.append(key)
                    rows, steps, wave = batches[key]
                    assert all(r["warmup"] == warmup for r in [*rows, *steps, wave])
                    lookup[arm][phase] = batches[key]
            partial = None
            if c["workload"] == "long" and c["cycle"] == 1:
                partial = {}
                for phase in ("fill", "evicted", "hot"):
                    key = (*base, "native_partial", phase)
                    expected_refs.append(key)
                    partial[phase] = refs[key]
                assert [r["initial_gpu_hit_tokens"] for r in partial["evicted"][0]] == [2048, 2048, 2048, 0]
                assert [r["initial_gpu_hit_tokens"] for r in partial["hot"][0]] == [2048] * 4
            verify(c, lookup, partial, 8 if warmup else 128)
    assert list(batches) == expected_waves and list(refs) == expected_refs
    assert [(c["seed"], c["cycle"], c["workload"], c["arm"]) for c in cycles] == expected_cycles
    for load in ("mixed", "long"):
        for cycle in (0, 1):
            runs = [c for c in trace["formal"] if c["workload"] == load and c["cycle"] == cycle]
            for i in range(4):
                assert {c["arms"][i] for c in runs} == set(ARMS)


def compare(samples, draws):
    values, boot, by_seed, by_cycle = {}, {}, {}, {}
    for arm, rows in samples.items():
        groups = np.asarray([[r["ttft_ms"] for r in rows if r["seed"] == seed] for seed in SEEDS])
        values[arm] = np.percentile(groups, 95)
        boot[arm] = np.percentile(groups[draws].reshape(len(draws), -1), 95, axis=1)
        by_seed[arm] = np.percentile(groups, 95, axis=1)
        paired = defaultdict(list)
        for r in rows:
            paired[r["seed"], r["cycle"]].append(r["ttft_ms"])
        by_cycle[arm] = np.asarray([np.percentile(paired[k], 95) for k in sorted(paired)])
    changes = {}
    for arm in ARMS[1:]:
        changes[arm] = {}
        for base in ("A", "36"):
            changes[arm][base] = dict(p95_change_pct=float(percent(values[arm], values[base])),
                by_seed_pct=dict(zip(SEEDS, percent(by_seed[arm], by_seed[base]).tolist(), strict=True)),
                lower_p95_cycles=int(np.sum(by_cycle[arm] < by_cycle[base])), paired_cycles=len(by_cycle[arm]),
                paired_seed_interval_pct=np.percentile(percent(boot[arm], boot[base]), [2.5, 97.5]).tolist())
    return changes


def summarize(rows, waves):
    data = stats(rows, waves)
    data["waves"] = len(waves)
    data["wave_wall_s"] = sum(w["elapsed_s"] for w in waves)
    data["full_cpu_restores"] = sum(r["cpu_restored_tokens"] == r["prompt_tokens"] - 128 for r in rows)
    data["target_prefix_blocks"] = sum((r["prompt_tokens"] - 128) // 256 for r in rows)
    data["d2h_batches"] = sum(r["d2h_batches"] for r in rows)
    data["batched_d2h_blocks"] = sum(r["batched_d2h_blocks"] for r in rows)
    data["end_cpu_resident_blocks"] = sorted({w.get("cpu_resident_blocks", 0) for w in waves})
    data["peak_active_gpu_blocks"] = max(w["used_blocks_peak"] for w in waves)
    data["max_cpu_evictions_since_reset"] = max(w.get("cpu_evictions_since_reset", 0) for w in waves)
    return data


def report(out):
    files = ("design.json", "trace.json", "environment.json", "engine.json", "status.json", "cycles.jsonl",
             "requests.jsonl", "waves.jsonl", "steps.jsonl", "reference/requests.jsonl",
             "reference/waves.jsonl", "reference/steps.jsonl")
    hashes = {name: sha256(out / name) for name in files}
    design, trace, status = [json.loads((out / name).read_text()) for name in ("design.json", "trace.json", "status.json")]
    assert status["measurement_complete"] and hashes["trace.json"] == design["trace_sha256"]
    requests, waves, batches = load_batches(out)
    _, _, refs = load_batches(out / "reference")
    cycles = jsonl(out / "cycles.jsonl")
    audit(out, trace, batches, refs, cycles)
    draws = np.random.default_rng(1800).integers(len(SEEDS), size=(4000, len(SEEDS)))
    workloads = {}
    for load in ("mixed", "long"):
        def sample(source, arm):
            return [r for r in source if not r["warmup"] and r["workload"] == load and r["arm"] == arm]
        stages = ["fill", "hot", "pressure1", "no_reuse", "evicted", "evicted_before_hot", "post_fill", "all"]
        if load == "mixed":
            stages.insert(3, "pressure2")
        arms, comparisons, gates = {}, {}, {}
        for arm in ARMS:
            raw, times = sample(requests, arm), sample(waves, arm)
            wall = sum(c["wall_s"] for c in sample(cycles, arm))
            groups = {p: summarize(subset(raw, p), subset(times, p)) for p in stages}
            expected_n = 160 if load == "mixed" else 128
            assert len(raw) == expected_n and len(times) == expected_n // 4
            worst = sorted(raw, key=lambda r: r["ttft_ms"], reverse=True)[:int(np.ceil(0.05 * len(raw)))]
            arms[arm] = dict(groups=groups, cycle_wall_s=wall,
                             output_tokens_per_wall_s=sum(r["output_tokens"] for r in raw) / wall,
                             slowest_5pct_by_phase=Counter(r["phase"] for r in worst))
        for phase in stages:
            comparisons[phase] = compare({arm: subset(sample(requests, arm), phase) for arm in ARMS}, draws)
        for arm in ARMS[1:]:
            g = arms[arm]["groups"]
            early = g["evicted_before_hot"]
            gates[arm] = dict(no_reuse=comparisons["no_reuse"][arm]["A"]["p95_change_pct"] <= design["goals_pct"]["no_reuse_vs_A_max"],
                              evicted=comparisons["evicted"][arm]["A"]["p95_change_pct"] <= design["goals_pct"]["evicted_vs_A_max"],
                              retains_early_replay=early["full_cpu_restores"] == early["ttft_ms"]["n"])
            arms[arm]["d2h_blocks_saved_vs_36"] = arms["36"]["groups"]["all"]["d2h_blocks"] - g["all"]["d2h_blocks"]
            arms[arm]["early_blocks_missing"] = early["target_prefix_blocks"] - early["h2d_blocks"]
        workloads[load] = dict(arms=arms, comparisons=comparisons, gates=gates)
    eligible = [int(arm) for arm in ARMS[1:] if all(all(w["gates"][arm].values()) for w in workloads.values())]
    formal = [r for r in requests if not r["warmup"]]
    assert len(formal) == status["formal_requests"]
    cpu_n = sum(r["arm"] != "A" for r in formal)
    assert cpu_n == status["formal_cpu_requests_with_native_reference"]
    return dict(workloads=workloads, smallest_passing_budget=min(eligible, default=None),
                audit=dict(timed_requests=len(formal), cpu_requests_with_same_shape_native_greedy=cpu_n,
                           formal_reference_requests=sum(len(rows) for rows, _, w in refs.values() if not w["warmup"]),
                           excluded_warmup_requests=len(requests)-len(formal)),
                raw_sha256=hashes, report_sha256=sha256(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(report(parser.parse_args().run), indent=2))
