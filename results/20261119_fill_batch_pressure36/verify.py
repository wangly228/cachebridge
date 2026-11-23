"""Run from the repository root. Recheck the frozen inputs, outputs and paired p95s."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

run = Path(__file__).resolve().parent
design = json.loads((run / "design.json").read_text())
baseline = Path(design["trace"]).parent
summary = json.loads((run / "summary.json").read_text())


def formal(folder):
    rows = map(json.loads, (folder / "requests.jsonl").read_text().splitlines())
    return [r for r in rows if r["phase"] != "warmup"]


def p95(rows):
    values = sorted(r["ttft_ms"] for r in rows)
    pos = (len(values) - 1) * .95
    i = int(pos)
    return values[i] + (values[ceil(pos)] - values[i]) * (pos - i)


for folder, hashes in ((run, summary["raw_sha256"]), (baseline, design["frozen_sha256"])):
    for name, expected in hashes.items():
        assert sha256((folder / name).read_bytes()).hexdigest() == expected

refs = defaultdict(list)
for row in formal(baseline):
    refs[row["seed"], row["cycle"], row["phase"]].append(row)
raw = formal(run)
counts = Counter()
for r in raw:
    arm, seed, cycle, phase = (r[k] for k in ("arm", "seed", "cycle", "phase"))
    tag = arm, seed, cycle, phase
    i = counts[tag]
    counts[tag] += 1
    ref = refs[seed, cycle, "hot" if r["cpu_restored_tokens"] else phase][i]
    assert r["input_token_ids"] == refs[seed, cycle, phase][i]["input_token_ids"] == ref["input_token_ids"]
    assert r["output_token_ids"] == ref["output_token_ids"] and len(r["output_token_ids"]) == 128
    assert abs(r["ttft_ms"] - (r["token_s"][0] - r["enqueue_s"]) * 1000) < 1e-5
    assert r["preemptions"] == r["cpu_save_failures"] == r["cpu_restore_failures"] == 0
    assert r["d2h_batches"] == int(arm == "batch" and phase == "fill")
    assert r["batched_d2h_blocks"] == 8 * r["d2h_batches"]
assert len(counts) == 72 and set(counts.values()) == {4}

by_arm = {arm: [r for r in raw if r["arm"] == arm] for arm in design["arms"]}
for arm, requests in by_arm.items():
    for phase, group in summary["arms"][arm]["groups"].items():
        sample = [r for r in requests if phase == "all" or r["phase"] == phase]
        assert len(sample) == group["ttft_ms"]["n"]
        assert abs(p95(sample) - group["ttft_ms"]["p95"]) < 1e-7
        for field in ("d2h_blocks", "h2d_blocks", "d2h_batches", "batched_d2h_blocks", "prompt_computed_tokens"):
            assert sum(r[field] for r in sample) == group[field]
for old, new in zip(by_arm["reuse"], by_arm["batch"], strict=True):
    for field in ("seed", "cycle", "phase", "d2h_blocks", "h2d_blocks", "prompt_computed_tokens",
                  "initial_cpu_hit_tokens", "initial_gpu_hit_tokens", "output_token_ids"):
        assert old[field] == new[field]

pairs = []
for case in design["orders"]:
    seed, cycle = case["seed"], case["cycle"]
    delta = {}
    for phase in ("fill", "all"):
        latencies = {arm: p95([r for r in by_arm[arm] if (r["seed"], r["cycle"]) == (seed, cycle)
                               and (phase == "all" or r["phase"] == phase)]) for arm in ("reuse", "batch")}
        delta[phase] = latencies["batch"] - latencies["reuse"]
    pairs.append(dict(seed=seed, cycle=cycle, batch_minus_serial_p95_ms=delta))
slowest = sorted(by_arm["batch"], key=lambda r: r["ttft_ms"])[-6:]
print(json.dumps(dict(formal_requests=len(raw), validated_p95_groups=15, paired_cycles=pairs,
                      six_slowest_batch_phases=[r["phase"] for r in slowest],
                      matched_128_token_outputs=len(raw)), indent=2))
