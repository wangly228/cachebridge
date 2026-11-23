"""Reconstruct cold-fill TTFT from the recorded 32/36-block mixed workload, offline."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np

from benchmarks.common import sha256, write_json, write_jsonl


ARCHIVE = Path("results/20261119_first_use_budget36")
ARMS = ("32", "36")
PARTS = ("ttft_ms", "runner_ms", "write_ms", "other_ms", "runner_before_admission_ms",
         "runner_from_admission_ms", "own_write_ms", "peer_write_ms", "outside_steps_ms", "inside_other_ms")
EPS = 1e-5  # milliseconds; tolerate conversion between absolute and wave-relative clocks


def selected(run, name):
    waves = defaultdict(list)
    with (run / name).open() as f:
        for line in f:
            r = json.loads(line)
            if not r["warmup"] and r["workload"] == "mixed" and r["phase"] == "fill" and r["arm"] in ARMS:
                waves[r["seed"], r["cycle"], r["arm"]].append(r)
    return waves


def timeline(rows, steps, prompts, block, budget):
    positions = {r["seq_id"]: i for i, r in enumerate(rows)}
    assert len(positions) == len(rows) == 4 and [r["input_token_ids"] for r in rows] == prompts
    prefill = [s for s in steps if s["is_prefill"]]
    assert steps[:len(prefill)] == prefill and max(r["token_s"][0] for r in rows) < steps[len(prefill)]["start_s"]
    for r in rows:
        assert r["initial_gpu_hit_tokens"] == r["initial_cpu_hit_tokens"] == r["h2d_blocks"] == r["preemptions"] == 0
        assert r["cpu_save_failures"] == r["cpu_restore_failures"] == 0 and r["allocations"] == r["d2h_batches"] == 1
        assert r["prefill_tokens"] == r["prompt_computed_tokens"] == r["prompt_tokens"] == len(r["input_token_ids"])
        assert r["d2h_blocks"] == r["batched_d2h_blocks"] == r["prompt_tokens"] // block
        assert r["enqueue_s"] <= prefill[0]["start_s"] and r["output_tokens"] == len(r["output_token_ids"]) == 128

    computed = [0] * len(rows)
    runner_at_admission, first_token, step_data = {}, {}, []
    runner = write = gaps = 0.0
    end = prefill[0]["start_s"]
    for j, s in enumerate(prefill, 1):
        start = s["start_s"]
        assert start >= end and 0 < s["run_ms"] <= s["step_ms"]
        gaps += 1000 * (start - end)
        end = start + s["step_ms"] / 1000
        step_tokens, writes, finishes = [], [], []
        step_write = 0.0
        for seq_id in s["seq_ids"]:  # Scheduler.postprocess publishes tokens in this order.
            i = positions[seq_id]
            r = rows[i]
            runner_at_admission.setdefault(i, runner)
            before = computed[i]
            n = min(r["prompt_tokens"] - before, budget - sum(step_tokens))
            assert n > 0
            step_tokens.append(n)
            computed[i] += n
            full_blocks = computed[i] // block - before // block
            writes.append(full_blocks)
            if full_blocks:
                # Each request made exactly one timed group write in these waves.
                assert full_blocks == r["d2h_blocks"] and r["d2h_ms"] > 0
                step_write += r["d2h_ms"]
            if computed[i] != r["prompt_tokens"]:
                assert r["token_s"][0] > end
                continue
            first = r["token_s"][0]
            assert start <= first <= end and i not in first_token
            finishes.append(i + 1)
            ttft = 1000 * (first - r["enqueue_s"])
            run_ms, write_ms = runner + s["run_ms"], write + step_write
            other = ttft - run_ms - write_ms
            outside = 1000 * (prefill[0]["start_s"] - r["enqueue_s"]) + gaps
            assert abs(ttft - r["ttft_ms"]) < EPS and other - outside >= -EPS
            assert 1000 * (first - start) - s["run_ms"] - step_write >= -EPS
            first_token[i] = dict(ttft_ms=ttft, runner_ms=run_ms, write_ms=write_ms, other_ms=other,
                runner_before_admission_ms=runner_at_admission[i], runner_from_admission_ms=run_ms - runner_at_admission[i],
                own_write_ms=r["d2h_ms"], peer_write_ms=write_ms - r["d2h_ms"],
                outside_steps_ms=outside, inside_other_ms=other - outside, token_step=j, seq_id=seq_id)
        assert sum(step_tokens) == s["scheduled_tokens"] and s["step_ms"] - s["run_ms"] - step_write >= -EPS
        step_data.append(dict(fifo=[positions[q] + 1 for q in s["seq_ids"]], query_tokens=step_tokens,
                              written_blocks=writes, first_tokens=finishes, used_gpu_blocks=s["used_blocks"],
                              start_s=start, step_ms=s["step_ms"], runner_ms=s["run_ms"], write_ms=step_write))
        runner += s["run_ms"]
        write += step_write
    assert len(first_token) == 4 and abs(write - sum(r["d2h_ms"] for r in rows)) < EPS
    return first_token, step_data


def subtract(a, b):
    return {k: a[k] - b[k] for k in PARTS}


def at_p95(pairs, arm):
    ranked = sorted(pairs, key=lambda p: p[arm]["ttft_ms"])
    rank = 0.95 * (len(ranked) - 1)
    lo = int(rank)
    weight = rank - lo
    supports = [(ranked[lo], 1 - weight), (ranked[lo + 1], weight)]
    values = {k: sum(w * row[arm][k] for row, w in supports) for k in PARTS}
    assert abs(values["ttft_ms"] - np.percentile([p[arm]["ttft_ms"] for p in pairs], 95)) < EPS
    # Also hold these input identities and weights fixed in the opposite arm.
    paired = {k: sum(w * row["delta"][k] for row, w in supports) for k in PARTS}
    return dict(ms=values, matched_delta_ms=paired, supports=[
        dict(rank=lo + 1 + j, key=row["key"], weight=w) for j, (row, w) in enumerate(supports)])


def compare(pairs):
    quantiles = {a: at_p95(pairs, a) for a in ARMS}
    delta = subtract(quantiles["32"]["ms"], quantiles["36"]["ms"])
    assert abs(delta["ttft_ms"] - sum(delta[k] for k in ("runner_ms", "write_ms", "other_ms"))) < EPS
    return dict(n_pairs=len(pairs), p95=quantiles, p95_delta_ms=delta,
                paired_mean_delta_ms={k: float(np.mean([p["delta"][k] for p in pairs])) for k in PARTS})


def seed_intervals(pairs, seeds):
    # Same seed clusters and draws as the frozen performance report. Re-rank each draw.
    draws = np.random.default_rng(1800).integers(len(seeds), size=(4000, len(seeds)))
    quantiles = {}
    for a in ARMS:
        v = np.array([[[p[a][k] for k in PARTS] for p in pairs if p["key"][0] == seed] for seed in seeds])
        sampled = v[draws].reshape(len(draws), len(pairs), len(PARTS))
        order = np.argsort(sampled[:, :, 0], axis=1, kind="stable")
        rank = 0.95 * (len(pairs) - 1)
        lo = int(rank)
        q = [sampled[np.arange(len(draws)), order[:, i]] for i in (lo, lo + 1)]
        quantiles[a] = q[0] + (rank - lo) * (q[1] - q[0])
    delta = quantiles["32"] - quantiles["36"]
    intervals = np.percentile(delta, [2.5, 97.5], axis=0).T.tolist()
    pct = 100 * (quantiles["32"][:, 0] / quantiles["36"][:, 0] - 1)
    return dict(delta_ms=dict(zip(PARTS, intervals, strict=True)), ttft_pct=np.percentile(pct, [2.5, 97.5]).tolist())


def analyze(run):
    archive = json.loads((run / "summary.json").read_text())
    names = ("requests.jsonl", "steps.jsonl", "waves.jsonl", "trace.json", "engine.json", "environment.json", "design.json")
    hashes = {name: sha256(run / name) for name in names}
    assert all(h == archive["raw_sha256"][name] for name, h in hashes.items())
    env = json.loads((run / "environment.json").read_text())
    recorder_files = ("nanovllm/engine/llm_engine.py", "nanovllm/engine/model_runner.py", "nanovllm/engine/scheduler.py",
                      "nanovllm/engine/cpu_prefix_cache.py", "nanovllm/engine/kv_transfer.py", "benchmarks/phase1.py")
    for name in recorder_files:
        assert sha256(run / "source" / name) == env["source_sha256"][name]
    engine = json.loads((run / "engine.json").read_text())
    trace = json.loads((run / "trace.json").read_text())["formal"]
    seeds = json.loads((run / "design.json").read_text())["seeds"]
    block, budget = engine["block_tokens"], engine["config"]["max_num_batched_tokens"]
    requests, steps, waves = (selected(run, n) for n in names[:3])
    pairs, step_pairs = [], []
    for c in sorted((c for c in trace if c["workload"] == "mixed"), key=lambda c: (c["seed"], c["cycle"])):
        key = c["seed"], c["cycle"]
        assert c["phases"][0][0] == "fill"
        prompts = dict(c["phases"])["fill"]
        assert sorted(map(len, prompts)) == [1152, 1152, 2176, 2176]
        per_arm, per_step = {}, {}
        for a in ARMS:
            arm_key = (*key, a)
            wave, = waves[arm_key]
            assert wave["cpu_resident_blocks"] == 24 < int(a) and wave["cpu_evictions_since_reset"] == 0
            assert not wave["profiler_enabled"] and wave["cpu_capacity_blocks"] == 64
            per_arm[a], per_step[a] = timeline(requests[arm_key], steps[arm_key], prompts, block, budget)
        for j, (left, right) in enumerate(zip(per_step["32"], per_step["36"], strict=True), 1):
            shape = {k: left[k] for k in ("fifo", "query_tokens", "written_blocks", "first_tokens", "used_gpu_blocks")}
            assert all(v == right[k] for k, v in shape.items())
            step_pairs.append(dict(key=[*key, j], **shape, times={
                a: {k: s[k] for k in ("start_s", "step_ms", "runner_ms", "write_ms")}
                for a, s in zip(ARMS, (left, right), strict=True)}))
        for i, prompt in enumerate(prompts):
            left, right = (per_arm[a][i] for a in ARMS)
            assert requests[(*key, "32")][i]["output_token_ids"] == requests[(*key, "36")][i]["output_token_ids"]
            pairs.append({"key": [*key, i + 1], "prompt_tokens": len(prompt), "32": left, "36": right,
                          "delta": subtract(left, right)})
    assert len(pairs) == len({tuple(p["key"]) for p in pairs}) == 32
    pooled = compare(pairs)
    ci = seed_intervals(pairs, seeds)
    old = archive["workloads"]["mixed"]
    for a in ARMS:
        fill = old["arms"][a]["groups"]["fill"]
        assert abs(fill["ttft_ms"]["p95"] - pooled["p95"][a]["ms"]["ttft_ms"]) < EPS
        assert abs(sum(p[a]["own_write_ms"] for p in pairs) - fill["d2h_ms"]) < EPS
    np.testing.assert_allclose(ci["ttft_pct"], old["comparisons"]["fill"]["32"]["36"]["paired_seed_interval_pct"])
    by_seed = {s: compare([p for p in pairs if p["key"][0] == s]) for s in seeds}
    by_fifo = {i: compare([p for p in pairs if p["key"][2] == i]) for i in range(1, 5)}
    summary = dict(pooled=pooled, paired_seed_bootstrap_95=ci, by_seed=by_seed, by_fifo=by_fifo,
                   cycles_per_arm=len(pairs) // 4, prefill_steps_per_arm=len(step_pairs),
                   write_blocks_per_arm=sum(sum(s["written_blocks"]) for s in step_pairs),
                   archive=str(run), input_sha256={**hashes, "summary.json": sha256(run / "summary.json")},
                   recording_head=env["git_head"], analysis_sha256=sha256(__file__))
    return pairs, step_pairs, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True, help="new directory for offline results")
    args = parser.parse_args()
    pairs, steps, summary = analyze(args.run)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "requests_paired.jsonl", pairs)
    write_jsonl(out / "steps_paired.jsonl", steps)
    write_json(out / "summary.json", summary)
    shutil.copyfile(__file__, out / "cold_fill_timing.py")
    (out / "command.txt").write_text(shlex.join([sys.executable, "-m", "benchmarks.cold_fill_timing", *sys.argv[1:]]) + "\n")
