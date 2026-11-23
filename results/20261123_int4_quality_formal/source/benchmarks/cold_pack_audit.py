"""Audit FIFO delay, greedy outputs and prefill packing in a completed cold_pack GPU run."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import ceil, floor
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index(run, kind):
    by_wave = defaultdict(list)
    for directory in (run, run / "reference"):
        with (directory / f"{kind}.jsonl").open() as stream:
            for line in stream:
                r = json.loads(line)
                if not r["warmup"]:
                    key = (r["seed"], r["cycle"], r["workload"], r["phase"], r["arm"])
                    by_wave[key].append(r)
    return by_wave


def percentiles(sample):
    values = sorted(sample)
    result = {}
    for pct in (50, 95):
        rank = (len(values) - 1) * pct / 100
        i = floor(rank)
        result[f"p{pct}"] = values[i] + (values[ceil(rank)] - values[i]) * (rank - i)
    return result


def layout(rows, steps):
    fifo = {r["seq_id"]: i for i, r in enumerate(rows)}
    return [(s["is_prefill"], s["scheduled_tokens"], [fifo[x] for x in s["seq_ids"]]) for s in steps]


def first_token_boundary(r, steps):
    stamp = r["token_s"][0]
    indices = [i for i, s in enumerate(steps) if s["start_s"] <= stamp <= s["start_s"] + s["step_ms"] / 1000]
    assert len(indices) == 1
    i = indices[0]
    s = steps[i]
    assert s["is_prefill"] and r["seq_id"] in s["seq_ids"] and i + 1 < len(steps)
    end = s["start_s"] + s["step_ms"] / 1000
    following = steps[i + 1]["start_s"]
    assert stamp <= end <= following and abs(r["ttft_ms"] - (stamp - r["enqueue_s"]) * 1000) < 1e-5
    return dict(ttft_ms=r["ttft_ms"], enqueue_to_step_end_ms=(end - r["enqueue_s"]) * 1000,
                enqueue_to_next_step_start_ms=(following - r["enqueue_s"]) * 1000,
                internal_token_to_step_end_ms=(end - stamp) * 1000,
                step_end_to_next_start_ms=(following - end) * 1000)


def analyze(run):
    assert read(run / "status.json")["complete"]
    design, engine, main = (read(run / f"{n}.json") for n in ("design", "engine", "summary"))
    assert design["candidate"] == "cold_pack" and design["baseline"] == "overlap"
    budget = engine["config"]["max_num_batched_tokens"]
    for name, expected in main["raw_sha256"].items():
        assert fingerprint(run / name) == expected, name
    rows, steps = index(run, "requests"), index(run, "steps")
    assert rows.keys() == steps.keys()
    greedy = defaultdict(Counter)
    delays = defaultdict(list)
    work = defaultdict(Counter)
    layouts = defaultdict(Counter)
    for key, packed in rows.items():
        seed, cycle, load, phase, arm = key
        if arm != "candidate":
            continue
        baseline_key = (*key[:-1], "32")
        current = rows[baseline_key]
        phase_group = "no_reuse" if phase in ("pressure1", "pressure2") else phase
        per_arm_times = {}
        for tag, batch_key, ref_arm in (("current", baseline_key, "A"), ("packed", key, "A_packed")):
            batch, schedule = rows[batch_key], steps[batch_key]
            ref_key = (seed, cycle, load, "hot" if phase == "evicted" else phase, ref_arm)
            ref = rows[ref_key]
            assert len(batch) == len(ref) == 4 and layout(batch, schedule) == layout(ref, steps[ref_key])
            assert all(a["enqueue_s"] <= b["enqueue_s"] for a, b in zip(batch, batch[1:]))
            for r, native in zip(batch, ref, strict=True):
                assert r["input_token_ids"] == native["input_token_ids"]
                assert r["prefill_tokens"] == native["prefill_tokens"] and r["decode_tokens"] == native["decode_tokens"]
                assert r["output_token_ids"] == native["output_token_ids"] and len(r["output_token_ids"]) == 128
            per_arm_times[tag] = [first_token_boundary(r, schedule) for r in batch]
            prefill = [s for s in schedule if s["is_prefill"]]
            tokens = sum(s["scheduled_tokens"] for s in prefill)
            assert tokens == sum(r["prefill_tokens"] for r in batch)
            assert all(0 < s["scheduled_tokens"] <= budget for s in prefill)
            for group in (phase_group, "all"):
                totals = work[load, group, tag]
                totals.update(waves=1, prefill_steps=len(prefill), query_tokens=tokens,
                              budget_tokens=len(prefill) * budget, unused_tokens=len(prefill) * budget - tokens)
                layouts[load, group, tag].update((s["scheduled_tokens"], len(s["seq_ids"])) for s in prefill)
        for i, (a, b) in enumerate(zip(current, packed, strict=True)):
            assert a["input_token_ids"] == b["input_token_ids"]
            for group in (phase_group, "all"):
                greedy[load, group].update(requests=1, first_token_changed=a["output_token_ids"][0] != b["output_token_ids"][0],
                          sequence_changed=a["output_token_ids"] != b["output_token_ids"],
                          current_exact_vs_native=1, packed_exact_vs_native_packed=1)
            old, new = per_arm_times["current"][i], per_arm_times["packed"][i]
            delays[load, "all", "all"].append((old, new))
            if phase_group in ("fill", "no_reuse"):
                delays[load, phase_group, str(i + 1)].append((old, new))
    loads = {k: dict(greedy={}, latency_ms={}, prefill={}) for k in main["workloads"]}
    for (load, group), count in greedy.items():
        loads[load]["greedy"][group] = dict(count)
        if group == "all":
            check = main["candidate_vs_current_greedy"][load]
            assert count["requests"] == check["requests"] and count["sequence_changed"] == check["different_greedy_outputs"]
    for (load, group, position), pairs in delays.items():
        item = dict(n_pairs=len(pairs))
        for i, tag in enumerate(("current", "packed")):
            item[tag] = {metric: percentiles(p[i][metric] for p in pairs) for metric in pairs[0][i]}
        item["paired_ttft_delta_ms"] = percentiles(new["ttft_ms"] - old["ttft_ms"] for old, new in pairs)
        loads[load]["latency_ms"].setdefault(group, {})[position] = item
    for key, counts in work.items():
        load, group, tag = key
        loads[load]["prefill"].setdefault(group, {})[tag] = dict(counts,
             utilized_fraction=counts["query_tokens"] / counts["budget_tokens"],
             step_shapes=[dict(query_tokens=q, requests=n, steps=num) for (q, n), num in sorted(layouts[key].items())])
    return dict(stage=design["stage"], seeds=main["seeds"], loads=loads, token_budget_per_step=budget,
                ttft="Engine enqueue to first real token timestamp inside scheduler.postprocess, in milliseconds.",
                step_bounds="The actual step() return was not timestamped. step_end is recorded after postprocess and before creating outputs. "
                   "Its return occurs between step_end and the following step start. Neither bound measures client streaming.",
                reference="Each CPU arm is checked against its matching native scheduler. For evicted CPU restores use its native GPU-hot wave.",
                fifo="FIFO positions 1 to 4 are the observed insertion order within each four-request wave. Percentiles use linear interpolation. "
                   "Paired TTFT deltas are packed minus current on identical requests.",
                input_run=str(run), input_summary_sha256=fingerprint(run/"summary.json"), input_sha256=main["raw_sha256"],
                script_sha256=fingerprint(Path(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
