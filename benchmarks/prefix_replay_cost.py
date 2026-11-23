"""Cost-weighted sensitivity of complete, cold-start prefix-cache traces on this 3090."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, floor
from pathlib import Path
import sys

import numpy as np


def read(file):
    return json.loads(file.read_text())


def records(file):
    with file.open() as stream:
        for line in stream:
            yield json.loads(line)


def groups(rows, fields):
    result = defaultdict(list)
    for row in rows:
        if not row.get("warmup", False):
            result[tuple(row[f] for f in fields)].append(row)
    return result


def write_json(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def q95(values):
    return float(np.percentile(values, 95))


def percentile_support(rows, value):
    ordered = sorted(rows, key=lambda r: r[value])
    rank = (len(rows) - 1) * .95
    return [dict(seed=ordered[i]["seed"], wave=ordered[i]["wave"],
                 **{k: ordered[i][k] for k in ("measured_step_shapes","step_of_first_token","prefill_runner_before_first_token_ms","affected_wave","missing_tokens") if k in ordered[i]},
                 fifo=ordered[i]["fifo"], value_ms=ordered[i][value], weight=w)
            for i, w in ((floor(rank), 1 - rank % 1), (ceil(rank), rank % 1)) if w]


def paired_ranges(rows, value):
    data = {s: [r for r in rows if r["seed"] == s] for s in sorted({r["seed"] for r in rows})}
    observed = np.array([[r["observed_ms"] for r in rs] for rs in data.values()])
    alt = np.array([[r[value] for r in rs] for rs in data.values()])
    draws = np.random.default_rng(20261123).integers(len(data), size=(4000, len(data)))
    change = 100 * (np.percentile(alt[draws], 95, axis=(1, 2)) /
                    np.percentile(observed[draws], 95, axis=(1, 2)) - 1)
    return dict(p95_ms=q95(alt), change_pct=100*(q95(alt)/q95(observed)-1),
                bootstrap_sensitivity_pct=np.percentile(change, [2.5, 97.5]).tolist(),
                support=percentile_support(rows, value))


def calibration(folders):
    raw = []
    for folder in folders:
        design = read(folder / "design.json")
        assert design["config"]["max_num_batched_tokens"] == 2048
        assert design["environment"]["block_size"] == 256
        assert design["environment"]["gpu"] == "NVIDIA GeForce RTX 3090"
        for p in records(folder / "pairs.jsonl"):
            s = p["shape"]
            if (s["outstanding"] != 1 or s["query_tokens"] != 128 or
                    s["cpu_tokens"] not in (1024, 2048) or s["gpu_prefix_tokens"]):
                continue
            # Both measures include scheduler/host overhead; subtract the synchronous
            # transfer actually measured on each side of the matched, isolated pair.
            work = p["recompute_ms"] - p["recompute_d2h_ms"] - (p["restore_ms"] - p["restore_h2d_ms"])
            raw.append(dict(prefix=s["cpu_tokens"], seed=p["seed"], source=str(folder),
                            recompute_ms=p["recompute_ms"], restore_ms=p["restore_ms"],
                            nontransfer_delta_ms=work, h2d_ms=p["restore_h2d_ms"],
                            sync_d2h_ms=p["recompute_d2h_ms"]))
    table = {}
    for length in (1024, 2048):
        cases = [r for r in raw if r["prefix"] == length]
        if not cases:
            raise ValueError(f"No paired isolated calibration for {length} tokens")
        table[length] = {k: float(np.median([r[k] for r in cases]))
                         for k in ("nontransfer_delta_ms", "h2d_ms", "sync_d2h_ms")}
        table[length]["pairs"] = len(cases)
    return raw, table


def formal_rows(folder, block_tokens, budget):
    fields = ("workload", "seed", "cycle", "phase", "arm")
    by_wave = groups(records(folder / "requests.jsonl"), fields)
    steps = groups((s for s in records(folder / "steps.jsonl") if s["is_prefill"]), fields)
    output, fully_restored = [], defaultdict(list)
    for key, wave in by_wave.items():
        workload, seed, cycle, phase, arm = key
        if arm != "candidate":
            continue
        native = by_wave[(*key[:-1], "A")]
        pw = steps[key]
        writes = sum(r["d2h_blocks"] for r in wave)
        if writes:
            assert not any(r["cpu_restored_tokens"] or r["initial_gpu_hit_tokens"] for r in wave)
            assert [r["prefill_tokens"] for r in wave] == [r["prefill_tokens"] for r in native]
            assert [(s["scheduled_tokens"], len(s["seq_ids"])) for s in pw] == [
                (s["scheduled_tokens"], len(s["seq_ids"])) for s in steps[(*key[:-1], "A")]]
        if workload == "mixed" and phase == "evicted":
            assert all(r["cpu_restored_tokens"] == r["prompt_tokens"] - 128 for r in wave)
            assert len(pw) == 1 and len(pw[0]["seq_ids"]) == len(wave)
            fully_restored[tuple(r["prompt_tokens"] for r in wave)].append([r["ttft_ms"] for r in wave])
        scheduled = sum(s["scheduled_tokens"] for s in pw)
        step_end = [s["start_s"] + s["step_ms"] / 1000 for s in pw]
        for fifo, (r, a) in enumerate(zip(wave, native, strict=True)):
            # First token is timestamped inside the step, before postprocess ends.
            token_step = next(i for i, (s, end) in enumerate(zip(pw, step_end))
                              if s["start_s"] <= r["token_s"][0] <= end + 1e-6
                              and r["seq_id"] in s["seq_ids"])
            assert r["input_token_ids"] == a["input_token_ids"]
            output.append(dict(workload=workload, seed=seed, cycle=cycle, wave=f"{cycle}/{phase}",
                               phase=phase, fifo=fifo, observed_ms=r["ttft_ms"], a_ms=a["ttft_ms"],
                               written_blocks=r["d2h_blocks"], wave_written_blocks=writes,
                               write_service_ms=r["d2h_ms"], wave_prefill_steps=len(pw),
                               computed_tokens=r["prompt_computed_tokens"], restored_blocks=r["h2d_blocks"],
                               packed_step_lower_bound=ceil(scheduled / budget),
                               step_of_first_token=token_step+1,
                               prefill_runner_before_first_token_ms=sum(s["run_ms"] for s in pw[:token_step+1]),
                               wave_prefill_runner_ms=sum(s["run_ms"] for s in pw),
                               unused_wave_token_budget=len(pw)*budget-scheduled,
                               measured_step_shapes=[s["scheduled_tokens"] for s in pw],
                               measured_runner_per_step_ms=[s["run_ms"] for s in pw],
                               native_speed_on_writing_waves_ms=a["ttft_ms"] if writes else r["ttft_ms"],
                               faster_observation_on_writing_waves_ms=min(a["ttft_ms"],r["ttft_ms"]) if writes else r["ttft_ms"],
                               zero_on_writing_waves_ms=0.0 if writes else r["ttft_ms"]))
    return output, fully_restored


def boundary_rows(folder, block_tokens, table, references):
    by_wave = groups(records(folder / "requests.jsonl"), ("revisit_waves", "seed", "wave_index", "arm"))
    past = defaultdict(set)
    output = []
    for (level, seed, wave_id, arm), wave in sorted(by_wave.items()):
        if arm != "CPU":
            continue
        native = by_wave[level, seed, wave_id, "A"]
        info = []
        for r, a in zip(wave, native, strict=True):
            assert r["input_token_ids"] == a["input_token_ids"]
            assert r["initial_gpu_hit_tokens"] == 0
            prefix = tuple(r["input_token_ids"][:-128])
            revisit = prefix in past[level, seed]
            past[level, seed].add(prefix)
            assert r["prefill_tokens"] == r["prompt_tokens"]-r["h2d_blocks"]*block_tokens
            missing = len(prefix) - r["cpu_restored_tokens"] if revisit else 0
            info.append((revisit, missing))
        affected = any(missing > 0 for _, missing in info)
        lengths = tuple(r["prompt_tokens"] for r in wave)
        ref = references.get(lengths)
        if not ref:
            raise ValueError(f"Missing fully restored reference wave for FIFO lengths {lengths}")
        ref = np.median(ref, axis=0)
        if affected:
            assert all(revisit for revisit, _ in info)
        for fifo, ((revisit, missing), r, a) in enumerate(zip(info, wave, native, strict=True)):
            entry = table[lengths[fifo] - 128]
            # Linear within a length only for a partial prefix; this is an
            # isolated-request service estimate, not a queue or p95 estimate.
            fraction = missing / (lengths[fifo]-128)
            work = fraction * entry["nontransfer_delta_ms"]
            h2d = fraction * entry["h2d_ms"]
            extra_d2h = fraction * entry["sync_d2h_ms"]
            output.append(dict(level=level, seed=seed, wave=wave_id, fifo=fifo,
                               seq_id=r["seq_id"], revisit=revisit, missing_tokens=missing,
                               affected_wave=affected, computed_tokens=r["prompt_computed_tokens"],
                               restored_blocks=r["h2d_blocks"], written_blocks=r["d2h_blocks"],
                               write_service_ms=r["d2h_ms"], read_service_ms=r["h2d_ms"],
                               observed_ms=r["ttft_ms"], a_ms=a["ttft_ms"],
                               estimated_avoidable_nontransfer_ms=work, estimated_extra_h2d_ms=h2d,
                               one_additional_sync_write_ms=extra_d2h,
                               zero_incomplete_only_ms=0.0 if missing else r["ttft_ms"],
                               zero_affected_wave_ms=0.0 if affected else r["ttft_ms"],
                               full_replay_reference_ms=float(ref[fifo]) if affected else r["ttft_ms"]))
    return output


def run(args):
    boundary = Path(args.boundary)
    formal = Path(args.formal)
    folders = [Path(n) for n in args.calibrations]
    raw, table = calibration(folders)
    b = read(boundary / "engine.json")
    assert b["block_tokens"] == 256 and b["config"]["max_num_batched_tokens"] == 2048
    assert b["actual_gpu_budget"]["actual_blocks"] == 36 and b["cpu_slots"] == 64
    f = read(formal / "engine.json")
    assert f["kv_budget"]["actual_blocks"] == 36 and f["cpu_slots"] == 64
    assert b["dtype"] == f["dtype"] == "torch.bfloat16"
    formal_data, references = formal_rows(formal, b["block_tokens"], b["config"]["max_num_batched_tokens"])
    boundary_data = boundary_rows(boundary, b["block_tokens"], table, references)
    inputs = [folder / name for folder in (boundary, formal)
              for name in ("requests.jsonl", "engine.json", "environment.json")]
    inputs += [formal / "steps.jsonl"]
    inputs += [folder / name for folder in folders for name in ("design.json", "pairs.jsonl")]
    result = dict(boundary={}, formal={}, calibration_table=table, calibration_pairs=raw,
                  fully_restored_reference_counts={str(k):len(v) for k,v in references.items()},
                  input_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in inputs},
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    for (level,), rows in groups(boundary_data, ("level",)).items():
        affected = [r for r in rows if r["affected_wave"]]
        summary = result["boundary"][level] = dict(
            n=len(rows), n_revisits=sum(r["revisit"] for r in rows),
            incomplete=sum(r["missing_tokens"]>0 for r in rows), missing_tokens=sum(r["missing_tokens"] for r in rows),
            affected_wave_count=len({(r["seed"],r["wave"]) for r in affected}), affected_requests=len(affected),
            observed_p95_ms=q95([r["observed_ms"] for r in rows]), observed_support=percentile_support(rows, "observed_ms"),
            extra_h2d_blocks=sum(r["missing_tokens"]//b["block_tokens"] for r in rows),
            estimated_nontransfer_ms=sum(r["estimated_avoidable_nontransfer_ms"] for r in rows),
            estimated_extra_h2d_ms=sum(r["estimated_extra_h2d_ms"] for r in rows),
            one_extra_sync_d2h_ms=sum(r["one_additional_sync_write_ms"] for r in rows),
            costs_by_wave={})
        for kind in ("affected", "cold", "complete_revisit"):
            sample = [r for r in rows if ("affected" if r["affected_wave"] else
                                           "complete_revisit" if r["revisit"] else "cold") == kind]
            if sample:
                summary["costs_by_wave"][kind] = dict(n=len(sample),p95_ms=q95([r["observed_ms"] for r in sample]),
                      d2h_blocks=sum(r["written_blocks"] for r in sample), d2h_ms=sum(r["write_service_ms"] for r in sample))
        for column in ("zero_affected_wave_ms", "full_replay_reference_ms", "a_ms"):
            summary[column] = paired_ranges(rows, column)
    for (load,), rows in groups(formal_data, ("workload",)).items():
        summary = result["formal"][load] = dict(n=len(rows),observed_p95_ms=q95([r["observed_ms"] for r in rows]),
                                                observed_support=percentile_support(rows, "observed_ms"), phases={})
        for (phase,), rs in groups(rows, ("phase",)).items():
            # One FIFO row represents a wave for step counts.
            waves = [r for r in rs if r["fifo"] == 0]
            summary["phases"][phase] = dict(n=len(rs),p95_ms=q95([r["observed_ms"] for r in rs]),
                    native_p95_ms=q95([r["a_ms"] for r in rs]),writes=sum(r["written_blocks"] for r in rs),
                    write_service_ms=sum(r["write_service_ms"] for r in rs),
                    median_wave_prefill_runner_ms=float(np.median([r["wave_prefill_runner_ms"] for r in waves])),
                    unused_wave_token_budget=sum(r["unused_wave_token_budget"] for r in waves),
                    single_step_128token_runner_ms=[ms for r in waves for tokens,ms in
                        zip(r["measured_step_shapes"],r["measured_runner_per_step_ms"],strict=True) if tokens==128],
                    prefill_steps=sum(r["wave_prefill_steps"] for r in waves),
                    token_budget_lower_bound=sum(r["packed_step_lower_bound"] for r in waves))
        for column in ("zero_on_writing_waves_ms", "native_speed_on_writing_waves_ms",
                       "faster_observation_on_writing_waves_ms", "a_ms"):
            summary[column] = paired_ranges(rows, column)
    out = Path(args.output)
    out.mkdir(exist_ok=False, parents=True)
    for name, data in (("boundary_requests",boundary_data), ("formal_requests",formal_data)):
        with (out / f"{name}.jsonl").open("w") as stream:
            for row in data:
                stream.write(json.dumps(row,allow_nan=False)+"\n")
    write_json(out / "summary.json", result)
    write_json(out / "command.json", [sys.executable,"-m","benchmarks.prefix_replay_cost",*sys.argv[1:]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", default="results/20261123_revisit_boundary")
    parser.add_argument("--formal", default="results/20261123_overlap_formal")
    parser.add_argument("--calibrations", nargs="+", default=["results/20261118_queue_c_calibration",
                                                              "results/20261118_stage2_c_calibration"])
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
