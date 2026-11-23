"""Rebuild the CPU INT4 acceptance checks from archived runs, without CUDA."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def roots(directory):
    return {tuple(p[:256]) for f in directory.glob("trace-*.json")
            for phases in read(f).values() for _, inputs in phases for p in inputs}


def finite_logits(bits):
    assert bits.dtype == np.int16
    logits = (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    assert logits.shape[-1] == 151936 and np.isfinite(logits).all()
    return logits


def check_quality(directory, pilot):
    cases = rows(directory / "cases.jsonl")
    assert len(cases) == 36
    assert {(c["seed"], c["task"], c["depth"], c["length"]) for c in cases} == {
        (s,t,d,n) for s in (3201,3202) for t in ("direct","crossref") for d in (10,50,90)
        for n in (2048,8192,12288)}
    old = {tuple(c["input_ids"][:256]) for c in rows(pilot / "cases.jsonl")}
    assert all(tuple(c["input_ids"][:256]) not in old for c in cases)
    errors = defaultdict(list)
    for case in cases:
        n = case["length"]
        assert case["bf16"]["gpu_hit_tokens"] == case["int4"]["restored_tokens"] == n
        with np.load(directory / case["tensors"]) as archive:
            for tag, prefix, extra in (("teacher","",len(case["teacher_lead"])), ("natural","natural_",0)):
                r = case[tag]
                a = finite_logits(archive[prefix+"bf16_logits"])
                b = finite_logits(archive[prefix+"int4_logits"])
                assert len(a) == len(b) == len(r)
                assert [e["position"] for e in r] == list(range(len(case["input_ids"])+extra-1,
                                                                 len(case["input_ids"])+extra-1+len(r)))
                assert a.argmax(1).tolist() == [e["bf16_top1"] for e in r]
                assert b.argmax(1).tolist() == [e["int4_top1"] for e in r]
                assert np.allclose(np.max(np.abs(a-b),axis=1), [e["max_abs"] for e in r])
                for ref, actual, row in zip(a, b, r, strict=True):
                    x = ref.astype(np.float64) - float(ref.max())
                    y = actual.astype(np.float64) - float(actual.max())
                    logp = x - np.log(np.exp(x).sum())
                    logq = y - np.log(np.exp(y).sum())
                    target = row["target"]
                    observed = [-logp[target], -logq[target], (np.exp(logp)*(logp-logq)).sum()]
                    archived = [row["bf16_nll"], row["int4_nll"], row["kl_bf16_int4"]]
                    np.testing.assert_allclose(observed, archived, atol=1e-8, rtol=1e-8)
        for entry in case["kv"]:
            for k in ("K","V"):
                values = entry[k]
                assert len(values) == 36 and np.isfinite(values).all()
                errors[(entry["metric"],k)].extend(values)
    summary = read(directory / "summary.json")
    gates = {}
    breakdown = {}
    for name in ("direct","crossref"):
        s = summary["by_task"][name]
        group = [c for c in cases if c["task"] == name]
        assert s["bf16_correct"] == sum(c["bf16"]["exact"] for c in group)
        assert s["int4_correct"] == sum(c["int4"]["exact"] for c in group)
        gates[name] = dict(reference_floor=s["bf16_correct"] >= 9, semantic_losses=s["lost"] <= 1,
                           digit_agreement=s["top1_agree"]/s["positions"] >= .95,
                           digit_delta_nll=s["mean_delta_nll"] <= .15,
                           natural_kl=s["natural_mean_kl"] <= .20, natural_delta_nll=s["natural_mean_delta_nll"] <= .25)
        breakdown[name] = {}
        for length in (2048,8192,12288):
            part = [c for c in group if c["length"] == length]
            breakdown[name][length] = dict(n=len(part), bf16=sum(c["bf16"]["exact"] for c in part),
                int4=sum(c["int4"]["exact"] for c in part),
                lost=sum(c["bf16"]["exact"] and not c["int4"]["exact"] for c in part))
    long_loss = sum(c["bf16"]["exact"] and not c["int4"]["exact"] for c in cases if c["length"]==12288)
    numeric = {f"{k}_{metric}":dict(n=len(v), min=min(v), median=float(np.median(v)), max=max(v))
               for (metric,k),v in errors.items()}
    return dict(by_task=gates, longest_loss=long_loss, longest_gate=long_loss<=1, kv_samples=numeric,
                positions_checked=sum(len(c["teacher"])+len(c["natural"]) for c in cases),
                cases_by_task_length=breakdown)


def check_performance(directory, pilot):
    design = read(directory / "design.json")
    assert sorted(design["seeds"]) == list(range(3501,3507))
    assert len({tuple(p["order"]) for p in design["plan"]}) == 6
    assert roots(directory).isdisjoint(roots(pilot))
    dma = {}
    for arm in ("bf16_64","int4_64"):
        copies = read(pilot / f"3400-{arm}/dma_events.json")
        assert len(copies) == 2 and "DtoH" in copies[0]["name"] and "HtoD" in copies[1]["name"]
        assert copies[0]["args"]["bytes"] == copies[1]["args"]["bytes"]
        dma[arm] = copies[0]["args"]["bytes"]
    dma["bf16_18"] = dma["bf16_64"]
    sizes, phases = {}, defaultdict(lambda: [0,0,0])
    replica = defaultdict(list)
    raw_count = 0
    for entry in design["plan"]:
        seed = entry["seed"]
        f = directory / f"trace-{seed}.json"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == entry["trace_sha256"]
        trace = read(f)
        matched = {}
        for arm in entry["order"]:
            run = directory / f"{seed}-{arm}"
            engine = read(run / "engine.json")
            assert engine["dma_per_page"] == engine["slot_stride"] == engine["gpu_stage_bytes"] == dma[arm]
            assert engine["payload_bytes"] + engine["quant_metadata_bytes"] == dma[arm]
            assert engine["cpu_pageable_bytes"] == engine["slot_count"]*dma[arm]
            assert engine["pinned_bytes"] == [dma[arm]]*2
            assert engine["gpu_pool_bytes"] == 36*37748736
            assert engine["attention_cache_dtype"] == engine["weight_dtype"] == "torch.bfloat16"
            sizes[arm] = engine["cpu_pageable_bytes"]
            raw = rows(run / "requests.jsonl")
            matched[arm] = raw
            raw_count += len(raw)
            for r in raw:
                phase, prompts = trace[r["workload"]][r["wave_index"]]
                assert r["phase"] == phase and r["input_token_ids"] == prompts[r["fifo"]]
                assert r["h2d_bytes"] == r["h2d_blocks"]*dma[arm] and r["d2h_bytes"] == r["d2h_blocks"]*dma[arm]
                assert r["cpu_restore_failures"] == r["cpu_save_failures"] == r["preemptions"] == 0
            for w in rows(run / "waves.jsonl"):
                replica[arm].append(w["also_indexed_on_gpu"])
        for a,b in zip(matched["bf16_64"],matched["int4_64"],strict=True):
            phase = a["workload"] + "/" + a["phase"]
            for k in ("prompt_computed_tokens","initial_gpu_hit_tokens","cpu_restored_tokens","d2h_blocks"):
                assert a[k] == b[k]
            x = phases[phase]
            x[0] += 1
            x[1] += int(a["output_token_ids"][0] != b["output_token_ids"][0])
            x[2] += int(a["output_token_ids"] != b["output_token_ids"])
    assert raw_count == 1224
    assert sizes["int4_64"] == sizes["bf16_18"]
    s = read(directory / "summary.json")
    def benefit(w, scope, arm, threshold):
        m = s[w][scope]["int4_vs"][arm]
        return m["change_pct"] <= -threshold and m["paired_seed_95pct"][1] < 0
    cap = s["capacity"]["whole"]["arms"]
    revisit = s["capacity"]["revisit"]["arms"]
    gates = dict(bytes=dma["int4_64"] <= dma["bf16_64"]/3, effective_capacity=(
        cap["int4_64"]["max_resident_slots"] == 64 and cap["bf16_18"]["max_resident_slots"] == 18),
        mixed_restore=benefit("mixed","evicted","bf16_64",15),
        same_byte_revisit=benefit("capacity","revisit","bf16_18",30) and
                          revisit["int4_64"]["full_cpu_hits"] > revisit["bf16_18"]["full_cpu_hits"],
        guardrails=all(s[w][p]["int4_vs"]["bf16_64"]["change_pct"] <= 5 for w,p in (
            ("mixed","whole"),("mixed","no_reuse"),("capacity","whole"))))
    return dict(gates=gates, requests=raw_count, transfers_without_failures=True, dma_bytes=dma, pageable_bytes=sizes,
                max_blocks_also_indexed_on_gpu={a:max(v) for a,v in replica.items()},
                output_differences={p:dict(n=x[0],first=x[1],sequence=x[2]) for p,x in phases.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("timing","timing-pilot","quality","quality-pilot","output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    protocol = "docs/INT4_PREFIX_PROTOCOL.md"
    envs = [read(run / "environment.json") for run in (args.timing,args.quality)]
    for k in ("git_head","gpu","capability","packages","model_files","cuda_runtime"):
        assert envs[0][k] == envs[1][k]
    for k,v in envs[0]["source_sha256"].items():
        if k.startswith("nanovllm/") or k in (protocol,"benchmarks/int4_quality.py","benchmarks/int4_prefix.py"):
            assert envs[1]["source_sha256"][k] == v
    out = dict(protocol_sha256=envs[0]["source_sha256"][protocol],
               performance=check_performance(args.timing,args.timing_pilot),
               quality=check_quality(args.quality,args.quality_pilot))
    args.output.write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    main()
