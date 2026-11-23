"""Verify the public first-use setting against the archived budget experiment."""
import argparse
import json
from pathlib import Path
import traceback

import torch

from nanovllm import LLM
from benchmarks.common import prepare_run, sha256, write_json
from benchmarks.first_use_budget import normalized_steps
from benchmarks.first_use_budget_report import load_batches, wave_key
from benchmarks.phase1 import reset_cache, save_wave, wave


ARCHIVE = Path("results/20261119_first_use_budget36")
REQUEST_FIELDS = ("input_token_ids", "output_token_ids", "output_tokens", "prefill_tokens", "prompt_computed_tokens",
                  "decode_tokens", "allocations", "initial_gpu_hit_tokens", "allocation_gpu_hit_tokens",
                  "initial_cpu_hit_tokens", "cpu_restored_tokens", "d2h_blocks", "h2d_blocks", "d2h_batches",
                  "batched_d2h_blocks", "preemptions", "cpu_save_failures", "cpu_restore_failures", "restore_decisions")


def selections(trace):
    warm = {c["workload"]: c for c in trace["warmup"]}
    return [(None, [(warm["long"], True)]), (32, [(warm["mixed"], True)] + [(c, False) for c in trace["formal"]])]


def compare_wave(batch, ref):
    assert normalized_steps(batch) == normalized_steps(ref), wave_key(batch[2])
    for field in ("used_blocks_peak", "cpu_resident_blocks", "cpu_capacity_blocks", "cpu_evictions_since_reset"):
        assert batch[2][field] == ref[2][field], (wave_key(batch[2]), field)
    for i, (new, old) in enumerate(zip(batch[0], ref[0], strict=True)):
        for field in REQUEST_FIELDS:
            assert new[field] == old[field], (wave_key(new), i, field)
        assert abs(new["ttft_ms"] - 1000 * (new["token_s"][0] - new["enqueue_s"])) < 1e-5


def verify_archive():
    old = json.loads((ARCHIVE / "summary.json").read_text())
    names = ("trace.json", "requests.jsonl", "waves.jsonl", "steps.jsonl", "environment.json", "engine.json")
    hashes = {n: sha256(ARCHIVE / n) for n in names}
    assert all(h == old["raw_sha256"][n] for n, h in hashes.items())
    return hashes


def audit(out, reference):
    raw, _, batches = load_batches(out)
    design = json.loads((out / "design.json").read_text())
    assert design["reference_sha256"] == verify_archive()
    trace = json.loads((ARCHIVE / "trace.json").read_text())
    engines = json.loads((out / "engine.json").read_text())
    keys = []
    for limit, items in selections(trace):
        mode, arm = ("auto", "36") if limit is None else ("explicit", "32")
        engine = engines[mode]
        assert engine["effective_blocks"] == int(arm)
        assert engine["kwargs"].get("cpu_prefix_first_use_blocks") == limit
        assert ("cpu_prefix_first_use_blocks" in engine["kwargs"]) == (limit is not None)
        for c, warm in items:
            for phase, inputs in c["phases"]:
                key = c["seed"], c["cycle"], c["workload"], arm, phase
                keys.append(key)
                batch = batches[key]
                assert batch[2]["entry"] == mode and batch[2]["warmup"] == warm
                assert [r["input_token_ids"] for r in batch[0]] == inputs
                compare_wave(batch, reference[key])
    assert list(batches) == keys
    results = {}
    for load, expected_n, expected_early_blocks in (("mixed", 160, 96), ("long", 128, 128)):
        rows = [r for r in raw if not r["warmup"] and r["workload"] == load]
        early = [r for r in rows if r["cycle"] == 1 and r["phase"] == "evicted"]
        written = sum(r["d2h_blocks"] for r in rows)
        early_full = sum(r["cpu_restored_tokens"] == r["prompt_tokens"] - 128 for r in early)
        blocks = sum(r["h2d_blocks"] for r in early)
        assert (len(rows), written, len(early), early_full, blocks) == (expected_n, 256, 16, 16, expected_early_blocks)
        results[load] = dict(requests=len(rows), d2h_blocks=written, early_replays=len(early),
                             fully_restored_early_replays=early_full, early_h2d_blocks=blocks)
    auto = [r for r in raw if r["entry"] == "auto"]
    assert len(auto) == 16 and sum(r["d2h_blocks"] for r in auto) == 36
    return dict(workloads=results, exact_formal_requests=sum(not r["warmup"] for r in raw),
                exact_automatic_smoke_requests=len(auto), exact_explicit_smoke_requests=sum(
                    r["warmup"] and r["entry"] == "explicit" for r in raw), exact_waves=len(batches),
                raw_sha256={n: sha256(out / n) for n in ("requests.jsonl", "waves.jsonl", "steps.jsonl", "engine.json", "design.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", help="new directory for the model replay")
    action.add_argument("--summarize", type=Path, help="rebuild the audit from captured logs")
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    _, _, reference = load_batches(ARCHIVE)
    if args.summarize:
        print(json.dumps(audit(args.summarize, reference), indent=2))
        return
    args.module = "benchmarks.first_use_config"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        hashes = verify_archive()
        old, now = (json.loads((p / "environment.json").read_text()) for p in (ARCHIVE, out))
        for name in ("gpu", "capability", "packages", "cuda_runtime", "model_files"):
            assert old[name] == now[name], name
        engine = json.loads((ARCHIVE / "engine.json").read_text())
        trace = json.loads((ARCHIVE / "trace.json").read_text())
        write_json(out / "design.json", dict(reference=str(ARCHIVE), reference_sha256=hashes,
                   purpose="configuration and exact-output regression", seeds=[1801, 1802, 1803, 1804],
                   warmup_seed=1800, output_tokens=128, warmup_output_tokens=8))
        engines = {}
        for limit, items in selections(trace):
            kwargs = dict(engine["config"])
            if limit is not None:
                kwargs["cpu_prefix_first_use_blocks"] = limit
            mode = "auto" if limit is None else "explicit"
            torch.manual_seed(0)
            llm = LLM(args.model, **kwargs)
            runner = llm.model_runner
            effective = llm.scheduler.block_manager.first_use_limit
            assert effective == (36 if limit is None else 32) and llm.config.cpu_prefix_first_use_blocks == limit
            assert runner.kv_cache_budget["actual_blocks"] == 36 and runner.kv_transfer.capacity == 64
            assert llm.config.kvcache_block_size == 256 and llm.scheduler.block_manager.namespace.hex() == engine["namespace"]
            assert runner.kv_cache.dtype == next(runner.model.parameters()).dtype == torch.bfloat16
            assert not runner.enforce_eager and runner.graph_bs == engine["graph_batch_sizes"]
            engines[mode] = dict(kwargs=kwargs, effective_blocks=effective, kv_budget=runner.kv_cache_budget,
                                 cpu_slots=runner.kv_transfer.capacity, dtype=str(runner.kv_cache.dtype),
                                 namespace=llm.scheduler.block_manager.namespace.hex(), graph_batch_sizes=runner.graph_bs)
            write_json(out / "engine.json", engines)
            for i, (case, warm) in enumerate(items):
                if i:
                    reset_cache(llm)
                manager = llm.scheduler.block_manager
                assert manager.first_use_limit == effective and not manager.entries and not manager.used_block_ids
                tag = dict(seed=case["seed"], cycle=case["cycle"], workload=case["workload"], entry=mode,
                           arm=str(effective), warmup=warm)
                torch.manual_seed(case["seed"])
                for phase, inputs in case["phases"]:
                    batch = wave(llm, inputs, 8 if warm else 128, dict(tag, phase=phase))
                    save_wave(out, batch)
                    compare_wave(batch, reference[wave_key(batch[2])])
                print(f"{mode} seed={case['seed']} {case['workload']} cycle={case['cycle']} checked", flush=True)
            llm.exit()
            llm = None
            del runner
        write_json(out / "summary.json", audit(out, reference))
        write_json(out / "status.json", dict(verification_complete=True))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(verification_complete=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
