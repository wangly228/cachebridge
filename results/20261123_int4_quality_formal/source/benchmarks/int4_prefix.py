"""Same-slot and same-byte Qwen3 CPU prefix cache comparisons on one GPU."""
import argparse
from dataclasses import fields
import gzip
from itertools import permutations
import json
from pathlib import Path
from random import Random
import subprocess
import sys

import numpy as np
import torch

from benchmarks.common import distribution, memory, prepare_run, sha256, write_json, write_jsonl
from benchmarks.mixed_fill import make_case
from benchmarks.phase1 import reset_cache, save_wave, wave
from nanovllm import LLM


ARMS = {"bf16_64": ("native", 64), "int4_64": ("int4", 64), "bf16_18": ("native", 18)}
OUTPUT = 16


def make_traces(seed, index):
    phases = make_case(seed, index % 2, index % 6)["phases"]
    rng = Random(f"cachebridge-capacity/{seed}")
    def tokens(n):
        return [rng.randrange(100, 10000) for _ in range(n)]
    banks = [[tokens(1024) for _ in range(4)] for _ in range(4)]
    capacity = []
    for i, bank in enumerate(banks):
        for phase in ("fill", "touch"):
            capacity.append((f"{phase}{i}", [p + tokens(128) for p in bank]))
    for i, bank in enumerate(banks):
        capacity.append((f"revisit{i}", [p + tokens(128) for p in bank]))
    return dict(mixed=phases, capacity=capacity)


def index_bytes(manager):
    # Inclusive reachable Python objects (some token IDs/lists are shared with GPU metadata).
    seen = set()
    def size(x):
        if id(x) in seen:
            return 0
        seen.add(id(x))
        n = sys.getsizeof(x)
        if isinstance(x, dict):
            n += sum(size(k) + size(v) for k, v in x.items())
        elif isinstance(x, (list, tuple)):
            n += sum(size(v) for v in x)
        return n
    return sum(size(x) for x in (manager.entries, manager.recent_uses, manager.free_slots))


def inventory(llm):
    runner = llm.model_runner
    t = runner.kv_transfer
    pool = runner.kv_cache
    assert t.device_stage.is_contiguous() and t.slots.flags.c_contiguous
    assert len(t.write_stages) == 2 and t.pinned is t.write_stages[0][0]
    stages = [buf for buf, _, _ in t.write_stages]
    assert all(s.is_pinned() for s in stages)
    page = pool.numel() // pool.size(2)
    codes = page // 2 if t.cache_format == "int4" else page * pool.element_size()
    maxima = page // 32 * 2 if t.cache_format == "int4" else 0
    assert codes + maxima == t.block_bytes == t.slots.strides[0]
    assert pool.dtype == next(runner.model.parameters()).dtype == torch.bfloat16
    assert all(m.k_cache.dtype == m.v_cache.dtype == pool.dtype for m in runner.model.modules()
               if hasattr(m, "k_cache"))
    assert llm.config.num_kvcache_blocks == 36 and t.slots.nbytes == t.capacity * t.block_bytes
    host = torch.cuda.memory.host_memory_stats()
    return dict(config={f.name:getattr(llm.config, f.name) for f in fields(llm.config) if f.name != "hf_config"},
                engine_module=str(Path(sys.modules["nanovllm"].__file__).resolve()),
                gpu=str(torch.cuda.get_device_name()), graph_batches=runner.graph_bs,
                weight_dtype=str(next(runner.model.parameters()).dtype), attention_cache_dtype=str(pool.dtype),
                namespace=llm.scheduler.block_manager.namespace.hex(), kv_budget=runner.kv_cache_budget,
                page_elements=page, payload_bytes=codes, quant_metadata_bytes=maxima,
                slot_padding=t.slots.strides[0]-codes-maxima, slot_count=t.capacity,
                slot_stride=t.slots.strides[0], slot_base_mod64=t.slots.ctypes.data % 64,
                cpu_pageable_bytes=t.slots.nbytes, pinned_bytes=[s.untyped_storage().nbytes() for s in stages],
                pinned_allocator={k:v for k,v in host.items() if k.endswith(".current")},
                gpu_stage_bytes=t.device_stage.untyped_storage().nbytes(),
                gpu_pool_bytes=pool.untyped_storage().nbytes(), pool_shape=list(pool.shape),
                stage_dtype=str(t.device_stage.dtype), dma_per_page=t.block_bytes, process_memory=memory())


def dma_probe(llm, out):
    # Diagnostic after all measured requests, both IDs inactive, index never used again.
    manager = llm.scheduler.block_manager
    assert not manager.used_block_ids and manager.pending_write is None
    t = llm.model_runner.kv_transfer
    src = next(iter(manager.hash_to_block_id.values()))
    dst = (src + 1) % len(manager.blocks)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        t.save(src, 0)
        t.restore(0, dst)
        torch.cuda.synchronize()
    trace = out / "dma_trace.json"
    prof.export_chrome_trace(str(trace))
    events = [e for e in json.loads(trace.read_text())["traceEvents"] if e.get("cat") == "gpu_memcpy"]
    write_json(out / "dma_events.json", events)
    with gzip.open(out / "dma_trace.json.gz", "wb") as f:
        f.write(trace.read_bytes())
    trace.unlink()


def worker(args):
    run = Path(args.output)
    design = json.loads((run / "design.json").read_text())
    out = run / f"{args.seed}-{args.worker}"
    out.mkdir()
    fmt, slots = ARMS[args.worker]
    torch.manual_seed(0)
    cfg = dict(max_kvcache_blocks=36, cpu_prefix_cache_blocks=slots, max_num_batched_tokens=2048,
               max_num_seqs=8, max_model_len=9216, enable_metrics=True, gpu_memory_utilization=0.8,
               cpu_prefix_cache_format=fmt, cpu_prefix_batch_writes=True, cpu_prefix_transfer_mode="overlap",
               cpu_prefix_pack_cold=True, cpu_prefix_admission="reuse", cpu_prefix_first_use_blocks=32)
    llm = LLM(args.model, **cfg)
    try:
        t = llm.model_runner.kv_transfer
        t.slots.fill(0)
        for _, inputs in make_traces(3499, 0)["mixed"]:
            wave(llm, inputs, OUTPUT, {})
        write_json(out / "engine.json", inventory(llm))
        traces = json.loads((run / f"trace-{args.seed}.json").read_text())
        for workload, phases in traces.items():
            reset_cache(llm)
            manager = llm.scheduler.block_manager
            assert not manager.entries and not manager.used_block_ids
            torch.manual_seed(args.seed)
            for i, (phase, inputs) in enumerate(phases):
                tag = dict(seed=args.seed, arm=args.worker, workload=workload, phase=phase, wave_index=i)
                batch = wave(llm, inputs, OUTPUT, tag)
                assert manager.pending_write is None and not manager.used_block_ids
                for pos, row in enumerate(batch[0]):
                    assert row["input_token_ids"] == inputs[pos] and row["output_tokens"] == OUTPUT
                    assert row["initial_gpu_hit_tokens"] + row["cpu_restored_tokens"] + row["prompt_computed_tokens"] == len(inputs[pos])
                    assert row["initial_cpu_hit_tokens"] == row["cpu_restored_tokens"] == row["h2d_blocks"] * 256
                    assert row["cpu_save_failures"] == row["cpu_restore_failures"] == row["preemptions"] == 0
                    assert abs(row["ttft_ms"] - 1000 * (row["token_s"][0] - row["enqueue_s"])) < 1e-5
                    row.update(fifo=pos, d2h_bytes=row["d2h_blocks"]*t.block_bytes,
                               h2d_bytes=row["h2d_blocks"]*t.block_bytes)
                batch[2].update(cpu_index_reachable_bytes=index_bytes(manager),
                                cpu_occupied_slot_bytes=len(manager.entries)*t.block_bytes,
                                also_indexed_on_gpu=len(manager.entries.keys() &
                                    {manager.gpu_keys[j] for j in manager.hash_to_block_id.values()}))
                save_wave(out, batch)
            print(f"{args.seed} {args.worker} {workload} done", flush=True)
        if design["stage"] == "screen" and args.worker != "bf16_18":
            dma_probe(llm, out)
    finally:
        llm.exit()


def read_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def paired(groups, seeds):
    rng = np.random.default_rng(3400)
    draws = rng.integers(0, len(seeds), size=(4000, len(seeds)))
    p95, boots = {}, {}
    for arm, rows in groups.items():
        data = np.array([[r["ttft_ms"] for r in rows if r["seed"] == seed] for seed in seeds])
        p95[arm] = np.percentile(data, 95)
        boots[arm] = np.percentile(data[draws].reshape(len(draws), -1), 95, axis=1)
    result = {}
    for base in ("bf16_64", "bf16_18"):
        changes = 100 * (boots["int4_64"] / boots[base] - 1)
        result[base] = dict(change_pct=100*(p95["int4_64"]/p95[base]-1),
                            paired_seed_95pct=np.percentile(changes,[2.5,97.5]).tolist() if len(seeds)>1 else None)
    return result


def report(out):
    design = json.loads((out / "design.json").read_text())
    seeds = design["seeds"]
    raw, waves = [], []
    for seed in seeds:
        arms = {}
        for arm in ARMS:
            directory = out / f"{seed}-{arm}"
            group = read_lines(directory / "requests.jsonl")
            arms[arm] = group
            raw.extend(group)
            waves.extend(read_lines(directory / "waves.jsonl"))
            assert len(group) == (5+12)*4
        for a,b,c in zip(*arms.values(), strict=True):
            assert a["arm"] == "bf16_64" and b["arm"] == "int4_64"
            for k in ("input_token_ids","seed","phase","wave_index","fifo","output_tokens"):
                assert a[k] == b[k] == c[k], k
            for k in ("initial_gpu_hit_tokens","cpu_restored_tokens","d2h_blocks","prompt_computed_tokens"):
                assert a[k] == b[k], k
    def stats(rows):
        restores = [r for r in rows if r["h2d_blocks"]]
        return dict(ttft_ms=distribution([r["ttft_ms"] for r in rows]),
                    gpu_hit_tokens=sum(r["initial_gpu_hit_tokens"] for r in rows),
                    restored_tokens=sum(r["cpu_restored_tokens"] for r in rows),
                    computed_tokens=sum(r["prompt_computed_tokens"] for r in rows),
                    full_cpu_hits=sum(r["cpu_restored_tokens"]==r["prompt_tokens"]-128 for r in rows),
                    h2d_bytes=sum(r["h2d_bytes"] for r in rows), d2h_bytes=sum(r["d2h_bytes"] for r in rows),
                    h2d_blocks=sum(r["h2d_blocks"] for r in rows), d2h_blocks=sum(r["d2h_blocks"] for r in rows),
                    h2d_ms_per_block=sum(r["h2d_ms"] for r in restores)/sum(r["h2d_blocks"] for r in restores) if restores else None,
                    preemptions=sum(r["preemptions"] for r in rows),
                    failures=sum(r["cpu_restore_failures"]+r["cpu_save_failures"] for r in rows))
    scopes = {"mixed":{"whole":lambda p:True, "fill":lambda p:p=="fill","no_reuse":lambda p:p.startswith("pressure"),
                       "evicted":lambda p:p=="evicted","hot":lambda p:p=="hot"},
              "capacity":{"whole":lambda p:True,"fill":lambda p:p.startswith("fill"),"touch":lambda p:p.startswith("touch"),
                          "revisit":lambda p:p.startswith("revisit")}}
    summary = {}
    for workload, filters in scopes.items():
        summary[workload] = {}
        for scope, keep in filters.items():
            selected = [r for r in raw if r["workload"] == workload and keep(r["phase"])]
            groups = {arm:[r for r in selected if r["arm"]==arm] for arm in ARMS}
            results = {arm:stats(rows) for arm,rows in groups.items()}
            for arm in ARMS:
                w = [e for e in waves if e["arm"]==arm and e["workload"]==workload and keep(e["phase"])]
                results[arm].update(output_tokens_per_s=sum(e["output_tokens"] for e in w)/sum(e["elapsed_s"] for e in w),
                    max_resident_slots=max(e["cpu_resident_blocks"] for e in w),
                    max_cpu_index_reachable_bytes=max(e["cpu_index_reachable_bytes"] for e in w),
                    gpu_peak_allocated=max(e["memory"]["gpu_peak_allocated"] for e in w),
                    gpu_peak_reserved=max(e["memory"]["gpu_peak_reserved"] for e in w),
                    cpu_peak_rss=max(e["memory"]["cpu_peak_rss"] for e in w))
            summary[workload][scope] = dict(arms=results, int4_vs=paired(groups,seeds))
    write_json(out / "summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    p.add_argument("--stage", choices=("screen","formal"), default="screen")
    p.add_argument("--worker", choices=tuple(ARMS), help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.summarize:
        report(Path(args.output))
        return
    args.module = "benchmarks.int4_prefix"
    out = prepare_run(args.output, args, args.model)
    seeds = [3400] if args.stage == "screen" else list(range(3501,3507))
    orders = list(permutations(ARMS))
    plan = []
    for index,seed in enumerate(seeds):
        f = out / f"trace-{seed}.json"
        write_json(f, make_traces(seed,index))
        plan.append(dict(seed=seed, order=orders[index%6], trace_sha256=sha256(f)))
    write_json(out / "design.json", dict(stage=args.stage,seeds=seeds,plan=plan,arms=ARMS,output=OUTPUT,
        block_size=256, concurrency=4, model_weight_dtype="BF16", gpu_blocks=36, graph=True,
        init="fresh process per arm/seed, prefault all slots, run separate warm mixed cycle, reset both indices per workload",
        workload="mixed FIFO 1024/2048 +128; capacity has 4 banks of 4 x 1024+128, immediate touch then delayed revisit",
        timing="from enqueue through first real token; whole includes cold fill; wave wall through all writes, no profiler",
        bootstrap="4000 paired whole-seed resamples, recompute pooled request p95; one seed descriptive only"))
    for row in plan:
        for arm in row["order"]:
            subprocess.run([sys.executable,"-m","benchmarks.int4_prefix","--output",str(out),
                            "--model",args.model,"--worker",arm,"--seed",str(row["seed"])], check=True)
    report(out)


if __name__ == "__main__":
    main()
