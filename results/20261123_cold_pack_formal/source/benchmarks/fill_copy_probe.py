"""Measure completed writes to pageable CPU KV slots and isolate copy overlap."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoConfig

from nanovllm.engine.kv_transfer import KVTransfer
from benchmarks.common import distribution, prepare_run, write_json, write_jsonl


def overlap_trace(transfer, out):
    pairs = [(k, k) for k in range(8)]
    original = np.copyto

    def marked_copy(*args):
        with torch.profiler.record_function("pageable_slot_write"):
            return original(*args)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        with patch.object(np, "copyto", marked_copy):
            with torch.profiler.record_function("serial_group"):
                for gpu, slot in pairs:
                    transfer.save(gpu, slot)
            with torch.profiler.record_function("two_buffer_group"):
                transfer.save_many(pairs)
    trace = out / "copy_trace.json"
    profile.export_chrome_trace(str(trace))
    return analyze_overlap(trace)


def analyze_overlap(trace):
    events = [e for e in json.loads(trace.read_text())["traceEvents"] if e.get("ph") == "X"]
    trace_stats = {}
    for name in ("serial_group", "two_buffer_group"):
        group, = (e for e in events if e["name"] == name and e.get("cat") == "user_annotation")
        start, stop = group["ts"], group["ts"] + group["dur"]
        members = [e for e in events if start <= e["ts"] < stop]
        cpu = sorted((e for e in members if e["name"] == "pageable_slot_write" and e.get("cat") == "user_annotation"),
                     key=lambda e: e["ts"])
        d2h = sorted((e for e in members if e.get("cat") == "gpu_memcpy" and "dtoh" in e["name"].lower()),
                     key=lambda e: e["ts"])
        assert len(cpu) == len(d2h) == 8
        # The two-buffer pipeline overlaps the CPU write of i with the GPU copy of i+1.
        overlaps = [max(0., min(c["ts"] + c["dur"], g["ts"] + g["dur"]) - max(c["ts"], g["ts"]))
                    for c, g in zip(cpu[:-1], d2h[1:], strict=True)]
        trace_stats[name] = dict(cpu_copies=len(cpu), gpu_d2h=len(d2h), adjacent_overlapping_pairs=sum(t > 0 for t in overlaps),
                                 adjacent_overlap_ms=[t / 1000 for t in overlaps],
                                 overlap_ms=sum(overlaps) / 1000, d2h_ms=sum(g["dur"] for g in d2h) / 1000,
                                 cpu_copy_ms=sum(c["dur"] for c in cpu) / 1000)
    return trace_stats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--pipeline", action="store_true", help="compare serial and bounded two-buffer copies, then trace one group")
    args = parser.parse_args()
    args.module = "benchmarks.fill_copy_probe"
    out = prepare_run(args.output, args, args.model)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    torch.manual_seed(1952)
    shape = (2, cfg.num_hidden_layers, 36, 256, cfg.num_key_value_heads, cfg.head_dim)
    cache = torch.randn(shape, device="cuda", dtype=cfg.dtype)
    transfer = KVTransfer(cache, 64, batch_writes=args.pipeline)
    transfer.slots.fill(0)
    host = torch.from_numpy(transfer.slots).view(cfg.dtype).reshape(64, *transfer.pinned.shape)
    assert not host.is_pinned() and transfer.pinned.is_pinned()
    arms = ("staged_numpy", "two_buffer") if args.pipeline else ("staged_numpy", "direct_pageable", "staged_torch")
    orders = list(permutations(arms))
    write_json(out / "design.json", dict(shape=shape, dtype=str(cfg.dtype), cpu_slots=64, block_bytes=transfer.block_bytes,
               synthetic_seed=1952, copies_per_batch=8, timed_batches_per_arm=12, cpu_threads=torch.get_num_threads(),
               orders=orders, endpoint="synchronous completed writes, including CPU copy; verification outside timing"))
    times = {arm: [] for arm in arms}
    for repeat in range(-1, 12):
        # Disjoint destinations between arms; a wrong or missing write must fail verification.
        src = [(repeat * 3 + k * 5) % shape[2] for k in range(8)]
        for arm in orders[repeat % len(orders)]:
            slots = [arms.index(arm) * 16 + (repeat % 2) * 8 + k for k in range(8)]
            torch.cuda.synchronize()
            start = perf_counter()
            pairs = list(zip(src, slots, strict=True))
            if arm == "two_buffer":
                transfer.save_many(pairs)
            elif arm == "staged_numpy":
                for gpu, slot in pairs:
                    transfer.save(gpu, slot)
            else:
                for gpu, slot in pairs:
                    transfer.device_stage.copy_(cache[:, :, gpu])
                    if arm == "direct_pageable":
                        host[slot].copy_(transfer.device_stage)
                    else:
                        transfer.pinned.copy_(transfer.device_stage)
                        host[slot].copy_(transfer.pinned)
            torch.cuda.synchronize()
            elapsed = (perf_counter() - start) * 1000
            if repeat >= 0:
                times[arm].append(elapsed)
                write_jsonl(out / "batches.jsonl", [dict(repeat=repeat, arm=arm, gpu_ids=src, slots=slots, ms=elapsed)])
        for k, gpu in enumerate(src):
            expected = cache[:, :, gpu].contiguous().cpu().view(torch.uint8)
            for index in range(len(arms)):
                actual = host[index * 16 + (repeat % 2) * 8 + k].view(torch.uint8)
                assert torch.equal(actual, expected), (repeat, gpu, index)
    write_json(out / "summary.json", {k: distribution(v) for k, v in times.items()})
    if args.pipeline:
        write_json(out / "overlap.json", overlap_trace(transfer, out))
    write_json(out / "status.json", dict(passed=True, verified_block_writes=13 * 8 * len(arms)))
    transfer.close()


if __name__ == "__main__":
    main()
