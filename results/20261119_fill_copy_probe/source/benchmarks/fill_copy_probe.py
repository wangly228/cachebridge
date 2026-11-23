"""Compare three synchronous writes to the same pageable CPU KV slots."""
import argparse
from itertools import permutations
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoConfig

from nanovllm.engine.kv_transfer import KVTransfer
from benchmarks.common import distribution, prepare_run, write_json, write_jsonl


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.fill_copy_probe"
    out = prepare_run(args.output, args, args.model)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    torch.manual_seed(1952)
    shape = (2, cfg.num_hidden_layers, 36, 256, cfg.num_key_value_heads, cfg.head_dim)
    cache = torch.randn(shape, device="cuda", dtype=cfg.dtype)
    transfer = KVTransfer(cache, 64)
    transfer.slots.fill(0)
    host = torch.from_numpy(transfer.slots).view(cfg.dtype).reshape(64, *transfer.pinned.shape)
    assert not host.is_pinned() and transfer.pinned.is_pinned()
    arms = ("staged_numpy", "direct_pageable", "staged_torch")
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
            for gpu, slot in zip(src, slots):
                if arm == "staged_numpy":
                    transfer.save(gpu, slot)
                else:
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
    write_json(out / "status.json", dict(passed=True, verified_block_writes=13 * 8 * len(arms)))
    transfer.close()


if __name__ == "__main__":
    main()
