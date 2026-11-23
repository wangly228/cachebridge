"""Synchronous scratch KV transfer cost in the runner's actual six-axis layout.

No cache store or policy is implemented. GPU source/target block IDs are
scattered. The direct path copies each contiguous layer/block view; the packed
path includes gather, pinned D2H/H2D, and scatter. Allocation is outside timing.
"""
import argparse
from pathlib import Path
from time import perf_counter
import traceback

import torch
from transformers import AutoConfig

from benchmarks.common import distribution, memory, prepare_run, write_json, write_jsonl


def measure(operation):
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start = perf_counter()
    begin.record()
    operation()
    end.record()
    end.synchronize()
    return dict(wall_ms=(perf_counter() - start) * 1000, cuda_event_ms=begin.elapsed_time(end))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[1, 4, 8, 32])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.module = "benchmarks.kv_transfer"
    if min(args.blocks) <= 0 or max(args.blocks) > 32 or args.repeats < 1:
        parser.error("blocks must be in [1, 32], repeats >= 1 (bounded scratch memory)")
    output = prepare_run(args.output, args, args.model)
    try:
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        assert config.dtype == torch.bfloat16
        pool_blocks = 2 * max(args.blocks)
        shape = (2, config.num_hidden_layers, pool_blocks, 256, config.num_key_value_heads, config.head_dim)
        torch.manual_seed(1234)
        source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        target = torch.empty_like(source)
        block_bytes = source.numel() * source.element_size() // pool_blocks
        write_json(output / "layout.json", dict(shape=shape, stride=source.stride(), dtype=str(source.dtype),
                   block_bytes=block_bytes, source="scratch tensor with Qwen3-4B runner layout; not model-generated KV",
                   pinned=True, copies_synchronous=True, allocation_excluded=True,
                   timing="wall includes CPU loop/launch and end-event wait; CUDA event includes stream idle gaps",
                   profiler_enabled=False, cpu_cache_enabled=False))
        summaries = []
        for count in args.blocks:
            src_ids = list(range(0, count * 2, 2))
            dst_ids = list(range(1, count * 2, 2))
            src_index = torch.tensor(src_ids, device="cuda")
            dst_index = torch.tensor(dst_ids, device="cuda")
            transfer_shape = (shape[0], shape[1], count, *shape[3:])
            host = torch.empty(transfer_shape, dtype=source.dtype, pin_memory=True)
            staging = torch.empty(transfer_shape, device="cuda", dtype=source.dtype)
            # Prepare views outside timing, matching reusable transfer descriptors.
            source_views = [source[k, layer, block] for k in range(2) for layer in range(shape[1]) for block in src_ids]
            target_views = [target[k, layer, block] for k in range(2) for layer in range(shape[1]) for block in dst_ids]
            host_views = [host[k, layer, i] for k in range(2) for layer in range(shape[1]) for i in range(count)]

            def direct_d2h():
                for cpu, gpu in zip(host_views, source_views):
                    cpu.copy_(gpu, non_blocking=True)

            def direct_h2d():
                for gpu, cpu in zip(target_views, host_views):
                    gpu.copy_(cpu, non_blocking=True)

            def packed_d2h():
                torch.index_select(source, 2, src_index, out=staging)
                host.copy_(staging, non_blocking=True)

            def packed_h2d():
                staging.copy_(host, non_blocking=True)
                target.index_copy_(2, dst_index, staging)

            for strategy, d2h, h2d in (("direct", direct_d2h, direct_h2d), ("packed", packed_d2h, packed_h2d)):
                records = []
                target.zero_()
                for repeat in range(args.repeats + 2):
                    down, up = measure(d2h), measure(h2d)
                    if repeat >= 2:
                        record = dict(strategy=strategy, blocks=count, tokens=count * 256,
                                      bytes_per_direction=block_bytes * count, repeat=repeat - 2,
                                      d2h=down, h2d=up)
                        records.append(record)
                        write_jsonl(output / "transfer.jsonl", [record])
                # Bitwise equality via int16 views; not approximate numeric equality.
                assert torch.equal(source.index_select(2, src_index).view(torch.int16),
                                   target.index_select(2, dst_index).view(torch.int16))
                assert torch.count_nonzero(target.index_select(2, src_index)).item() == 0
                summary = dict(strategy=strategy, blocks=count, tokens=count * 256,
                               bytes_per_direction=block_bytes * count, roundtrip_bitwise_equal=True,
                               untouched_blocks_zero=True,
                               d2h_wall_ms=distribution([r["d2h"]["wall_ms"] for r in records]),
                               h2d_wall_ms=distribution([r["h2d"]["wall_ms"] for r in records]), memory=memory(),
                               pinned_bytes=host.numel() * host.element_size(),
                               staging_bytes=staging.numel() * staging.element_size())
                summaries.append(summary)
                print(f"{strategy} {count} blocks: D2H {summary['d2h_wall_ms']['p50']:.3f}ms, H2D {summary['h2d_wall_ms']['p50']:.3f}ms", flush=True)
            del host_views, source_views, target_views, host, staging
        write_json(output / "summary.json", summaries)
        write_json(output / "status.json", dict(success=True, failures=0))
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        write_json(output / "status.json", dict(success=False))
        raise


if __name__ == "__main__":
    main()
