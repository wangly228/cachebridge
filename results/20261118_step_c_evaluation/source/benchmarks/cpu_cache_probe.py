"""Verify evicted model KV and pre-sampling logits at the same input and execution shape."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence
from benchmarks.common import prepare_run, write_json, write_jsonl
from benchmarks.phase1 import prompts, save_wave, wave


def aligned_tail(llm, out):
    manager = llm.scheduler.block_manager
    runner = llm.model_runner
    tokens = prompts(8130, 1, 512)[0]
    longer = tokens + [142]
    other = prompts(8131, 1, 513)[0]
    cases = [("cold", tokens, 0, 0), ("recomputed_tail", tokens, 256, 0),
             ("hot", longer, 512, 0), ("evict", other, 0, 0), ("restore", longer, 0, 512)]
    snapshots, outputs, logits = {}, {}, {}
    original_model = runner.run_model

    def capture(ids, positions, is_prefill):
        row = original_model(ids, positions, is_prefill)
        if case in ("hot", "restore"):
            logits[case] = (ids.tolist(), positions.tolist(), row.detach().cpu().contiguous())
        return row

    runner.run_model = capture
    for case, prompt, gpu, cpu in cases:
        result = wave(llm, [prompt], 1, dict(phase=case))
        save_wave(out, result)
        row, = result[0]
        assert (row["initial_gpu_hit_tokens"], row["initial_cpu_hit_tokens"]) == (gpu, cpu)
        outputs[case] = row["output_token_ids"]
        if case == "recomputed_tail":
            assert row["d2h_blocks"] == 1
        if case in ("cold", "recomputed_tail", "restore"):
            tail = manager._lookup(Sequence(longer))[1]
            kv = runner.kv_cache[:, :, tail.gpu_id].contiguous().view(torch.uint8).cpu().numpy().ravel()
            slot, _ = manager.entries[tail.key]
            saved = runner.kv_transfer.slots[slot]
            snapshots[case] = kv
            different = int(np.count_nonzero(kv != saved))
            write_jsonl(out / "kv.jsonl", [dict(case=case, gpu_id=tail.gpu_id, cpu_slot=slot, bytes=kv.size,
                        gpu_sha256=sha256(kv).hexdigest(), cpu_sha256=sha256(saved).hexdigest(),
                        cpu_gpu_different_bytes=different)])
            assert different == 0
    hot_ids, hot_positions, reference = logits["hot"]
    ids, positions, restored = logits["restore"]
    assert (ids, positions) == (hot_ids, hot_positions)
    assert torch.equal(restored.view(torch.uint8), reference.view(torch.uint8))
    assert outputs["hot"] == outputs["restore"]
    assert np.array_equal(snapshots["recomputed_tail"], snapshots["restore"])
    torch.save(dict(hot=reference, restore=restored), out / "full_logits.pt")
    write_json(out / "status.json", dict(passed=True, waves=len(cases), restored_logit_rows_verified=1,
               recomputed_vs_original_tail_different_bytes=int(np.count_nonzero(
                   snapshots["cold"] != snapshots["recomputed_tail"])), restored_vs_recomputed_tail_different_bytes=0))


def default_off(model, out):
    baseline = Path("results/20261118_pressure36")
    config = json.loads((baseline / "engine.json").read_text())["config"]
    refs = [json.loads(s) for s in (baseline / "requests.jsonl").read_text().splitlines()]
    llm = LLM(model, **config)
    try:
        assert llm.config.cpu_prefix_cache_blocks == 0 and llm.model_runner.kv_transfer is None
        write_json(out / "engine.json", dict(config=config, kv_budget=llm.model_runner.kv_cache_budget))
        for state in ("fill", "hot"):
            rows = [r for r in refs if r.get("cache") == state and r["phase"] == "warmup"]
            assert len(rows) == 4
            result = wave(llm, [r["input_token_ids"] for r in rows], 4, dict(phase=state))
            save_wave(out, result)
            for actual, ref in zip(result[0], rows):
                for field in ("output_token_ids", "initial_gpu_hit_tokens", "prefill_tokens", "decode_tokens"):
                    assert actual[field] == ref[field]
                assert actual["h2d_blocks"] == actual["d2h_blocks"] == 0
        write_json(out / "status.json", dict(passed=True, frozen_A_equal_requests=8, cpu_store_allocated=False))
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    parser.add_argument("--case", choices=("eviction", "aligned-tail", "default-off"), default="eviction")
    args = parser.parse_args()
    args.module = "benchmarks.cpu_cache_probe"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        torch.manual_seed(0)
        if args.case == "default-off":
            return default_off(args.model, out)
        config = dict(max_kvcache_blocks=3, cpu_prefix_cache_blocks=4, enable_metrics=True,
                      max_num_batched_tokens=512, max_model_len=768, max_num_seqs=8,
                      tensor_parallel_size=1, enforce_eager=False, gpu_memory_utilization=0.80)
        llm = LLM(args.model, **config)
        runner = llm.model_runner
        manager = llm.scheduler.block_manager
        assert runner.kv_cache.dtype == torch.bfloat16 and llm.config.num_kvcache_blocks == 3
        write_json(out / "engine.json", dict(config=config, kv_budget=runner.kv_cache_budget,
                   namespace=manager.namespace.hex(), dtype=str(runner.kv_cache.dtype), graph_bs=runner.graph_bs,
                   output_tokens=1 if args.case == "aligned-tail" else 4, temperature=0,
                   ignore_eos=True, concurrency=1, diagnostic_hooks=True))
        if args.case == "aligned-tail":
            return aligned_tail(llm, out)
        original_run, original_model = runner.run, runner.run_model
        reference_kv, reference_logits, actual_logits = [], [], {}
        checks, logit_checks = [], []
        case = ""
        row_index = 0

        def checked_run(seqs, is_prefill):
            if case in ("hot", "restore1", "restore2") and is_prefill:
                seq, = seqs
                assert (seq.num_cached_tokens, seq.num_scheduled_tokens) == (512, 1)
                for i, gpu_id in enumerate(seq.block_table[:2]):
                    kv = runner.kv_cache[:, :, gpu_id].contiguous().view(torch.uint8).cpu().numpy().ravel()
                    key = manager.gpu_keys[gpu_id]
                    slot, tokens = manager.entries[key]
                    assert tokens == seq.block(i)
                    saved = runner.kv_transfer.slots[slot]
                    if case == "hot":
                        reference_kv.append(kv)
                    ref = reference_kv[i]
                    check = dict(case=case, block=i, gpu_id=gpu_id, cpu_slot=slot, bytes=kv.size,
                                 reference_sha256=sha256(ref).hexdigest(), saved_sha256=sha256(saved).hexdigest(),
                                 gpu_sha256=sha256(kv).hexdigest(),
                                 gpu_different_bytes=int(np.count_nonzero(ref != kv)),
                                 cpu_different_bytes=int(np.count_nonzero(ref != saved)))
                    checks.append(check)
                    write_jsonl(out / "kv.jsonl", [check])
                    assert check["gpu_different_bytes"] == check["cpu_different_bytes"] == 0
            return original_run(seqs, is_prefill)

        def checked_model(ids, positions, is_prefill):
            nonlocal row_index
            logits = original_model(ids, positions, is_prefill)
            if case in ("hot", "restore1", "restore2"):
                path = (ids.tolist(), positions.tolist(), is_prefill)
                row = logits.detach().cpu().contiguous()
                if case == "hot":
                    reference_logits.append((path, row))
                else:
                    expected_path, expected = reference_logits[row_index]
                    actual_logits[case].append(row)
                    differences = (row.view(torch.uint8) != expected.view(torch.uint8)).sum().item()
                    check = dict(case=case, position=row_index, input_ids=path[0], positions=path[1],
                                 is_prefill=is_prefill, path_equal=path == expected_path,
                                 shape=list(row.shape), dtype=str(row.dtype), different_bytes=differences)
                    logit_checks.append(check)
                    write_jsonl(out / "logits.jsonl", [check])
                    assert check["path_equal"] and differences == 0
                row_index += 1
            return logits

        runner.run, runner.run_model = checked_run, checked_model
        seed_prompt = prompts(8111, 1, 513)[0]
        target = seed_prompt[:512] + prompts(8112, 1, 1)[0]
        other, third, fourth = prompts(8113, 3, 513)
        cases = [("fill", seed_prompt, 0, 0), ("hot", target, 512, 0),
                 ("evict1", other, 0, 0), ("restore1", target, 0, 512),
                 ("evict2", other, 0, 512), ("restore2", target, 0, 512),
                 ("lru_fill1", third, 0, 0), ("lru_fill2", fourth, 0, 0),
                 ("lru_miss", target, 0, 0)]
        outputs = {}
        for case, tokens, gpu, cpu in cases:
            actual_logits[case] = []
            row_index = 0
            result = wave(llm, [tokens], 4, dict(phase=case))
            save_wave(out, result)
            row, = result[0]
            outputs[case] = row["output_token_ids"]
            assert (row["initial_gpu_hit_tokens"], row["initial_cpu_hit_tokens"]) == (gpu, cpu)
            assert row["prefill_tokens"] == 513 - gpu - cpu
            assert row["cpu_save_failures"] == row["cpu_restore_failures"] == 0
        assert outputs["hot"] == outputs["restore1"] == outputs["restore2"]
        assert len(checks) == 6 and len(logit_checks) == 8 and manager.evictions > 0
        torch.save(dict(hot=[t for _, t in reference_logits], restore1=actual_logits["restore1"],
                        restore2=actual_logits["restore2"]), out / "full_logits.pt")
        write_json(out / "status.json", dict(passed=True, waves=len(cases), restored_kv_blocks_verified=4,
                   restored_logit_rows_verified=8, greedy_replays_exact=2, cpu_evictions=manager.evictions))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", dict(passed=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
