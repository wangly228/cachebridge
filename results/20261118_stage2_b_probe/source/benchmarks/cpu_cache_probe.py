"""Verify evicted model KV and pre-sampling logits at the same input and execution shape."""
import argparse
from hashlib import sha256
from pathlib import Path
import traceback

import numpy as np
import torch

from nanovllm import LLM
from benchmarks.common import prepare_run, write_json, write_jsonl
from benchmarks.phase1 import prompts, save_wave, wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = parser.parse_args()
    args.module = "benchmarks.cpu_cache_probe"
    out = prepare_run(args.output, args, args.model)
    llm = None
    try:
        torch.manual_seed(0)
        config = dict(max_kvcache_blocks=3, cpu_prefix_cache_blocks=4, enable_metrics=True,
                      max_num_batched_tokens=512, max_model_len=768, max_num_seqs=8,
                      tensor_parallel_size=1, enforce_eager=False, gpu_memory_utilization=0.80)
        llm = LLM(args.model, **config)
        runner = llm.model_runner
        manager = llm.scheduler.block_manager
        assert runner.kv_cache.dtype == torch.bfloat16 and llm.config.num_kvcache_blocks == 3
        write_json(out / "engine.json", dict(config=config, kv_budget=runner.kv_cache_budget,
                   namespace=manager.namespace.hex(), dtype=str(runner.kv_cache.dtype), graph_bs=runner.graph_bs,
                   output_tokens=4, temperature=0, ignore_eos=True, concurrency=1, diagnostic_hooks=True))
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
