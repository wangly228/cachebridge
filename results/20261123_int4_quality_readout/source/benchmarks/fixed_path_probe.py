"""Position-aligned logits and immutable real-KV diagnostics; no performance data."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.correctness_protocol import PROTOCOL, bitwise_equal, compare_logits
from benchmarks.phase1 import prompts, reset_cache


class FixedPathProbe:
    def __init__(self, llm, output):
        self.llm, self.output = llm, output
        self.runner = llm.model_runner
        self.original_run, self.original_model = self.runner.run, self.runner.run_model
        self.runner.run, self.runner.run_model = self.run, self.model
        self.snapshots = {}
        self.active = []

    def model(self, input_ids, positions, is_prefill):
        logits = self.original_model(input_ids, positions, is_prefill)
        self.current_logits = {}
        for row, seq in enumerate(self.active):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                self.current_logits[seq.seq_id] = logits[row].detach().cpu().clone()
        return logits

    def gather_prefix(self, seq):
        count = (seq.num_prompt_tokens - 1) // 256
        index = torch.tensor(seq.block_table[:count], dtype=torch.int64, device="cuda")
        return self.runner.kv_cache.index_select(2, index)

    def check_prefix(self, seq):
        index = self.indices[seq.seq_id]
        snapshot = self.snapshots[index]
        manager = self.llm.scheduler.block_manager
        prefix_hash = -1
        for i, expected_tokens in enumerate(snapshot["tokens"]):
            block = manager.blocks[seq.block_table[i]]
            prefix_hash = BlockManager.compute_hash(expected_tokens, prefix_hash)
            assert block.token_ids == expected_tokens and block.hash == prefix_hash
            assert block.ref_count > 0 and block.block_id in manager.used_block_ids
        assert bitwise_equal(self.gather_prefix(seq), snapshot["kv"]), (self.group, self.case, index, "prefix modified")
        self.prefix_checks += 1

    def snapshot(self, seq):
        index = self.indices[seq.seq_id]
        count = (seq.num_prompt_tokens - 1) // 256
        kv = self.gather_prefix(seq)
        tokens = [seq.block(i) for i in range(count)]
        hashes, digests = [], []
        prefix_hash = -1
        for i, token_ids in enumerate(tokens):
            prefix_hash = BlockManager.compute_hash(token_ids, prefix_hash)
            hashes.append(prefix_hash)
            cpu_bytes = kv[:, :, i].contiguous().cpu().view(torch.uint8).numpy()
            digests.append(hashlib.sha256(cpu_bytes).hexdigest())
        self.snapshots[index] = dict(kv=kv, tokens=tokens, hashes=hashes)
        write_jsonl(self.output / "prefix_snapshots.jsonl", [dict(group=self.group, request=index,
                    blocks=count, bytes=kv.numel() * kv.element_size(), block_sha256=digests,
                    logical_hash_chain=hashes, original_block_ids=seq.block_table[:count])])

    def run(self, seqs, is_prefill):
        self.active = seqs
        for seq in seqs:
            index = self.indices[seq.seq_id]
            assert seq.prompt_token_ids == self.inputs[index]
            assert seq.metrics.preemptions == 0
            if self.forced is not None:
                assert seq.completion_token_ids == self.forced[index][:seq.num_completion_tokens]
            if self.case != "cold":
                expected = ((len(self.inputs[index]) - 1) // 256) * 256
                assert seq.metrics.initial_gpu_hit_tokens == expected
                self.check_prefix(seq)
        sampled = self.original_run(seqs, is_prefill)
        for row, seq in enumerate(seqs):
            if self.case != "cold":
                self.check_prefix(seq)
            if seq.seq_id not in self.current_logits:
                continue  # chunked prefill without a real output position
            index, position = self.indices[seq.seq_id], seq.num_completion_tokens
            logits = self.current_logits[seq.seq_id]
            assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
            assert sampled[row] == logits.argmax().item()
            if self.case == "cold" and position == 0:
                self.snapshot(seq)
            self.logits[index, position] = logits
            forced_token = sampled[row] if self.forced is None else self.forced[index][position]
            self.events.append(dict(group=self.group, case=self.case, request=index, position=position,
                absolute_query_position=len(seq) - 1, greedy_token=sampled[row], used_token=forced_token,
                is_prefill=is_prefill, cached_before=seq.num_cached_tokens,
                scheduled_tokens=seq.num_scheduled_tokens, block_table=seq.block_table.copy()))
            sampled[row] = forced_token
        return sampled

    def execute(self, group, case, inputs, forced=None):
        self.group, self.case, self.inputs, self.forced = group, case, inputs, forced
        self.logits, self.events, self.indices, self.prefix_checks = {}, [], {}, 0
        self.llm.clear_metrics()
        ids = []
        for index, prompt in enumerate(inputs):
            seq_id = self.llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
            self.indices[seq_id] = index
            ids.append(seq_id)
        outputs, steps = {}, 0
        try:
            while not self.llm.is_finished():
                finished, _ = self.llm.step()
                outputs.update(finished)
                steps += 1
                assert steps < 200, "unexpected lack of scheduler progress"
        finally:
            write_jsonl(self.output / "positions.jsonl", self.events)
        arrays = torch.stack([torch.stack([self.logits[i, p] for p in range(16)]) for i in range(len(inputs))])
        path = [outputs[seq_id] for seq_id in ids]
        if forced is not None:
            assert path == forced
        write_jsonl(self.output / "cases.jsonl", [dict(group=group, case=case, steps=steps,
            output_tokens=path, prefix_bitwise_checks=self.prefix_checks,
            initial_gpu_hits=[self.llm.request_metrics[i].initial_gpu_hit_tokens for i in ids],
            prefill_tokens=[self.llm.request_metrics[i].prefill_tokens for i in ids],
            timing_valid_for_performance=False)])
        print(f"{group}/{case}: {len(inputs) * 16} positions, {self.prefix_checks} prefix bit checks", flush=True)
        return arrays, path

    def roundtrip(self):
        assert self.llm.is_finished()
        manager, cache = self.llm.scheduler.block_manager, self.runner.kv_cache
        assert not manager.used_block_ids
        pinned = torch.empty((2, cache.shape[1], *cache.shape[3:]), dtype=cache.dtype, pin_memory=True)
        for index, snapshot in self.snapshots.items():
            for i, prefix_hash in enumerate(snapshot["hashes"]):
                block_id = manager.hash_to_block_id[prefix_hash]
                block = manager.blocks[block_id]
                assert block.ref_count == 0 and block.token_ids == snapshot["tokens"][i]
                assert block.hash == prefix_hash and block_id not in manager.used_block_ids
                view, expected = cache[:, :, block_id], snapshot["kv"][:, :, i]
                assert bitwise_equal(view, expected)
                pinned.copy_(view, non_blocking=True)
                torch.cuda.synchronize()
                assert bitwise_equal(pinned, expected.cpu())
                # Ensure a no-op H2D cannot pass this diagnostic. No request is active.
                view.fill_(float("nan"))
                assert not bitwise_equal(view, expected)
                view.copy_(pinned, non_blocking=True)
                torch.cuda.synchronize()
                assert bitwise_equal(view, expected)
                write_jsonl(self.output / "roundtrip.jsonl", [dict(group=self.group, request=index,
                    logical_block=i, gpu_block=block_id, ref_count=0, bytes=pinned.numel() * pinned.element_size(),
                    d2h_bitwise_equal=True, poison_detected=True, h2d_bitwise_equal=True)])
        print(f"{self.group}: real KV roundtrip passed; pinned scratch {pinned.numel() * pinned.element_size()} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", default="results/20261117_phase1_smoke_eager")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.module = "benchmarks.fixed_path_probe"
    source = Path(args.input_run)
    config = json.loads((source / "engine.json").read_text())["config"]
    rows = [r for r in map(json.loads, (source / "requests.jsonl").read_text().splitlines())
            if r["phase"] == "baseline" and r["length"] == 2048 and r["concurrency"] == 4 and r["cache"] == "cold"]
    assert len(rows) == 4 and all(len(r["output_token_ids"]) == 16 for r in rows)
    groups = [dict(name="regression_2048_c4", inputs=[r["input_token_ids"] for r in rows],
                   forced=[r["output_token_ids"] for r in rows], held_out=False),
              dict(name="heldout_2048_c4", inputs=prompts(20261118, 4, 2048), forced=None, held_out=True),
              dict(name="heldout_8192_c1", inputs=prompts(20261119, 1, 8192), forced=None, held_out=True)]
    output = prepare_run(args.output, args, config["model"])
    write_json(output / "protocol.json", dict(**PROTOCOL, document_sha256=sha256("docs/CORRECTNESS_PROTOCOL.md")))
    write_json(output / "inputs.json", groups)
    llm = None
    try:
        config.pop("num_kvcache_blocks")
        config["enforce_eager"] = True
        torch.manual_seed(0)
        llm = LLM(**config)
        write_json(output / "engine.json", dict(config=config, kv_budget=llm.model_runner.kv_cache_budget,
                   kv_shape=list(llm.model_runner.kv_cache.shape), kv_dtype=str(llm.model_runner.kv_cache.dtype)))
        probe = FixedPathProbe(llm, output)
        summaries = []
        for group in groups:
            probe.snapshots.clear()
            reset_cache(llm)
            cold, path = probe.execute(group["name"], "cold", group["inputs"], group["forced"])
            write_json(output / f"{group['name']}_path.json", path)
            warm, _ = probe.execute(group["name"], "warm", group["inputs"], path)
            repeat, _ = probe.execute(group["name"], "warm_repeat", group["inputs"], path)
            # Keep logits even if an exact control below fails.
            np.savez_compressed(output / f"{group['name']}_logits_bf16.npz",
                cold=cold.view(torch.int16).numpy(), warm=warm.view(torch.int16).numpy(),
                warm_repeat=repeat.view(torch.int16).numpy())
            assert bitwise_equal(warm, repeat), (group["name"], "same-shape logits differ")
            probe.roundtrip()
            restored, _ = probe.execute(group["name"], "warm_after_roundtrip", group["inputs"], path)
            np.savez_compressed(output / f"{group['name']}_roundtrip_logits_bf16.npz",
                                warm_after_roundtrip=restored.view(torch.int16).numpy())
            assert bitwise_equal(warm, restored), (group["name"], "logits changed after roundtrip")
            comparisons = [dict(group=group["name"], held_out=group["held_out"], request=i, position=p,
                                **compare_logits(cold[i, p], warm[i, p]))
                           for i in range(len(group["inputs"])) for p in range(16)]
            write_jsonl(output / "comparisons.jsonl", comparisons)
            summaries.append(dict(group=group["name"], held_out=group["held_out"], positions=len(comparisons),
                exact_repeat=True, exact_after_roundtrip=True,
                numerical_screen_passed=all(r["passed"] for r in comparisons),
                numerical_failures=sum(not r["passed"] for r in comparisons),
                argmax_mismatches=sum(not r["argmax_equal"] for r in comparisons),
                max_abs_error=max(r["max_abs"] for r in comparisons),
                max_relative_l2=max(r["relative_l2"] for r in comparisons)))
            write_json(output / "summary.json", summaries)
        passed = all(r["numerical_screen_passed"] for r in summaries)
        write_json(output / "status.json", dict(success=passed, exact_invariants_passed=True,
                   numerical_screen_passed=passed, baseline_frozen=False,
                   note="scope limited to this fixed-path protocol; historic strict greedy failure remains failed"))
        if not passed:
            raise RuntimeError("predeclared numerical screening failed; thresholds left unchanged")
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        if not (output / "status.json").exists():
            write_json(output / "status.json", dict(success=False, baseline_frozen=False))
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
