"""Localize BF16 differences on the recomputed suffix and intervene on decode reads."""
import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch
from flash_attn import flash_attn_with_kvcache

from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.phase1 import reset_cache
from nanovllm import LLM, SamplingParams
from nanovllm.layers import attention


def difference(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    changes = (a.view(torch.int16) != b.view(torch.int16)).reshape(a.shape[0], -1).any(dim=1)
    indices = changes.nonzero().flatten()
    delta = a.float() - b.float()
    return dict(changed_rows=len(indices), first_changed=int(indices[0]) if len(indices) else None,
                max_abs=delta.abs().max().item(),
                relative_l2=(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)).item())


class Replay:
    def __init__(self, llm, out, group, path, archive, targets):
        self.llm, self.runner, self.out = llm, llm.model_runner, out
        self.name, self.inputs = group["name"], group["inputs"]
        self.path, self.archive, self.targets = path, archive, targets
        self.block = llm.config.kvcache_block_size
        self.cold_rows, self.cold_tails = {}, {}
        self.layers = self.runner.model.model.layers
        self.layer_for_key = {layer.self_attn.attn.k_cache.data_ptr(): i for i, layer in enumerate(self.layers)}

    def layer_hook(self, label, layer, module, args, output):
        if not self.slices:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        for i, span in self.slices:
            key = (i, layer, label)
            rows = tensor[span].detach().contiguous()
            assert len(rows) == self.block
            if self.mode == "cold":
                self.cold_rows[key] = rows.clone()
            else:
                diff = difference(self.cold_rows.pop(key), rows)
                self.diffs.append(dict(group=self.name, request=i, layer=layer, stage=label,
                    absolute_first_changed=(len(self.inputs[i]) - self.block + diff["first_changed"]
                                            if diff["first_changed"] is not None else None), **diff))

    def install_probes(self, stack):
        for n, layer in enumerate(self.layers):
            stages = [("block_output", layer)]
            if n == 0:
                a = layer.self_attn
                stages = [("embedding", self.runner.model.model.embed_tokens),
                          ("normalized_input", layer.input_layernorm), ("qkv_projection", a.qkv_proj),
                          ("attention_output", a.attn), ("output_projection", a.o_proj),
                          ("post_attention_norm", layer.post_attention_layernorm)] + stages
            for label, mod in stages:
                handle = mod.register_forward_hook(partial(self.layer_hook, label, n))
                stack.callback(handle.remove)

    def redirected_read(self, q, k_cache, v_cache, **kwargs):
        # Work on fresh storage. Original physical pages and block tables stay live.
        table = kwargs["block_table"]
        layer = self.layer_for_key[k_cache.data_ptr()]
        assert (table >= 0).all()  # groups have identical sequence lengths here
        pages = table.flatten().long()
        keys, values = k_cache.index_select(0, pages), v_cache.index_select(0, pages)
        scratch_table = torch.arange(pages.numel(), device=table.device, dtype=table.dtype).reshape_as(table)
        if self.mode == "cold_tail":
            for row, seq in enumerate(self.active):
                i = self.index[seq.seq_id]
                tail = seq.num_prompt_tokens // self.block - 1
                dest = row * table.size(1) + tail
                keys[dest].copy_(self.cold_tails[i][0, layer])
                values[dest].copy_(self.cold_tails[i][1, layer])
        return flash_attn_with_kvcache(q, keys, values, **dict(kwargs, block_table=scratch_table))

    def run(self, mode):
        self.mode = mode
        self.diffs, self.verdicts, self.shapes = [], [], []
        self.slices, self.active, self.index = [], [], {}
        llm, runner = self.llm, self.runner
        llm.clear_metrics()
        for i, prompt in enumerate(self.inputs):
            seq_id = llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=len(self.path[i]), ignore_eos=True))
            self.index[seq_id] = i
        base_run, base_model = runner.run, runner.run_model

        def model(ids, positions, prefill):
            logits = base_model(ids, positions, prefill)
            for row, seq in enumerate(self.active):
                if seq.num_cached_tokens + seq.num_scheduled_tokens != len(seq):
                    continue
                i, pos = self.index[seq.seq_id], seq.num_completion_tokens
                # A redirected decode still has the native warm prefill at position zero.
                expected = "cold" if mode == "cold" or (mode == "cold_tail" and pos > 0) else "warm"
                actual = logits[row].view(torch.int16).detach().cpu()
                ref = self.archive[expected][i, pos]
                equal = np.array_equal(actual.numpy(), ref)
                self.verdicts.append(dict(group=self.name, mode=mode, request=i, position=pos,
                    expected=expected, equal=equal, v1_outlier=[i, pos] in self.targets))
            return logits

        def run(seqs, prefill):
            self.active, self.slices = seqs, []
            offset = 0
            for seq in seqs:
                i, pos = self.index[seq.seq_id], seq.num_completion_tokens
                assert seq.completion_token_ids == self.path[i][:pos]
                end = offset + seq.num_scheduled_tokens
                if prefill and seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                    assert seq.num_scheduled_tokens >= self.block
                    self.shapes.append(dict(group=self.name, mode=mode, request=i,
                        scheduled_query=seq.num_scheduled_tokens, key_length=len(seq),
                        packed_query_rows=sum(s.num_scheduled_tokens for s in seqs),
                        initial_gpu_hit_tokens=seq.metrics.initial_gpu_hit_tokens))
                    if mode in ("cold", "warm") and any(t[0] == i for t in self.targets):
                        self.slices.append((i, slice(end-self.block, end)))
                offset = end
            output = base_run(seqs, prefill)
            for row, seq in enumerate(seqs):
                if seq.num_cached_tokens + seq.num_scheduled_tokens != len(seq):
                    continue
                i, pos = self.index[seq.seq_id], seq.num_completion_tokens
                if prefill:
                    assert len(seq) % self.block == 0
                    tail = runner.kv_cache[:, :, seq.block_table[-1]]
                    if mode == "cold":
                        self.cold_tails[i] = tail.clone()
                    elif mode == "warm" and any(t[0] == i for t in self.targets):
                        for layer in range(len(self.layers)):
                            for axis, label in enumerate(("stored_key", "stored_value")):
                                diff = difference(self.cold_tails[i][axis, layer], tail[axis, layer])
                                self.diffs.append(dict(group=self.name, request=i, layer=layer, stage=label,
                                    absolute_first_changed=(len(seq)-self.block+diff["first_changed"]
                                        if diff["first_changed"] is not None else None), **diff))
                output[row] = self.path[i][pos]
            return output

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "run", run))
            stack.enter_context(patch.object(runner, "run_model", model))
            if mode in ("cold", "warm"):
                self.install_probes(stack)
            else:
                stack.enter_context(patch.object(attention, "flash_attn_with_kvcache", self.redirected_read))
            while not llm.is_finished():
                llm.step()
        write_jsonl(self.out / "layer_differences.jsonl", self.diffs)
        write_jsonl(self.out / "logit_comparisons.jsonl", self.verdicts)
        write_jsonl(self.out / "shapes.jsonl", self.shapes)
        assert len(self.verdicts) == sum(map(len, self.path))
        assert all(s["initial_gpu_hit_tokens"] == (0 if mode == "cold" else s["key_length"] - self.block)
                   for s in self.shapes)
        if mode == "warm":
            assert not self.cold_rows
        if mode != "cold_tail":
            assert all(r["equal"] for r in self.verdicts), f"{mode} did not reproduce archive"
        return dict(group=self.name, mode=mode, positions=len(self.verdicts),
                    exact=sum(r["equal"] for r in self.verdicts),
                    exact_at_outliers=sum(r["equal"] and r["v1_outlier"] for r in self.verdicts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="results/20261118_fixed_path_probe")
    args = parser.parse_args()
    args.module = "benchmarks.prefill_acceptance"
    source = Path(args.source)
    config = json.loads((source / "engine.json").read_text())["config"]
    out = prepare_run(args.output, args, config["model"])
    failures = [r for r in map(json.loads, (source / "comparisons.jsonl").read_text().splitlines()) if not r["passed"]]
    write_json(out / "design.json", dict(config=config, source_sha256=sha256(source / "comparisons.jsonl"),
               changed_input="Only last prompt page for decode attention reads; all generated tokens teacher-forced",
               controls=["original cold", "original warm", "scratch remap with native warm pages"],
               criterion="bitwise equality of all vocabulary logits at matched positions"))
    llm = None
    try:
        torch.manual_seed(0)
        llm = LLM(**config)
        summaries = []
        for group in json.loads((source / "inputs.json").read_text()):
            name = group["name"]
            path = json.loads((source / f"{name}_path.json").read_text())
            targets = [[r["request"], r["position"]] for r in failures if r["group"] == name]
            with np.load(source / f"{name}_logits_bf16.npz") as data:
                archive = {k: data[k] for k in ("cold", "warm")}
            reset_cache(llm)
            probe = Replay(llm, out, group, path, archive, targets)
            for mode in ("cold", "warm", "warm_remap", "cold_tail"):
                result = probe.run(mode)
                summaries.append(result)
                write_json(out / "summary.json", summaries)
                print(result, flush=True)
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
