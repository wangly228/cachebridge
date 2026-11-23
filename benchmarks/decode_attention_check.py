"""Check the live paged decode output on archived outliers against the same Q/K/V."""
import argparse
from functools import partial
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch
from flash_attn import flash_attn_func

from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.phase1 import reset_cache
from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import get_context


def rel_l2(a, ref):
    return (torch.linalg.vector_norm(a.double() - ref) / torch.linalg.vector_norm(ref)).item()


def measure_layer(layer, targets, records, module, args, actual):
    ctx = get_context()
    for row, request, pos, length in targets:
        assert not ctx.is_prefill and ctx.context_lens[row].item() == length
        block_size = module.k_cache.size(1)
        ids = ctx.block_tables[row, :(length + block_size - 1) // block_size].long()
        def gather(cache):
            return cache.index_select(0, ids).flatten(0, 1)[:length]
        q, k, v = args[0][row].contiguous(), gather(module.k_cache), gather(module.v_cache)
        dense = flash_attn_func(q[None, None], k[None], v[None], softmax_scale=module.scale, causal=True)[0, 0]
        # Single last-token query sees all valid KV. Group heads as in Qwen3 GQA.
        q64 = q.double().reshape(module.num_kv_heads, -1, module.head_dim)
        scores = q64 @ k.double().permute(1, 2, 0) * module.scale
        ref = (scores.softmax(-1) @ v.double().transpose(0, 1)).reshape_as(q)
        paged = actual[row].reshape_as(q)
        assert torch.isfinite(ref).all() and torch.isfinite(paged).all() and torch.isfinite(dense).all()
        records.append(dict(request=request, position=pos, layer=layer, kv_tokens=length,
                            paged_vs_fp64_rel_l2=rel_l2(paged, ref),
                            dense_vs_fp64_rel_l2=rel_l2(dense, ref),
                            paged_vs_dense_rel_l2=rel_l2(paged, dense.double()),
                            paged_dense_equal=torch.equal(paged, dense)))


def check_group(llm, inputs, path, wanted, archive):
    reset_cache(llm)
    llm.generate(inputs, SamplingParams(temperature=0, ignore_eos=True, max_tokens=1), use_tqdm=False)
    llm.clear_metrics()
    ids = [llm.add_request(p, SamplingParams(temperature=0, ignore_eos=True, max_tokens=len(path[i])))
           for i, p in enumerate(inputs)]
    runner = llm.model_runner
    original_run, original_model = runner.run, runner.run_model
    targets, records, matched = [], [], set()
    def model(*args):
        logits = original_model(*args)
        for row, i, pos, _ in targets:
            assert torch.equal(logits[row].view(torch.int16).cpu(), torch.from_numpy(archive[i, pos]))
            matched.add((i, pos))
        return logits
    def run(seqs, prefill):
        targets.clear()
        for row, seq in enumerate(seqs):
            i, pos = ids.index(seq.seq_id), seq.num_completion_tokens
            assert seq.completion_token_ids == path[i][:pos]
            if not prefill and (i, pos) in wanted:
                targets.append((row, i, pos, len(seq)))
        output = original_run(seqs, prefill)
        for row, seq in enumerate(seqs):
            output[row] = path[ids.index(seq.seq_id)][seq.num_completion_tokens]
        return output
    layers = [m for m in runner.model.modules() if isinstance(m, Attention)]
    handles = [m.register_forward_hook(partial(measure_layer, n, targets, records)) for n, m in enumerate(layers)]
    try:
        with patch.object(runner, "run", run), patch.object(runner, "run_model", model):
            while not llm.is_finished():
                llm.step()
    finally:
        for handle in handles:
            handle.remove()
    assert matched == wanted and len(records) == len(wanted) * len(layers)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-run", default="results/20261118_fixed_path_probe")
    args = parser.parse_args()
    args.module = "benchmarks.decode_attention_check"
    source = Path(args.input_run)
    config = json.loads((source / "engine.json").read_text())["config"]
    out = prepare_run(args.output, args, config["model"])
    failures = [r for r in map(json.loads, (source / "comparisons.jsonl").read_text().splitlines()) if not r["passed"]]
    llm = None
    try:
        torch.manual_seed(0)
        llm = LLM(**config)
        records = []
        for group in json.loads((source / "inputs.json").read_text()):
            name = group["name"]
            wanted = {(r["request"], r["position"]) for r in failures if r["group"] == name}
            path = json.loads((source / f"{name}_path.json").read_text())
            with np.load(source / f"{name}_logits_bf16.npz") as archive:
                measured = check_group(llm, group["inputs"], path, wanted, archive["warm"])
            tagged = [dict(group=name, **r) for r in measured]
            write_jsonl(out / "layers.jsonl", tagged)
            records.extend(tagged)
        summary = {k: max(r[k] for r in records) for k in (
            "paged_vs_fp64_rel_l2", "dense_vs_fp64_rel_l2", "paged_vs_dense_rel_l2")}
        write_json(out / "summary.json", dict(maxima=summary, layers_measured=len(records),
                   archived_positions_reproduced=len(failures), equal_outputs=sum(r["paged_dense_equal"] for r in records),
                   input_comparisons_sha256=sha256(source / "comparisons.jsonl"), config=config,
                   actual_kv_blocks=llm.config.num_kvcache_blocks))
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
