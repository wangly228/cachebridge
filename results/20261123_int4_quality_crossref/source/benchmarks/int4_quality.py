"""Paired BF16 hot-page and INT4 CPU-restore diagnostics on long ledger lookups."""
import argparse
import json
from pathlib import Path
from random import Random
import re

import numpy as np
import torch

from benchmarks.common import prepare_run, write_json, write_jsonl
from benchmarks.phase1 import reset_cache, wave
from nanovllm import LLM
from nanovllm.engine.sequence import Sequence


def ledger(tokenizer, seed, length, depth, task):
    rng = Random(f"cachebridge-ledger/{seed}/{length}/{depth}")
    ids = rng.sample(range(100000, 999999), length // 8 + 5)
    target = ids.pop()
    secret = str(rng.randint(100000, 999999))
    ticket = None
    if task == "crossref":
        ticket_ids = rng.sample(range(100000, 999999), len(ids)+1)
        ticket = ticket_ids.pop()
        lines = []
        for rid, tid in zip(ids, ticket_ids, strict=True):
            lines.append(f"Ticket T-{tid} is assigned to record R-{rid}.\n")
            lines.append(f"Record R-{rid} | authorization code {rng.randint(100000,999999)}.\n")
        rng.shuffle(lines)
        subject = f"the record assigned to ticket T-{ticket}"
    else:
        subject = f"record R-{target}"
    user = ("The following ledger pairs record IDs with authorization codes.\n[[LEDGER]]\n"
            f"End of ledger. What is the authorization code for {subject}? "
            "Answer with only its six digits.")
    chat = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    head, query = chat.split("[[LEDGER]]")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    prefix = encode(head)
    inserted = pointer = None
    pointer_depth = 85 if depth <= 50 else 15
    while len(prefix) < length:
        if inserted is None and len(prefix) >= length * depth / 100:
            inserted = len(prefix)
            line = f"Record R-{target} | authorization code {secret}.\n"
        elif ticket is not None and pointer is None and len(prefix) >= length * pointer_depth / 100:
            pointer = len(prefix)
            line = f"Ticket T-{ticket} is assigned to record R-{target}.\n"
        elif ticket is not None:
            line = lines.pop()
        else:
            line = f"Record R-{ids.pop()} | authorization code {rng.randint(100000,999999)}.\n"
        prefix.extend(encode(line))
    assert inserted is not None and inserted + 32 < length
    assert ticket is None or (pointer is not None and pointer + 32 < length)
    prompt = prefix[:length] + encode(query)
    teacher_lead = encode("The authorization code is ")
    assert length < len(prompt) < length + 240
    return dict(seed=seed, length=length, depth=depth, task=task, ticket=ticket, pointer_token=pointer,
                needle_token=inserted, record=target,
                expected=secret, answer_ids=encode(secret), input_ids=prompt, teacher_lead=teacher_lead)


def page_ids(llm, prompt):
    manager = llm.scheduler.block_manager
    hits = manager._lookup(Sequence(prompt))
    assert all(h.gpu_id is not None for h in hits)
    return [h.gpu_id for h in hits]


def clear_gpu_index(llm):
    # Isolate a full CPU replay at exactly the same query shape as the GPU-hot control.
    manager = llm.scheduler.block_manager
    assert llm.is_finished() and not manager.used_block_ids and manager.pending_write is None
    manager.hash_to_block_id.clear()


def run_answer(llm, case, *, restored, force=None, lead=False, capture=False):
    prompt = case["input_ids"] + (case["teacher_lead"] if lead else [])
    rows = []
    def record(module, args, result):
        if len(rows) < 32:
            rows.append(args[0].detach()[0].cpu())
        if force is not None:
            return torch.full_like(result, force[len(rows) - 1])

    if restored:
        clear_gpu_index(llm)
    handle = llm.model_runner.sampler.register_forward_hook(record) if capture or force is not None else None
    try:
        batch = wave(llm, [prompt], len(force) if force is not None else 64, {})
    finally:
        if handle is not None:
            handle.remove()
    request = batch[0][0]
    length = case["length"]
    assert request["cpu_restored_tokens"] == request["h2d_blocks"] * 256 == (length if restored else 0)
    assert request["initial_gpu_hit_tokens"] == (0 if restored else length)
    assert request["prompt_computed_tokens"] == len(prompt) - length and request["allocations"] == 1
    assert request["preemptions"] == request["cpu_save_failures"] == request["cpu_restore_failures"] == 0
    if force is not None:
        assert len(force) <= 32 and request["output_token_ids"] == force
    output = llm.tokenizer.decode(request["output_token_ids"], skip_special_tokens=False)
    # Score the first answer only. Keep all tokens (including forced decode beyond EOS) for auditing.
    answer = output.split(llm.tokenizer.eos_token, 1)[0].strip()
    without_id = re.sub(rf"R[- ]{case['record']}\b", "", answer, flags=re.I)
    if case["ticket"] is not None:
        without_id = re.sub(rf"T[- ]{case['ticket']}\b", "", without_id, flags=re.I)
    candidates = sorted(set(re.findall(r"(?<!\d)\d{6}(?!\d)", without_id)))
    return dict(ids=request["output_token_ids"], text=output, exact=candidates == [case["expected"]],
                answer=answer, candidates=candidates, exact_digits_only=answer == case["expected"],
                restored_tokens=request["cpu_restored_tokens"], gpu_hit_tokens=request["initial_gpu_hit_tokens"],
                query_tokens=request["prompt_computed_tokens"]), rows


def logit_errors(reference, actual, target):
    a, b = reference.double(), actual.double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    p, q = a.log_softmax(0), b.log_softmax(0)
    return dict(target=target, bf16_top1=a.argmax().item(), int4_top1=b.argmax().item(),
                bf16_top2_margin=float(a.topk(2).values.diff().abs().item()),
                max_abs=(a-b).abs().max().item(), relative_l2=float((a-b).norm() / a.norm()),
                kl_bf16_int4=float((p.exp() * (p-q)).sum()),
                bf16_nll=-p[target].item(), int4_nll=-q[target].item())


def inspect_case(llm, out, case):
    reset_cache(llm)
    length = case["length"]
    prefix = case["input_ids"][:length]
    # A distinct incomplete tail leaves all answer positions unseen at fill time.
    fill = wave(llm, [prefix + [198]], 1, {})[0][0]
    assert fill["d2h_blocks"] == length // 256
    bef = page_ids(llm, case["input_ids"])
    assert len(bef) == length // 256
    indexes = [0, len(bef) // 2, len(bef) - 1]
    cache = llm.model_runner.kv_cache
    original = [cache[:, :, bef[i]].clone() for i in indexes]
    clean, natural_ref = run_answer(llm, case, restored=False, capture=True)
    _, ref = run_answer(llm, case, restored=False, force=case["answer_ids"], lead=True)
    lossy, _ = run_answer(llm, case, restored=True)
    _, quant = run_answer(llm, case, restored=True, force=case["answer_ids"], lead=True)
    natural_ids = clean["ids"][:32]
    eos = llm.tokenizer.eos_token_id
    if eos in natural_ids:
        natural_ids = natural_ids[:natural_ids.index(eos)+1]
    natural_ref = natural_ref[:len(natural_ids)]
    _, natural_quant = run_answer(llm, case, restored=True, force=natural_ids)
    aft = page_ids(llm, case["input_ids"])
    name = f"{case['seed']}-{length}-{case['depth']}-{case['task']}"
    sample = dict(bf16_logits=torch.stack(ref).view(torch.int16).numpy(),
                  int4_logits=torch.stack(quant).view(torch.int16).numpy(),
                  natural_bf16_logits=torch.stack(natural_ref).view(torch.int16).numpy(),
                  natural_int4_logits=torch.stack(natural_quant).view(torch.int16).numpy())
    kv_errors = []
    for i, before in zip(indexes, original):
        a = before.float().flatten(2)
        b = cache[:, :, aft[i]].float().flatten(2)
        delta = a - b
        assert torch.isfinite(delta).all()
        rel = delta.norm(dim=-1) / a.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        for metric, values in (("relative_l2", rel), ("cosine", cos), ("max_abs", delta.abs().amax(-1)),
                               ("rmse", delta.square().mean(-1).sqrt())):
            kv_errors.append(dict(block=i, metric=metric, K=values[0].tolist(), V=values[1].tolist()))
        sample[f"bf16_page{i}_head0"] = before[:, :, 0, 0].cpu().view(torch.int16).numpy()
        sample[f"int4_page{i}_head0"] = cache[:, :, aft[i], 0, 0].cpu().view(torch.int16).numpy()
    np.savez_compressed(out / f"{name}.npz", **sample)
    pos = [dict(position=len(case["input_ids"])+len(case["teacher_lead"])-1+j, **logit_errors(a, b, token))
           for j, (a, b, token) in enumerate(zip(ref, quant, case["answer_ids"], strict=True))]
    natural = [dict(position=len(case["input_ids"])-1+j, **logit_errors(a,b,token))
               for j,(a,b,token) in enumerate(zip(natural_ref,natural_quant,natural_ids,strict=True))]
    assert all(row["bf16_top1"] == token for row,token in zip(natural,natural_ids,strict=True))
    write_jsonl(out / "cases.jsonl", [dict(case, bf16=clean, int4=lossy, teacher=pos, kv=kv_errors,
                                          natural=natural, tensors=f"{name}.npz")])
    print(f"{name} expected={case['expected']} BF16={clean['text']!r} INT4={lossy['text']!r}", flush=True)


def summarize(out):
    cases = [json.loads(s) for s in (out / "cases.jsonl").read_text().splitlines()]
    def stats(group):
        pos = [r for c in group for r in c["teacher"]]
        natural = [r for c in group for r in c["natural"]]
        rel = [r for c in group for r in c["kv"] if r["metric"] == "relative_l2"]
        return dict(n=len(group), positions=len(pos), bf16_correct=sum(c["bf16"]["exact"] for c in group),
                    int4_correct=sum(c["int4"]["exact"] for c in group),
                    lost=sum(c["bf16"]["exact"] and not c["int4"]["exact"] for c in group),
                    gained=sum(c["int4"]["exact"] and not c["bf16"]["exact"] for c in group),
                    equal_sequences=sum(c["bf16"]["ids"] == c["int4"]["ids"] for c in group),
                    equal_answer_text=sum(c["bf16"]["answer"] == c["int4"]["answer"] for c in group),
                    digits_only={a:sum(c[a]["exact_digits_only"] for c in group) for a in ("bf16","int4")},
                    natural_positions=len(natural),
                    natural_top1_agree=sum(r["bf16_top1"]==r["int4_top1"] for r in natural),
                    natural_mean_kl=float(np.mean([r["kl_bf16_int4"] for r in natural])),
                    natural_mean_delta_nll=float(np.mean([r["int4_nll"]-r["bf16_nll"] for r in natural])),
                    top1_agree=sum(r["bf16_top1"] == r["int4_top1"] for r in pos),
                    mean_KL=float(np.mean([r["kl_bf16_int4"] for r in pos])),
                    max_KL=max(r["kl_bf16_int4"] for r in pos),
                    mean_delta_nll=float(np.mean([r["int4_nll"]-r["bf16_nll"] for r in pos])),
                    max_logit_abs=max(r["max_abs"] for r in pos),
                    max_logit_relative_l2=max(r["relative_l2"] for r in pos),
                    max_layer_page_rel_l2={k:max(max(r[k]) for r in rel) for k in ("K","V")})
    summary = dict(overall=stats(cases), by_length={l:stats([c for c in cases if c["length"] == l])
                   for l in sorted({c["length"] for c in cases})},
                   by_task={t:stats([c for c in cases if c["task"]==t]) for t in ("direct","crossref")},
                   failures=[{k:c[k] for k in ("seed","length","depth","task","expected","bf16","int4")}
                             for c in cases if not (c["bf16"]["exact"] and c["int4"]["exact"])])
    write_json(out / "summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    p.add_argument("--stage", choices=("screen", "formal"), default="screen")
    args = p.parse_args()
    args.module = "benchmarks.int4_quality"
    out = prepare_run(args.output, args, args.model)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg = dict(max_kvcache_blocks=52, cpu_prefix_cache_blocks=64, max_num_batched_tokens=2048,
               max_num_seqs=8, max_model_len=13312, enable_metrics=True, gpu_memory_utilization=0.8,
               cpu_prefix_cache_format="int4", cpu_prefix_batch_writes=True, cpu_prefix_transfer_mode="overlap",
               cpu_prefix_pack_cold=True, cpu_prefix_admission="reuse", cpu_prefix_first_use_blocks=48)
    llm = LLM(args.model, **cfg)
    try:
        assert llm.config.num_kvcache_blocks == 52
        assert llm.model_runner.kv_cache.dtype == next(llm.model_runner.model.parameters()).dtype == torch.bfloat16
        screen = args.stage == "screen"
        seeds = [3100] if screen else [3201, 3202]
        lengths = [2048, 12288] if screen else [2048, 8192, 12288]
        depths = [10, 50] if screen else [10, 50, 90]
        cases = [ledger(llm.tokenizer, seed, length, depth, task) for seed in seeds for length in lengths
                 for depth in depths for task in ("direct","crossref")]
        write_json(out / "design.json", dict(config=cfg, seeds=seeds, lengths=lengths, depths=depths,
                   tokenizer="Qwen3 chat template, thinking disabled",
                   control="original BF16 GPU pages, single request with identical position and query tokens",
                   replay="invalidate the idle GPU hash index, restore the full contiguous prefix through LLM",
                   teacher="append identical assistant lead, force gold digits and record full-vocabulary logits",
                   natural="force the BF16 greedy answer including EOS, capped at 32 positions, without adding a lead",
                   layer_sample="first/middle/last complete prefix page, all 36 layers, both K and V",
                   scoring="free greedy 64 tokens; before EOS strip the queried record ID, require exactly one distinct six-digit code equal to gold; retain strict digits-only score"))
        for case in cases:
            inspect_case(llm, out, case)
        summarize(out)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
