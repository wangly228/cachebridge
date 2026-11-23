"""Extend the output window for every case that missed the frozen 64-token quality gate."""
import argparse
import json
from pathlib import Path
import re

import torch

from benchmarks.common import prepare_run, sha256, write_json, write_jsonl
from benchmarks.int4_quality import clear_gpu_index
from benchmarks.phase1 import reset_cache
from nanovllm import LLM, SamplingParams


def readout(tokenizer, ids, case):
    text = tokenizer.decode(ids, skip_special_tokens=False).split(tokenizer.eos_token, 1)[0].strip()
    body = re.sub(rf"R[- ]{case['record']}\b", "", text, flags=re.I)
    if case["ticket"] is not None:
        body = re.sub(rf"T[- ]{case['ticket']}\b", "", body, flags=re.I)
    codes = sorted(set(re.findall(r"(?<!\d)\d{6}(?!\d)", body)))
    return dict(text=text, candidates=codes, exact=codes == [case["expected"]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(Path.home() / "models/huggingface/Qwen3-4B"))
    args = p.parse_args()
    args.module = "benchmarks.int4_answer_length"
    source = Path(args.input)
    cases = [json.loads(l) for l in (source / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 36 and (source/"summary.json").is_file()
    selected = [c for c in cases if not (c["bf16"]["exact"] and c["int4"]["exact"])]
    out = prepare_run(args.output, args, args.model)
    write_json(out / "design.json", dict(case_sha256=sha256(source/"cases.jsonl"), n=len(selected),
               selection="every case with either 64-token semantic miss", max_tokens=192, ignore_eos=False,
               check="both greedy paths must reproduce all original tokens through EOS or position 64",
               scope="post-hoc diagnosis of reading the code before the output limit"))
    cfg = json.loads((source/"design.json").read_text())["config"]
    torch.manual_seed(0)
    torch.set_num_threads(2)
    llm = LLM(args.model, **cfg)
    records = []
    try:
        params = SamplingParams(temperature=0, max_tokens=192, ignore_eos=False)
        for c in selected:
            reset_cache(llm)
            prefix = c["input_ids"][:c["length"]]
            llm.generate([prefix + [198]], SamplingParams(temperature=0,max_tokens=1), use_tqdm=False)
            record = {k:c[k] for k in ("seed","task","length","depth","expected")}
            eos = llm.tokenizer.eos_token_id
            for arm in ("bf16","int4"):
                if arm == "int4":
                    clear_gpu_index(llm)
                llm.clear_metrics()
                ids = llm.generate([c["input_ids"]], params, use_tqdm=False)[0]["token_ids"]
                m, = llm.request_metrics.values()
                assert m.cpu_restored_tokens == (c["length"] if arm == "int4" else 0)
                assert m.initial_gpu_hit_tokens + m.cpu_restored_tokens == c["length"]
                previous = c[arm]["ids"]
                if eos in previous:
                    previous = previous[:previous.index(eos)+1]
                assert ids[:len(previous)] == previous, (c["seed"], c["length"], c["depth"], arm)
                short = readout(llm.tokenizer, ids[:64], c)
                assert short["exact"] == c[arm]["exact"] and short["candidates"] == c[arm]["candidates"]
                record[arm] = dict(**readout(llm.tokenizer, ids, c), short=short["exact"],
                                   ids=ids, reached_eos=ids[-1]==eos, restored=m.cpu_restored_tokens)
            records.append(record)
            write_jsonl(out / "cases.jsonl", [record])
        def stats(arm):
            failed = [r[arm] for r in records if not r[arm]["short"]]
            return dict(short_misses=len(failed), recovered=sum(r["exact"] for r in failed),
                        extended_misses=sum(not r["exact"] for r in failed),
                        capped_without_eos=sum(not r["reached_eos"] for r in failed))
        write_json(out / "summary.json", dict(paired_cases=len(records), bf16=stats("bf16"), int4=stats("int4"),
                   extended_bf16_correct_int4_wrong=sum(r["bf16"]["exact"] and not r["int4"]["exact"] for r in records),
                   extended_bf16_wrong_int4_correct=sum(r["int4"]["exact"] and not r["bf16"]["exact"] for r in records)))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
