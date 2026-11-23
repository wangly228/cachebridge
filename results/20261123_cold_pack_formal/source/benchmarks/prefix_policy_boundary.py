"""Nine-wave cold sequences comparing native GPU, overlap32 and cold FIFO packing."""
import argparse
from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from nanovllm import LLM
from nanovllm.engine.cpu_prefix_cache import TieredBlockManager
from nanovllm.engine.metrics import RequestMetrics
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from benchmarks.common import ROOT, prepare_run, sha256, write_json, write_jsonl
from benchmarks.first_use_budget import normalized_steps
from benchmarks.mixed_fill import make_case
from benchmarks.phase1 import save_wave, wave
from benchmarks.prefix_replay import COUNTERS, NoPayload
from benchmarks.revisit_boundary import ARCHIVE, LEVELS, trace_for
from benchmarks.write_admission import jsonl, select_arm

ARMS = ("A", "overlap32", "cold_pack")
FIELDS = ("warmup", "seed", "revisit_waves", "arm", "wave_index")


def metadata(run, arm, observed=None):
    """Plan prefill shape warmup or replay every real step with archived greedy IDs."""
    check = observed is not None
    output = 8 if run["warmup"] else 128
    config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048, eos=-1, kvcache_block_size=256,
                             num_kvcache_blocks=36, cpu_prefix_pack_cold=arm=="cold_pack")
    scheduler = Scheduler(config)
    if arm != "A":
        scheduler.block_manager = TieredBlockManager(36, 256, NoPayload(64), b"boundary", "reuse", True, 32, "overlap")
    shapes = set()
    for i, prompts in enumerate(run["inputs"]):
        expected = observed[i] if check else None
        sequences = []
        outputs = {}
        for fifo, prompt in enumerate(prompts):
            seq = Sequence(prompt, SamplingParams(temperature=0, max_tokens=output, ignore_eos=True))
            if check:
                r = expected[0][fifo]
                assert r["input_token_ids"] == prompt and len(r["output_token_ids"]) == output
                seq.seq_id = r["seq_id"]
                outputs[seq.seq_id] = r["output_token_ids"]
            seq.metrics = RequestMetrics(seq.seq_id, len(seq), 0)
            scheduler.add(seq)
            sequences.append(seq)
        step_count = 0
        while not scheduler.is_finished():
            seqs, prefill = scheduler.schedule()
            tokens = sum(s.num_scheduled_tokens for s in seqs)
            if prefill:
                shapes.add((tokens, len(seqs)))
            if check:
                step = expected[1][step_count]
                assert (step["is_prefill"], step["scheduled_tokens"], step["seq_ids"], step["used_blocks"]) == (
                    prefill,tokens,[s.seq_id for s in seqs],len(scheduler.block_manager.used_block_ids))
            generated = [outputs[s.seq_id][s.num_completion_tokens] if check else 17 for s in seqs]
            scheduler.postprocess(seqs, generated, prefill)
            step_count += 1
        assert getattr(scheduler.block_manager, "pending_write", None) is None
        if check:
            assert step_count == len(expected[1])
            if arm != "A":
                assert (len(scheduler.block_manager.entries), scheduler.block_manager.evictions) == (
                    expected[2]["cpu_resident_blocks"], expected[2]["cpu_evictions_since_reset"])
            for seq, r in zip(sequences, expected[0], strict=True):
                assert all(getattr(seq.metrics,k) == r[k] for k in COUNTERS), (r["seq_id"],arm)
    return shapes


def build_runs(seeds):
    # Rotate position by complete seed and revisit level; a fourth seed leaves
    # one extra placement per arm/level. All three run positions are exercised.
    orders = tuple(permutations(ARMS))
    runs = []
    for i, seed in enumerate(seeds):
        case = make_case(seed, 0, (0,2,3,5)[i % 4])
        for j in range(len(LEVELS)):
            level = LEVELS[(i+j) % len(LEVELS)]
            order = orders[((0,3,4,1)[i % 4]+2*LEVELS.index(level)) % 6]
            runs.append(dict(seed=seed, revisit_waves=level, warmup=False, arms=order, inputs=trace_for(case,level)))
    # Choose a small cover of the actual (query tokens, batch size) prefill shapes
    # by running only metadata. Warm sibling traces translate every token by a
    # fixed offset (preserves prefix equality); test coverage again from real logs.
    all_shapes = []
    for run in runs:
        all_shapes.append({(arm,*s) for arm in ARMS for s in metadata(run,arm)})
    uncovered = set.union(*all_shapes)
    warm = []
    while uncovered:
        best = max(range(len(runs)),key=lambda i:len(all_shapes[i]&uncovered))
        run = runs[best]
        prompts = [[[v+16000 for v in p] for p in batch] for batch in run["inputs"]]
        warm.append(dict(run,seed=run["seed"]+10000,warmup=True,inputs=prompts,arms=ARMS))
        uncovered -= all_shapes[best]
    return warm+runs


def measure(llm, out, run):
    for arm in run["arms"]:
        native = arm == "A"
        llm.config.cpu_prefix_pack_cold = arm == "cold_pack"
        llm.config.cpu_prefix_transfer_mode = "sync" if native else "overlap"
        select_arm(llm,"A" if native else "batch")
        manager = llm.scheduler.block_manager
        assert llm.scheduler.pack_cold == (arm == "cold_pack")
        assert not manager.used_block_ids and not manager.hash_to_block_id and len(manager.free_block_ids)==36
        if not native:
            assert manager.transfer_mode=="overlap" and manager.first_use_limit==32 and not manager.entries
        torch.manual_seed(run["seed"])
        tag=dict(seed=run["seed"],revisit_waves=run["revisit_waves"],warmup=run["warmup"],arm=arm)
        count=8 if run["warmup"] else 128
        batches=[]
        start=perf_counter()
        for i, prompts in enumerate(run["inputs"]):
            batches.append(wave(llm,prompts,count,dict(tag,wave_index=i)))
            assert getattr(manager,"pending_write",None) is None
        torch.cuda.synchronize()
        elapsed=perf_counter()-start
        for batch in batches:
            save_wave(out,batch)
        write_jsonl(out/"cycles.jsonl",[dict(tag,elapsed_s=elapsed)])
        print(f"{tag} {elapsed:.2f}s",flush=True)


def keyed(rows):
    result=defaultdict(list)
    for r in rows:
        result[tuple(r[k] for k in FIELDS)].append(r)
    return result


def report(out):
    design=read_json(out/"design.json")
    trace=read_json(out/"trace.json")
    assert design["trace_sha256"]==sha256(out/"trace.json")
    req, steps, waves=[keyed(jsonl(out/f"{name}.jsonl")) for name in ("requests","steps","waves")]
    cycles=jsonl(out/"cycles.jsonl")
    shapes=defaultdict(set)
    matched=defaultdict(int)
    numbers=defaultdict(list)
    ordered_keys=[]
    for run in trace:
        batches={}
        for arm in run["arms"]:
            header=(run["warmup"],run["seed"],run["revisit_waves"],arm)
            ordered_keys.append(header)
            seen=set()
            batches[arm]=[]
            for i, prompts in enumerate(run["inputs"]):
                key=(*header,i)
                wave_rows=req.pop(key)
                step_rows=steps.pop(key)
                summary,=waves.pop(key)
                assert len(wave_rows)==4 and not summary["profiler_enabled"]
                assert summary["cpu_cache_enabled"]==(arm!="A")
                for r,p in zip(wave_rows,prompts,strict=True):
                    prefix=tuple(p[:-128])
                    is_revisit = prefix in seen
                    seen.add(prefix)
                    assert r["initial_gpu_hit_tokens"]==0 and r["allocations"]==1
                    assert r["prefill_tokens"]==r["prompt_computed_tokens"]==len(p)-r["cpu_restored_tokens"]
                    assert r["initial_cpu_hit_tokens"]==r["cpu_restored_tokens"]==r["h2d_blocks"]*256
                    assert abs(r["ttft_ms"]-1000*(r["token_s"][0]-r["enqueue_s"]))<1e-5
                    assert r["preemptions"]==r["cpu_save_failures"]==r["cpu_restore_failures"]==0
                    if not run["warmup"]:
                        numbers[run["revisit_waves"],arm].append((r,is_revisit,summary))
                shapes[arm,run["warmup"]].update((s["scheduled_tokens"],len(s["seq_ids"])) for s in step_rows if s["is_prefill"])
                batches[arm].append((wave_rows,step_rows,summary))
            metadata(run,arm,batches[arm])
        for pair in (("A","overlap32"),("A","cold_pack"),("overlap32","cold_pack")):
            for a,b in zip(batches[pair[0]],batches[pair[1]],strict=True):
                if normalized_steps(a)==normalized_steps(b) and all(
                    x["prefill_tokens"]==y["prefill_tokens"] for x,y in zip(a[0],b[0],strict=True)):
                    assert [r["output_token_ids"] for r in a[0]]==[r["output_token_ids"] for r in b[0]]
                    if not run["warmup"]:
                        matched["/".join(pair)] += len(a[0])
    assert not req and not steps and not waves
    assert [tuple(r[k] for k in FIELDS[:-1]) for r in cycles]==ordered_keys
    assert all(shapes[a,False] <= shapes[a,True] for a in ARMS),shapes
    seeds=design["seeds"]
    draws=np.random.default_rng(2200).integers(len(seeds),size=(4000,len(seeds)))
    result=[]
    for level in LEVELS:
        detail={}
        simulated={}
        for arm in ARMS:
            rows=numbers[level,arm]
            group=np.array([[r["ttft_ms"] for r,_,_ in rows if r["seed"]==s] for s in seeds])
            assert group.shape==(len(seeds),36)
            boot=group[draws].reshape(len(draws),-1)
            simulated[arm]=np.percentile(boot,95,axis=1)
            revisits=[r for r,rev,_ in rows if rev]
            assert len(revisits)==len(seeds)*level*4
            wall=sum(r["elapsed_s"] for r in cycles if r["revisit_waves"]==level and r["arm"]==arm and not r["warmup"])
            detail[arm]=dict(n=len(rows),revisited=len(revisits),actual_revisit_pct=100*len(revisits)/len(rows),
                p95_ms=float(np.percentile(group,95)),by_seed_ms=dict(zip(seeds,np.percentile(group,95,axis=1).tolist())),
                output_tokens_per_s=len(rows)*128/wall,wall_s=wall,full_revisits=sum(r["cpu_restored_tokens"]==r["prompt_tokens"]-128 for r in revisits),
                restored_blocks=sum(r["h2d_blocks"] for r,_,_ in rows),written_blocks=sum(r["d2h_blocks"] for r,_,_ in rows),
                prompt_computed_tokens=sum(r["prompt_computed_tokens"] for r,_,_ in rows),
                first_wave_writes=sum(r["d2h_blocks"] for r,_,_ in rows if r["wave_index"]==0),
                max_cpu_evictions=max(w.get("cpu_evictions_since_reset",0) for _,_,w in rows),
                gpu_peak_bytes=max(w["memory"]["gpu_peak_allocated"] for _,_,w in rows),
                cpu_peak_rss=max(w["memory"]["cpu_peak_rss"] for _,_,w in rows))
        changes={}
        for base in ("A","overlap32"):
            changes[f"cold_pack_vs_{base}"]=dict(change_pct=100*(detail["cold_pack"]["p95_ms"]/detail[base]["p95_ms"]-1),
                paired_95pct=np.percentile(100*(simulated["cold_pack"]/simulated[base]-1),[2.5,97.5]).tolist())
        result.append(dict(revisit_waves=level,arms=detail,comparisons=changes))
    return dict(levels=result,same_shape_output_checks=dict(matched),all_metadata_calibrated=True,
            measured_shapes_preheated=True,raw_sha256={n:sha256(out/n) for n in ("requests.jsonl","steps.jsonl",
            "waves.jsonl","cycles.jsonl","trace.json","engine.json","environment.json","design.json")})


def read_json(p):
    return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    target=p.add_mutually_exclusive_group(required=True)
    target.add_argument("--output")
    target.add_argument("--summarize",type=Path)
    p.add_argument("--model",default=str(Path.home()/"models/huggingface/Qwen3-4B"))
    args=p.parse_args()
    if args.summarize:
        print(json.dumps(report(args.summarize),indent=2))
        return
    args.module="benchmarks.prefix_policy_boundary"
    out=prepare_run(args.output,args,args.model)
    import nanovllm
    assert Path(nanovllm.__file__).resolve().parent==ROOT/"nanovllm" and Path.cwd()==ROOT
    current=read_json(out/"environment.json")
    prior=read_json(ARCHIVE/"environment.json")
    for field in ("gpu","capability","cuda_runtime","packages","model_files"):
        assert current[field]==prior[field],field
    seeds=list(range(2201,2205))
    runs=build_runs(seeds)
    write_json(out/"trace.json",runs)
    write_json(out/"design.json",dict(seeds=seeds,trace_sha256=sha256(out/"trace.json"),arms=ARMS,
        preload="metadata set cover using disjoint input IDs; no timed GPU data used",
        bootstrap="4000 paired whole-seed draws, RNG 2200",
        cold="empty GPU and CPU metadata each 9-wave sequence; preheated engine; no inter-wave cache resets",
        endpoint="Enqueue to first generated token; wall covers all transfers and waits. Output 128; four queued per wave."))
    previous=read_json(ARCHIVE/"engine.json")
    config=dict(previous["config"],cpu_prefix_transfer_mode="overlap",cpu_prefix_first_use_blocks=32)
    torch.manual_seed(0)
    llm=LLM(args.model,**config)
    try:
        runner=llm.model_runner
        assert runner.kv_cache_budget["actual_blocks"]==36 and runner.kv_transfer.capacity==64
        assert not runner.enforce_eager and runner.graph_bs==previous["graph_batch_sizes"]
        assert runner.kv_cache.dtype==next(runner.model.parameters()).dtype==torch.bfloat16
        write_json(out/"engine.json",dict(config=config,actual_gpu_budget=runner.kv_cache_budget,cpu_slots=64,block_tokens=256,
                    graph_batch_sizes=runner.graph_bs,dtype=str(runner.kv_cache.dtype),engine_import=str(nanovllm.__file__),
                    modes=dict(A="native GPU",overlap32="overlap",cold_pack="overlap with partial cold peer")))
        runner.kv_transfer.slots.fill(0)
        for run in runs:
            measure(llm,out,run)
        write_json(out/"summary.json",report(out))
        write_json(out/"status.json",dict(complete=True))
    finally:
        llm.exit()


if __name__=="__main__":
    main()
