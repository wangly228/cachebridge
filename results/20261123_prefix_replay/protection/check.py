"""From the cachebridge root, compare a token-only transcript and exercise deferred D2H and shared refs on CPU. Pass a new output directory."""
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import Future
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from benchmarks import prefix_replay as replay

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
source = Path('results/20261123_cold_pack_screen')
input_dir = root/'configuration_only'
input_dir.mkdir(exist_ok=True)
shutil.copyfile(source/'engine.json', input_dir/'engine.json')
assert list(input_dir.iterdir()) == [input_dir/'engine.json']
source_rows = [r for r in replay.read_lines(source/'requests.jsonl') if
               (r['warmup'], r['arm'], r['workload'], r['seed'], r['cycle']) == (False, 'candidate', 'mixed', 2099, 1)]
by_phase = defaultdict(list)
for row in source_rows:
    by_phase[row['phase']].append({k:row[k] for k in ('input_token_ids','output_token_ids')})
transcript = root/'held_out_waves.jsonl'
transcript.write_text(''.join(json.dumps(dict(cycle=1, wave=w, requests=rows))+'\n' for w,rows in by_phase.items()))
flags = ['--input',str(input_dir),'--arm','candidate','--transcript',str(transcript)]
subprocess.run([sys.executable, '-m', 'benchmarks.prefix_replay', *flags, '--output', str(root/'cli')], check=True)
counts = Counter()
class DelayedCopy(Future):
    def __init__(self, transfer, pairs):
        super().__init__()
        self.transfer = transfer
        self.pairs = pairs
        self.expected = {gpu_id: tuple(transfer.manager.blocks[gpu_id].token_ids) for gpu_id, _ in pairs}

    def protection(self):
        manager = self.transfer.manager
        for gpu_id, slot in self.pairs:
            assert gpu_id in manager.used_block_ids and gpu_id not in manager.free_block_ids
            assert manager.blocks[gpu_id].ref_count > 0
            assert tuple(manager.blocks[gpu_id].token_ids) == self.expected[gpu_id]
            assert slot not in manager.free_slots and slot not in {s for s, _ in manager.entries.values()}
            counts['reserved_block_checks'] += 1

    def result(self, timeout=None):
        if not self.done():
            self.protection()
            for gpu_id,slot in self.pairs:
                self.transfer.saved[slot] = self.expected[gpu_id]
            self.set_result(perf_counter())
            counts['delayed_batches_completed'] += 1
        return super().result(timeout)

class GuardedTransfer(replay.NoPayload):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.saved = {}

    def save(self, gpu_id, slot):
        self.saved[slot] = tuple(self.manager.blocks[gpu_id].token_ids)
        counts['sync_saved_blocks'] += 1

    def submit_save_many(self, pairs):
        counts['reserved_blocks'] += len(pairs)
        assert self.manager.pending_write is None
        return DelayedCopy(self, tuple(pairs))

    def restore(self, slot, gpu_id):
        manager = self.manager
        assert gpu_id in manager.used_block_ids and gpu_id not in manager.free_block_ids
        assert manager.blocks[gpu_id].ref_count == 1
        assert slot not in manager.free_slots
        assert manager.pending_write is None
        owners = [tuple(ids) for resident_slot,ids in manager.entries.values() if resident_slot == slot]
        assert owners == [self.saved[slot]]
        counts['restores_to_exclusively_claimed_targets'] += 1

start = replay.ObservedCache.start_cycle
def attach(self, tag, policy):
    self.transfer.manager = self
    return start(self,tag,policy)

post = replay.Scheduler.postprocess
def postprocess(self,*args):
    output = post(self,*args)
    manager = self.block_manager
    if manager.pending_write is not None:
        counts['step_observations_with_pending_write'] += 1
        manager.pending_write[0].protection()
    active = [s for s in (*self.waiting,*self.running) if s.block_table]
    refs = Counter(b for seq in active for b in seq.block_table)
    assert set(refs) == set(manager.used_block_ids)
    assert set(manager.free_block_ids) == set(range(len(manager.blocks))) - set(refs)
    assert len(manager.free_block_ids) == len(set(manager.free_block_ids))
    assert all(refs[b.block_id] == b.ref_count for b in manager.blocks)
    counts['steps_with_refcounts_verified'] += 1
    counts['steps_with_shared_block'] += int(any(v > 1 for v in refs.values()))
    return output

with patch.object(replay,'NoPayload',GuardedTransfer),patch.object(replay.ObservedCache,'start_cycle',attach),patch.object(replay.Scheduler,'postprocess',postprocess):
    replay.run(input_dir,root/'guarded','baseline',transcript,'candidate')

fields = replay.COUNTERS
for name in ('cli','guarded'):
    rows = replay.read_lines(root/name/'requests.jsonl')
    assert len(rows) == len(source_rows)
    assert all(actual['wave'] == expected['phase'] and all(actual[k] == expected[k] for k in fields)
               for actual,expected in zip(rows,source_rows,strict=True))
    assert json.loads((root/name/'summary.json').read_text())['pack_cold'] is True
real_trace_counts = dict(counts)
counts.clear()
shared_ids = source_rows[0]["input_token_ids"][:1024]
shared_wave = root/'shared_prompt.jsonl'
requests = [dict(input_token_ids=shared_ids+[9]*128, output_token_ids=[5]*8),
            dict(input_token_ids=shared_ids+[8]*128, output_token_ids=[7]*8)]
shared_wave.write_text(json.dumps(dict(cycle=0,wave=0,requests=requests))+"\n")
with patch.object(replay,'NoPayload',GuardedTransfer),patch.object(replay.ObservedCache,'start_cycle',attach),patch.object(replay.Scheduler,'postprocess',postprocess):
    replay.run(input_dir,root/'guarded_shared','baseline',shared_wave,'candidate')
shared_rows = replay.read_lines(root/'guarded_shared/requests.jsonl')
assert counts['steps_with_shared_block'] > 0
assert counts['reserved_blocks'] == 4
assert shared_rows[1]['initial_gpu_hit_tokens'] == 1024
s = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
evidence = dict(source=str(source),selector=dict(warmup=False,arm='candidate',workload='mixed',seed=2099,cycle=1),
                waves=list(by_phase), requests=len(source_rows), counters_per_request=len(fields),
                cli_field_equalities=len(source_rows)*len(fields),guarded_field_equalities=len(source_rows)*len(fields),
                source_requests_sha256=s(source/'requests.jsonl'),config_sha256=s(input_dir/'engine.json'),
                transcript_sha256=s(transcript), cli_summary_sha256=s(root/'cli/summary.json'),
                guarded_summary_sha256=s(root/'guarded/summary.json'),guarded_counts=real_trace_counts, synthetic_simultaneous_shared_counts=dict(counts), synthetic_peer_gpu_hit_tokens=shared_rows[1]['initial_gpu_hit_tokens'],
                metadata_limits='CPU-only checks real allocation, publication and release order; guarded transfer compares token labels, not KV bits or device timings.')
(root/'audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence))
