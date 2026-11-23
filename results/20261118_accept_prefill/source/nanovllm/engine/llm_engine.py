import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.metrics import RequestMetrics


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.request_metrics = {}
        self.step_metrics = []
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("prompt plus output exceeds max_model_len")
        required_blocks = (len(prompt) + sampling_params.max_tokens - 2) // Sequence.block_size + 1
        if required_blocks > self.config.num_kvcache_blocks:
            raise ValueError("request cannot fit in the GPU KV pool even without other requests")
        enqueue_time = perf_counter() if self.config.enable_metrics else None
        seq = Sequence(prompt, sampling_params)
        if self.config.enable_metrics:
            seq.metrics = RequestMetrics(seq.seq_id, len(prompt), enqueue_time)
            self.request_metrics[seq.seq_id] = seq.metrics
        self.scheduler.add(seq)
        return seq.seq_id

    def clear_metrics(self):
        if not self.is_finished():
            raise RuntimeError("cannot clear metrics with unfinished requests")
        self.request_metrics.clear()
        self.step_metrics.clear()

    def step(self):
        start = perf_counter() if self.config.enable_metrics else None
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        if self.config.enable_metrics:
            run_start = perf_counter()
            used_blocks = len(self.scheduler.block_manager.used_block_ids)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if self.config.enable_metrics:
            run_end = perf_counter()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.config.enable_metrics:
            self.step_metrics.append(dict(
                is_prefill=is_prefill, scheduled_tokens=abs(num_tokens),
                seq_ids=[seq.seq_id for seq in seqs], used_blocks=used_blocks,
                start_time=start, run_ms=(run_end - run_start) * 1000,
                step_ms=(perf_counter() - start) * 1000,
            ))
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
