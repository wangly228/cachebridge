import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_kvcache_blocks: int | None = None
    enable_metrics: bool = False
    cpu_prefix_cache_blocks: int = 0
    cpu_restore_cost_table: str | None = None
    cpu_prefix_admission: str = "always"
    cpu_prefix_batch_writes: bool = False
    # Optional upper bound on first-use writes under reuse admission, in KV blocks.
    cpu_prefix_first_use_blocks: int | None = None

    cpu_prefix_transfer_mode: str = "sync"
    cpu_prefix_pack_cold: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 0 < self.gpu_memory_utilization <= 1
        assert self.max_num_seqs > 0 and self.max_num_batched_tokens > 0
        assert self.max_model_len > 0
        assert self.max_kvcache_blocks is None or self.max_kvcache_blocks > 0
        if type(self.cpu_prefix_cache_blocks) is not int or self.cpu_prefix_cache_blocks < 0:
            raise ValueError("cpu_prefix_cache_blocks must be a nonnegative integer")
        if self.cpu_prefix_admission not in ("always", "reuse"):
            raise ValueError("cpu_prefix_admission must be 'always' or 'reuse'")
        if type(self.cpu_prefix_batch_writes) is not bool:
            raise ValueError("cpu_prefix_batch_writes must be a boolean")
        if type(self.cpu_prefix_pack_cold) is not bool:
            raise ValueError("cpu_prefix_pack_cold must be a boolean")
        if self.cpu_prefix_transfer_mode not in ("sync", "overlap"):
            raise ValueError("cpu_prefix_transfer_mode must be 'sync' or 'overlap'")
        if self.cpu_prefix_transfer_mode == "overlap" and not self.cpu_prefix_batch_writes:
            raise ValueError("overlap requires cpu_prefix_batch_writes")
        limit = self.cpu_prefix_first_use_blocks
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("cpu_prefix_first_use_blocks must be None or a nonnegative integer")
        if self.cpu_prefix_cache_blocks and self.tensor_parallel_size != 1:
            raise ValueError("CPU prefix caching requires tensor_parallel_size=1")
        if self.cpu_restore_cost_table is not None:
            if not self.cpu_prefix_cache_blocks or not os.path.isfile(self.cpu_restore_cost_table):
                raise ValueError("cpu_restore_cost_table requires CPU cache blocks and an existing table file")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
