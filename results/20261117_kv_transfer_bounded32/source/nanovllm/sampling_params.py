from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert isfinite(self.temperature) and (self.temperature == 0 or self.temperature > 1e-10)
        assert self.max_tokens > 0
