from dataclasses import dataclass

@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        if not (0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
