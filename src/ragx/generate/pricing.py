from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_input_token_upper_bound(prompt: str) -> int:
    """Conservative upper bound for byte-level/BPE tokenizers plus chat framing."""
    return len(prompt.encode("utf-8")) + 256


def calculate_cost_usd(
    usage: TokenUsage,
    *,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    return (
        usage.input_tokens * input_usd_per_million
        + usage.output_tokens * output_usd_per_million
    ) / 1_000_000.0
