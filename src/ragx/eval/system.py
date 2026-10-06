from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SystemMetrics:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.rerank_ms + self.generation_ms

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["total_ms"] = self.total_ms
        return data
