from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ragx.generate.citations import CitationValidation, validate_citations
from ragx.generate.pricing import TokenUsage
from ragx.generate.prompt import ABSTENTION, build_prompt
from ragx.generate.providers import LLMProvider
from ragx.index.base import SearchHit


@dataclass(slots=True)
class AskResult:
    answer: str
    citations: CitationValidation
    retrieved: list[SearchHit]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    abstained: bool

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.rerank_ms + self.generation_ms


def answer_with_context(
    question: str,
    hits: list[SearchHit],
    *,
    provider: LLMProvider,
    retrieval_ms: float = 0.0,
    rerank_ms: float = 0.0,
) -> AskResult:
    prompt = build_prompt(question, hits)
    generated_at = perf_counter()
    answer = provider.generate(prompt, temperature=0.0, seed=0).strip()
    generation_ms = (perf_counter() - generated_at) * 1000
    citations = validate_citations(answer, len(hits))
    usage = getattr(provider, "last_usage", TokenUsage())
    cost_usd = float(getattr(provider, "last_cost_usd", 0.0))
    return AskResult(
        answer=answer,
        citations=citations,
        retrieved=hits,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        generation_ms=generation_ms,
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        cost_usd=cost_usd,
        abstained=ABSTENTION in answer.lower(),
    )
