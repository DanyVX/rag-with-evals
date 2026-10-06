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
    preprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.retrieval_ms + self.rerank_ms + self.generation_ms


def answer_with_context(
    question: str,
    hits: list[SearchHit],
    *,
    provider: LLMProvider,
    preprocess_ms: float = 0.0,
    retrieval_ms: float = 0.0,
    rerank_ms: float = 0.0,
    prior_prompt_tokens: int = 0,
    prior_completion_tokens: int = 0,
    prior_cost_usd: float = 0.0,
) -> AskResult:
    if not hits:
        answer = ABSTENTION
        citations = validate_citations(answer, 0)
        return AskResult(
            answer=answer,
            citations=citations,
            retrieved=[],
            preprocess_ms=preprocess_ms,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=0.0,
            prompt_tokens=prior_prompt_tokens,
            completion_tokens=prior_completion_tokens,
            cost_usd=prior_cost_usd,
            abstained=True,
        )

    prompt = build_prompt(question, hits)
    generated_at = perf_counter()
    answer = provider.generate(prompt, temperature=0.0, seed=0).strip()
    generation_ms = (perf_counter() - generated_at) * 1000
    citations = validate_citations(answer, len(hits))
    usage = getattr(provider, "last_usage", TokenUsage())
    generation_cost_usd = float(getattr(provider, "last_cost_usd", 0.0))
    return AskResult(
        answer=answer,
        citations=citations,
        retrieved=hits,
        preprocess_ms=preprocess_ms,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        generation_ms=generation_ms,
        prompt_tokens=prior_prompt_tokens + usage.input_tokens,
        completion_tokens=prior_completion_tokens + usage.output_tokens,
        cost_usd=prior_cost_usd + generation_cost_usd,
        abstained=ABSTENTION in answer.lower(),
    )
