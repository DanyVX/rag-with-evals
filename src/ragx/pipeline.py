from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ragx.generate.citations import CitationValidation, validate_citations
from ragx.generate.prompt import ABSTENTION, build_prompt
from ragx.generate.providers import LLMProvider
from ragx.index.base import SearchHit


@dataclass(slots=True)
class AskResult:
    answer: str
    citations: CitationValidation
    retrieved: list[SearchHit]
    retrieval_ms: float
    generation_ms: float
    abstained: bool


def answer_with_context(
    question: str,
    hits: list[SearchHit],
    *,
    provider: LLMProvider,
) -> AskResult:
    retrieval_ms = 0.0
    prompt = build_prompt(question, hits)
    generated_at = perf_counter()
    answer = provider.generate(prompt, temperature=0.0, seed=0).strip()
    generation_ms = (perf_counter() - generated_at) * 1000
    citations = validate_citations(answer, len(hits))
    return AskResult(
        answer=answer,
        citations=citations,
        retrieved=hits,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        abstained=ABSTENTION in answer.lower(),
    )
