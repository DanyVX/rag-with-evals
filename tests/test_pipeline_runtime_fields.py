import pytest

from ragx.chunk.core import Chunk
from ragx.generate.pricing import TokenUsage
from ragx.generate.prompt import ABSTENTION
from ragx.index.base import SearchHit
from ragx.pipeline import answer_with_context


class CountingProvider:
    provider_id = "counting"

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = TokenUsage(input_tokens=12, output_tokens=4)
        self.last_cost_usd = 0.002

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        del prompt, temperature, seed
        self.calls += 1
        return "Supported answer [1]"


def test_empty_evidence_abstains_without_provider_call() -> None:
    provider = CountingProvider()
    result = answer_with_context(
        "missing?",
        [],
        provider=provider,
        preprocess_ms=1.0,
        retrieval_ms=2.0,
        rerank_ms=3.0,
    )
    assert result.answer == ABSTENTION
    assert result.abstained is True
    assert provider.calls == 0
    assert result.generation_ms == 0.0
    assert result.total_ms == pytest.approx(6.0)


def test_pipeline_reports_provider_usage_cost_and_stage_timings() -> None:
    provider = CountingProvider()
    hit = SearchHit(
        chunk=Chunk(
            id="c1",
            doc_hash="d1",
            source="fixture.txt",
            text="Evidence",
            position=0,
        ),
        score=0.9,
        rank=1,
    )
    result = answer_with_context(
        "question?",
        [hit],
        provider=provider,
        preprocess_ms=1.5,
        retrieval_ms=2.5,
        rerank_ms=0.5,
        prior_prompt_tokens=3,
        prior_completion_tokens=2,
        prior_cost_usd=0.001,
    )
    assert provider.calls == 1
    assert result.citations.valid
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 6
    assert result.cost_usd == pytest.approx(0.003)
    assert result.generation_ms >= 0.0
    assert result.total_ms >= 4.5
