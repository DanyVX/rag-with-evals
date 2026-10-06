import pytest

from ragx.chunk.core import Chunk
from ragx.generate.pricing import TokenUsage
from ragx.index.base import SearchHit
from ragx.pipeline import answer_with_context


class FakeProvider:
    provider_id = "fake"

    def __init__(self) -> None:
        self.last_usage = TokenUsage(input_tokens=10, output_tokens=4)
        self.last_cost_usd = 0.002

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        assert "QUESTION:" in prompt
        return "Grounded answer [1]"


def test_pipeline_preserves_real_stage_timings_usage_and_cost() -> None:
    chunk = Chunk(id="c1", doc_hash="d", source="x", text="evidence", position=0)
    hit = SearchHit(chunk=chunk, score=0.9, rank=1)
    result = answer_with_context(
        "question?",
        [hit],
        provider=FakeProvider(),
        retrieval_ms=12.5,
        rerank_ms=3.25,
    )
    assert result.retrieval_ms == pytest.approx(12.5)
    assert result.rerank_ms == pytest.approx(3.25)
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4
    assert result.cost_usd == pytest.approx(0.002)
    assert result.citations.valid
    assert result.total_ms >= 15.75
