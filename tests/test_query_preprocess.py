import pytest

from ragx.generate.pricing import TokenUsage
from ragx.retrieve.query import preprocess_query


class FakeExpander:
    provider_id = "fake"

    def __init__(self) -> None:
        self.last_usage = TokenUsage(input_tokens=12, output_tokens=4)
        self.last_cost_usd = 0.001

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        assert "ABC-123" in prompt
        return "ABC-123 connection timeout network failure"


def test_none_normalizes_unicode_and_whitespace() -> None:
    result = preprocess_query("  Ｐython  ", mode="none")
    assert result.processed == "Python"


def test_lowercase_preserves_content_but_normalizes_case() -> None:
    result = preprocess_query("HTTP Error ABC-123", mode="lowercase")
    assert result.processed == "http error abc-123"


def test_empty_query_stays_empty() -> None:
    result = preprocess_query("   ", mode="none")
    assert result.processed == ""


def test_long_query_is_bounded() -> None:
    result = preprocess_query("x" * 100, mode="none", max_chars=20)
    assert result.processed == "x" * 20
    assert result.truncated


def test_query_expansion_uses_provider() -> None:
    result = preprocess_query("ABC-123 timeout", mode="query_expansion", expander=FakeExpander())
    assert result.processed == "ABC-123 connection timeout network failure"


def test_query_expansion_requires_provider() -> None:
    with pytest.raises(ValueError, match="requires an LLM provider"):
        preprocess_query("query", mode="query_expansion")
