import pytest

from ragx.cost import BudgetExceededError, CostTracker
from ragx.generate.providers import OpenAICompatibleProvider


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


def test_openai_compatible_commits_actual_reported_cost(monkeypatch) -> None:
    monkeypatch.setattr("ragx.generate.providers.httpx.post", lambda *args, **kwargs: FakeResponse())
    tracker = CostTracker(max_spend_usd=1.0)
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="example",
        cost_tracker=tracker,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        max_output_tokens=32,
    )
    assert provider.generate("hello") == "answer"
    assert provider.last_usage.input_tokens == 10
    assert provider.last_usage.output_tokens == 4
    assert provider.last_cost_usd == pytest.approx(18 / 1_000_000)
    assert tracker.spent_usd == pytest.approx(provider.last_cost_usd)
    assert tracker.reserved_usd == pytest.approx(0.0)


def test_paid_provider_refuses_call_before_http_when_budget_is_too_small(monkeypatch) -> None:
    called = {"value": False}

    def fake_post(*args, **kwargs):
        called["value"] = True
        return FakeResponse()

    monkeypatch.setattr("ragx.generate.providers.httpx.post", fake_post)
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="example",
        cost_tracker=CostTracker(max_spend_usd=0.000001),
        input_usd_per_million=100.0,
        output_usd_per_million=100.0,
        max_output_tokens=32,
    )
    with pytest.raises(BudgetExceededError):
        provider.generate("hello")
    assert called["value"] is False


def test_zero_cost_local_provider_can_run_without_paid_budget(monkeypatch) -> None:
    monkeypatch.setattr("ragx.generate.providers.httpx.post", lambda *args, **kwargs: FakeResponse())
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="",
        model="local",
        cost_tracker=CostTracker(max_spend_usd=0.0),
        zero_cost_local=True,
    )
    assert provider.generate("hello") == "answer"
    assert provider.last_cost_usd == 0.0


def test_retry_conservatively_charges_ambiguous_failed_attempt(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise __import__("httpx").TimeoutException("timeout")
        return FakeResponse()

    monkeypatch.setattr("ragx.generate.providers.httpx.post", fake_post)
    monkeypatch.setattr("ragx.generate.providers.time.sleep", lambda *_: None)
    tracker = CostTracker(max_spend_usd=1.0)
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        model="example",
        cost_tracker=tracker,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        max_output_tokens=32,
        max_retries=1,
    )
    maximum_first_attempt = provider._maximum_cost("hello")
    provider.generate("hello")
    actual_success = 18 / 1_000_000
    assert tracker.spent_usd == pytest.approx(maximum_first_attempt + actual_success)
    assert provider.last_cost_usd == pytest.approx(maximum_first_attempt + actual_success)
    assert tracker.reserved_usd == pytest.approx(0.0)
