import pytest

from ragx.generate.pricing import TokenUsage, calculate_cost_usd, estimate_input_token_upper_bound


def test_cost_calculation_uses_input_and_output_rates() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    cost = calculate_cost_usd(
        usage,
        input_usd_per_million=2.0,
        output_usd_per_million=8.0,
    )
    assert cost == pytest.approx(6.0)


def test_input_upper_bound_is_conservative_for_ascii() -> None:
    prompt = "hello world"
    assert estimate_input_token_upper_bound(prompt) >= len(prompt)
