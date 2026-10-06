import pytest
from pydantic import ValidationError

from ragx.config import Settings


def test_paid_remote_provider_requires_budget_and_rates() -> None:
    with pytest.raises(ValidationError):
        Settings(
            RAGX_PROVIDER="openai-compatible",
            RAGX_MODEL="remote-model",
            OPENAI_BASE_URL="https://example.test/v1",
            RAGX_MAX_SPEND_USD=0,
        )


def test_local_compatible_endpoint_can_be_zero_cost() -> None:
    settings = Settings(
        RAGX_PROVIDER="openai-compatible",
        RAGX_MODEL="local-model",
        OPENAI_BASE_URL="http://127.0.0.1:11434/v1",
        RAGX_MAX_SPEND_USD=0,
        RAGX_INPUT_USD_PER_MILLION=0,
        RAGX_OUTPUT_USD_PER_MILLION=0,
    )
    assert settings.zero_cost_local_endpoint


def test_anthropic_requires_key() -> None:
    with pytest.raises(ValidationError):
        Settings(
            RAGX_PROVIDER="anthropic",
            RAGX_MODEL="model",
            RAGX_MAX_SPEND_USD=1,
            RAGX_INPUT_USD_PER_MILLION=1,
            RAGX_OUTPUT_USD_PER_MILLION=1,
        )
