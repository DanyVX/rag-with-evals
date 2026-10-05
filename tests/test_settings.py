from decimal import Decimal

import pytest
from pydantic import ValidationError

from ragx.settings import Settings


def test_default_configuration_is_offline() -> None:
    settings = Settings()

    assert settings.provider == "mock"
    assert settings.max_provider_spend_usd == Decimal("0")


def test_billable_provider_requires_positive_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_PROVIDER", "anthropic")
    monkeypatch.setenv("RAGX_ANTHROPIC_API_KEY", "test-key")

    with pytest.raises(ValidationError, match="MAX_PROVIDER_SPEND_USD"):
        Settings()


def test_openai_compatible_requires_endpoint_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RAGX_MAX_PROVIDER_SPEND_USD", "1")

    with pytest.raises(ValidationError, match="base URL and API key"):
        Settings()
