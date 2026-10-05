"""Validated, environment-only runtime configuration."""

from decimal import Decimal
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings that make external model access opt-in and bounded."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAGX_", extra="ignore")

    provider: Literal["mock", "openai_compatible", "anthropic"] = "mock"
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    max_provider_spend_usd: Decimal = Field(default=Decimal("0"), ge=0, decimal_places=4)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    @field_validator("openai_base_url")
    @classmethod
    def require_http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("RAGX_OPENAI_BASE_URL must start with http:// or https://")
        return value

    @model_validator(mode="after")
    def validate_provider_access(self) -> "Settings":
        if self.provider == "mock":
            return self
        if self.max_provider_spend_usd <= 0:
            raise ValueError("A non-mock provider requires RAGX_MAX_PROVIDER_SPEND_USD > 0")
        if self.provider == "openai_compatible" and (
            self.openai_base_url is None or self.openai_api_key is None
        ):
            raise ValueError("openai_compatible requires base URL and API key")
        if self.provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("anthropic requires RAGX_ANTHROPIC_API_KEY")
        return self
