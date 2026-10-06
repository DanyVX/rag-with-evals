from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment only."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="dev", alias="RAGX_ENV")
    max_spend_usd: float = Field(default=0.0, alias="RAGX_MAX_SPEND_USD")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    @model_validator(mode="after")
    def validate_budget(self) -> "Settings":
        if self.max_spend_usd < 0:
            raise ValueError("RAGX_MAX_SPEND_USD cannot be negative")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
