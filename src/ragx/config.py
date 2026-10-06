from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment only."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="dev", alias="RAGX_ENV")
    max_spend_usd: float = Field(default=0.0, alias="RAGX_MAX_SPEND_USD")

    provider: str = Field(default="mock", alias="RAGX_PROVIDER")
    allow_mock_api: bool = Field(default=False, alias="RAGX_ALLOW_MOCK_API")
    model: str = Field(default="mock", alias="RAGX_MODEL")
    provider_input_usd_per_million: float = Field(
        default=0.0, alias="RAGX_INPUT_USD_PER_MILLION"
    )
    provider_output_usd_per_million: float = Field(
        default=0.0, alias="RAGX_OUTPUT_USD_PER_MILLION"
    )
    provider_max_output_tokens: int = Field(default=2048, alias="RAGX_MAX_OUTPUT_TOKENS")

    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    chunks_path: Path = Field(
        default=Path("data/processed/chunks.jsonl"), alias="RAGX_CHUNKS_PATH"
    )
    index_dir: Path = Field(default=Path("data/index"), alias="RAGX_INDEX_DIR")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="RAGX_EMBEDDING_MODEL"
    )
    embedding_cache_path: Path = Field(
        default=Path("cache/embeddings.sqlite3"), alias="RAGX_EMBEDDING_CACHE_PATH"
    )
    embedding_preprocess_version: str = Field(
        default="v1", alias="RAGX_EMBEDDING_PREPROCESS_VERSION"
    )
    retriever: str = Field(default="hybrid", alias="RAGX_RETRIEVER")
    retrieve_top_n: int = Field(default=20, alias="RAGX_RETRIEVE_TOP_N")
    top_k: int = Field(default=5, alias="RAGX_TOP_K")
    score_floor: float | None = Field(default=None, alias="RAGX_SCORE_FLOOR")
    reranker_model: str | None = Field(default=None, alias="RAGX_RERANKER_MODEL")

    @property
    def zero_cost_local_endpoint(self) -> bool:
        if self.provider != "openai-compatible" or not self.openai_base_url:
            return False
        host = (urlparse(self.openai_base_url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    @model_validator(mode="after")
    def validate_runtime(self) -> "Settings":
        if self.max_spend_usd < 0:
            raise ValueError("RAGX_MAX_SPEND_USD cannot be negative")
        if self.provider_input_usd_per_million < 0 or self.provider_output_usd_per_million < 0:
            raise ValueError("provider token prices cannot be negative")
        if self.provider_max_output_tokens < 1:
            raise ValueError("RAGX_MAX_OUTPUT_TOKENS must be positive")
        if self.retrieve_top_n < 1 or self.top_k < 1:
            raise ValueError("retrieval limits must be positive")
        if self.top_k > self.retrieve_top_n:
            raise ValueError("RAGX_TOP_K cannot exceed RAGX_RETRIEVE_TOP_N")
        if self.retriever not in {"dense", "bm25", "hybrid"}:
            raise ValueError("RAGX_RETRIEVER must be dense, bm25, or hybrid")
        if self.provider not in {"mock", "openai-compatible", "anthropic"}:
            raise ValueError("RAGX_PROVIDER must be mock, openai-compatible, or anthropic")

        remote_paid = self.provider != "mock" and not self.zero_cost_local_endpoint
        if remote_paid and self.max_spend_usd <= 0:
            raise ValueError("remote paid providers require a positive RAGX_MAX_SPEND_USD")
        if remote_paid and (
            self.provider_input_usd_per_million <= 0
            or self.provider_output_usd_per_million <= 0
        ):
            raise ValueError(
                "remote paid providers require explicit positive input/output token prices"
            )
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic provider")
        if self.provider == "openai-compatible" and not self.openai_base_url:
            raise ValueError("OPENAI_BASE_URL is required for the OpenAI-compatible provider")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
