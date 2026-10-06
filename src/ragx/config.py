from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment only."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="dev", alias="RAGX_ENV")
    max_spend_usd: float = Field(default=0.0, alias="RAGX_MAX_SPEND_USD")

    provider: str = Field(default="mock", alias="RAGX_PROVIDER")
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
    retriever: str = Field(default="hybrid", alias="RAGX_RETRIEVER")
    retrieve_top_n: int = Field(default=20, alias="RAGX_RETRIEVE_TOP_N")
    top_k: int = Field(default=5, alias="RAGX_TOP_K")
    score_floor: float | None = Field(default=None, alias="RAGX_SCORE_FLOOR")
    reranker_model: str | None = Field(default=None, alias="RAGX_RERANKER_MODEL")

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
        if self.provider != "mock" and self.max_spend_usd <= 0:
            raise ValueError("remote providers require a positive RAGX_MAX_SPEND_USD")
        if self.provider != "mock" and (
            self.provider_input_usd_per_million <= 0
            or self.provider_output_usd_per_million <= 0
        ):
            raise ValueError(
                "remote providers require explicit positive input/output token prices"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
