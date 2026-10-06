from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ragx.chunk.core import Chunk
from ragx.config import Settings
from ragx.cost import CostTracker
from ragx.embed.cache import CachedEmbedder, EmbeddingCache
from ragx.embed.sentence_transformer import SentenceTransformerEmbedder
from ragx.generate.anthropic_provider import AnthropicProvider
from ragx.generate.pricing import TokenUsage
from ragx.generate.providers import MockProvider, OpenAICompatibleProvider
from ragx.index.base import SearchHit
from ragx.index.bm25 import BM25Index
from ragx.index.faiss_sqlite import FaissSQLiteStore
from ragx.pipeline import AskResult, answer_with_context
from ragx.retrieve.advanced import apply_score_floor
from ragx.retrieve.core import dense_search, hybrid_rrf, sparse_search
from ragx.retrieve.query import preprocess_query
from ragx.retrieve.rerank import CrossEncoderReranker


@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    processed_query: str
    preprocess_ms: float
    retrieval_ms: float
    rerank_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    query_truncated: bool = False


def _load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise RuntimeError(f"chunk file does not exist: {path}")
    chunks = [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise RuntimeError(f"chunk file is empty: {path}")
    return chunks


class RAGRuntime:
    """Long-lived configured retrieval + generation runtime for the API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = _load_chunks(settings.chunks_path)
        base_embedder = SentenceTransformerEmbedder(settings.embedding_model)
        self.embedder = CachedEmbedder(
            base_embedder,
            EmbeddingCache(settings.embedding_cache_path),
            preprocess_version=settings.embedding_preprocess_version,
        )
        self.store = FaissSQLiteStore(settings.index_dir, settings.embedding_model)
        if self.store.size() == 0:
            raise RuntimeError(
                f"dense index is empty at {settings.index_dir}; run 'ragx index' first"
            )
        if not self.store.matches_chunks(self.chunks):
            raise RuntimeError(
                "dense index is stale relative to the configured chunk file; rebuild with ragx index"
            )
        self.bm25 = BM25Index(self.chunks)
        self.reranker = (
            CrossEncoderReranker(settings.reranker_model)
            if settings.reranker_model
            else None
        )
        self.cost_tracker = CostTracker(settings.max_spend_usd)
        self.provider = self._build_provider()

    def _build_provider(self):
        settings = self.settings
        if settings.provider == "mock":
            if not settings.allow_mock_api:
                raise RuntimeError(
                    "RAGX_PROVIDER=mock is disabled for the API; configure a real provider "
                    "or explicitly set RAGX_ALLOW_MOCK_API=true for tests."
                )
            return MockProvider()
        if settings.provider == "anthropic":
            assert settings.anthropic_api_key is not None
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
                model=settings.model,
                cost_tracker=self.cost_tracker,
                input_usd_per_million=settings.provider_input_usd_per_million,
                output_usd_per_million=settings.provider_output_usd_per_million,
                max_output_tokens=settings.provider_max_output_tokens,
            )
        assert settings.openai_base_url is not None
        return OpenAICompatibleProvider(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key or "",
            model=settings.model,
            cost_tracker=self.cost_tracker,
            input_usd_per_million=settings.provider_input_usd_per_million,
            output_usd_per_million=settings.provider_output_usd_per_million,
            max_output_tokens=settings.provider_max_output_tokens,
            zero_cost_local=settings.zero_cost_local_endpoint,
        )

    def retrieve(self, question: str) -> RetrievalResult:
        preprocess_started = perf_counter()
        expander = self.provider if self.settings.query_preprocess == "query_expansion" else None
        processed = preprocess_query(
            question,
            mode=self.settings.query_preprocess,
            expander=expander,
        )
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

        preprocessing_usage = TokenUsage()
        preprocessing_cost = 0.0
        if self.settings.query_preprocess == "query_expansion":
            preprocessing_usage = getattr(self.provider, "last_usage", TokenUsage())
            preprocessing_cost = float(getattr(self.provider, "last_cost_usd", 0.0))

        query = processed.processed
        if not query:
            return RetrievalResult(
                hits=[],
                processed_query="",
                preprocess_ms=preprocess_ms,
                retrieval_ms=0.0,
                rerank_ms=0.0,
                prompt_tokens=preprocessing_usage.input_tokens,
                completion_tokens=preprocessing_usage.output_tokens,
                cost_usd=preprocessing_cost,
                query_truncated=processed.truncated,
            )

        started = perf_counter()
        top_n = self.settings.retrieve_top_n
        if self.settings.retriever == "dense":
            hits = dense_search(
                query,
                embedder=self.embedder,
                store=self.store,
                k=top_n,
            )
        elif self.settings.retriever == "bm25":
            hits = sparse_search(query, index=self.bm25, k=top_n)
        else:
            dense = dense_search(
                query,
                embedder=self.embedder,
                store=self.store,
                k=top_n,
            )
            sparse = sparse_search(query, index=self.bm25, k=top_n)
            hits = hybrid_rrf(dense, sparse, k=top_n)

        hits = apply_score_floor(hits, self.settings.score_floor)
        retrieval_ms = (perf_counter() - started) * 1000.0

        rerank_ms = 0.0
        if self.reranker and hits:
            rerank_started = perf_counter()
            hits = self.reranker.rerank(query, hits, self.settings.top_k)
            rerank_ms = (perf_counter() - rerank_started) * 1000.0
        else:
            hits = hits[: self.settings.top_k]
            for rank, hit in enumerate(hits, 1):
                hit.rank = rank

        return RetrievalResult(
            hits=hits,
            processed_query=query,
            preprocess_ms=preprocess_ms,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            prompt_tokens=preprocessing_usage.input_tokens,
            completion_tokens=preprocessing_usage.output_tokens,
            cost_usd=preprocessing_cost,
            query_truncated=processed.truncated,
        )

    def ask(self, question: str) -> AskResult:
        retrieval = self.retrieve(question)
        return answer_with_context(
            question,
            retrieval.hits,
            provider=self.provider,
            preprocess_ms=retrieval.preprocess_ms,
            retrieval_ms=retrieval.retrieval_ms,
            rerank_ms=retrieval.rerank_ms,
            prior_prompt_tokens=retrieval.prompt_tokens,
            prior_completion_tokens=retrieval.completion_tokens,
            prior_cost_usd=retrieval.cost_usd,
        )

    def status(self) -> dict[str, object]:
        return {
            "chunks": len(self.chunks),
            "index_size": self.store.size(),
            "index_matches_chunks": self.store.matches_chunks(self.chunks),
            "embedding_model": self.settings.embedding_model,
            "embedding_cache_path": str(self.settings.embedding_cache_path),
            "embedding_preprocess_version": self.settings.embedding_preprocess_version,
            "retriever": self.settings.retriever,
            "query_preprocess": self.settings.query_preprocess,
            "reranker": self.settings.reranker_model,
            "provider": self.provider.provider_id,
            "budget": {
                "max_spend_usd": self.cost_tracker.max_spend_usd,
                "spent_usd": self.cost_tracker.spent_usd,
                "reserved_usd": self.cost_tracker.reserved_usd,
                "remaining_usd": self.cost_tracker.remaining_usd,
            },
        }
