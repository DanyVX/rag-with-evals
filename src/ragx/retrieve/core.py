from __future__ import annotations

from collections import defaultdict

from ragx.embed.base import Embedder
from ragx.index.base import SearchHit, VectorStore
from ragx.index.bm25 import BM25Index


def dense_search(query: str, *, embedder: Embedder, store: VectorStore, k: int) -> list[SearchHit]:
    if not query.strip():
        return []
    vector = embedder.encode([query])[0]
    return store.search(vector, k)


def sparse_search(query: str, *, index: BM25Index, k: int) -> list[SearchHit]:
    return index.search(query, k)


def hybrid_rrf(
    dense: list[SearchHit],
    sparse: list[SearchHit],
    *,
    k: int,
    rrf_k: int = 60,
) -> list[SearchHit]:
    scores: dict[str, float] = defaultdict(float)
    chunks = {}
    for hits in (dense, sparse):
        for rank, hit in enumerate(hits, 1):
            scores[hit.chunk.id] += 1.0 / (rrf_k + rank)
            chunks[hit.chunk.id] = hit.chunk
    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [
        SearchHit(chunk=chunks[cid], score=scores[cid], rank=rank + 1)
        for rank, cid in enumerate(ranked)
    ]
