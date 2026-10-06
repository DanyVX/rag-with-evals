from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ragx.chunk.core import Chunk
from ragx.index.base import SearchHit


def _terms(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = chunks or []
        self._bm25 = BM25Okapi([_terms(c.text) for c in self._chunks]) if self._chunks else None

    def rebuild(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([_terms(c.text) for c in self._chunks]) if self._chunks else None

    def search(self, query: str, k: int) -> list[SearchHit]:
        if not query.strip() or self._bm25 is None or k <= 0:
            return []
        scores = self._bm25.get_scores(_terms(query))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self._chunks[i].id))
        return [
            SearchHit(chunk=self._chunks[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(order[: min(k, len(order))])
        ]
