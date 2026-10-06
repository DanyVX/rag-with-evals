from __future__ import annotations

import numpy as np

from ragx.chunk.core import Chunk
from ragx.index.base import SearchHit


class InMemoryVectorStore:
    def __init__(self, embedding_model_id: str) -> None:
        self.embedding_model_id = embedding_model_id
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        if len(self._chunks) == 0:
            self._vectors = vectors
        else:
            if self._vectors.shape[1] != vectors.shape[1]:
                raise ValueError("embedding dimensionality changed; reindex required")
            self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if not self._chunks or k <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError("query embedding dimension does not match index")
        scores = self._vectors @ q
        order = np.argsort(-scores, kind="stable")[: min(k, len(self._chunks))]
        return [
            SearchHit(chunk=self._chunks[int(i)], score=float(scores[int(i)]), rank=rank + 1)
            for rank, i in enumerate(order)
        ]

    def size(self) -> int:
        return len(self._chunks)
