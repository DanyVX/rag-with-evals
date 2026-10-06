from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel

from ragx.chunk.core import Chunk


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0


class VectorStore(Protocol):
    embedding_model_id: str

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]: ...

    def size(self) -> int: ...
