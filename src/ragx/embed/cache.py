from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from ragx.embed.base import Embedder


def embedding_cache_key(model_id: str, preprocess_version: str, text: str) -> str:
    payload = (
        model_id.encode("utf-8")
        + b"\0"
        + preprocess_version.encode("utf-8")
        + b"\0"
        + text.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, key: str) -> np.ndarray | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dimension, vector FROM embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        dimension, blob = row
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if len(vector) != int(dimension):
            raise RuntimeError("embedding cache row has invalid dimensionality")
        return vector

    def put(self, key: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
                (key, len(vector), vector.tobytes()),
            )

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


class CachedEmbedder:
    def __init__(
        self,
        base: Embedder,
        cache: EmbeddingCache,
        *,
        preprocess_version: str = "v1",
    ) -> None:
        self.base = base
        self.cache = cache
        self.preprocess_version = preprocess_version
        self.model_id = base.model_id
        self.max_sequence_length = base.max_sequence_length

    def tokenize(self, text: str) -> list[str]:
        return self.base.tokenize(text)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        keys = [
            embedding_cache_key(self.model_id, self.preprocess_version, text)
            for text in texts
        ]
        vectors: list[np.ndarray | None] = [self.cache.get(key) for key in keys]
        missing_indices = [i for i, vector in enumerate(vectors) if vector is None]

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            generated = self.base.encode(missing_texts)
            for index, vector in zip(missing_indices, generated):
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                self.cache.put(keys[index], vector)
                vectors[index] = vector

        resolved = [vector for vector in vectors if vector is not None]
        if len(resolved) != len(texts):
            raise RuntimeError("embedding cache failed to resolve all vectors")
        dimensions = {len(vector) for vector in resolved}
        if len(dimensions) != 1:
            raise RuntimeError("embedding cache contains mixed vector dimensions")
        return np.vstack(resolved).astype(np.float32)
