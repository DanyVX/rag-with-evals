from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from ragx.chunk.core import Chunk
from ragx.index.base import SearchHit


def chunk_id_hash(chunk_ids: list[str]) -> str:
    payload = "\n".join(sorted(chunk_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FaissSQLiteStore:
    """Persistent cosine-similarity FAISS index with chunk metadata in SQLite."""

    def __init__(self, directory: str | Path, embedding_model_id: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model_id = embedding_model_id
        self.db_path = self.directory / "chunks.sqlite3"
        self.index_path = self.directory / "vectors.faiss"
        self.manifest_path = self.directory / "manifest.json"
        self._index: faiss.IndexFlatIP | None = None
        self._init_db()
        self._load_if_present()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    row_id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )

    def _load_if_present(self) -> None:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest["embedding_model_id"] != self.embedding_model_id:
                raise ValueError("embedding model changed; refusing to mix indexes")
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            if self._index.ntotal != self.size():
                raise RuntimeError(
                    "FAISS vector count and SQLite metadata count differ; rebuild the index"
                )

    def _write_manifest(self) -> None:
        dimension = int(self._index.d) if self._index is not None else 0
        self.manifest_path.write_text(
            json.dumps(
                {
                    "embedding_model_id": self.embedding_model_id,
                    "dimension": dimension,
                    "count": self.size(),
                    "chunk_id_hash": self.chunk_id_hash(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        faiss.normalize_L2(vectors)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        elif self._index.d != vectors.shape[1]:
            raise ValueError("embedding dimensionality changed; reindex required")

        with self._connect() as conn:
            start = conn.execute("SELECT COALESCE(MAX(row_id), -1) + 1 FROM chunks").fetchone()[0]
            rows = [(start + i, chunk.id, chunk.model_dump_json()) for i, chunk in enumerate(chunks)]
            conn.executemany("INSERT INTO chunks(row_id, chunk_id, payload) VALUES (?, ?, ?)", rows)

        self._index.add(vectors)
        faiss.write_index(self._index, str(self.index_path))
        self._write_manifest()

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if self._index is None or self._index.ntotal == 0 or k <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._index.d:
            raise ValueError("query embedding dimension does not match index")
        faiss.normalize_L2(q)
        scores, ids = self._index.search(q, min(k, self._index.ntotal))
        hits: list[SearchHit] = []
        with self._connect() as conn:
            for rank, (row_id, score) in enumerate(zip(ids[0], scores[0]), 1):
                row = conn.execute(
                    "SELECT payload FROM chunks WHERE row_id = ?", (int(row_id),)
                ).fetchone()
                if row:
                    hits.append(
                        SearchHit(
                            chunk=Chunk.model_validate_json(row[0]),
                            score=float(score),
                            rank=rank,
                        )
                    )
        return hits

    def size(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def chunk_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks ORDER BY row_id").fetchall()
        return [str(row[0]) for row in rows]

    def chunk_id_hash(self) -> str:
        return chunk_id_hash(self.chunk_ids())

    def matches_chunks(self, chunks: list[Chunk]) -> bool:
        if len(chunks) != self.size():
            return False
        return self.chunk_id_hash() == chunk_id_hash([chunk.id for chunk in chunks])
