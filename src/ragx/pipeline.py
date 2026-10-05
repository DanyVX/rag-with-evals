"""Composable offline RAG baseline with persisted chunk indexes."""

import json
from dataclasses import asdict
from pathlib import Path

from ragx.chunk import Chunk, FixedTokenChunker
from ragx.ingest.service import ingest_paths
from ragx.retrieve import BM25Retriever, RetrievedChunk


def build_index(inputs: list[Path], destination: Path, *, chunk_size: int = 256) -> dict[str, int]:
    """Ingest and persist a deterministic local index without model downloads."""
    ingested = ingest_paths(inputs)
    chunker = FixedTokenChunker(size=chunk_size)
    chunks = [chunk for document in ingested.documents for chunk in chunker.chunk(document)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False), encoding="utf-8"
    )
    return {
        "documents": len(ingested.documents),
        "chunks": len(chunks),
        "warnings": len(ingested.warnings),
    }


def load_index(path: Path) -> BM25Retriever:
    """Load an index written by :func:`build_index`, rejecting malformed state."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload]
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as error:
        raise ValueError(f"Unable to load index {path}: {error}") from error
    return BM25Retriever(chunks)


def answer(
    retriever: BM25Retriever, query: str, *, top_k: int = 3
) -> tuple[str, list[RetrievedChunk]]:
    """Return extractive evidence only; never invent a generated answer offline."""
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return "not found in the provided documents", []
    citations = " ".join(f"[{number}]" for number in range(1, len(hits) + 1))
    return f"Evidence retrieved from the provided documents: {citations}", hits
