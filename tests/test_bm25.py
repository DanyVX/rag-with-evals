from pathlib import Path

from ragx.chunk import FixedTokenChunker
from ragx.ingest.models import Document
from ragx.retrieve import BM25Retriever


def _chunks() -> list:
    first = Document("Python has a package manager named pip.", Path("one"), "one")
    second = Document("Rust uses Cargo for package management.", Path("two"), "two")
    return FixedTokenChunker(size=20).chunk(first) + FixedTokenChunker(size=20).chunk(second)


def test_bm25_returns_relevant_chunk_and_handles_empty_query() -> None:
    retriever = BM25Retriever(_chunks())

    result = retriever.search("python pip", top_k=1)

    assert result[0].chunk.document_hash == "one"
    assert retriever.search("   ") == []


def test_bm25_honors_score_floor_and_top_k_larger_than_corpus() -> None:
    retriever = BM25Retriever(_chunks())

    assert retriever.search("python", score_floor=100) == []
    assert len(retriever.search("package", top_k=100)) == 2
