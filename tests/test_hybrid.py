from pathlib import Path

from ragx.chunk import FixedTokenChunker
from ragx.ingest.models import Document
from ragx.retrieve import BM25Retriever, HashingDenseRetriever, mmr_select, reciprocal_rank_fusion


def test_dense_fusion_and_mmr_are_deterministic() -> None:
    documents = [
        Document("Python package pip", Path("a"), "a"),
        Document("Python package poetry", Path("b"), "b"),
    ]
    chunks = [
        chunk for document in documents for chunk in FixedTokenChunker(size=10).chunk(document)
    ]
    bm25, dense = BM25Retriever(chunks), HashingDenseRetriever(chunks)
    fused = reciprocal_rank_fusion([bm25.search("python package"), dense.search("python package")])
    assert len(fused) == 2
    assert len(mmr_select("python", fused, top_k=1)) == 1
    assert dense.search("") == []
