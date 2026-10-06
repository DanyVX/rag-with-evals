from pathlib import Path

import numpy as np

from ragx.chunk.core import ChunkConfig, chunk_document
from ragx.chunk.tokenizer import WhitespaceTokenCodec
from ragx.generate.providers import MockProvider
from ragx.index.bm25 import BM25Index
from ragx.index.memory import InMemoryVectorStore
from ragx.ingest.loaders import load_document
from ragx.pipeline import answer_with_context
from ragx.retrieve.core import hybrid_rrf, sparse_search

FIXTURE = Path(__file__).parent / "fixtures" / "corpus"


def deterministic_vector(text: str) -> np.ndarray:
    lowered = text.lower()
    vector = np.asarray(
        [
            float("aurora" in lowered or "blue_orchid" in lowered),
            float("borealis" in lowered or "silver_pine" in lowered),
            0.1,
        ],
        dtype=np.float32,
    )
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def test_tiny_full_pipeline_is_deterministic() -> None:
    documents = []
    for path in sorted(FIXTURE.glob("*.txt")):
        documents.extend(load_document(path))

    codec = WhitespaceTokenCodec(max_sequence_length=64)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            ChunkConfig(strategy="fixed", size=32, overlap=4),
            codec=codec,
        )
    ]
    vectors = np.vstack([deterministic_vector(chunk.text) for chunk in chunks])
    store = InMemoryVectorStore("fixture-v1")
    store.add(chunks, vectors)
    bm25 = BM25Index(chunks)

    question = "What codename does Project Aurora use for its stable release channel?"
    query_vector = deterministic_vector(question)
    dense = store.search(query_vector, 5)
    sparse = sparse_search(question, index=bm25, k=5)
    hits = hybrid_rrf(dense, sparse, k=2)

    assert hits[0].chunk.source.endswith("alpha.txt")
    provider = MockProvider("Project Aurora uses BLUE_ORCHID [1].")
    first = answer_with_context(question, hits, provider=provider)
    second = answer_with_context(question, hits, provider=provider)

    assert first.answer == second.answer
    assert first.citations.valid
    assert first.citations.cited == [1]
    assert not first.abstained
    assert [hit.chunk.id for hit in first.retrieved] == [hit.chunk.id for hit in second.retrieved]
