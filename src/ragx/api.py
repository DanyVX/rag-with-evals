from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ragx.generate.providers import MockProvider
from ragx.index.base import SearchHit
from ragx.pipeline import answer_with_context

app = FastAPI(title="rag-with-evals", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float
    rank: int


class AskResponse(BaseModel):
    answer: str
    citations_valid: bool
    abstained: bool
    retrieved_chunks: list[RetrievedChunk]
    timing_ms: dict[str, float]


def _retrieve(question: str) -> list[SearchHit]:
    """Dependency seam replaced by configured retrieval in production."""
    del question
    return []


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    hits = _retrieve(request.question)
    result = answer_with_context(request.question, hits, provider=MockProvider())
    return AskResponse(
        answer=result.answer,
        citations_valid=result.citations.valid,
        abstained=result.abstained,
        retrieved_chunks=[
            RetrievedChunk(
                id=h.chunk.id,
                text=h.chunk.text,
                source=h.chunk.source,
                score=h.score,
                rank=h.rank,
            )
            for h in hits
        ],
        timing_ms={
            "retrieval": result.retrieval_ms,
            "generation": result.generation_ms,
        },
    )
