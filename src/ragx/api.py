"""Thin API for the deterministic baseline."""

from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragx.pipeline import answer, load_index

app = FastAPI(title="ragx", version="0.1.0")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    index_path: Path
    top_k: int = Field(default=3, ge=1, le=50)


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    """Return abstention or cited evidence with timing and source metadata."""
    started = perf_counter()
    try:
        response, hits = answer(load_index(request.index_path), request.query, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "answer": response,
        "citations": [number for number in range(1, len(hits) + 1)],
        "chunks": [
            {"id": hit.chunk.id, "score": hit.score, "source": str(hit.chunk.document_hash)}
            for hit in hits
        ],
        "latency_ms": round((perf_counter() - started) * 1000, 3),
    }
