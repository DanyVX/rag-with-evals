from __future__ import annotations

from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragx.config import get_settings
from ragx.cost import BudgetExceededError
from ragx.runtime import RAGRuntime

app = FastAPI(title="rag-with-evals", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float
    rank: int
    page: int | None = None
    section: str | None = None


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    invalid_citations: list[int]
    citations_valid: bool
    abstained: bool
    retrieved_chunks: list[RetrievedChunk]
    timing_ms: dict[str, float]
    usage: UsageResponse


@lru_cache(maxsize=1)
def get_runtime() -> RAGRuntime:
    return RAGRuntime(get_settings())


@app.get("/health")
def health() -> dict[str, object]:
    try:
        return {"status": "ok", **get_runtime().status()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = get_runtime().ask(request.question)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"provider returned HTTP {exc.response.status_code}",
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        citations=result.citations.cited,
        invalid_citations=result.citations.invalid,
        citations_valid=result.citations.valid,
        abstained=result.abstained,
        retrieved_chunks=[
            RetrievedChunk(
                id=hit.chunk.id,
                text=hit.chunk.text,
                source=hit.chunk.source,
                score=hit.score,
                rank=hit.rank,
                page=hit.chunk.page,
                section=hit.chunk.section,
            )
            for hit in result.retrieved
        ],
        timing_ms={
            "retrieval": result.retrieval_ms,
            "rerank": result.rerank_ms,
            "generation": result.generation_ms,
            "total": result.total_ms,
        },
        usage=UsageResponse(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
        ),
    )
