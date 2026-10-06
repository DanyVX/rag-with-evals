from fastapi.testclient import TestClient

import ragx.api as api_module
from ragx.chunk.core import Chunk
from ragx.generate.citations import CitationValidation
from ragx.index.base import SearchHit
from ragx.pipeline import AskResult

client = TestClient(api_module.app)


class FakeRuntime:
    def status(self) -> dict[str, object]:
        return {"provider": "fake", "index_size": 1}

    def ask(self, question: str) -> AskResult:
        assert question == "What is this?"
        chunk = Chunk(
            id="c1",
            doc_hash="d1",
            source="fixture.txt",
            text="This is evidence.",
            position=0,
        )
        return AskResult(
            answer="This is the answer [1]",
            citations=CitationValidation(cited=[1], invalid=[], valid=True),
            retrieved=[SearchHit(chunk=chunk, score=0.9, rank=1)],
            retrieval_ms=4.0,
            rerank_ms=1.0,
            generation_ms=5.0,
            prompt_tokens=20,
            completion_tokens=8,
            cost_usd=0.001,
            abstained=False,
        )


def test_health_reports_runtime(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "get_runtime", lambda: FakeRuntime())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["provider"] == "fake"


def test_ask_returns_evidence_timing_usage_and_citations(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "get_runtime", lambda: FakeRuntime())
    response = client.post("/ask", json={"question": "What is this?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This is the answer [1]"
    assert body["citations"] == [1]
    assert body["citations_valid"] is True
    assert body["retrieved_chunks"][0]["id"] == "c1"
    assert body["timing_ms"]["retrieval"] == 4.0
    assert body["timing_ms"]["rerank"] == 1.0
    assert body["timing_ms"]["total"] == 10.0
    assert body["usage"]["prompt_tokens"] == 20
    assert body["usage"]["cost_usd"] == 0.001
