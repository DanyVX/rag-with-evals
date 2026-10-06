from pathlib import Path

from ragx.api import AskRequest, ask
from ragx.pipeline import answer, build_index, load_index


def test_pipeline_persists_retrieves_and_abstains(tmp_path: Path) -> None:
    source = tmp_path / "guide.txt"
    source.write_text("Python uses pip to install packages.", encoding="utf-8")
    index = tmp_path / "index.json"

    summary = build_index([source], index, chunk_size=20)

    assert summary["chunks"] == 1
    assert load_index(index).search("pip")[0].chunk.document_hash
    response, hits = answer(load_index(index), "missing")
    assert response == "not found in the provided documents"
    assert hits == []
    payload = ask(AskRequest(query="pip", index_path=index))
    assert payload["citations"] == [1]
