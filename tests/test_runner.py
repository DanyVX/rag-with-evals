import json
from pathlib import Path

import pytest

from ragx.eval.runner import evaluate
from ragx.pipeline import build_index


def test_evaluation_runner_writes_retrieval_metrics(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Python uses pip for packages.", encoding="utf-8")
    index = tmp_path / "index.json"
    build_index([source], index, chunk_size=20)
    chunk_id = json.loads(index.read_text(encoding="utf-8"))[0]["id"]
    data = tmp_path / "eval.jsonl"
    data.write_text(
        json.dumps({"question": "pip", "gold_chunk_ids": [chunk_id]}) + "\n", encoding="utf-8"
    )
    metrics = evaluate(index, data)
    assert metrics["n"] == 1
    assert metrics["hit_at_5"]["mean"] == 1.0


def test_evaluation_runner_rejects_bad_rows(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    data = tmp_path / "bad.jsonl"
    data.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid evaluation row"):
        evaluate(index, data)
