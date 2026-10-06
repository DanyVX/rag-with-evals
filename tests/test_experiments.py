import json
from pathlib import Path

from ragx.eval.experiments import ExperimentConfig, run_experiment
from ragx.pipeline import build_index


def test_experiment_cache_is_hash_addressed_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Package installation uses pip.", encoding="utf-8")
    index = tmp_path / "index.json"
    build_index([source], index, chunk_size=20)
    chunk_id = json.loads(index.read_text(encoding="utf-8"))[0]["id"]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps({"question": "pip", "gold_chunk_ids": [chunk_id]}) + "\n", encoding="utf-8"
    )
    config = ExperimentConfig(str(index), str(dataset))

    output, cached = run_experiment(config, tmp_path / "results")
    repeated, repeated_cached = run_experiment(config, tmp_path / "results")

    assert output == repeated
    assert not cached
    assert repeated_cached
