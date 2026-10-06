"""Resumable, hash-addressed experiment execution."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from ragx.eval.runner import evaluate


@dataclass(frozen=True)
class ExperimentConfig:
    index_path: str
    dataset_path: str
    k: int = 5

    @property
    def identifier(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()[:16]


def run_experiment(config: ExperimentConfig, results_directory: Path) -> tuple[Path, bool]:
    """Evaluate once per immutable config hash and reuse completed artifacts."""
    results_directory.mkdir(parents=True, exist_ok=True)
    output = results_directory / f"{config.identifier}.json"
    if output.exists():
        return output, True
    metrics = evaluate(Path(config.index_path), Path(config.dataset_path), k=config.k)
    output.write_text(
        json.dumps({"config": asdict(config), "metrics": metrics}, indent=2), encoding="utf-8"
    )
    return output, False
