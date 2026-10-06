from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any


@dataclass(slots=True)
class ExperimentResult:
    config_hash: str
    config: dict[str, Any]
    result: dict[str, Any]
    cached: bool = False


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class ExperimentRunner:
    """Resumable JSON-cache experiment runner keyed by canonical config hashes."""

    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        config: dict[str, Any],
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> ExperimentResult:
        key = config_hash(config)
        path = self.results_dir / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ExperimentResult(key, config, payload["result"], cached=True)

        try:
            result = fn(config)
            payload = {"config": config, "result": result, "status": "ok"}
        except Exception as exc:
            payload = {
                "config": config,
                "result": {"error": repr(exc)},
                "status": "failed",
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            raise
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return ExperimentResult(key, config, result, cached=False)
