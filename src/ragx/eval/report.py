"""Versioned, provenance-rich result artifact writing."""

import json
from pathlib import Path

from ragx.provenance import collect_environment


def write_result(path: Path, metrics: dict[str, object], project_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metrics": metrics, "environment": collect_environment(project_root)}, indent=2
        ),
        encoding="utf-8",
    )
