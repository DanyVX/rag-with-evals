from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ManifestEntry(BaseModel):
    source: str
    content_hash: str


class IngestManifest(BaseModel):
    version: int = 1
    files: dict[str, str]


def load_manifest(path: str | Path) -> IngestManifest:
    path = Path(path)
    if not path.exists():
        return IngestManifest(files={})
    return IngestManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: IngestManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def changed_sources(current: dict[str, str], previous: IngestManifest) -> tuple[set[str], set[str]]:
    changed = {source for source, digest in current.items() if previous.files.get(source) != digest}
    deleted = set(previous.files) - set(current)
    return changed, deleted
