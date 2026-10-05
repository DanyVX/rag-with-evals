"""Stable ingestion data structures."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Normalized text and provenance for one source document."""

    text: str
    source: Path
    content_hash: str
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestWarning:
    source: Path
    message: str


@dataclass(frozen=True)
class IngestResult:
    documents: list[Document]
    warnings: list[IngestWarning]
    duplicates_skipped: int
