"""Filesystem ingestion with content-hash deduplication."""

from hashlib import sha256
from pathlib import Path

from ragx.ingest.clean import clean_text, is_substantive
from ragx.ingest.loaders import UnsupportedDocumentError, load_text
from ragx.ingest.models import Document, IngestResult, IngestWarning


def _content_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def ingest_paths(paths: list[Path]) -> IngestResult:
    """Load files recursively, skip non-textual data, and deduplicate by normalized content."""
    documents: list[Document] = []
    warnings: list[IngestWarning] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    candidates: list[Path] = []
    for path in paths:
        candidates.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw, metadata = load_text(path)
        except (OSError, UnsupportedDocumentError) as error:
            warnings.append(IngestWarning(path, str(error)))
            continue
        text = clean_text(raw)
        if not is_substantive(text):
            warnings.append(IngestWarning(path, "Skipped empty or non-substantive extracted text"))
            continue
        digest = _content_hash(text)
        if digest in seen_hashes:
            duplicate_count += 1
            warnings.append(IngestWarning(path, "Skipped exact duplicate content"))
            continue
        seen_hashes.add(digest)
        documents.append(Document(text=text, source=path, content_hash=digest, metadata=metadata))
    return IngestResult(documents=documents, warnings=warnings, duplicates_skipped=duplicate_count)
