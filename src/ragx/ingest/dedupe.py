from __future__ import annotations

import re
from collections.abc import Iterable

from ragx.ingest.models import Document


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: str, b: str, n: int = 5) -> float:
    sa, sb = _shingles(a, n), _shingles(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(
    documents: Iterable[Document],
    *,
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[Document], list[tuple[str, str, str]]]:
    """Exact hash dedupe plus deterministic near-duplicate shingle detection."""
    kept: list[Document] = []
    removed: list[tuple[str, str, str]] = []
    seen_hashes: dict[str, Document] = {}
    for doc in documents:
        if doc.doc_hash in seen_hashes:
            removed.append((doc.source, seen_hashes[doc.doc_hash].source, "exact"))
            continue
        duplicate_of: Document | None = None
        for candidate in kept:
            if jaccard_similarity(doc.text, candidate.text) >= near_duplicate_threshold:
                duplicate_of = candidate
                break
        if duplicate_of is not None:
            removed.append((doc.source, duplicate_of.source, "near"))
            continue
        seen_hashes[doc.doc_hash] = doc
        kept.append(doc)
    return kept, removed
