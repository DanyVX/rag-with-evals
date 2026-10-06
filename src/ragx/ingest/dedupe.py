from __future__ import annotations

import hashlib
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


def simhash64(text: str, n: int = 5) -> int:
    shingles = _shingles(text, n)
    if not shingles:
        return 0
    vector = [0] * 64
    for shingle in shingles:
        payload = "\x1f".join(shingle).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _band_keys(fingerprint: int) -> list[tuple[int, int]]:
    mask = (1 << 16) - 1
    return [(band, (fingerprint >> (band * 16)) & mask) for band in range(4)]


def deduplicate(
    documents: Iterable[Document],
    *,
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[Document], list[tuple[str, str, str]]]:
    """Exact hash dedupe plus banded 64-bit SimHash near-duplicate detection."""
    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    max_distance = max(1, round((1.0 - near_duplicate_threshold) * 64))

    kept: list[Document] = []
    fingerprints: list[int] = []
    removed: list[tuple[str, str, str]] = []
    seen_hashes: dict[str, Document] = {}
    buckets: dict[tuple[int, int], set[int]] = {}

    for doc in documents:
        if doc.doc_hash in seen_hashes:
            removed.append((doc.source, seen_hashes[doc.doc_hash].source, "exact"))
            continue

        fingerprint = simhash64(doc.text)
        candidates: set[int] = set()
        for key in _band_keys(fingerprint):
            candidates.update(buckets.get(key, set()))

        duplicate_index = None
        for idx in sorted(candidates):
            if hamming_distance(fingerprint, fingerprints[idx]) <= max_distance:
                duplicate_index = idx
                break

        if duplicate_index is not None:
            removed.append((doc.source, kept[duplicate_index].source, "near"))
            continue

        idx = len(kept)
        kept.append(doc)
        fingerprints.append(fingerprint)
        seen_hashes[doc.doc_hash] = doc
        for key in _band_keys(fingerprint):
            buckets.setdefault(key, set()).add(idx)

    return kept, removed
