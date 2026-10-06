from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ragx.chunk.core import Chunk, ChunkConfig, chunk_document
from ragx.chunk.tokenizer import TokenCodec
from ragx.ingest.dedupe import deduplicate
from ragx.ingest.loaders import load_document
from ragx.ingest.manifest import IngestManifest, changed_sources, load_manifest, save_manifest
from ragx.ingest.models import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}


@dataclass(slots=True)
class IncrementalIngestResult:
    files_seen: int
    changed_files: int
    deleted_files: int
    documents: int
    deduplicated_documents: int
    chunks: int
    reused_chunks: int
    rebuilt_chunks: int


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_fingerprint(config: ChunkConfig, codec: TokenCodec, tokenizer_id: str) -> str:
    payload = json.dumps(
        {
            "config": config.model_dump(),
            "tokenizer_id": tokenizer_id,
            "max_sequence_length": codec.max_sequence_length,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_documents(path: Path) -> list[Document]:
    if not path.exists():
        return []
    return [
        Document.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    temporary.replace(path)


def incremental_ingest_directory(
    source: str | Path,
    *,
    output_chunks: str | Path,
    document_store: str | Path,
    manifest_path: str | Path,
    config: ChunkConfig,
    codec: TokenCodec,
    tokenizer_id: str,
) -> IncrementalIngestResult:
    source = Path(source)
    output_chunks = Path(output_chunks)
    document_store = Path(document_store)
    manifest_path = Path(manifest_path)

    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    current_hashes = {str(path): _file_hash(path) for path in files}
    previous = load_manifest(manifest_path)
    changed, deleted = changed_sources(current_hashes, previous)

    previous_documents = _read_documents(document_store)
    retained_documents = [
        document
        for document in previous_documents
        if document.source not in changed and document.source not in deleted
    ]

    new_documents: list[Document] = []
    changed_lookup = {str(path): path for path in files if str(path) in changed}
    for source_name in sorted(changed_lookup):
        new_documents.extend(load_document(changed_lookup[source_name]))

    all_documents = sorted(
        retained_documents + new_documents,
        key=lambda document: (document.source, document.page or 0, document.doc_hash),
    )
    canonical_documents, removed = deduplicate(all_documents)

    fingerprint = _chunk_fingerprint(config, codec, tokenizer_id)
    previous_chunks = _read_chunks(output_chunks)
    reusable_by_hash: dict[str, list[Chunk]] = {}
    if previous.chunk_config_hash == fingerprint:
        canonical_hashes = {document.doc_hash for document in canonical_documents}
        for chunk in previous_chunks:
            if chunk.doc_hash in canonical_hashes:
                reusable_by_hash.setdefault(chunk.doc_hash, []).append(chunk)

    chunks: list[Chunk] = []
    reused = 0
    rebuilt = 0
    for document in canonical_documents:
        reusable = sorted(
            reusable_by_hash.get(document.doc_hash, []),
            key=lambda chunk: chunk.position,
        )
        if reusable:
            chunks.extend(reusable)
            reused += len(reusable)
            continue
        generated = chunk_document(document, config, codec=codec)
        chunks.extend(generated)
        rebuilt += len(generated)

    chunks.sort(key=lambda chunk: (chunk.source, chunk.page or 0, chunk.position, chunk.id))
    _atomic_jsonl(
        document_store,
        [document.model_dump_json() for document in canonical_documents],
    )
    _atomic_jsonl(output_chunks, [chunk.model_dump_json() for chunk in chunks])
    save_manifest(
        IngestManifest(
            files=current_hashes,
            chunk_config_hash=fingerprint,
        ),
        manifest_path,
    )

    return IncrementalIngestResult(
        files_seen=len(files),
        changed_files=len(changed),
        deleted_files=len(deleted),
        documents=len(canonical_documents),
        deduplicated_documents=len(removed),
        chunks=len(chunks),
        reused_chunks=reused,
        rebuilt_chunks=rebuilt,
    )
