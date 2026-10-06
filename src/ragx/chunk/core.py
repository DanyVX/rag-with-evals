from __future__ import annotations

import re
from hashlib import sha256
from typing import Callable, Literal

from pydantic import BaseModel, Field

from ragx.ingest.models import Document

Tokenizer = Callable[[str], list[str]]


class ChunkConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "sentence", "structure"] = "fixed"
    size: int = Field(default=256, ge=8)
    overlap: int = Field(default=32, ge=0)
    prepend_section: bool = False


class Chunk(BaseModel):
    id: str
    doc_hash: str
    source: str
    text: str
    position: int
    page: int | None = None
    section: str | None = None


def whitespace_tokenizer(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _stable_id(doc_hash: str, position: int, text: str) -> str:
    raw = f"{doc_hash}:{position}:{sha256(text.encode()).hexdigest()}"
    return sha256(raw.encode()).hexdigest()[:24]


def _fixed(text: str, size: int, overlap: int, tokenize: Tokenizer) -> list[str]:
    tokens = tokenize(text)
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    step = size - overlap
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens), step)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _sentence_window(text: str, size: int, tokenize: Tokenizer) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for sentence in _sentences(text):
        n = len(tokenize(sentence))
        if buf and count + n > size:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.append(sentence)
        count += n
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _recursive(text: str, size: int, tokenize: Tokenizer) -> list[str]:
    blocks = re.split(r"\n{2,}", text)
    out: list[str] = []
    for block in blocks:
        if len(tokenize(block)) <= size:
            if block.strip():
                out.append(block.strip())
        else:
            out.extend(_sentence_window(block, size, tokenize))
    return out


def _structure(text: str, size: int, tokenize: Tokenizer) -> list[tuple[str | None, str]]:
    current: str | None = None
    sections: list[tuple[str | None, list[str]]] = []
    body: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if body:
                sections.append((current, body))
                body = []
            current = re.sub(r"^#{1,6}\s+", "", line).strip()
        else:
            body.append(line)
    if body:
        sections.append((current, body))
    result: list[tuple[str | None, str]] = []
    for heading, lines in sections:
        for part in _recursive("\n".join(lines), size, tokenize):
            result.append((heading, part))
    return result


def chunk_document(
    document: Document,
    config: ChunkConfig,
    *,
    tokenize: Tokenizer = whitespace_tokenizer,
) -> list[Chunk]:
    if not document.text.strip():
        return []
    if config.strategy == "fixed":
        raw = [(None, x) for x in _fixed(document.text, config.size, config.overlap, tokenize)]
    elif config.strategy == "recursive":
        raw = [(None, x) for x in _recursive(document.text, config.size, tokenize)]
    elif config.strategy == "sentence":
        raw = [(None, x) for x in _sentence_window(document.text, config.size, tokenize)]
    else:
        raw = _structure(document.text, config.size, tokenize)

    chunks: list[Chunk] = []
    for pos, (section, text) in enumerate(raw):
        text = text.strip()
        if not text:
            continue
        rendered = f"{section}\n\n{text}" if config.prepend_section and section else text
        chunks.append(
            Chunk(
                id=_stable_id(document.doc_hash, pos, rendered),
                doc_hash=document.doc_hash,
                source=document.source,
                text=rendered,
                position=pos,
                page=document.page,
                section=section,
            )
        )
    return chunks
