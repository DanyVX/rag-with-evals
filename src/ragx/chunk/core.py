from __future__ import annotations

import re
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field

from ragx.chunk.tokenizer import TokenCodec, WhitespaceTokenCodec
from ragx.ingest.models import Document


class ChunkConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "sentence", "structure"] = "fixed"
    size: int = Field(default=256, ge=8)
    overlap: int = Field(default=0, ge=0)
    prepend_section: bool = False


class Chunk(BaseModel):
    id: str
    doc_hash: str
    source: str
    text: str
    position: int
    page: int | None = None
    section: str | None = None
    token_count: int | None = None


def _stable_id(doc_hash: str, position: int, text: str) -> str:
    raw = f"{doc_hash}:{position}:{sha256(text.encode()).hexdigest()}"
    return sha256(raw.encode()).hexdigest()[:24]


def _fixed(text: str, size: int, overlap: int, codec: TokenCodec) -> list[str]:
    tokens = codec.encode(text)
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    if not tokens:
        return []
    step = size - overlap
    return [
        codec.decode(tokens[i : i + size]).strip()
        for i in range(0, len(tokens), step)
        if tokens[i : i + size]
    ]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if sentence.strip()
    ]


def _logical_blocks(text: str) -> list[str]:
    """Split paragraphs while keeping fenced code blocks and table rows together."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""
    backtick_fence = chr(96) * 3

    def flush() -> None:
        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not in_fence and (
            stripped.startswith(backtick_fence) or stripped.startswith("~~~")
        ):
            flush()
            in_fence = True
            fence_marker = stripped[:3]
            current.append(line)
            continue
        if in_fence:
            current.append(line)
            if stripped.startswith(fence_marker):
                in_fence = False
                flush()
            continue
        if not stripped:
            flush()
            continue
        current.append(line)

    flush()
    return blocks


def _suffix(text: str, overlap: int, codec: TokenCodec) -> str:
    if overlap <= 0:
        return ""
    tokens = codec.encode(text)
    return codec.decode(tokens[-overlap:]).strip()


def _pack_blocks(
    blocks: list[str],
    *,
    size: int,
    overlap: int,
    codec: TokenCodec,
    separator: str,
) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    output: list[str] = []
    current: list[str] = []

    def rendered(parts: list[str]) -> str:
        return separator.join(part for part in parts if part).strip()

    def flush() -> None:
        text = rendered(current)
        if text:
            output.append(text)
        current.clear()

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_count = len(codec.encode(block))
        if block_count > size:
            flush()
            segments = _fixed(block, size, overlap, codec)
            output.extend(segment for segment in segments if segment)
            continue

        candidate = rendered(current + [block])
        if current and len(codec.encode(candidate)) > size:
            previous = rendered(current)
            flush()
            carry = _suffix(previous, overlap, codec)
            if carry:
                with_carry = rendered([carry, block])
                if len(codec.encode(with_carry)) <= size:
                    current.extend([carry, block])
                    continue
            current.append(block)
        else:
            current.append(block)

    flush()
    return output


def _sentence_window(text: str, size: int, overlap: int, codec: TokenCodec) -> list[str]:
    return _pack_blocks(
        _sentences(text),
        size=size,
        overlap=overlap,
        codec=codec,
        separator=" ",
    )


def _recursive(text: str, size: int, overlap: int, codec: TokenCodec) -> list[str]:
    return _pack_blocks(
        _logical_blocks(text),
        size=size,
        overlap=overlap,
        codec=codec,
        separator="\n\n",
    )


def _structure(
    text: str,
    size: int,
    overlap: int,
    codec: TokenCodec,
) -> list[tuple[str | None, str]]:
    current_heading: str | None = None
    body: list[str] = []
    sections: list[tuple[str | None, str]] = []

    def flush_section() -> None:
        nonlocal body
        section_text = "\n".join(body).strip()
        if section_text:
            sections.append((current_heading, section_text))
        body = []

    for line in text.splitlines():
        heading_match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if heading_match:
            flush_section()
            current_heading = heading_match.group(1).strip()
        else:
            body.append(line)
    flush_section()

    result: list[tuple[str | None, str]] = []
    for heading, section_text in sections:
        parts = _recursive(section_text, size, overlap, codec)
        result.extend((heading, part) for part in parts)
    return result


def chunk_document(
    document: Document,
    config: ChunkConfig,
    *,
    codec: TokenCodec | None = None,
) -> list[Chunk]:
    if not document.text.strip():
        return []

    codec = codec or WhitespaceTokenCodec()
    size = min(config.size, codec.max_sequence_length)
    if config.overlap >= size:
        raise ValueError(
            f"overlap ({config.overlap}) must be smaller than effective chunk size ({size})"
        )

    if config.strategy == "fixed":
        raw = [(None, part) for part in _fixed(document.text, size, config.overlap, codec)]
    elif config.strategy == "recursive":
        raw = [
            (None, part)
            for part in _recursive(document.text, size, config.overlap, codec)
        ]
    elif config.strategy == "sentence":
        raw = [
            (None, part)
            for part in _sentence_window(document.text, size, config.overlap, codec)
        ]
    else:
        raw = _structure(document.text, size, config.overlap, codec)

    rendered_parts: list[tuple[str | None, str]] = []
    for section, text in raw:
        text = text.strip()
        if not text:
            continue
        rendered = f"{section}\n\n{text}" if config.prepend_section and section else text
        if len(codec.encode(rendered)) <= codec.max_sequence_length:
            rendered_parts.append((section, rendered))
            continue
        for split in _fixed(rendered, codec.max_sequence_length, 0, codec):
            if split:
                rendered_parts.append((section, split))

    chunks: list[Chunk] = []
    for position, (section, text) in enumerate(rendered_parts):
        count = len(codec.encode(text))
        if count > codec.max_sequence_length:
            raise RuntimeError("chunk exceeds tokenizer max sequence length")
        chunks.append(
            Chunk(
                id=_stable_id(document.doc_hash, position, text),
                doc_hash=document.doc_hash,
                source=document.source,
                text=text,
                position=position,
                page=document.page,
                section=section,
                token_count=count,
            )
        )
    return chunks
