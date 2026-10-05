"""A conservative fixed-token baseline with stable chunk identities."""

import re
from dataclasses import dataclass
from hashlib import sha256

from ragx.ingest.models import Document

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    document_hash: str
    start_token: int
    end_token: int
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class FixedTokenChunker:
    """Whitespace-token baseline; swap in a model tokenizer for experimental runs."""

    size: int = 256
    overlap: int = 0
    prepend_title: bool = False

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be at least one token")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be non-negative and smaller than size")

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = _TOKEN.findall(document.text)
        output: list[Chunk] = []
        step = self.size - self.overlap
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.size]
            if not window:
                break
            body = " ".join(window)
            title = document.metadata.get("title")
            text = f"{title}\n\n{body}" if self.prepend_title and title else body
            digest = sha256(f"{document.content_hash}:{start}:{body}".encode()).hexdigest()
            output.append(
                Chunk(
                    id=digest,
                    text=text,
                    document_hash=document.content_hash,
                    start_token=start,
                    end_token=start + len(window),
                    metadata=document.metadata.copy(),
                )
            )
            if start + self.size >= len(tokens):
                break
        return output
