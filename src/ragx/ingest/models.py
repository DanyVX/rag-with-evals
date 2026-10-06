from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field


class Document(BaseModel):
    source: str
    text: str
    media_type: str
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    url: str | None = None
    language: str | None = None
    warnings: list[str] = Field(default_factory=list)
    doc_hash: str

    @classmethod
    def from_text(
        cls,
        *,
        source: str | Path,
        text: str,
        media_type: str,
        **kwargs: object,
    ) -> "Document":
        normalized_source = str(source)
        digest = sha256(text.encode("utf-8")).hexdigest()
        return cls(
            source=normalized_source,
            text=text,
            media_type=media_type,
            doc_hash=digest,
            **kwargs,
        )
