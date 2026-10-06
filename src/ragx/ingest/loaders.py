from __future__ import annotations

from pathlib import Path

import fitz
import trafilatura
from bs4 import BeautifulSoup

from ragx.ingest.clean import clean_text
from ragx.ingest.models import Document

_TEXT_SUFFIXES = {".txt": "text/plain", ".md": "text/markdown"}


class UnsupportedDocumentError(ValueError):
    pass


def _load_text(path: Path) -> Document:
    media_type = _TEXT_SUFFIXES[path.suffix.lower()]
    raw = path.read_text(encoding="utf-8", errors="replace")
    return Document.from_text(source=path, text=clean_text(raw), media_type=media_type)


def _load_html(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
    if not extracted:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        extracted = soup.get_text("\n")
    return Document.from_text(
        source=path,
        text=clean_text(extracted),
        media_type="text/html",
    )


def _load_pdf(path: Path) -> list[Document]:
    pdf = fitz.open(path)
    pages: list[Document] = []
    for index, page in enumerate(pdf):
        raw = page.get_text("text")
        cleaned = clean_text(raw)
        warnings: list[str] = []
        if not cleaned:
            warnings.append("no_text_layer")
        pages.append(
            Document.from_text(
                source=path,
                text=cleaned,
                media_type="application/pdf",
                page=index + 1,
                warnings=warnings,
            )
        )
    return pages


def load_document(path: str | Path) -> list[Document]:
    """Load a supported local document into one or more canonical records."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return [_load_text(path)]
    if suffix in {".html", ".htm"}:
        return [_load_html(path)]
    if suffix == ".pdf":
        return _load_pdf(path)
    raise UnsupportedDocumentError(f"Unsupported document type: {suffix or '<none>'}")
