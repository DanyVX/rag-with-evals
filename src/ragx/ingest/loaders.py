from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import fitz
import trafilatura
from bs4 import BeautifulSoup

from ragx.ingest.clean import clean_text
from ragx.ingest.models import Document
from ragx.ingest.quality import looks_like_broken_table, probable_language

_TEXT_SUFFIXES = {".txt": "text/plain", ".md": "text/markdown"}
_EXTREMELY_LONG_CHARS = 2_000_000
_VERY_LONG_LINE_CHARS = 20_000


class UnsupportedDocumentError(ValueError):
    pass


def _quality_metadata(raw: str, cleaned: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if "\ufffd" in raw:
        warnings.append("encoding_replacement_character")
    if not cleaned:
        warnings.append("empty_document")
    if len(cleaned) > _EXTREMELY_LONG_CHARS:
        warnings.append("extremely_long_document")
    if any(len(line) > _VERY_LONG_LINE_CHARS for line in cleaned.splitlines()):
        warnings.append("very_long_line")
    if looks_like_broken_table(cleaned):
        warnings.append("possible_broken_table")
    language = probable_language(cleaned)
    if language == "non_latin_or_mixed":
        warnings.append("non_english_or_mixed")
    return language, warnings


def _load_text(path: Path) -> Document:
    media_type = _TEXT_SUFFIXES[path.suffix.lower()]
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    language, warnings = _quality_metadata(raw, cleaned)
    return Document.from_text(
        source=path,
        text=cleaned,
        media_type=media_type,
        language=language,
        warnings=warnings,
    )


def _load_html(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    extracted = trafilatura.extract(
        raw,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        output_format="markdown",
    )
    warnings: list[str] = []
    if not extracted:
        warnings.append("trafilatura_fallback")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        extracted = soup.get_text("\n")
    cleaned = clean_text(extracted or "")
    language, quality = _quality_metadata(raw, cleaned)
    warnings.extend(quality)
    return Document.from_text(
        source=path,
        text=cleaned,
        media_type="text/html",
        language=language,
        warnings=warnings,
    )


def _page_text_with_layout(page: fitz.Page) -> tuple[str, list[str]]:
    blocks = [
        block
        for block in page.get_text("blocks", sort=False)
        if len(block) >= 5 and str(block[4]).strip()
    ]
    if not blocks:
        return "", ["no_text_layer"]

    width = float(page.rect.width)
    midpoint = width / 2.0
    body = [block for block in blocks if (float(block[2]) - float(block[0])) < width * 0.70]
    left = [block for block in body if float(block[2]) <= midpoint * 1.08]
    right = [block for block in body if float(block[0]) >= midpoint * 0.92]
    two_column = len(left) >= 2 and len(right) >= 2

    warnings: list[str] = []
    if two_column:
        warnings.append("two_column_layout_detected")
        spanning = [block for block in blocks if block not in body]
        top_spanning = [
            block for block in spanning if float(block[1]) < float(page.rect.height) * 0.18
        ]
        bottom_spanning = [
            block for block in spanning if float(block[1]) >= float(page.rect.height) * 0.18
        ]
        ordered = (
            sorted(top_spanning, key=lambda b: (float(b[1]), float(b[0])))
            + sorted(left, key=lambda b: (float(b[1]), float(b[0])))
            + sorted(right, key=lambda b: (float(b[1]), float(b[0])))
            + sorted(bottom_spanning, key=lambda b: (float(b[1]), float(b[0])))
        )
    else:
        ordered = sorted(blocks, key=lambda b: (float(b[1]), float(b[0])))

    text = "\n".join(str(block[4]).strip() for block in ordered if str(block[4]).strip())
    return text, warnings


def _normalize_edge_line(line: str) -> str:
    compact = re.sub(r"\s+", " ", line.strip().lower())
    compact = re.sub(r"\b\d+\b", "<num>", compact)
    return compact


def _repeated_page_furniture(page_texts: list[str]) -> set[str]:
    if len(page_texts) < 2:
        return set()
    candidates: Counter[str] = Counter()
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        edge = lines[:2] + lines[-2:]
        for line in {_normalize_edge_line(item) for item in edge if item}:
            candidates[line] += 1
    threshold = max(2, math.ceil(len(page_texts) * 0.60))
    return {line for line, count in candidates.items() if count >= threshold and line}


def _strip_page_furniture(text: str, repeated: set[str]) -> tuple[str, bool]:
    if not repeated:
        return text, False
    lines = text.splitlines()
    keep = list(lines)
    changed = False
    for index in list(range(min(2, len(lines)))) + list(
        range(max(0, len(lines) - 2), len(lines))
    ):
        if 0 <= index < len(lines) and _normalize_edge_line(lines[index]) in repeated:
            keep[index] = ""
            changed = True
    return "\n".join(keep), changed


def _load_pdf(path: Path) -> list[Document]:
    with fitz.open(path) as pdf:
        extracted = [_page_text_with_layout(page) for page in pdf]

    raw_pages = [text for text, _ in extracted]
    repeated = _repeated_page_furniture(raw_pages)
    pages: list[Document] = []
    for index, (raw, layout_warnings) in enumerate(extracted):
        stripped, removed = _strip_page_furniture(raw, repeated)
        cleaned = clean_text(stripped)
        language, quality = _quality_metadata(raw, cleaned)
        warnings = list(layout_warnings)
        warnings.extend(quality)
        if removed:
            warnings.append("repeated_header_footer_removed")
        warnings = list(dict.fromkeys(warnings))
        pages.append(
            Document.from_text(
                source=path,
                text=cleaned,
                media_type="application/pdf",
                page=index + 1,
                language=language,
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
