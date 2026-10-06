from __future__ import annotations

import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_SPACES_RE = re.compile(r"[ \t]+")
_MANY_BLANKS_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\n(?=\w)")


def clean_text(text: str) -> str:
    """Apply conservative normalization without rewriting semantic content."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK_RE.sub("", text)
    lines = [_SPACES_RE.sub(" ", line).rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    return _MANY_BLANKS_RE.sub("\n\n", text)
