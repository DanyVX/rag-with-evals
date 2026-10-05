"""Conservative text normalization that preserves semantic content."""

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHENATED_NEWLINE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
_SPACE = re.compile(r"[^\S\r\n]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(value: str) -> str:
    """Normalize Unicode and repair common extraction artifacts deterministically."""
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    normalized = _CONTROL.sub("", normalized)
    normalized = _HYPHENATED_NEWLINE.sub("", normalized)
    normalized = _SPACE.sub(" ", normalized)
    return _BLANKS.sub("\n\n", normalized).strip()


def is_substantive(value: str, minimum_alphabetic_characters: int = 10) -> bool:
    """Reject empty and extraction-garbage documents without guessing at content."""
    return sum(character.isalpha() for character in value) >= minimum_alphabetic_characters
