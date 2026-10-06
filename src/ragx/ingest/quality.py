from __future__ import annotations

import re


def alphabetic_ratio(text: str) -> float:
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return 0.0
    return sum(c.isalpha() for c in nonspace) / len(nonspace)


def looks_like_broken_table(text: str, *, threshold: float = 0.25) -> bool:
    if len(text) < 40:
        return False
    return alphabetic_ratio(text) < threshold and bool(re.search(r"\d", text))


def probable_language(text: str) -> str:
    """Conservative script-level language handling without mandatory language packages."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    ascii_letters = sum(c.isascii() for c in letters)
    return "en_or_latin" if ascii_letters / len(letters) > 0.85 else "non_latin_or_mixed"
