from __future__ import annotations

import re

from pydantic import BaseModel


class CitationValidation(BaseModel):
    cited: list[int]
    invalid: list[int]
    valid: bool


def parse_citations(answer: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", answer)]


def validate_citations(answer: str, source_count: int) -> CitationValidation:
    cited = parse_citations(answer)
    invalid = sorted({i for i in cited if i < 1 or i > source_count})
    return CitationValidation(cited=cited, invalid=invalid, valid=not invalid)
