from pathlib import Path

import pytest

from ragx.chunk import FixedTokenChunker
from ragx.ingest.models import Document


def _document(text: str) -> Document:
    return Document(
        text=text,
        source=Path("source.txt"),
        content_hash="source-hash",
        metadata={"title": "Guide"},
    )


def test_fixed_chunking_has_overlap_and_stable_ids() -> None:
    chunker = FixedTokenChunker(size=3, overlap=1)

    first = chunker.chunk(_document("one two three four five"))
    second = chunker.chunk(_document("one two three four five"))

    assert [chunk.text for chunk in first] == ["one two three", "three four five"]
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[1].start_token == 2


def test_title_is_optional_and_invalid_overlap_is_rejected() -> None:
    chunks = FixedTokenChunker(size=3, prepend_title=True).chunk(_document("one two"))

    assert chunks[0].text == "Guide\n\none two"
    with pytest.raises(ValueError, match="smaller than size"):
        FixedTokenChunker(size=2, overlap=2)
