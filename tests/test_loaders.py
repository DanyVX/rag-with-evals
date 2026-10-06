from pathlib import Path

import pytest

from ragx.ingest.loaders import UnsupportedDocumentError, load_document


def test_txt_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text(" hello\\n\\nworld ", encoding="utf-8")

    docs = load_document(path)

    assert len(docs) == 1
    assert docs[0].text == "hello\\n\\nworld"
    assert docs[0].doc_hash


def test_markdown_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\\nBody", encoding="utf-8")
    docs = load_document(path)
    assert docs[0].media_type == "text/markdown"


def test_unsupported_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path)
