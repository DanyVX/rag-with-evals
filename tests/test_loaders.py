from pathlib import Path

import fitz
import pytest

from ragx.ingest.loaders import UnsupportedDocumentError, load_document


def test_txt_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text(" hello\n\nworld ", encoding="utf-8")

    docs = load_document(path)

    assert len(docs) == 1
    assert docs[0].text == "hello\n\nworld"
    assert docs[0].doc_hash


def test_markdown_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\nBody", encoding="utf-8")
    docs = load_document(path)
    assert docs[0].media_type == "text/markdown"


def test_html_loader_preserves_heading_structure(tmp_path: Path) -> None:
    path = tmp_path / "sample.html"
    path.write_text(
        "<html><body><nav>menu</nav><h1>Main title</h1><h2>Section</h2>"
        "<p>Useful body text.</p></body></html>",
        encoding="utf-8",
    )
    doc = load_document(path)[0]
    assert "Main title" in doc.text
    assert "Section" in doc.text
    assert "Useful body text" in doc.text
    assert "menu" not in doc.text


def test_encoding_replacement_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "broken.txt"
    path.write_bytes(b"valid\xfftext")
    doc = load_document(path)[0]
    assert "encoding_replacement_character" in doc.warnings


def test_pdf_without_text_layer_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(path)
    doc = load_document(path)[0]
    assert "no_text_layer" in doc.warnings
    assert "empty_document" in doc.warnings
    assert doc.text == ""


def test_repeated_pdf_header_footer_are_removed(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    with fitz.open() as pdf:
        for page_number in range(1, 4):
            page = pdf.new_page(width=600, height=800)
            page.insert_text((50, 40), "ACME ANNUAL REPORT 2026")
            page.insert_text((50, 150), f"Unique body content for page {page_number}.")
            page.insert_text((50, 760), f"Page {page_number}")
        pdf.save(path)

    docs = load_document(path)
    assert len(docs) == 3
    assert all("ACME ANNUAL REPORT 2026" not in doc.text for doc in docs)
    assert all("repeated_header_footer_removed" in doc.warnings for doc in docs)
    assert all("Unique body content" in doc.text for doc in docs)


def test_unsupported_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path)
