from pathlib import Path

from ragx.ingest.service import ingest_paths


def test_ingest_cleans_deduplicates_and_preserves_provenance(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    duplicate = tmp_path / "duplicate.md"
    first.write_text("Hello\u200b world\n\n\nco-\noperate", encoding="utf-8")
    duplicate.write_text("Hello world\n\ncooperate", encoding="utf-8")

    result = ingest_paths([tmp_path])

    assert len(result.documents) == 1
    assert result.documents[0].text == "Hello world\n\ncooperate"
    assert result.documents[0].source == duplicate
    assert result.duplicates_skipped == 1


def test_html_removes_boilerplate_and_empty_files_are_skipped(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    empty = tmp_path / "empty.txt"
    page.write_text(
        "<nav>menu</nav><h1>Useful title</h1><p>Useful body text.</p>", encoding="utf-8"
    )
    empty.write_text("\u200b\x00", encoding="utf-8")

    result = ingest_paths([page, empty])

    assert result.documents[0].text == "Useful title\nUseful body text."
    assert len(result.warnings) == 1
