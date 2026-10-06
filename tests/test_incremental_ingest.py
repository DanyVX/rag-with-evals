from pathlib import Path

from ragx.chunk.core import ChunkConfig
from ragx.chunk.tokenizer import WhitespaceTokenCodec
from ragx.ingest.incremental import incremental_ingest_directory


def run_ingest(source: Path, root: Path):
    return incremental_ingest_directory(
        source,
        output_chunks=root / "chunks.jsonl",
        document_store=root / "documents.jsonl",
        manifest_path=root / "manifest.json",
        config=ChunkConfig(strategy="fixed", size=8, overlap=2),
        codec=WhitespaceTokenCodec(max_sequence_length=32),
        tokenizer_id="test-whitespace-v1",
    )


def test_unchanged_second_run_reuses_chunks(tmp_path: Path) -> None:
    source = tmp_path / "src"
    state = tmp_path / "state"
    source.mkdir()
    (source / "a.txt").write_text(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        encoding="utf-8",
    )

    first = run_ingest(source, state)
    second = run_ingest(source, state)

    assert first.changed_files == 1
    assert first.rebuilt_chunks > 0
    assert second.changed_files == 0
    assert second.rebuilt_chunks == 0
    assert second.reused_chunks == first.chunks


def test_changed_file_rebuilds_only_changed_document(tmp_path: Path) -> None:
    source = tmp_path / "src"
    state = tmp_path / "state"
    source.mkdir()
    (source / "a.txt").write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")
    (source / "b.txt").write_text("one two three four five six", encoding="utf-8")
    first = run_ingest(source, state)

    (source / "b.txt").write_text("one two three four five six seven changed", encoding="utf-8")
    second = run_ingest(source, state)

    assert first.changed_files == 2
    assert second.changed_files == 1
    assert second.reused_chunks > 0
    assert second.rebuilt_chunks > 0


def test_deleted_source_is_tombstoned_from_chunks(tmp_path: Path) -> None:
    source = tmp_path / "src"
    state = tmp_path / "state"
    source.mkdir()
    a = source / "a.txt"
    b = source / "b.txt"
    a.write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")
    b.write_text("one two three four five six", encoding="utf-8")
    run_ingest(source, state)

    b.unlink()
    result = run_ingest(source, state)
    text = (state / "chunks.jsonl").read_text(encoding="utf-8")

    assert result.deleted_files == 1
    assert str(b) not in text
    assert str(a) in text


def test_chunk_config_change_invalidates_chunk_reuse(tmp_path: Path) -> None:
    source = tmp_path / "src"
    state = tmp_path / "state"
    source.mkdir()
    (source / "a.txt").write_text(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        encoding="utf-8",
    )
    run_ingest(source, state)

    result = incremental_ingest_directory(
        source,
        output_chunks=state / "chunks.jsonl",
        document_store=state / "documents.jsonl",
        manifest_path=state / "manifest.json",
        config=ChunkConfig(strategy="fixed", size=10, overlap=2),
        codec=WhitespaceTokenCodec(max_sequence_length=32),
        tokenizer_id="test-whitespace-v1",
    )
    assert result.changed_files == 0
    assert result.reused_chunks == 0
    assert result.rebuilt_chunks > 0
