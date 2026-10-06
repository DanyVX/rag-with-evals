from ragx.chunk.core import ChunkConfig, chunk_document
from ragx.chunk.tokenizer import WhitespaceTokenCodec
from ragx.ingest.models import Document


def _doc(text: str) -> Document:
    return Document.from_text(source="x.md", text=text, media_type="text/markdown")


def test_chunk_ids_are_stable() -> None:
    doc = _doc(
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen sixteen"
    )
    config = ChunkConfig(strategy="fixed", size=8, overlap=1)
    a = chunk_document(doc, config)
    b = chunk_document(doc, config)
    assert [x.id for x in a] == [x.id for x in b]


def test_structure_chunking_preserves_heading() -> None:
    doc = _doc("# Intro\nAlpha beta gamma.\n\n# Next\nDelta epsilon.")
    chunks = chunk_document(doc, ChunkConfig(strategy="structure", size=20))
    assert chunks[0].section == "Intro"


def test_model_max_sequence_length_is_never_exceeded() -> None:
    doc = _doc("one two three four five six seven eight nine ten eleven twelve")
    codec = WhitespaceTokenCodec(max_sequence_length=5)
    chunks = chunk_document(
        doc,
        ChunkConfig(strategy="fixed", size=10, overlap=1),
        codec=codec,
    )
    assert chunks
    assert all(chunk.token_count is not None and chunk.token_count <= 5 for chunk in chunks)


def test_recursive_chunking_keeps_small_fenced_code_block_intact() -> None:
    fence = chr(96) * 3
    text = (
        "Intro paragraph.\n\n"
        + fence
        + "python\n"
        + "def hello():\n    return 'world'\n"
        + fence
        + "\n\nEnding paragraph."
    )
    chunks = chunk_document(
        _doc(text),
        ChunkConfig(strategy="recursive", size=30, overlap=2),
        codec=WhitespaceTokenCodec(max_sequence_length=30),
    )
    assert any("def hello()" in chunk.text and "return 'world'" in chunk.text for chunk in chunks)


def test_sentence_overlap_is_applied_without_exceeding_limit() -> None:
    doc = _doc(
        "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    )
    codec = WhitespaceTokenCodec(max_sequence_length=6)
    chunks = chunk_document(
        doc,
        ChunkConfig(strategy="sentence", size=6, overlap=2),
        codec=codec,
    )
    assert len(chunks) >= 2
    assert all(len(codec.encode(chunk.text)) <= 6 for chunk in chunks)
