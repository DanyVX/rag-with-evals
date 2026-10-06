from ragx.chunk.core import ChunkConfig, chunk_document
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
