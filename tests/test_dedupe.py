from ragx.ingest.dedupe import deduplicate
from ragx.ingest.models import Document


def _doc(source: str, text: str) -> Document:
    return Document.from_text(source=source, text=text, media_type="text/plain")


def test_exact_duplicates_removed() -> None:
    kept, removed = deduplicate([_doc("a", "same"), _doc("b", "same")])
    assert len(kept) == 1
    assert removed[0][2] == "exact"


def test_near_duplicates_removed() -> None:
    base = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    near = base + " extra"
    kept, removed = deduplicate(
        [_doc("a", base), _doc("b", near)],
        near_duplicate_threshold=0.7,
    )
    assert len(kept) == 1
    assert removed[0][2] == "near"
