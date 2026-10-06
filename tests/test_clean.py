from ragx.ingest.clean import clean_text


def test_clean_text_removes_zero_width_and_controls() -> None:
    assert clean_text("a\u200bb\x00c") == "abc"


def test_clean_text_repairs_simple_hyphenation() -> None:
    assert clean_text("retriev-\nal") == "retrieval"


def test_clean_text_collapses_excess_blank_lines() -> None:
    assert clean_text("a\n\n\n\nb") == "a\n\nb"
