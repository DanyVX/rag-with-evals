import pytest

from ragx.eval.dataset import EvalItem, dataset_checksum, validate_human_subset


def test_dataset_checksum_is_order_independent() -> None:
    a = EvalItem(
        id="a",
        question="q",
        question_type="factoid",
        answerable=True,
        split="synthetic",
        gold_chunk_ids=["c"],
        gold_answer="x",
    )
    b = a.model_copy(update={"id": "b"})
    assert dataset_checksum([a, b]) == dataset_checksum([b, a])


def test_human_subset_requires_100() -> None:
    with pytest.raises(ValueError):
        validate_human_subset([])
