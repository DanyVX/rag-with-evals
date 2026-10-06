from math import log2

import pytest

from ragx.eval.metrics import (
    abstention_scores,
    citation_scores,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragx.eval.stats import bootstrap_mean_ci


def test_retrieval_metrics() -> None:
    assert recall_at_k(["a", "b"], {"b", "c"}, 2) == 0.5
    assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a", "b"], {"b"}, 2) == pytest.approx(1 / log2(3))
    assert citation_scores([1, 3], 2)["precision"] == 0.5


def test_abstention_and_bootstrap_are_deterministic() -> None:
    assert abstention_scores([True, False], [True, False]) == {"precision": 1.0, "recall": 1.0}
    assert bootstrap_mean_ci([0.0, 1.0], samples=20) == bootstrap_mean_ci([0.0, 1.0], samples=20)
    with pytest.raises(ValueError):
        abstention_scores([True], [])
