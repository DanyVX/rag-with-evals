import pytest

from ragx.eval.metrics import abstention_metrics, bootstrap_ci, retrieval_metrics


def test_retrieval_metrics() -> None:
    result = retrieval_metrics(["a", "b", "c"], {"b"}, ks=(1, 3))
    assert result.recall_at_k[1] == 0.0
    assert result.hit_at_k[3] == 1.0
    assert result.mrr == pytest.approx(0.5)


def test_abstention_metrics() -> None:
    result = abstention_metrics([True, False, True], [True, False, False])
    assert result["abstention_precision"] == pytest.approx(0.5)
    assert result["abstention_recall"] == pytest.approx(1.0)


def test_bootstrap_ci_is_deterministic() -> None:
    a = bootstrap_ci([0.0, 1.0, 1.0], samples=100, seed=3)
    b = bootstrap_ci([0.0, 1.0, 1.0], samples=100, seed=3)
    assert a == b
