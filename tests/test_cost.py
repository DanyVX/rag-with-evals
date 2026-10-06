import pytest

from ragx.cost import BudgetExceededError, CostTracker


def test_zero_budget_disables_calls() -> None:
    tracker = CostTracker(max_spend_usd=0)
    with pytest.raises(BudgetExceededError):
        tracker.reserve(0.01)


def test_cost_tracker_rejects_over_budget() -> None:
    tracker = CostTracker(max_spend_usd=1.0)
    tracker.commit(0.75)
    with pytest.raises(BudgetExceededError):
        tracker.reserve(0.26)


def test_cost_tracker_tracks_remaining_budget() -> None:
    tracker = CostTracker(max_spend_usd=2.0)
    tracker.commit(0.5)
    assert tracker.remaining_usd == pytest.approx(1.5)
