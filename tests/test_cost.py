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


def test_reservation_reduces_available_budget() -> None:
    tracker = CostTracker(max_spend_usd=1.0)
    reservation = tracker.reserve(0.6)
    assert tracker.remaining_usd == pytest.approx(0.4)
    with pytest.raises(BudgetExceededError):
        tracker.reserve(0.5)
    tracker.cancel_reservation(reservation)
    assert tracker.remaining_usd == pytest.approx(1.0)


def test_commit_preserves_other_concurrent_reservations() -> None:
    tracker = CostTracker(max_spend_usd=1.0)
    first = tracker.reserve(0.4)
    tracker.reserve(0.4)
    tracker.commit(0.3, reserved_usd=first)
    assert tracker.spent_usd == pytest.approx(0.3)
    assert tracker.reserved_usd == pytest.approx(0.4)
    assert tracker.remaining_usd == pytest.approx(0.3)
