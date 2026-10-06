from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


class BudgetExceededError(RuntimeError):
    """Raised before a provider call would exceed the configured budget."""


@dataclass(slots=True)
class CostTracker:
    max_spend_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    _lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return max(0.0, self.max_spend_usd - self.spent_usd - self.reserved_usd)

    def reserve(self, estimated_cost_usd: float) -> float:
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd cannot be negative")
        with self._lock:
            if self.max_spend_usd <= 0:
                raise BudgetExceededError(
                    "Real provider calls are disabled. Set RAGX_MAX_SPEND_USD to a positive value."
                )
            if self.spent_usd + self.reserved_usd + estimated_cost_usd > self.max_spend_usd:
                raise BudgetExceededError("Call would exceed configured provider budget")
            self.reserved_usd += estimated_cost_usd
        return estimated_cost_usd

    def commit(self, actual_cost_usd: float, *, reserved_usd: float = 0.0) -> None:
        if actual_cost_usd < 0 or reserved_usd < 0:
            raise ValueError("cost values cannot be negative")
        with self._lock:
            released = min(reserved_usd, self.reserved_usd)
            other_reservations = self.reserved_usd - released
            projected_total = self.spent_usd + actual_cost_usd + other_reservations
            if projected_total > self.max_spend_usd:
                self.reserved_usd -= released
                raise BudgetExceededError("Actual provider cost exceeded configured budget")
            self.reserved_usd -= released
            self.spent_usd += actual_cost_usd

    def cancel_reservation(self, reserved_usd: float) -> None:
        if reserved_usd < 0:
            raise ValueError("reserved_usd cannot be negative")
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved_usd)
