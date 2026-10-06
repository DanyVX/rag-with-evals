from __future__ import annotations

from dataclasses import dataclass


class BudgetExceededError(RuntimeError):
    """Raised before a provider call would exceed the configured budget."""


@dataclass(slots=True)
class CostTracker:
    max_spend_usd: float
    spent_usd: float = 0.0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_spend_usd - self.spent_usd)

    def reserve(self, estimated_cost_usd: float) -> None:
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd cannot be negative")
        if self.max_spend_usd <= 0:
            raise BudgetExceededError(
                "Real provider calls are disabled. Set RAGX_MAX_SPEND_USD to a positive value."
            )
        if self.spent_usd + estimated_cost_usd > self.max_spend_usd:
            raise BudgetExceededError("Call would exceed configured provider budget")

    def commit(self, actual_cost_usd: float) -> None:
        if actual_cost_usd < 0:
            raise ValueError("actual_cost_usd cannot be negative")
        if self.spent_usd + actual_cost_usd > self.max_spend_usd:
            raise BudgetExceededError("Actual provider cost exceeded configured budget")
        self.spent_usd += actual_cost_usd
