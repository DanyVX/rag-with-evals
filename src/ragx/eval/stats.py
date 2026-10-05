"""Seeded bootstrap confidence intervals for headline metrics."""

import random
from collections.abc import Sequence


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int = 1_000, seed: int = 0
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty metric sample")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return sum(values) / len(values), lower, upper
