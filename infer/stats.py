"""Small-sample summaries with confidence intervals (no SciPy needed)."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def bootstrap_ci(
    samples: Sequence[float], level: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    xs = list(samples)
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    lo = means[int((1 - level) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return lo, hi


def summarize(samples: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Mean, bootstrap CI of the mean, median and count."""
    xs = [float(x) for x in samples]
    lo, hi = bootstrap_ci(xs, level)
    return {
        "mean": statistics.fmean(xs) if xs else math.nan,
        "ci_low": lo,
        "ci_high": hi,
        "median": statistics.median(xs) if xs else math.nan,
        "n": len(xs),
    }


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(samples)
    if not xs:
        return math.nan
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
