"""Beginner-friendly diagnostics for the cumulative residual spreads we trade."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MeanReversionEstimate:
    """A simple regression summary for one cumulative residual spread."""

    beta: float
    one_sided_p_value: float
    half_life_days: float


def cumulative_test_spreads(
    residuals: pd.DataFrame, split_date: pd.Timestamp
) -> pd.DataFrame:
    """Recreate the post-split cumulative residuals used by the signal script."""
    test_residuals = residuals[residuals.index > split_date]
    return test_residuals.cumsum()


def estimate_mean_reversion(spread: pd.Series) -> MeanReversionEstimate:
    """Estimate whether deviations in a spread tend to shrink on the next day.

    The regression is: change today = intercept + beta * yesterday's level.
    A negative beta is consistent with mean reversion.  The p-value assumes
    independent daily errors, so it is an exploratory diagnostic only.
    """
    previous_level = spread.shift(1).dropna()
    daily_change = spread.diff().dropna()
    slope, _, _, two_sided_p_value, _ = stats.linregress(
        previous_level, daily_change
    )
    one_sided_p_value = two_sided_p_value / 2 if slope < 0 else 1 - two_sided_p_value / 2

    rho = 1 + slope
    half_life_days = np.nan
    if 0 < rho < 1:
        half_life_days = np.log(0.5) / np.log(rho)

    return MeanReversionEstimate(
        beta=slope,
        one_sided_p_value=one_sided_p_value,
        half_life_days=half_life_days,
    )
