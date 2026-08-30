"""Time-series-aware, frequentist inference helpers for offline backtests."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS_PER_YEAR = 252


@dataclass
class HacTestResult:
    """One-sided test of whether the mean daily return is positive."""

    mean_daily_return: float
    standard_error: float
    t_statistic: float
    one_sided_p_value: float


def annualized_sharpe(returns: pd.Series) -> float:
    """Calculate the usual zero-cash-rate annualized Sharpe ratio."""
    daily_volatility = returns.std(ddof=1)
    if daily_volatility == 0:
        raise ValueError("Sharpe ratio is undefined when returns have zero volatility")
    return returns.mean() * TRADING_DAYS_PER_YEAR / (
        daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def newey_west_positive_mean_test(
    returns: pd.Series, max_lag: int
) -> HacTestResult:
    """Test for a positive mean return using a Newey-West/HAC standard error.

    The adjustment allows returns close in time to be related.  The null is a
    non-positive mean return and the alternative is a positive mean return.
    """
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("At least two returns are required")
    if not 0 <= max_lag < len(values):
        raise ValueError("max_lag must be between 0 and the sample length - 1")

    centered = values - values.mean()
    long_run_variance = np.mean(centered**2)
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        autocovariance = np.mean(centered[lag:] * centered[:-lag])
        long_run_variance += 2 * weight * autocovariance

    standard_error = np.sqrt(max(long_run_variance, 0) / len(values))
    t_statistic = values.mean() / standard_error if standard_error else np.nan
    one_sided_p_value = stats.norm.sf(t_statistic)
    return HacTestResult(
        mean_daily_return=values.mean(),
        standard_error=standard_error,
        t_statistic=t_statistic,
        one_sided_p_value=one_sided_p_value,
    )


def sample_moving_blocks(
    values: np.ndarray, block_size: int, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Build one resample from consecutive blocks drawn with replacement."""
    if not 1 <= block_size <= len(values):
        raise ValueError("block_size must be between 1 and the sample length")
    starts = rng.integers(0, len(values) - block_size + 1, size=int(np.ceil(sample_size / block_size)))
    blocks = [values[start : start + block_size] for start in starts]
    return np.concatenate(blocks)[:sample_size]


def moving_block_bootstrap_sharpes(
    returns: pd.Series,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Resample consecutive return blocks and calculate a Sharpe for each sample."""
    values = returns.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = pd.Series(sample_moving_blocks(values, block_size, len(values), rng))
        sharpes.append(annualized_sharpe(sample))
    return np.array(sharpes)
