"""Small, transparent robustness summaries for a fixed backtest return series."""

import numpy as np
import pandas as pd


def performance_summary(returns: pd.Series) -> dict[str, float]:
    """Return the core performance measures without making an inference claim."""
    cumulative_value = (1 + returns).cumprod()
    annualized_mean = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe = annualized_mean / annualized_volatility if annualized_volatility else 0.0
    return {
        "observations": len(returns),
        "annualized_mean": annualized_mean,
        "cagr": cumulative_value.iloc[-1] ** (252 / len(returns)) - 1,
        "sharpe": sharpe,
        "max_drawdown": (cumulative_value / cumulative_value.cummax() - 1).min(),
    }


def selection_change_rate(eligibility: pd.DataFrame, rebalance_days: int) -> float:
    """Measure how often adjacent five-day selected-stock sets differ."""
    snapshots = eligibility.iloc[::rebalance_days]
    if len(snapshots) < 2:
        return 0.0
    changes = (snapshots.iloc[1:].to_numpy() != snapshots.iloc[:-1].to_numpy()).any(axis=1)
    return changes.mean()
