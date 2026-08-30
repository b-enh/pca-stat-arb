"""Version 2's frozen short-term PCA residual-reversal strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters frozen before the UK external evaluation."""

    pca_window: int = 126
    explained_variance_target: float = 0.55
    pca_refit_days: int = 10
    initial_history_days: int = 252
    signal_window: int = 5
    volatility_window: int = 20
    rebalance_days: int = 20
    stocks_per_side: int = 3


def dynamic_pca_residuals(
    returns: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> pd.DataFrame:
    """Create past-only residuals using enough PCA factors to explain 55% variance."""
    residual_rows = []
    residual_dates = []
    for start in range(config.pca_window, len(returns), config.pca_refit_days):
        training = returns.iloc[start - config.pca_window : start]
        mean = training.mean()
        standard_deviation = training.std(ddof=0)
        if (standard_deviation == 0).any():
            raise ValueError("PCA training window contains a zero-volatility stock")
        standardized = (training - mean) / standard_deviation
        _, singular_values, right_vectors = np.linalg.svd(
            standardized.to_numpy(), full_matrices=False
        )
        explained_variance = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        component_count = int(
            np.searchsorted(explained_variance, config.explained_variance_target) + 1
        )
        loadings = right_vectors[:component_count].T

        for row in range(start, min(start + config.pca_refit_days, len(returns))):
            current = ((returns.iloc[row] - mean) / standard_deviation).to_numpy()
            fitted = current @ loadings @ loadings.T
            residual_rows.append(current - fitted)
            residual_dates.append(returns.index[row])

    return pd.DataFrame(
        residual_rows,
        index=residual_dates,
        columns=returns.columns,
    )


def dollar_neutral(positions: pd.DataFrame) -> pd.DataFrame:
    """Balance long and short target weights, staying flat if either side is absent."""
    longs = positions.clip(lower=0)
    shorts = -positions.clip(upper=0)
    long_total = longs.sum(axis=1).replace(0, np.nan)
    short_total = shorts.sum(axis=1).replace(0, np.nan)
    balanced = longs.div(long_total, axis=0) * 0.5 - shorts.div(short_total, axis=0) * 0.5
    return balanced.fillna(0.0)


def residual_reversal_signal(
    residuals: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> pd.DataFrame:
    """Estimate reversal from five-day residual pressure scaled by recent volatility."""
    pressure = residuals.rolling(config.signal_window).sum()
    volatility = residuals.rolling(config.volatility_window).std().replace(0, np.nan)
    return -pressure / volatility


def residual_reversal_positions(
    residuals: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> pd.DataFrame:
    """Long recent residual losers and short winners on a twenty-day schedule."""
    signal = residual_reversal_signal(residuals, config)
    positions = pd.DataFrame(0.0, index=residuals.index, columns=residuals.columns)
    for decision in range(
        config.initial_history_days, len(residuals), config.rebalance_days
    ):
        current = signal.iloc[decision].replace([np.inf, -np.inf], np.nan).dropna()
        if len(current) < 2 * config.stocks_per_side:
            continue
        longs = current.nlargest(config.stocks_per_side).index
        shorts = current.nsmallest(config.stocks_per_side).index
        holding_dates = positions.index[decision : decision + config.rebalance_days]
        positions.loc[holding_dates, longs] = 1.0
        positions.loc[holding_dates, shorts] = -1.0
    return dollar_neutral(positions)


def build_strategy(
    returns: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dynamic PCA residuals and their frozen reversal positions."""
    residuals = dynamic_pca_residuals(returns, config)
    return residuals, residual_reversal_positions(residuals, config)
