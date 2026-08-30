"""Version 2's fixed residual-correlation centrality strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """The fixed settings selected before the frozen external evaluation."""

    initial_history_days: int = 252
    correlation_window: int = 126
    rebalance_days: int = 20
    stocks_per_side: int = 3


def dollar_neutral(positions: pd.DataFrame) -> pd.DataFrame:
    """Balance long and short target weights, staying flat if either side is absent."""
    longs = positions.clip(lower=0)
    shorts = -positions.clip(upper=0)
    long_total = longs.sum(axis=1).replace(0, np.nan)
    short_total = shorts.sum(axis=1).replace(0, np.nan)
    balanced = longs.div(long_total, axis=0) * 0.5 - shorts.div(short_total, axis=0) * 0.5
    return balanced.fillna(0.0)


def residual_correlation_centrality(
    residuals: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> pd.DataFrame:
    """Hold residual-network peripheral stocks against central stocks."""
    positions = pd.DataFrame(0.0, index=residuals.index, columns=residuals.columns)
    for decision in range(config.initial_history_days, len(residuals), config.rebalance_days):
        history = residuals.iloc[decision - config.correlation_window : decision]
        correlation = history.corr().abs()
        centrality = (correlation.sum() - 1.0) / (len(correlation) - 1)
        longs = centrality.nsmallest(config.stocks_per_side).index
        shorts = centrality.nlargest(config.stocks_per_side).index
        holding_dates = positions.index[decision : decision + config.rebalance_days]
        positions.loc[holding_dates, longs] = 1.0
        positions.loc[holding_dates, shorts] = -1.0
    return dollar_neutral(positions)
