"""Shared portfolio accounting for the offline research scripts.

The input positions are trading signals: +1 for long, -1 for short and 0 for
flat.  This module converts them into equal-weight portfolio holdings, applies
a one-day trading lag, and charges costs on changes in those holdings.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResults:
    """The daily holdings and returns used by a backtest or statistical test."""

    weights: pd.DataFrame
    gross_exposure: pd.Series
    net_exposure: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    trading_costs: pd.Series
    net_returns: pd.Series


def calculate_portfolio_returns(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float,
) -> PortfolioResults:
    """Calculate lagged equal-weight portfolio returns and turnover-based costs.

    A cost is charged for every unit of portfolio weight that changes.  For
    example, opening two 50% positions has total turnover of 100%, so a
    5-basis-point cost rate deducts 0.05% of the portfolio on that day.
    """
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
    if not positions.index.equals(returns.index):
        raise ValueError("positions and returns must have the same dates")
    if not positions.columns.equals(returns.columns):
        raise ValueError("positions and returns must have the same columns")

    # A signal generated today can first be held tomorrow.
    lagged_positions = positions.shift(1).fillna(0.0)

    # Equal-weight the active positions so the absolute weights add up to 100%.
    active_count = lagged_positions.abs().sum(axis=1).replace(0, np.nan)
    weights = lagged_positions.div(active_count, axis=0).fillna(0.0)

    # The first row represents moving from an empty portfolio into its weights.
    weight_changes = weights.diff().fillna(weights)
    turnover = weight_changes.abs().sum(axis=1)
    trading_costs = turnover * transaction_cost

    # Gross exposure measures total position size; net exposure measures direction.
    gross_exposure = weights.abs().sum(axis=1)
    net_exposure = weights.sum(axis=1)
    gross_returns = (weights * returns).sum(axis=1)
    net_returns = gross_returns - trading_costs

    return PortfolioResults(
        weights=weights,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        gross_returns=gross_returns,
        turnover=turnover,
        trading_costs=trading_costs,
        net_returns=net_returns,
    )
