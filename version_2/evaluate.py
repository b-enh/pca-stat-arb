"""Evaluate Version 2 on the original development dataset."""

from pathlib import Path

import pandas as pd

from common.portfolio import calculate_portfolio_returns
from common.robustness import performance_summary
from version_2.strategy import StrategyConfig, residual_correlation_centrality


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSACTION_COST = 0.0005


def evaluate() -> dict[str, float]:
    residuals = pd.read_csv(PROJECT_ROOT / "data" / "residuals.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(PROJECT_ROOT / "data" / "returns.csv", index_col="Date", parse_dates=True)
    returns = returns.loc[residuals.index, residuals.columns]
    config = StrategyConfig()
    positions = residual_correlation_centrality(residuals, config).iloc[config.initial_history_days :]
    portfolio = calculate_portfolio_returns(positions, returns.loc[positions.index], TRANSACTION_COST)
    return {
        "average_daily_turnover": portfolio.turnover.mean(),
        "average_absolute_net_exposure": portfolio.net_exposure.abs().mean(),
        **performance_summary(portfolio.net_returns),
    }


if __name__ == "__main__":
    print(pd.Series(evaluate()).to_string())
