"""Canonical Version 1 evaluator using corrected portfolio accounting."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from common.portfolio import calculate_portfolio_returns
from common.robustness import performance_summary
from version_1.strategy import StrategyConfig, walk_forward_positions


def evaluate() -> dict[str, float]:
    config = StrategyConfig()
    residuals = pd.read_csv(
        PROJECT_ROOT / "data" / "residuals.csv", index_col=0, parse_dates=True
    )
    returns = pd.read_csv(
        PROJECT_ROOT / "data" / "returns.csv", index_col="Date", parse_dates=True
    )
    positions, _ = walk_forward_positions(residuals, config)
    positions = positions.iloc[config.initial_selection_days :]
    aligned_returns = returns.loc[positions.index, positions.columns]
    portfolio = calculate_portfolio_returns(positions, aligned_returns, 0.0005)
    return performance_summary(portfolio.net_returns)


if __name__ == "__main__":
    print(pd.Series(evaluate()).to_string())
