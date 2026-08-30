"""Evaluate frozen Version 2 rules on the three development panels."""

from pathlib import Path

import pandas as pd

from common.portfolio import calculate_portfolio_returns
from common.robustness import performance_summary
from version_2.strategy import build_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "version_2" / "outputs" / "discovery.csv"
TRANSACTION_COST = 0.0005


def load_discovery_panels() -> dict[str, pd.DataFrame]:
    paths = {
        "technology": PROJECT_ROOT / "data" / "returns.csv",
        "financial": PROJECT_ROOT / "version_2" / "data" / "discovery" / "financial_returns.csv",
        "energy": PROJECT_ROOT / "version_2" / "data" / "discovery" / "energy_returns.csv",
    }
    return {
        name: pd.read_csv(path, index_col="Date", parse_dates=True).dropna()
        for name, path in paths.items()
    }


def evaluate_panel(returns: pd.DataFrame) -> dict[str, float]:
    residuals, positions = build_strategy(returns)
    aligned_returns = returns.loc[residuals.index, residuals.columns]
    portfolio = calculate_portfolio_returns(positions, aligned_returns, TRANSACTION_COST)
    midpoint = len(portfolio.net_returns) // 2
    return {
        **performance_summary(portfolio.net_returns),
        "gross_sharpe_before_costs": performance_summary(portfolio.gross_returns)["sharpe"],
        "average_daily_turnover": portfolio.turnover.mean(),
        "average_absolute_net_exposure": portfolio.net_exposure.abs().mean(),
        "first_half_sharpe": performance_summary(portfolio.net_returns.iloc[:midpoint])["sharpe"],
        "second_half_sharpe": performance_summary(portfolio.net_returns.iloc[midpoint:])["sharpe"],
    }


def evaluate() -> pd.DataFrame:
    return pd.DataFrame(
        [{"universe": name, **evaluate_panel(returns)} for name, returns in load_discovery_panels().items()]
    )


def main() -> None:
    results = evaluate()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print("Version 2 discovery evaluation:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
