"""Reproduce Version 2's second external-universe diagnostic."""

from pathlib import Path

import pandas as pd

from common.portfolio import calculate_portfolio_returns
from common.robustness import performance_summary
from version_2.strategy import StrategyConfig, residual_correlation_centrality
from version_2.validate import OUTPUT_DIR, TRANSACTION_COST, infer


DATA_DIR = Path(__file__).resolve().parent / "data" / "external_energy_2017_2021"


def evaluate_secondary_data() -> pd.DataFrame:
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)
    residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
    returns = returns.loc[residuals.index, residuals.columns]
    config = StrategyConfig()
    positions = residual_correlation_centrality(residuals, config).iloc[config.initial_history_days :]
    portfolio = calculate_portfolio_returns(
        positions, returns.loc[positions.index, positions.columns], TRANSACTION_COST
    )
    result = pd.DataFrame([{
        "strategy": "residual_correlation_centrality",
        **performance_summary(portfolio.net_returns),
        **infer(portfolio.net_returns),
    }])
    result["passed_validation"] = False
    return result


def main() -> None:
    result = evaluate_secondary_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_DIR / "second_external_diagnostic.csv", index=False)
    print("Version 2 second external-universe diagnostic:")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
