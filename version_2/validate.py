"""Reproduce Version 2's frozen external evaluation and robustness checks."""

from pathlib import Path

import numpy as np
import pandas as pd

from common.inference import moving_block_bootstrap_sharpes, newey_west_positive_mean_test
from common.portfolio import calculate_portfolio_returns
from common.robustness import performance_summary
from version_2.strategy import StrategyConfig, residual_correlation_centrality


VERSION_DIR = Path(__file__).resolve().parent
DATA_DIR = VERSION_DIR / "data" / "external_financial_2017_2021"
OUTPUT_DIR = VERSION_DIR / "outputs"
TRANSACTION_COST = 0.0005
SEARCH_ADJUSTMENT_TESTS = 77


def load_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)
    residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
    return returns.loc[residuals.index, residuals.columns], residuals


def infer(net_returns: pd.Series) -> dict[str, float]:
    hac = newey_west_positive_mean_test(net_returns, 5)
    bootstrap = moving_block_bootstrap_sharpes(net_returns, 5, 5_000, 42)
    return {
        "hac_t_statistic": hac.t_statistic,
        "hac_one_sided_p_value": hac.one_sided_p_value,
        "bootstrap_sharpe_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_sharpe_high": float(np.percentile(bootstrap, 97.5)),
        "search_adjusted_p_value": min(hac.one_sided_p_value * SEARCH_ADJUSTMENT_TESTS, 1.0),
    }


def robustness(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in (0, 5, 10, 20):
        net_returns = calculate_portfolio_returns(positions, returns, cost / 10_000).net_returns
        rows.append({"check": "transaction_cost", "setting": f"{cost}_bp", **performance_summary(net_returns)})
    base = calculate_portfolio_returns(positions, returns, TRANSACTION_COST).net_returns
    midpoint = len(base) // 2
    for label, subset in (("first_half", base.iloc[:midpoint]), ("second_half", base.iloc[midpoint:])):
        rows.append({"check": "subperiod", "setting": label, **performance_summary(subset)})
    for stock in positions.columns:
        net_returns = calculate_portfolio_returns(
            positions.drop(columns=stock), returns.drop(columns=stock), TRANSACTION_COST
        ).net_returns
        rows.append({"check": "leave_one_stock_out", "setting": stock, **performance_summary(net_returns)})
    return pd.DataFrame(rows)


def evaluate_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns, residuals = load_frozen_data()
    config = StrategyConfig()
    positions = residual_correlation_centrality(residuals, config).iloc[config.initial_history_days :]
    evaluation_returns = returns.loc[positions.index, positions.columns]
    portfolio = calculate_portfolio_returns(positions, evaluation_returns, TRANSACTION_COST)
    headline = pd.DataFrame([{
        "strategy": "residual_correlation_centrality",
        "average_daily_turnover": portfolio.turnover.mean(),
        "average_absolute_net_exposure": portfolio.net_exposure.abs().mean(),
        **performance_summary(portfolio.net_returns),
        **infer(portfolio.net_returns),
    }])
    headline["passed_validation"] = (
        (headline.search_adjusted_p_value < 0.05) & (headline.bootstrap_sharpe_low > 0)
    )
    return headline, robustness(positions, evaluation_returns)


def main() -> None:
    headline, robustness_results = evaluate_frozen_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    headline.to_csv(OUTPUT_DIR / "external_validation.csv", index=False)
    robustness_results.to_csv(OUTPUT_DIR / "robustness.csv", index=False)
    print("Version 2 frozen external evaluation:")
    print(headline.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
