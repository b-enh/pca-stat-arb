"""Reproduce Version 2's frozen UK external evaluation and robustness checks."""

from pathlib import Path

import numpy as np
import pandas as pd

from common.inference import moving_block_bootstrap_sharpes, newey_west_positive_mean_test
from common.portfolio import calculate_portfolio_returns
from common.robustness import performance_summary
from version_2.strategy import build_strategy


VERSION_DIR = Path(__file__).resolve().parent
DATA_DIR = VERSION_DIR / "data" / "external_uk_2017_2021"
OUTPUT_DIR = VERSION_DIR / "outputs"
TRANSACTION_COST = 0.0005
SEARCH_ADJUSTMENT_TESTS = 360


def load_frozen_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True).dropna()


def infer(net_returns: pd.Series) -> dict[str, float]:
    hac = newey_west_positive_mean_test(net_returns, 5)
    bootstrap = moving_block_bootstrap_sharpes(net_returns, 5, 5_000, 42)
    return {
        "hac_t_statistic": hac.t_statistic,
        "hac_one_sided_p_value": hac.one_sided_p_value,
        "bootstrap_sharpe_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_sharpe_high": float(np.percentile(bootstrap, 97.5)),
        "search_adjusted_p_value": min(
            hac.one_sided_p_value * SEARCH_ADJUSTMENT_TESTS, 1.0
        ),
    }


def strategy_portfolio(returns: pd.DataFrame, cost: float = TRANSACTION_COST):
    residuals, positions = build_strategy(returns)
    aligned_returns = returns.loc[residuals.index, residuals.columns]
    return positions, aligned_returns, calculate_portfolio_returns(
        positions, aligned_returns, cost
    )


def robustness(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in (0, 5, 10, 20):
        _, _, portfolio = strategy_portfolio(returns, cost / 10_000)
        rows.append(
            {
                "check": "transaction_cost",
                "setting": f"{cost}_bp",
                **performance_summary(portfolio.net_returns),
            }
        )

    _, _, base_portfolio = strategy_portfolio(returns)
    midpoint = len(base_portfolio.net_returns) // 2
    for label, subset in (
        ("first_half", base_portfolio.net_returns.iloc[:midpoint]),
        ("second_half", base_portfolio.net_returns.iloc[midpoint:]),
    ):
        rows.append(
            {"check": "subperiod", "setting": label, **performance_summary(subset)}
        )

    for stock in returns.columns:
        _, _, portfolio = strategy_portfolio(returns.drop(columns=stock))
        rows.append(
            {
                "check": "leave_one_stock_out",
                "setting": stock,
                **performance_summary(portfolio.net_returns),
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = load_frozen_data()
    _, _, portfolio = strategy_portfolio(returns)
    midpoint = len(portfolio.net_returns) // 2
    headline = pd.DataFrame(
        [
            {
                "strategy": "short_term_pca_residual_reversal",
                "universe": "uk_large_cap_2017_2021",
                **performance_summary(portfolio.net_returns),
                "gross_sharpe_before_costs": performance_summary(
                    portfolio.gross_returns
                )["sharpe"],
                "average_daily_turnover": portfolio.turnover.mean(),
                "average_absolute_net_exposure": portfolio.net_exposure.abs().mean(),
                "first_half_sharpe": performance_summary(
                    portfolio.net_returns.iloc[:midpoint]
                )["sharpe"],
                "second_half_sharpe": performance_summary(
                    portfolio.net_returns.iloc[midpoint:]
                )["sharpe"],
                **infer(portfolio.net_returns),
            }
        ]
    )
    headline["passed_validation"] = (
        (headline.search_adjusted_p_value < 0.05)
        & (headline.bootstrap_sharpe_low > 0)
    )
    return headline, robustness(returns)


def main() -> None:
    headline, robustness_results = evaluate_frozen_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    headline.to_csv(OUTPUT_DIR / "external_validation.csv", index=False)
    robustness_results.to_csv(OUTPUT_DIR / "robustness.csv", index=False)
    print("Version 2 frozen UK external evaluation:")
    print(headline.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
