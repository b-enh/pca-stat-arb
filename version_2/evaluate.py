"""Evaluate the pre-stated Version 2 extensions with a shared accounting path."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import moving_block_bootstrap_sharpes, newey_west_positive_mean_test
from portfolio import calculate_portfolio_returns
from robustness import performance_summary
from strategy import (
    StrategyConfig,
    build_strategy_variants,
    finite_pressure_zscores,
    high_dispersion_regime,
    positions_from_entry_gate,
    strongest_dislocation_gate,
    version_one_eligibility,
)


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "version_2" / "outputs"
TRANSACTION_COST = 0.0005
HAC_LAG = 5
BLOCK_SIZE = 5
N_BOOTSTRAP = 5_000
SEED = 42


def evaluate_returns(net_returns: pd.Series) -> dict[str, float]:
    summary = performance_summary(net_returns)
    hac = newey_west_positive_mean_test(net_returns, HAC_LAG)
    bootstrapped = moving_block_bootstrap_sharpes(
        net_returns, BLOCK_SIZE, N_BOOTSTRAP, SEED
    )
    return {
        **summary,
        "hac_t_statistic": hac.t_statistic,
        "hac_one_sided_p_value": hac.one_sided_p_value,
        "bootstrap_sharpe_low": np.percentile(bootstrapped, 2.5),
        "bootstrap_sharpe_high": np.percentile(bootstrapped, 97.5),
    }


def main() -> None:
    config = StrategyConfig()
    residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)
    variants = build_strategy_variants(residuals, returns, config)
    aligned_returns = returns.loc[residuals.index, residuals.columns]

    headline_rows = []
    robustness_rows = []
    for name, (positions, entry_gate) in variants.items():
        evaluation_positions = positions.iloc[config.initial_selection_days :]
        evaluation_returns = aligned_returns.iloc[config.initial_selection_days :]
        portfolio = calculate_portfolio_returns(
            evaluation_positions, evaluation_returns, TRANSACTION_COST
        )
        headline_rows.append(
            {
                "strategy": name,
                **evaluate_returns(portfolio.net_returns),
                "average_entry_eligible_stocks": entry_gate.iloc[
                    config.initial_selection_days :
                ].sum(axis=1).mean(),
                "average_daily_turnover": portfolio.turnover.mean(),
                "average_net_exposure": portfolio.net_exposure.mean(),
            }
        )

        for cost_basis_points in (0, 5, 10, 20):
            cost_portfolio = calculate_portfolio_returns(
                evaluation_positions,
                evaluation_returns,
                cost_basis_points / 10_000,
            )
            robustness_rows.append(
                {
                    "strategy": name,
                    "check": "transaction_cost",
                    "setting": f"{cost_basis_points}_bp",
                    **performance_summary(cost_portfolio.net_returns),
                }
            )

        halfway = len(portfolio.net_returns) // 2
        for label, subperiod in (
            ("first_half", portfolio.net_returns.iloc[:halfway]),
            ("second_half", portfolio.net_returns.iloc[halfway:]),
        ):
            robustness_rows.append(
                {
                    "strategy": name,
                    "check": "subperiod",
                    "setting": label,
                    **performance_summary(subperiod),
                }
            )

        for ticker in evaluation_positions.columns:
            reduced_positions = evaluation_positions.drop(columns=ticker)
            reduced_returns = evaluation_returns.drop(columns=ticker)
            reduced_portfolio = calculate_portfolio_returns(
                reduced_positions, reduced_returns, TRANSACTION_COST
            )
            robustness_rows.append(
                {
                    "strategy": name,
                    "check": "leave_one_stock_out",
                    "setting": ticker,
                    **performance_summary(reduced_portfolio.net_returns),
                }
            )

    baseline_gate = version_one_eligibility(residuals, config)
    for pressure_window in (3, 5, 10):
        for normalization_window in (40, 60, 90):
            sensitivity_zscores = finite_pressure_zscores(
                residuals, pressure_window, normalization_window
            )
            sensitivity_positions = positions_from_entry_gate(
                sensitivity_zscores,
                baseline_gate,
                config.entry_threshold,
                config.exit_threshold,
            ).iloc[config.initial_selection_days :]
            sensitivity_portfolio = calculate_portfolio_returns(
                sensitivity_positions, evaluation_returns, TRANSACTION_COST
            )
            robustness_rows.append(
                {
                    "strategy": "finite_residual_pressure",
                    "check": "parameter_neighborhood",
                    "setting": f"pressure_{pressure_window}_normalization_{normalization_window}",
                    **performance_summary(sensitivity_portfolio.net_returns),
                }
            )

    dispersion_neighborhood = {
        (40, 0.5, 3),
        (60, 0.5, 3),
        (90, 0.5, 3),
        (60, 0.4, 3),
        (60, 0.6, 3),
        (60, 0.5, 2),
        (60, 0.5, 4),
    }
    finite_zscores = finite_pressure_zscores(residuals)
    for history_window, threshold_quantile, maximum_entries in sorted(
        dispersion_neighborhood
    ):
        regime_gate = high_dispersion_regime(
            finite_zscores, history_window, threshold_quantile
        )
        strongest_gate = strongest_dislocation_gate(
            finite_zscores,
            baseline_gate,
            config.entry_threshold,
            maximum_entries,
        )
        neighborhood_positions = positions_from_entry_gate(
            finite_zscores,
            regime_gate & strongest_gate,
            config.entry_threshold,
            config.exit_threshold,
        ).iloc[config.initial_selection_days :]
        neighborhood_portfolio = calculate_portfolio_returns(
            neighborhood_positions, evaluation_returns, TRANSACTION_COST
        )
        robustness_rows.append(
            {
                "strategy": "finite_pressure_dispersion_strongest",
                "check": "dispersion_neighborhood",
                "setting": (
                    f"history_{history_window}_quantile_{threshold_quantile:.1f}"
                    f"_entries_{maximum_entries}"
                ),
                **performance_summary(neighborhood_portfolio.net_returns),
            }
        )

    best_regime_gate = high_dispersion_regime(finite_zscores)
    best_strongest_gate = strongest_dislocation_gate(
        finite_zscores,
        baseline_gate,
        config.entry_threshold,
        maximum_entries=3,
    )
    for cooldown_days in (0, 1, 2, 3, 5):
        execution_positions = positions_from_entry_gate(
            finite_zscores,
            best_regime_gate & best_strongest_gate,
            config.entry_threshold,
            config.exit_threshold,
            allow_direct_reversal=False,
            cooldown_days=cooldown_days,
        ).iloc[config.initial_selection_days :]
        execution_portfolio = calculate_portfolio_returns(
            execution_positions, evaluation_returns, TRANSACTION_COST
        )
        robustness_rows.append(
            {
                "strategy": "dispersion_strongest_no_direct_reversal",
                "check": "execution_neighborhood",
                "setting": f"cooldown_{cooldown_days}",
                **performance_summary(execution_portfolio.net_returns),
            }
        )

    headline = pd.DataFrame(headline_rows).set_index("strategy")
    robustness = pd.DataFrame(robustness_rows)
    parameter_tests = robustness["check"].isin(
        [
            "parameter_neighborhood",
            "dispersion_neighborhood",
            "execution_neighborhood",
        ]
    ).sum()
    tested_specifications = len(headline) - 1 + parameter_tests
    headline["tested_specifications"] = tested_specifications
    headline["search_adjusted_hac_p_value"] = (
        headline["hac_one_sided_p_value"] * tested_specifications
    ).clip(upper=1.0)
    OUTPUT_DIR.mkdir(exist_ok=True)
    headline.to_csv(OUTPUT_DIR / "headline_results.csv")
    robustness.to_csv(OUTPUT_DIR / "robustness_results.csv", index=False)

    display_columns = [
        "cagr",
        "sharpe",
        "max_drawdown",
        "average_daily_turnover",
        "hac_one_sided_p_value",
        "search_adjusted_hac_p_value",
        "bootstrap_sharpe_low",
        "bootstrap_sharpe_high",
    ]
    print("Version 2 extensions (historical and exploratory):")
    print(headline[display_columns].round(4).to_string())
    print("\nAll attempted variants are shown; none is confirmatory evidence of an edge.")


if __name__ == "__main__":
    main()
