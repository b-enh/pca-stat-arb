"""Apply time-series-aware inference to the walk-forward return series."""

from pathlib import Path

import numpy as np
import pandas as pd

from common.inference import moving_block_bootstrap_sharpes, newey_west_positive_mean_test
from common.portfolio import calculate_portfolio_returns
from version_1.strategy import StrategyConfig, walk_forward_positions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Fixed research specification shared with walk_forward_evaluation.py.
INITIAL_SELECTION_DAYS = 252
REBALANCE_DAYS = 5
AUTOCORR_THRESHOLD = -0.02
ZSCORE_WINDOW = 20
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5
TRANSACTION_COST = 0.0005

# Five days matches the strategy's selection/rebalance interval.
HAC_MAX_LAG = 5
BOOTSTRAP_BLOCK_SIZE = 5
N_BOOTSTRAP = 5_000
SEED = 42

residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)
positions, _ = walk_forward_positions(residuals, StrategyConfig(
    INITIAL_SELECTION_DAYS, REBALANCE_DAYS, AUTOCORR_THRESHOLD,
    ZSCORE_WINDOW, ENTRY_THRESHOLD, EXIT_THRESHOLD,
))
evaluation_positions = positions.iloc[INITIAL_SELECTION_DAYS:]
evaluation_returns = returns.loc[evaluation_positions.index, evaluation_positions.columns]
net_returns = calculate_portfolio_returns(
    evaluation_positions, evaluation_returns, TRANSACTION_COST
).net_returns

hac_result = newey_west_positive_mean_test(net_returns, HAC_MAX_LAG)
bootstrap_sharpes = moving_block_bootstrap_sharpes(
    net_returns, BOOTSTRAP_BLOCK_SIZE, N_BOOTSTRAP, SEED
)

print("Walk-forward frequentist inference (historical and exploratory):")
print(f"HAC/Newey-West lag: {HAC_MAX_LAG} trading days")
print(f"Mean daily return: {hac_result.mean_daily_return:.5f}")
print(f"HAC standard error: {hac_result.standard_error:.5f}")
print(f"HAC t-statistic: {hac_result.t_statistic:.3f}")
print(f"One-sided p-value for positive mean return: {hac_result.one_sided_p_value:.4f}")
print(f"Moving-block bootstrap block size: {BOOTSTRAP_BLOCK_SIZE} trading days")
print(f"Observed Sharpe: {net_returns.mean() * 252 / (net_returns.std() * np.sqrt(252)):.2f}")
print(f"95% Sharpe confidence interval: [{np.percentile(bootstrap_sharpes, 2.5):.2f}, {np.percentile(bootstrap_sharpes, 97.5):.2f}]")
print(
    "\nThe p-value and interval are more cautious than an independent-day analysis, "
    "but they still describe historical exploratory evidence rather than a fresh test."
)
