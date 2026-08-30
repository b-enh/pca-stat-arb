"""Evaluate the unchanged strategy rules with repeated past-only selection."""

from pathlib import Path

import pandas as pd

from common.portfolio import calculate_portfolio_returns
from version_1.strategy import StrategyConfig, walk_forward_positions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# These are fixed before viewing walk-forward performance; they are not tuned here.
INITIAL_SELECTION_DAYS = 252
REBALANCE_DAYS = 5
AUTOCORR_THRESHOLD = -0.02
ZSCORE_WINDOW = 20
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5
TRANSACTION_COST = 0.0005

residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)

positions, eligibility = walk_forward_positions(residuals, StrategyConfig(
    INITIAL_SELECTION_DAYS, REBALANCE_DAYS, AUTOCORR_THRESHOLD,
    ZSCORE_WINDOW, ENTRY_THRESHOLD, EXIT_THRESHOLD,
))

evaluation_positions = positions.iloc[INITIAL_SELECTION_DAYS:]
evaluation_returns = returns.loc[evaluation_positions.index, evaluation_positions.columns]
portfolio = calculate_portfolio_returns(
    evaluation_positions, evaluation_returns, TRANSACTION_COST
)

net_returns = portfolio.net_returns
cumulative_value = (1 + net_returns).cumprod()
annualized_mean = net_returns.mean() * 252
annualized_volatility = net_returns.std() * (252**0.5)
sharpe = annualized_mean / annualized_volatility
cagr = cumulative_value.iloc[-1] ** (252 / len(net_returns)) - 1
max_drawdown = (cumulative_value / cumulative_value.cummax() - 1).min()

print("Walk-forward evaluation (historical and exploratory):")
print(f"Evaluation dates: {evaluation_positions.index[0].date()} to {evaluation_positions.index[-1].date()}")
print(f"Observations: {len(net_returns)}")
print(f"Average eligible stocks: {eligibility.iloc[INITIAL_SELECTION_DAYS:].sum(axis=1).mean():.2f}")
print(f"Annualized arithmetic mean return: {annualized_mean:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annualized volatility: {annualized_volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {max_drawdown:.2%}")
print(f"Average daily turnover: {portfolio.turnover.mean():.2%}")
print(f"Average net exposure: {portfolio.net_exposure.mean():.2%}")
print(
    "\nThis is a stronger historical design because each selection decision uses "
    "only earlier data. It remains exploratory because the settings were developed "
    "while examining this dataset."
)
