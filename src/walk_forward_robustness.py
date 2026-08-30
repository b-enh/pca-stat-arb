"""Robustness checks for the fixed, historical walk-forward strategy."""

from pathlib import Path

import pandas as pd

from portfolio import calculate_portfolio_returns
from robustness import performance_summary, selection_change_rate
from walk_forward import walk_forward_positions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Fixed Version 1 specification: do not tune these using the results below.
INITIAL_SELECTION_DAYS = 252
REBALANCE_DAYS = 5
AUTOCORR_THRESHOLD = -0.02
ZSCORE_WINDOW = 20
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5

residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)
positions, eligibility = walk_forward_positions(
    residuals,
    INITIAL_SELECTION_DAYS,
    REBALANCE_DAYS,
    AUTOCORR_THRESHOLD,
    ZSCORE_WINDOW,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
)
positions = positions.iloc[INITIAL_SELECTION_DAYS:]
returns = returns.loc[positions.index, positions.columns]

print("Walk-forward robustness checks (historical and exploratory):")

print("\nTransaction-cost sensitivity:")
for cost_basis_points in (0, 5, 10, 20, 30):
    portfolio = calculate_portfolio_returns(positions, returns, cost_basis_points / 10_000)
    summary = performance_summary(portfolio.net_returns)
    print(
        f"  {cost_basis_points:>2} bp: Sharpe {summary['sharpe']:.2f}, "
        f"CAGR {summary['cagr']:.2%}"
    )

base_portfolio = calculate_portfolio_returns(positions, returns, 0.0005)
halfway = len(base_portfolio.net_returns) // 2
print("\nEqual-length subperiods at 5 bp cost:")
for label, subperiod_returns in (
    ("First half", base_portfolio.net_returns.iloc[:halfway]),
    ("Second half", base_portfolio.net_returns.iloc[halfway:]),
):
    summary = performance_summary(subperiod_returns)
    print(
        f"  {label}: Sharpe {summary['sharpe']:.2f}, CAGR {summary['cagr']:.2%}, "
        f"max drawdown {summary['max_drawdown']:.2%}"
    )

print("\nLeave-one-stock-out results at 5 bp cost:")
for ticker in positions.columns:
    remaining_positions = positions.drop(columns=ticker)
    remaining_returns = returns.drop(columns=ticker)
    portfolio = calculate_portfolio_returns(remaining_positions, remaining_returns, 0.0005)
    summary = performance_summary(portfolio.net_returns)
    print(f"  Without {ticker}: Sharpe {summary['sharpe']:.2f}, CAGR {summary['cagr']:.2%}")

evaluation_eligibility = eligibility.iloc[INITIAL_SELECTION_DAYS:]
print("\nSelection stability:")
print(f"  Average eligible stocks: {evaluation_eligibility.sum(axis=1).mean():.2f}")
print(f"  Five-day rebalances that changed the eligible set: {selection_change_rate(evaluation_eligibility, REBALANCE_DAYS):.1%}")
