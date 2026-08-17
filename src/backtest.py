import pandas as pd
import numpy as np
import sys

positions_file = sys.argv[1] if len(sys.argv) > 1 else "../data/positions.csv"

returns = pd.read_csv("../data/returns.csv", index_col="Date", parse_dates=True)
positions = pd.read_csv(positions_file, index_col=0, parse_dates=True)

returns = returns.loc[positions.index]
positions_shifted = positions.shift(1).fillna(0)

active_count = positions_shifted.abs().sum(axis=1).replace(0, np.nan)
weights = positions_shifted.div(active_count, axis=0).fillna(0)

TRANSACTION_COST = 0.0005  # 5 basis points per trade

position_changes = positions_shifted.diff().abs()
trade_cost = (position_changes * TRANSACTION_COST).sum(axis=1)

strategy_returns_gross = (weights * returns).sum(axis=1)
strategy_returns_net = strategy_returns_gross - trade_cost

cumulative_gross = (1 + strategy_returns_gross).cumprod()
cumulative_net = (1 + strategy_returns_net).cumprod()

def print_stats(rets, cum, label):
    ann_return = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"\n--- {label} ---")
    print(f"Annualized return: {ann_return:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
    print(f"Final value of $1: ${cum.iloc[-1]:.2f}")

print(f"Backtest for: {positions_file}")
print_stats(strategy_returns_gross, cumulative_gross, "BEFORE transaction costs")
print_stats(strategy_returns_net, cumulative_net, "AFTER transaction costs")
print(f"\nTotal cost drag (annualized): {(strategy_returns_gross.mean() - strategy_returns_net.mean()) * 252:.2%}")