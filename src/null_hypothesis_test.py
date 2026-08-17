import pandas as pd
import numpy as np
from scipy import stats

positions = pd.read_csv("../data/positions.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("../data/returns.csv", index_col="Date", parse_dates=True)
returns = returns.loc[positions.index]

positions_shifted = positions.shift(1).fillna(0)
active_count = positions_shifted.abs().sum(axis=1).replace(0, np.nan)
weights = positions_shifted.div(active_count, axis=0).fillna(0)

TRANSACTION_COST = 0.0005
position_changes = positions_shifted.diff().abs()
trade_cost = (position_changes * TRANSACTION_COST).sum(axis=1)
strategy_returns = (weights * returns).sum(axis=1) - trade_cost


t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)

print(f"Mean daily return: {strategy_returns.mean():.5f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value (two-sided): {p_value:.4f}")
print(f"p-value (one-sided, since we have a directional hypothesis): {p_value/2:.4f}")

if p_value < 0.05:
    print("\nResult: reject the null at the 5% level (some evidence of a real edge)")
else:
    print("\nResult: fail to reject the null at the 5% level (cannot rule out 'no edge')")