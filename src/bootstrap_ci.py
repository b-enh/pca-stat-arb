import pandas as pd
import numpy as np

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

# Bootstrap: resample daily returns WITH replacement, many times,
# recompute annualized Sharpe each time, to see the plausible range
N_BOOTSTRAP = 5000
sharpes = []

np.random.seed(42)
for _ in range(N_BOOTSTRAP):
    sample = strategy_returns.sample(n=len(strategy_returns), replace=True)
    ann_return = sample.mean() * 252
    ann_vol = sample.std() * np.sqrt(252)
    sharpes.append(ann_return / ann_vol)

sharpes = np.array(sharpes)

print(f"Observed Sharpe (point estimate): {strategy_returns.mean() * 252 / (strategy_returns.std() * np.sqrt(252)):.2f}")
print(f"Bootstrap mean Sharpe: {sharpes.mean():.2f}")
print(f"95% confidence interval: [{np.percentile(sharpes, 2.5):.2f}, {np.percentile(sharpes, 97.5):.2f}]")
print(f"% of bootstrap samples with Sharpe > 0: {(sharpes > 0).mean():.1%}")