import pandas as pd

residuals = pd.read_csv("../data/residuals.csv", index_col=0, parse_dates=True)

SPLIT_FRACTION = 0.7
split_idx = int(len(residuals) * SPLIT_FRACTION)
split_date = residuals.index[split_idx]

selection_period = residuals.iloc[:split_idx]
test_period = residuals.iloc[split_idx:]

print(f"Selection period: {selection_period.index[0].date()} to {selection_period.index[-1].date()} ({len(selection_period)} days)")
print(f"Test period: {test_period.index[0].date()} to {test_period.index[-1].date()} ({len(test_period)} days)")

autocorr = selection_period.apply(lambda col: col.autocorr(lag=1))
print("\nLag-1 autocorrelation (selection period only):")
print(autocorr.sort_values())

AUTOCORR_THRESHOLD = -0.02
selected_stocks = autocorr[autocorr < AUTOCORR_THRESHOLD].index.tolist()

print(f"\nSelected stocks (autocorr < {AUTOCORR_THRESHOLD}): {selected_stocks}")


with open("../data/universe_config.txt", "w") as f:
    f.write(f"split_date={split_date.date()}\n")
    f.write(f"selected_stocks={','.join(selected_stocks)}\n")