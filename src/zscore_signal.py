import pandas as pd
import numpy as np

residuals = pd.read_csv("../data/residuals.csv", index_col=0, parse_dates=True)


with open("../data/universe_config.txt") as f:
    config = dict(line.strip().split("=") for line in f)
split_date = pd.Timestamp(config["split_date"])
selected_stocks = config["selected_stocks"].split(",")


residuals = residuals[selected_stocks]
residuals = residuals[residuals.index > split_date]

ZSCORE_WINDOW = 20
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5

cum_residuals = residuals.cumsum()
rolling_mean = cum_residuals.rolling(ZSCORE_WINDOW).mean()
rolling_std = cum_residuals.rolling(ZSCORE_WINDOW).std()
zscore = (cum_residuals - rolling_mean) / rolling_std

positions = pd.DataFrame(0, index=zscore.index, columns=zscore.columns)

for ticker in zscore.columns:
    z = zscore[ticker]
    pos = 0
    ticker_positions = []

    for value in z:
        if pd.isna(value):
            ticker_positions.append(0)
            continue

        if pos == 0:
            if value > ENTRY_THRESHOLD:
                pos = -1
            elif value < -ENTRY_THRESHOLD:
                pos = 1
        elif pos == 1:
            if value > ENTRY_THRESHOLD:
                pos = -1
            elif abs(value) < EXIT_THRESHOLD:
                pos = 0
        elif pos == -1:
            if value < -ENTRY_THRESHOLD:
                pos = 1
            elif abs(value) < EXIT_THRESHOLD:
                pos = 0

        ticker_positions.append(pos)

    positions[ticker] = ticker_positions

positions.to_csv("../data/positions.csv")
print(f"Positions computed for {len(selected_stocks)} stocks over {len(positions)} test-period days")
print(positions.tail())