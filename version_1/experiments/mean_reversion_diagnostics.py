"""Explore whether the cumulative residual spreads used by the strategy revert."""

from pathlib import Path

import pandas as pd

from version_1.experiments.mean_reversion import (
    cumulative_test_spreads,
    estimate_mean_reversion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STATIC_SPLIT_DIR = DATA_DIR / "experiments" / "static_split"


def read_universe_config() -> tuple[pd.Timestamp, list[str]]:
    with open(STATIC_SPLIT_DIR / "universe_config.txt") as config_file:
        config = dict(line.strip().split("=") for line in config_file)
    return pd.Timestamp(config["split_date"]), config["selected_stocks"].split(",")


residuals = pd.read_csv(DATA_DIR / "residuals.csv", index_col=0, parse_dates=True)
split_date, selected_stocks = read_universe_config()
spreads = cumulative_test_spreads(residuals[selected_stocks], split_date)

rows = []
for ticker in selected_stocks:
    estimate = estimate_mean_reversion(spreads[ticker])
    rows.append(
        {
            "ticker": ticker,
            "mean_reversion_beta": estimate.beta,
            "exploratory_one_sided_p_value": estimate.one_sided_p_value,
            "estimated_half_life_days": estimate.half_life_days,
        }
    )

results = pd.DataFrame(rows).set_index("ticker")
print("Cumulative residual-spread diagnostic (historical and exploratory):")
print(results.round(4).to_string())
print(
    "\nA negative beta is consistent with mean reversion. The p-values assume "
    "independent daily errors, so they are not final proof of an edge."
)
