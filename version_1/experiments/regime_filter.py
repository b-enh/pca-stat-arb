import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiments" / "regime_filter"

positions = pd.read_csv(DATA_DIR / "positions_full_period.csv", index_col=0, parse_dates=True)
cohesion = pd.read_csv(DATA_DIR / "cohesion.csv", index_col=0, parse_dates=True).squeeze("columns")

REGIME_PERCENTILE = 0.75


cohesion = cohesion.reindex(positions.index, method="ffill")

cohesion_full = pd.read_csv(DATA_DIR / "cohesion.csv", index_col=0, parse_dates=True).squeeze("columns")
cohesion_threshold_full = cohesion_full.expanding(min_periods=20).quantile(REGIME_PERCENTILE)
cohesion_threshold = cohesion_threshold_full.reindex(positions.index, method="ffill")

high_cohesion_regime = cohesion > cohesion_threshold

positions_filtered = positions.copy()
positions_filtered[high_cohesion_regime] = 0

positions_filtered.to_csv(DATA_DIR / "positions_filtered_full_period.csv")

print(f"Days flagged as high-cohesion regime: {high_cohesion_regime.sum()} / {len(high_cohesion_regime)}")
print(f"({high_cohesion_regime.sum() / len(high_cohesion_regime):.1%} of test-period trading days)")

overlap = (positions.abs().sum(axis=1) > 0) & high_cohesion_regime
print(f"\nDays with an open position AND flagged as high-cohesion: {overlap.sum()}")
