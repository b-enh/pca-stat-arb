"""Stable Version 1 interface backed by the cleaned research implementation."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from walk_forward import walk_forward_positions as _walk_forward_positions


@dataclass(frozen=True)
class StrategyConfig:
    initial_selection_days: int = 252
    rebalance_days: int = 5
    autocorr_threshold: float = -0.02
    zscore_window: int = 20
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5


def walk_forward_positions(
    residuals: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the finalized Version 1 entry-only walk-forward selection rules."""
    return _walk_forward_positions(
        residuals,
        config.initial_selection_days,
        config.rebalance_days,
        config.autocorr_threshold,
        config.zscore_window,
        config.entry_threshold,
        config.exit_threshold,
    )
