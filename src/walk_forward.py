"""Walk-forward stock selection using only residuals available at each date."""

import numpy as np
import pandas as pd


def calculate_zscores(
    residuals: pd.DataFrame,
    zscore_window: int,
) -> pd.DataFrame:
    """Calculate the past-only cumulative-residual Z-scores used by the signal."""
    cumulative_residuals = residuals.cumsum()
    rolling_mean = cumulative_residuals.rolling(zscore_window).mean()
    rolling_std = cumulative_residuals.rolling(zscore_window).std()
    return (cumulative_residuals - rolling_mean) / rolling_std


def positions_from_zscores(
    zscores: pd.DataFrame,
    eligibility: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.DataFrame:
    """Use eligibility for entries while allowing open trades to reach their exit.

    A stock must be eligible to open or reverse into a new trade.  If it later
    becomes ineligible, an existing position is retained until its Z-score
    reaches the ordinary exit band or reaches the opposite entry threshold.
    """
    positions = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)

    for ticker in zscores:
        position = 0.0
        for date, zscore in zscores[ticker].items():
            if pd.isna(zscore):
                positions.at[date, ticker] = position
                continue
            if position == 0:
                if eligibility.at[date, ticker] and zscore > entry_threshold:
                    position = -1.0
                elif eligibility.at[date, ticker] and zscore < -entry_threshold:
                    position = 1.0
            elif position == 1:
                if zscore > entry_threshold:
                    position = -1.0 if eligibility.at[date, ticker] else 0.0
                elif abs(zscore) < exit_threshold:
                    position = 0.0
            else:
                if zscore < -entry_threshold:
                    position = 1.0 if eligibility.at[date, ticker] else 0.0
                elif abs(zscore) < exit_threshold:
                    position = 0.0
            positions.at[date, ticker] = position

    return positions


def walk_forward_selection(
    residuals: pd.DataFrame,
    initial_selection_days: int,
    rebalance_days: int,
    autocorr_threshold: float,
) -> pd.DataFrame:
    """Return a daily eligibility mask using strictly earlier residual history.

    At each five-day decision point, the stock rule is calculated from all
    residuals before that date.  The selected universe is then held fixed for
    the following five dates.
    """
    eligibility = pd.DataFrame(False, index=residuals.index, columns=residuals.columns)

    for decision_index in range(initial_selection_days, len(residuals), rebalance_days):
        history = residuals.iloc[:decision_index]
        selected = history.apply(lambda column: column.autocorr(lag=1)) < autocorr_threshold
        end_index = min(decision_index + rebalance_days, len(residuals))
        eligibility.iloc[decision_index:end_index] = selected.to_numpy()

    return eligibility


def walk_forward_positions(
    residuals: pd.DataFrame,
    initial_selection_days: int,
    rebalance_days: int,
    autocorr_threshold: float,
    zscore_window: int,
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run past-only selection while keeping open trades until their signal exits."""
    eligibility = walk_forward_selection(
        residuals, initial_selection_days, rebalance_days, autocorr_threshold
    )
    zscores = calculate_zscores(residuals, zscore_window)
    positions = positions_from_zscores(
        zscores, eligibility, entry_threshold, exit_threshold
    )
    return positions, eligibility
