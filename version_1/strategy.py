"""Version 1's complete, past-only signal and stock-selection rules."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """The fixed Version 1 settings used for its historical evaluation."""

    initial_selection_days: int = 252
    rebalance_days: int = 5
    autocorr_threshold: float = -0.02
    zscore_window: int = 20
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5


def calculate_zscores(residuals: pd.DataFrame, zscore_window: int) -> pd.DataFrame:
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
    """Use eligibility for entries while letting existing positions reach an exit."""
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
    """Select stocks using only residuals available before each decision date."""
    eligibility = pd.DataFrame(False, index=residuals.index, columns=residuals.columns)
    for decision_index in range(initial_selection_days, len(residuals), rebalance_days):
        history = residuals.iloc[:decision_index]
        selected = history.apply(lambda column: column.autocorr(lag=1)) < autocorr_threshold
        end_index = min(decision_index + rebalance_days, len(residuals))
        eligibility.iloc[decision_index:end_index] = selected.to_numpy()
    return eligibility


def walk_forward_positions(
    residuals: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Version 1's past-only selection and signal rules."""
    eligibility = walk_forward_selection(
        residuals,
        config.initial_selection_days,
        config.rebalance_days,
        config.autocorr_threshold,
    )
    zscores = calculate_zscores(residuals, config.zscore_window)
    positions = positions_from_zscores(
        zscores, eligibility, config.entry_threshold, config.exit_threshold
    )
    return positions, eligibility
