"""Theory-led extensions of the Version 1 PCA residual strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    initial_selection_days: int = 252
    rebalance_days: int = 5
    autocorr_threshold: float = -0.02
    zscore_window: int = 20
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5
    mean_reversion_lookback: int = 120
    minimum_half_life: float = 2.0
    maximum_half_life: float = 30.0
    cohesion_lookback: int = 60
    cohesion_history: int = 60


def calculate_zscores(residuals: pd.DataFrame, window: int) -> pd.DataFrame:
    """Copy the Version 1 cumulative-residual signal without changing it."""
    spreads = residuals.cumsum()
    return (spreads - spreads.rolling(window).mean()) / spreads.rolling(window).std()


def finite_pressure_zscores(
    residuals: pd.DataFrame, pressure_window: int = 5, normalization_window: int = 60
) -> pd.DataFrame:
    """Measure a recent residual imbalance without accumulating it indefinitely."""
    pressure = residuals.rolling(pressure_window).sum()
    rolling_mean = pressure.rolling(normalization_window).mean()
    rolling_std = pressure.rolling(normalization_window).std()
    return (pressure - rolling_mean) / rolling_std


def version_one_eligibility(
    residuals: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """Reproduce the Version 1 past-only lag-one autocorrelation gate."""
    eligibility = pd.DataFrame(False, index=residuals.index, columns=residuals.columns)
    for decision in range(
        config.initial_selection_days, len(residuals), config.rebalance_days
    ):
        history = residuals.iloc[:decision]
        selected = history.apply(lambda series: series.autocorr(lag=1)) < config.autocorr_threshold
        end = min(decision + config.rebalance_days, len(residuals))
        eligibility.iloc[decision:end] = selected.to_numpy()
    return eligibility


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate an AR(1)-style half-life from spread changes and lagged levels."""
    clean = spread.dropna()
    if len(clean) < 3 or clean.shift(1).iloc[1:].std() == 0:
        return np.nan
    lagged_level = clean.shift(1).iloc[1:].to_numpy()
    change = clean.diff().iloc[1:].to_numpy()
    slope = np.polyfit(lagged_level, change, 1)[0]
    persistence = 1 + slope
    if not 0 < persistence < 1:
        return np.nan
    return float(np.log(0.5) / np.log(persistence))


def half_life_eligibility(
    residuals: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """Gate entries using the estimated speed of the spread actually traded."""
    eligibility = pd.DataFrame(False, index=residuals.index, columns=residuals.columns)
    for decision in range(
        config.initial_selection_days, len(residuals), config.rebalance_days
    ):
        history = residuals.iloc[:decision].tail(config.mean_reversion_lookback)
        local_spreads = history.cumsum()
        half_lives = local_spreads.apply(estimate_half_life)
        selected = half_lives.between(
            config.minimum_half_life, config.maximum_half_life, inclusive="both"
        )
        end = min(decision + config.rebalance_days, len(residuals))
        eligibility.iloc[decision:end] = selected.to_numpy()
    return eligibility


def turning_confirmation(zscores: pd.DataFrame) -> pd.DataFrame:
    """Confirm an extreme has started moving toward zero before allowing entry."""
    previous = zscores.shift(1)
    same_side = zscores * previous > 0
    moving_toward_zero = zscores.abs() < previous.abs()
    return (same_side & moving_toward_zero).fillna(False)


def persistent_extreme_gate(
    zscores: pd.DataFrame, entry_threshold: float
) -> pd.DataFrame:
    """Require an extreme to persist for two days in the same direction."""
    previous = zscores.shift(1)
    both_extreme = (zscores.abs() > entry_threshold) & (
        previous.abs() > entry_threshold
    )
    same_direction = zscores * previous > 0
    return (both_extreme & same_direction).fillna(False)


def strongest_dislocation_gate(
    zscores: pd.DataFrame,
    eligibility: pd.DataFrame,
    entry_threshold: float,
    maximum_entries: int,
) -> pd.DataFrame:
    """Allow only the strongest eligible cross-sectional dislocations each day."""
    candidates = eligibility & (zscores.abs() > entry_threshold)
    strengths = zscores.abs().where(candidates)
    ranks = strengths.rank(axis=1, method="first", ascending=False)
    return candidates & (ranks <= maximum_entries)


def balanced_dislocation_gate(
    zscores: pd.DataFrame,
    eligibility: pd.DataFrame,
    entry_threshold: float,
    entries_per_side: int = 1,
) -> pd.DataFrame:
    """Select equally many strongest positive and negative dislocations."""
    candidates = eligibility & (zscores.abs() > entry_threshold)
    positive_strength = zscores.where(candidates & (zscores > 0))
    negative_strength = (-zscores).where(candidates & (zscores < 0))
    positive_rank = positive_strength.rank(axis=1, method="first", ascending=False)
    negative_rank = negative_strength.rank(axis=1, method="first", ascending=False)
    return (positive_rank <= entries_per_side) | (negative_rank <= entries_per_side)


def residual_reversal_gate(
    zscores: pd.DataFrame, residuals: pd.DataFrame
) -> pd.DataFrame:
    """Enter only when today's residual pushes against the extreme pressure."""
    return (zscores * residuals < 0).fillna(False)


def high_dispersion_regime(
    zscores: pd.DataFrame, history_window: int = 60, threshold_quantile: float = 0.5
) -> pd.DataFrame:
    """Identify days with above-median cross-sectional signal dispersion."""
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between zero and one")
    dispersion = zscores.std(axis=1)
    threshold = (
        dispersion.shift(1)
        .rolling(history_window, min_periods=history_window)
        .quantile(threshold_quantile)
    )
    regime = (dispersion >= threshold).fillna(False)
    return pd.DataFrame(
        np.repeat(regime.to_numpy()[:, None], len(zscores.columns), axis=1),
        index=zscores.index,
        columns=zscores.columns,
    )


def multi_horizon_agreement_gate(
    short_zscores: pd.DataFrame,
    slow_zscores: pd.DataFrame,
    entry_threshold: float,
) -> pd.DataFrame:
    """Require short and slower residual-pressure signals to agree."""
    both_extreme = (short_zscores.abs() > entry_threshold) & (
        slow_zscores.abs() > entry_threshold
    )
    same_direction = short_zscores * slow_zscores > 0
    return (both_extreme & same_direction).fillna(False)


def stable_residual_gate(
    residuals: pd.DataFrame, volatility_window: int = 20
) -> pd.DataFrame:
    """Prefer stocks whose recent residual noise is below the daily median."""
    residual_volatility = residuals.rolling(volatility_window).std()
    daily_median = residual_volatility.median(axis=1)
    return residual_volatility.le(daily_median, axis=0).fillna(False)


def first_component_share(past_returns: pd.DataFrame) -> float:
    """Return the variance share explained by PC1 in a past-only return window."""
    standard_deviation = past_returns.std(ddof=0).replace(0, np.nan)
    standardized = ((past_returns - past_returns.mean()) / standard_deviation).dropna(axis=1)
    if standardized.empty:
        return np.nan
    singular_values = np.linalg.svd(standardized.to_numpy(), compute_uv=False)
    variances = singular_values**2
    return float(variances[0] / variances.sum())


def cohesion_regime(
    returns: pd.DataFrame, signal_dates: pd.DatetimeIndex, config: StrategyConfig
) -> pd.Series:
    """Allow entries when past PC1 cohesion is above its own trailing median."""
    cohesion = pd.Series(np.nan, index=signal_dates, dtype=float)
    for date in signal_dates:
        past_returns = returns.loc[returns.index < date].tail(config.cohesion_lookback)
        if len(past_returns) == config.cohesion_lookback:
            cohesion.at[date] = first_component_share(past_returns)
    threshold = cohesion.shift(1).rolling(
        config.cohesion_history, min_periods=config.cohesion_history
    ).median()
    return (cohesion >= threshold).fillna(False)


def positions_from_entry_gate(
    zscores: pd.DataFrame,
    entry_gate: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    allow_direct_reversal: bool = True,
    cooldown_days: int = 0,
) -> pd.DataFrame:
    """Apply Version 1 position management while changing only the entry gate."""
    if cooldown_days < 0:
        raise ValueError("cooldown_days must be non-negative")
    positions = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)
    for ticker in zscores:
        position = 0.0
        remaining_cooldown = 0
        for date, zscore in zscores[ticker].items():
            if pd.isna(zscore):
                positions.at[date, ticker] = position
                continue
            can_enter = bool(entry_gate.at[date, ticker])
            if position == 0:
                if remaining_cooldown > 0:
                    remaining_cooldown -= 1
                elif can_enter and zscore > entry_threshold:
                    position = -1.0
                elif can_enter and zscore < -entry_threshold:
                    position = 1.0
            elif position == 1:
                if zscore > entry_threshold:
                    position = -1.0 if can_enter and allow_direct_reversal else 0.0
                    if position == 0:
                        remaining_cooldown = cooldown_days
                elif abs(zscore) < exit_threshold:
                    position = 0.0
                    remaining_cooldown = cooldown_days
            else:
                if zscore < -entry_threshold:
                    position = 1.0 if can_enter and allow_direct_reversal else 0.0
                    if position == 0:
                        remaining_cooldown = cooldown_days
                elif abs(zscore) < exit_threshold:
                    position = 0.0
                    remaining_cooldown = cooldown_days
            positions.at[date, ticker] = position
    return positions


def inverse_volatility_scaled_positions(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Give lower-volatility stocks moderately larger position magnitudes."""
    volatility = returns.rolling(volatility_window).std()
    cross_sectional_median = volatility.median(axis=1)
    scale = volatility.rdiv(cross_sectional_median, axis=0).clip(0.5, 2.0)
    return positions * scale.fillna(1.0)


def signal_strength_scaled_positions(
    positions: pd.DataFrame,
    zscores: pd.DataFrame,
    entry_threshold: float,
) -> pd.DataFrame:
    """Tilt active position magnitudes toward larger residual dislocations."""
    scale = (zscores.abs() / entry_threshold).clip(1.0, 2.0).fillna(1.0)
    return positions * scale


def build_strategy_variants(
    residuals: pd.DataFrame, returns: pd.DataFrame, config: StrategyConfig
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build the pre-stated sequence of Version 2 entry-gate extensions."""
    zscores = calculate_zscores(residuals, config.zscore_window)
    baseline_gate = version_one_eligibility(residuals, config)
    half_life_gate = half_life_eligibility(residuals, config)
    turn_gate = turning_confirmation(zscores)
    regime = cohesion_regime(returns, residuals.index, config)
    regime_gate = pd.DataFrame(
        np.repeat(regime.to_numpy()[:, None], len(residuals.columns), axis=1),
        index=residuals.index,
        columns=residuals.columns,
    )

    finite_zscores = finite_pressure_zscores(residuals)
    persistent_gate = persistent_extreme_gate(
        finite_zscores, config.entry_threshold
    )
    strongest_gate = strongest_dislocation_gate(
        finite_zscores,
        baseline_gate,
        config.entry_threshold,
        maximum_entries=3,
    )
    stable_gate = stable_residual_gate(residuals)
    balanced_gate = balanced_dislocation_gate(
        finite_zscores, baseline_gate, config.entry_threshold
    )
    reversal_gate = residual_reversal_gate(finite_zscores, residuals)
    dispersion_gate = high_dispersion_regime(finite_zscores)
    slow_zscores = finite_pressure_zscores(
        residuals, pressure_window=10, normalization_window=90
    )
    agreement_gate = multi_horizon_agreement_gate(
        finite_zscores, slow_zscores, config.entry_threshold
    )
    specifications = {
        "v1_baseline": (zscores, baseline_gate),
        "half_life_gate": (zscores, half_life_gate),
        "half_life_plus_turn": (zscores, half_life_gate & turn_gate),
        "half_life_turn_plus_cohesion": (
            zscores,
            half_life_gate & turn_gate & regime_gate,
        ),
        "v1_gate_plus_turn": (zscores, baseline_gate & turn_gate),
        "v1_gate_plus_cohesion": (zscores, baseline_gate & regime_gate),
        "v1_gate_turn_plus_cohesion": (
            zscores,
            baseline_gate & turn_gate & regime_gate,
        ),
        "finite_residual_pressure": (finite_zscores, baseline_gate),
        "finite_pressure_persistent_extreme": (
            finite_zscores,
            baseline_gate & persistent_gate,
        ),
        "finite_pressure_strongest_three": (finite_zscores, strongest_gate),
        "finite_pressure_stable_residuals": (
            finite_zscores,
            baseline_gate & stable_gate,
        ),
        "finite_pressure_strongest_stable": (
            finite_zscores,
            strongest_gate & stable_gate,
        ),
        "finite_pressure_balanced_sides": (finite_zscores, balanced_gate),
        "finite_pressure_residual_reversal": (
            finite_zscores,
            baseline_gate & reversal_gate,
        ),
        "finite_pressure_high_dispersion": (
            finite_zscores,
            baseline_gate & dispersion_gate,
        ),
        "finite_pressure_strongest_reversal": (
            finite_zscores,
            strongest_gate & reversal_gate,
        ),
        "finite_pressure_dispersion_balanced": (
            finite_zscores,
            dispersion_gate & balanced_gate,
        ),
        "finite_pressure_dispersion_strongest": (
            finite_zscores,
            dispersion_gate & strongest_gate,
        ),
        "finite_pressure_dispersion_stable": (
            finite_zscores,
            baseline_gate & dispersion_gate & stable_gate,
        ),
        "finite_pressure_multi_horizon": (
            finite_zscores,
            baseline_gate & agreement_gate,
        ),
        "finite_pressure_multi_horizon_dispersion": (
            finite_zscores,
            baseline_gate & agreement_gate & dispersion_gate,
        ),
    }
    variants = {
        name: (
            positions_from_entry_gate(
                variant_zscores, gate, config.entry_threshold, config.exit_threshold
            ),
            gate,
        )
        for name, (variant_zscores, gate) in specifications.items()
    }
    best_gate = dispersion_gate & strongest_gate
    best_positions = variants["finite_pressure_dispersion_strongest"][0]
    aligned_returns = returns.loc[residuals.index, residuals.columns]
    risk_scaled = inverse_volatility_scaled_positions(
        best_positions, aligned_returns
    )
    strength_scaled = signal_strength_scaled_positions(
        best_positions, finite_zscores, config.entry_threshold
    )
    variants["dispersion_strongest_inverse_volatility"] = (
        risk_scaled,
        best_gate,
    )
    variants["dispersion_strongest_signal_weighted"] = (
        strength_scaled,
        best_gate,
    )
    variants["dispersion_strongest_combined_weighting"] = (
        inverse_volatility_scaled_positions(strength_scaled, aligned_returns),
        best_gate,
    )
    variants["dispersion_strongest_no_direct_reversal"] = (
        positions_from_entry_gate(
            finite_zscores,
            best_gate,
            config.entry_threshold,
            config.exit_threshold,
            allow_direct_reversal=False,
        ),
        best_gate,
    )
    variants["dispersion_strongest_three_day_cooldown"] = (
        positions_from_entry_gate(
            finite_zscores,
            best_gate,
            config.entry_threshold,
            config.exit_threshold,
            allow_direct_reversal=False,
            cooldown_days=3,
        ),
        best_gate,
    )
    return variants


def build_final_strategy(
    residuals: pd.DataFrame,
    returns: pd.DataFrame,
    config: StrategyConfig = StrategyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the selected Version 2 strategy without running research variants."""
    zscores = finite_pressure_zscores(residuals)
    eligibility = version_one_eligibility(residuals, config)
    dispersion_gate = high_dispersion_regime(zscores)
    strongest_gate = strongest_dislocation_gate(
        zscores,
        eligibility,
        config.entry_threshold,
        maximum_entries=3,
    )
    entry_gate = dispersion_gate & strongest_gate
    positions = positions_from_entry_gate(
        zscores,
        entry_gate,
        config.entry_threshold,
        config.exit_threshold,
        allow_direct_reversal=False,
    )
    return positions, entry_gate, zscores
