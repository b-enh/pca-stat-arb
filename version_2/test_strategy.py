"""Focused tests for the Version 2 entry-gate extensions."""

import unittest

import numpy as np
import pandas as pd

from strategy import (
    StrategyConfig,
    balanced_dislocation_gate,
    cohesion_regime,
    estimate_half_life,
    finite_pressure_zscores,
    high_dispersion_regime,
    inverse_volatility_scaled_positions,
    multi_horizon_agreement_gate,
    positions_from_entry_gate,
    persistent_extreme_gate,
    stable_residual_gate,
    strongest_dislocation_gate,
    residual_reversal_gate,
    signal_strength_scaled_positions,
    turning_confirmation,
)


class VersionTwoStrategyTests(unittest.TestCase):
    def test_decaying_spread_has_finite_half_life(self):
        spread = pd.Series(0.8 ** np.arange(30))
        self.assertAlmostEqual(estimate_half_life(spread), np.log(0.5) / np.log(0.8))

    def test_turn_requires_same_side_move_toward_zero(self):
        zscores = pd.DataFrame({"AAA": [-2.0, -1.7, -1.9, 0.2]})
        self.assertEqual(
            turning_confirmation(zscores)["AAA"].tolist(),
            [False, True, False, False],
        )

    def test_entry_gate_does_not_force_existing_position_out(self):
        index = pd.date_range("2024-01-01", periods=3)
        zscores = pd.DataFrame({"AAA": [-2.0, -1.0, -0.2]}, index=index)
        gate = pd.DataFrame({"AAA": [True, False, False]}, index=index)
        positions = positions_from_entry_gate(zscores, gate, 1.5, 0.5)
        self.assertEqual(positions["AAA"].tolist(), [1.0, 1.0, 0.0])

    def test_no_direct_reversal_closes_before_new_direction(self):
        index = pd.date_range("2024-01-01", periods=3)
        zscores = pd.DataFrame({"AAA": [-2.0, 2.0, 2.0]}, index=index)
        gate = pd.DataFrame(True, index=index, columns=["AAA"])
        positions = positions_from_entry_gate(
            zscores, gate, 1.5, 0.5, allow_direct_reversal=False
        )
        self.assertEqual(positions["AAA"].tolist(), [1.0, 0.0, -1.0])

    def test_cooldown_blocks_new_entries_for_requested_days(self):
        index = pd.date_range("2024-01-01", periods=5)
        zscores = pd.DataFrame({"AAA": [-2.0, -0.2, -2.0, -2.0, -2.0]}, index=index)
        gate = pd.DataFrame(True, index=index, columns=["AAA"])
        positions = positions_from_entry_gate(
            zscores, gate, 1.5, 0.5, cooldown_days=2
        )
        self.assertEqual(positions["AAA"].tolist(), [1.0, 0.0, 0.0, 0.0, 1.0])

    def test_finite_pressure_does_not_remember_distant_residuals(self):
        residuals = pd.DataFrame({"AAA": [100.0] + [0.0] * 11 + [1.0, -1.0, 1.0, -1.0]})
        changed = residuals.copy()
        changed.iloc[0, 0] = -100.0
        original = finite_pressure_zscores(residuals, 2, 5)
        altered = finite_pressure_zscores(changed, 2, 5)
        self.assertAlmostEqual(original.iloc[-1, 0], altered.iloc[-1, 0])

    def test_persistent_extreme_requires_two_same_direction_days(self):
        zscores = pd.DataFrame({"AAA": [1.6, 1.7, -1.8, -1.9, -1.0]})
        gate = persistent_extreme_gate(zscores, 1.5)
        self.assertEqual(gate["AAA"].tolist(), [False, True, False, True, False])

    def test_strongest_gate_limits_daily_candidates(self):
        zscores = pd.DataFrame([[2.0, -3.0, 1.8, 0.2]], columns=list("ABCD"))
        eligibility = pd.DataFrame(True, index=zscores.index, columns=zscores.columns)
        gate = strongest_dislocation_gate(zscores, eligibility, 1.5, 2)
        self.assertEqual(gate.sum(axis=1).iloc[0], 2)
        self.assertTrue(gate.loc[0, "A"])
        self.assertTrue(gate.loc[0, "B"])

    def test_stable_gate_selects_lower_half_of_residual_noise(self):
        residuals = pd.DataFrame(
            {
                "quiet": [0.0, 0.1, 0.0, 0.1],
                "noisy": [0.0, 2.0, -2.0, 2.0],
            }
        )
        gate = stable_residual_gate(residuals, 3)
        self.assertTrue(gate.iloc[-1, 0])
        self.assertFalse(gate.iloc[-1, 1])

    def test_balanced_gate_selects_one_candidate_each_side(self):
        zscores = pd.DataFrame([[2.0, 3.0, -1.8, -2.5]], columns=list("ABCD"))
        eligibility = pd.DataFrame(True, index=zscores.index, columns=zscores.columns)
        gate = balanced_dislocation_gate(zscores, eligibility, 1.5)
        self.assertEqual(gate.sum(axis=1).iloc[0], 2)
        self.assertTrue(gate.loc[0, "B"])
        self.assertTrue(gate.loc[0, "D"])

    def test_reversal_gate_requires_residual_opposite_to_pressure(self):
        zscores = pd.DataFrame({"AAA": [2.0, -2.0, 2.0]})
        residuals = pd.DataFrame({"AAA": [-0.1, 0.1, 0.1]})
        self.assertEqual(
            residual_reversal_gate(zscores, residuals)["AAA"].tolist(),
            [True, True, False],
        )

    def test_dispersion_regime_uses_only_prior_threshold_values(self):
        zscores = pd.DataFrame(
            {
                "AAA": [0.0, 1.0, 2.0, 3.0],
                "BBB": [0.0, -1.0, -2.0, -3.0],
            }
        )
        regime = high_dispersion_regime(zscores, history_window=2)
        self.assertFalse(regime.iloc[1].any())
        self.assertTrue(regime.iloc[-1].all())

    def test_dispersion_regime_rejects_invalid_quantile(self):
        with self.assertRaises(ValueError):
            high_dispersion_regime(pd.DataFrame({"AAA": [1.0]}), threshold_quantile=1.0)

    def test_multi_horizon_gate_requires_extreme_same_direction_signals(self):
        short = pd.DataFrame({"AAA": [2.0, 2.0, 1.0, -2.0]})
        slow = pd.DataFrame({"AAA": [1.8, -1.8, 2.0, -1.7]})
        gate = multi_horizon_agreement_gate(short, slow, 1.5)
        self.assertEqual(gate["AAA"].tolist(), [True, False, False, True])

    def test_inverse_volatility_scaling_favors_quieter_stock(self):
        returns = pd.DataFrame(
            {
                "quiet": [0.0, 0.01, -0.01, 0.01],
                "noisy": [0.0, 0.10, -0.10, 0.10],
            }
        )
        positions = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
        scaled = inverse_volatility_scaled_positions(positions, returns, 3)
        self.assertGreater(scaled.iloc[-1, 0], scaled.iloc[-1, 1])

    def test_signal_strength_scaling_is_capped(self):
        positions = pd.DataFrame({"AAA": [1.0, 1.0]})
        zscores = pd.DataFrame({"AAA": [1.5, 10.0]})
        scaled = signal_strength_scaled_positions(positions, zscores, 1.5)
        self.assertEqual(scaled["AAA"].tolist(), [1.0, 2.0])

    def test_cohesion_uses_only_returns_before_signal_date(self):
        dates = pd.date_range("2024-01-01", periods=8)
        original = pd.DataFrame(
            {"AAA": np.arange(8.0), "BBB": np.arange(8.0) * 0.5}, index=dates
        )
        changed = original.copy()
        changed.loc[dates[5]:, "BBB"] *= -1
        config = StrategyConfig(cohesion_lookback=3, cohesion_history=2)
        original_regime = cohesion_regime(original, dates, config)
        changed_regime = cohesion_regime(changed, dates, config)
        self.assertEqual(original_regime.loc[dates[5]], changed_regime.loc[dates[5]])


if __name__ == "__main__":
    unittest.main()
