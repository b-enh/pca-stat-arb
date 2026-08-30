import unittest

import numpy as np
import pandas as pd

from version_2.strategy import (
    StrategyConfig,
    dynamic_pca_residuals,
    residual_reversal_positions,
    residual_reversal_signal,
)


class ResidualReversalStrategyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(113)
        self.returns = pd.DataFrame(
            rng.normal(0, 0.01, (430, 15)),
            index=pd.date_range("2020-01-01", periods=430, freq="B"),
            columns=[f"stock_{number}" for number in range(15)],
        )

    def test_future_returns_do_not_change_past_residuals(self):
        changed = self.returns.copy()
        changed.iloc[-10:] += 1.0

        original = dynamic_pca_residuals(self.returns)
        revised = dynamic_pca_residuals(changed)

        pd.testing.assert_frame_equal(original.iloc[:-10], revised.iloc[:-10])

    def test_positions_are_aligned_finite_and_dollar_neutral(self):
        residuals = dynamic_pca_residuals(self.returns)
        positions = residual_reversal_positions(residuals)

        self.assertTrue(positions.index.equals(residuals.index))
        self.assertTrue(positions.columns.equals(residuals.columns))
        self.assertTrue(np.isfinite(positions.to_numpy()).all())
        np.testing.assert_allclose(positions.sum(axis=1), 0.0, atol=1e-12)

    def test_future_residuals_do_not_change_past_positions(self):
        residuals = dynamic_pca_residuals(self.returns)
        changed = residuals.copy()
        changed.iloc[-20:] += 100.0

        original_positions = residual_reversal_positions(residuals)
        revised_positions = residual_reversal_positions(changed)

        pd.testing.assert_frame_equal(original_positions.iloc[:-20], revised_positions.iloc[:-20])

    def test_rebalances_hold_three_stocks_per_side(self):
        config = StrategyConfig()
        residuals = dynamic_pca_residuals(self.returns, config)
        positions = residual_reversal_positions(residuals, config)
        first_rebalance = positions.iloc[config.initial_history_days]

        self.assertEqual((first_rebalance > 0).sum(), config.stocks_per_side)
        self.assertEqual((first_rebalance < 0).sum(), config.stocks_per_side)

    def test_negative_residual_pressure_has_positive_signal(self):
        residuals = pd.DataFrame(
            {"loser": [-1.0] * 20, "winner": [1.0] * 20},
            index=pd.date_range("2020-01-01", periods=20, freq="B"),
        )
        residuals.iloc[:15] *= np.linspace(0.5, 1.5, 15)[:, None]

        signal = residual_reversal_signal(residuals).iloc[-1]

        self.assertGreater(signal["loser"], 0)
        self.assertLess(signal["winner"], 0)


if __name__ == "__main__":
    unittest.main()
