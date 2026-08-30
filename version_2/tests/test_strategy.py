import unittest

import numpy as np
import pandas as pd

from version_2.strategy import StrategyConfig, residual_correlation_centrality


class CentralityStrategyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(113)
        self.residuals = pd.DataFrame(
            rng.normal(size=(400, 15)),
            index=pd.date_range("2020-01-01", periods=400, freq="B"),
            columns=[f"stock_{number}" for number in range(15)],
        )

    def test_positions_are_aligned_finite_and_dollar_neutral(self):
        positions = residual_correlation_centrality(self.residuals)

        self.assertTrue(positions.index.equals(self.residuals.index))
        self.assertTrue(positions.columns.equals(self.residuals.columns))
        self.assertTrue(np.isfinite(positions.to_numpy()).all())
        self.assertTrue(np.allclose(positions.sum(axis=1), 0.0))

    def test_future_residuals_do_not_change_past_positions(self):
        changed = self.residuals.copy()
        changed.iloc[300:] *= 100.0

        original_positions = residual_correlation_centrality(self.residuals)
        changed_positions = residual_correlation_centrality(changed)

        pd.testing.assert_frame_equal(original_positions.iloc[:300], changed_positions.iloc[:300])

    def test_rebalances_hold_three_stocks_per_side(self):
        config = StrategyConfig()
        positions = residual_correlation_centrality(self.residuals, config)
        first_rebalance = positions.iloc[config.initial_history_days]

        self.assertEqual((first_rebalance > 0).sum(), config.stocks_per_side)
        self.assertEqual((first_rebalance < 0).sum(), config.stocks_per_side)


if __name__ == "__main__":
    unittest.main()
