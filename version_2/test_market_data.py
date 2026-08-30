"""Offline tests for Version 2's isolated market-data transformations."""

import unittest

import numpy as np
import pandas as pd

from version_2.market_data import calculate_walk_forward_residuals


class MarketDataTests(unittest.TestCase):
    def test_residual_builder_covers_every_post_training_return(self):
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(
            rng.normal(size=(73, 5)),
            index=pd.date_range("2024-01-01", periods=73),
            columns=list("ABCDE"),
        )
        residuals = calculate_walk_forward_residuals(
            returns, training_window=20, refit_interval=5, components=3
        )
        self.assertEqual(len(residuals), 53)
        self.assertEqual(residuals.index[-1], returns.index[-1])

    def test_future_return_does_not_change_earlier_residual(self):
        rng = np.random.default_rng(9)
        returns = pd.DataFrame(
            rng.normal(size=(30, 5)),
            index=pd.date_range("2024-01-01", periods=30),
            columns=list("ABCDE"),
        )
        changed = returns.copy()
        changed.iloc[-1] *= 100
        original_residuals = calculate_walk_forward_residuals(
            returns, training_window=20, refit_interval=5, components=3
        )
        changed_residuals = calculate_walk_forward_residuals(
            changed, training_window=20, refit_interval=5, components=3
        )
        pd.testing.assert_series_equal(
            original_residuals.iloc[-2], changed_residuals.iloc[-2]
        )


if __name__ == "__main__":
    unittest.main()
