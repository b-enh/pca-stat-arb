import unittest

import pandas as pd

from experiments.mean_reversion import cumulative_test_spreads, estimate_mean_reversion


class MeanReversionTests(unittest.TestCase):
    def test_cumulative_spread_resets_at_the_test_split(self):
        residuals = pd.DataFrame({"AAA": [0.10, 0.20, -0.10]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
        spreads = cumulative_test_spreads(residuals, residuals.index[0])
        self.assertEqual(spreads["AAA"].tolist(), [0.20, 0.10])

    def test_negative_beta_has_a_positive_finite_half_life(self):
        estimate = estimate_mean_reversion(pd.Series([1.0, 0.8, 0.64, 0.512, 0.4096]))
        self.assertLess(estimate.beta, 0)
        self.assertGreater(estimate.half_life_days, 0)
