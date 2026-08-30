import unittest

import numpy as np
import pandas as pd

from common.inference import (
    moving_block_bootstrap_sharpes,
    newey_west_positive_mean_test,
    sample_moving_blocks,
)


class InferenceTests(unittest.TestCase):
    def test_moving_block_sample_keeps_each_drawn_block_consecutive(self):
        sample = sample_moving_blocks(np.arange(10.0), 2, 8, np.random.default_rng(7))
        self.assertTrue(np.all(np.diff(sample.reshape(-1, 2), axis=1) == 1))

    def test_newey_west_test_is_unfavourable_for_negative_returns(self):
        result = newey_west_positive_mean_test(pd.Series([-0.01, -0.02, -0.01, -0.03, -0.02]), 1)
        self.assertLess(result.t_statistic, 0)
        self.assertGreater(result.one_sided_p_value, 0.5)

    def test_block_bootstrap_returns_one_sharpe_per_resample(self):
        returns = pd.Series([0.01, -0.005, 0.002, -0.001, 0.004, -0.002])
        self.assertEqual(len(moving_block_bootstrap_sharpes(returns, 2, 10, 42)), 10)

    def test_all_zero_bootstrap_sample_has_zero_sharpe(self):
        sharpes = moving_block_bootstrap_sharpes(pd.Series([0.0] * 10), 2, 10, 42)
        np.testing.assert_array_equal(sharpes, np.zeros(10))
