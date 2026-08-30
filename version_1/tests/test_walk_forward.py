import unittest

import pandas as pd

from version_1.strategy import positions_from_zscores, walk_forward_selection


class WalkForwardTests(unittest.TestCase):
    def test_selection_does_not_use_future_residuals(self):
        index = pd.date_range("2024-01-01", periods=6, freq="D")
        original = pd.DataFrame({"AAA": [1, -1, 1, -1, 10, 10]}, index=index)
        changed_future = original.copy()
        changed_future.iloc[4:, 0] = [-10, -10]
        original_selection = walk_forward_selection(original, 4, 1, -0.02)
        changed_selection = walk_forward_selection(changed_future, 4, 1, -0.02)
        self.assertEqual(original_selection.iloc[4, 0], changed_selection.iloc[4, 0])

    def test_ineligible_stock_cannot_open_a_new_position(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        zscores = pd.DataFrame({"AAA": [-2.0, -2.0]}, index=index)
        eligibility = pd.DataFrame({"AAA": [False, False]}, index=index)

        positions = positions_from_zscores(zscores, eligibility, 1.5, 0.5)

        self.assertEqual(positions["AAA"].tolist(), [0.0, 0.0])

    def test_open_position_survives_later_loss_of_eligibility_until_exit(self):
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        zscores = pd.DataFrame({"AAA": [-2.0, -1.0, -0.2]}, index=index)
        eligibility = pd.DataFrame({"AAA": [True, False, False]}, index=index)

        positions = positions_from_zscores(zscores, eligibility, 1.5, 0.5)

        self.assertEqual(positions["AAA"].tolist(), [1.0, 1.0, 0.0])
