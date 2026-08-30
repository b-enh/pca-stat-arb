import unittest

import pandas as pd

from robustness import selection_change_rate


class RobustnessTests(unittest.TestCase):
    def test_selection_change_rate_is_zero_for_a_constant_selection(self):
        eligibility = pd.DataFrame(
            {"AAA": [True, True, True, True], "BBB": [False, False, False, False]}
        )

        self.assertEqual(selection_change_rate(eligibility, rebalance_days=1), 0.0)

    def test_selection_change_rate_detects_a_changed_stock_set(self):
        eligibility = pd.DataFrame(
            {"AAA": [True, True, False], "BBB": [False, False, True]}
        )

        self.assertEqual(selection_change_rate(eligibility, rebalance_days=1), 0.5)


if __name__ == "__main__":
    unittest.main()
