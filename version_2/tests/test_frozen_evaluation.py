import unittest

from version_2.validate import evaluate_frozen_data


class FrozenEvaluationTests(unittest.TestCase):
    def test_frozen_external_metrics_reproduce(self):
        headline, _ = evaluate_frozen_data()
        result = headline.iloc[0]

        self.assertAlmostEqual(result.sharpe, 0.8037008903709195)
        self.assertAlmostEqual(result.cagr, 0.0654773050853319)
        self.assertAlmostEqual(result.max_drawdown, -0.1354303442385597)
        self.assertAlmostEqual(result.average_daily_turnover, 0.06326889279437611)
        self.assertFalse(result.passed_validation)


if __name__ == "__main__":
    unittest.main()
