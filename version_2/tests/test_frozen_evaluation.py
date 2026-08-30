import unittest

from version_2.validate import evaluate_frozen_data
from version_2.validate_secondary import evaluate_secondary_data


class FrozenEvaluationTests(unittest.TestCase):
    def test_frozen_external_metrics_reproduce(self):
        headline, _ = evaluate_frozen_data()
        result = headline.iloc[0]

        self.assertAlmostEqual(result.sharpe, 0.4707244669937758)
        self.assertAlmostEqual(result.cagr, 0.03231266969307112)
        self.assertAlmostEqual(result.max_drawdown, -0.13939883584341373)
        self.assertAlmostEqual(result.average_daily_turnover, 0.040169133192389)
        self.assertFalse(result.passed_validation)

    def test_second_external_metrics_reproduce(self):
        result = evaluate_secondary_data().iloc[0]

        self.assertAlmostEqual(result.sharpe, 0.19215753780199438)
        self.assertAlmostEqual(result.cagr, 0.015463187187164884)
        self.assertAlmostEqual(result.max_drawdown, -0.22687759814923292)
        self.assertFalse(result.passed_validation)


if __name__ == "__main__":
    unittest.main()
