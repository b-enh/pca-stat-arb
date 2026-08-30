import unittest

import pandas as pd

from portfolio import calculate_portfolio_returns


class PortfolioAccountingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=3, freq="D")
        self.returns = pd.DataFrame(0.0, index=self.dates, columns=["AAA", "BBB"])

    def test_empty_portfolio_has_no_return_or_cost(self):
        result = calculate_portfolio_returns(
            pd.DataFrame(0.0, index=self.dates, columns=self.returns.columns), self.returns, 0.0005
        )
        self.assertTrue((result.net_returns == 0.0).all())
        self.assertTrue((result.trading_costs == 0.0).all())

    def test_opening_two_equal_positions_costs_five_basis_points(self):
        positions = pd.DataFrame([[1.0, -1.0]] * 3, index=self.dates, columns=self.returns.columns)
        result = calculate_portfolio_returns(positions, self.returns, 0.0005)
        self.assertAlmostEqual(result.turnover.iloc[1], 1.0)
        self.assertAlmostEqual(result.trading_costs.iloc[1], 0.0005)

    def test_reversing_a_two_stock_portfolio_has_200_percent_turnover(self):
        positions = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]], index=self.dates, columns=self.returns.columns)
        result = calculate_portfolio_returns(positions, self.returns, 0.0005)
        self.assertAlmostEqual(result.turnover.iloc[2], 2.0)
        self.assertAlmostEqual(result.trading_costs.iloc[2], 0.001)

    def test_equal_long_short_positions_have_zero_net_exposure(self):
        positions = pd.DataFrame([[1.0, -1.0]] * 3, index=self.dates, columns=self.returns.columns)
        result = calculate_portfolio_returns(positions, self.returns, 0.0005)
        self.assertAlmostEqual(result.gross_exposure.iloc[1], 1.0)
        self.assertAlmostEqual(result.net_exposure.iloc[1], 0.0)

    def test_net_return_equals_gross_return_minus_cost(self):
        positions = pd.DataFrame([[1.0, -1.0]] * 3, index=self.dates, columns=self.returns.columns)
        returns = self.returns.copy()
        returns.iloc[1] = [0.02, -0.02]
        result = calculate_portfolio_returns(positions, returns, 0.0005)
        pd.testing.assert_series_equal(result.net_returns, result.gross_returns - result.trading_costs)
