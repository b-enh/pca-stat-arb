"""Offline tests for the Alpaca paper-trading boundary."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from version_2.alpaca_paper import (
    build_order_plan,
    load_target_weights,
    validate_signal_freshness,
)


class AlpacaPaperTests(unittest.TestCase):
    def test_order_plan_rebalances_from_current_exposure(self):
        orders = build_order_plan(
            {"AAA": 0.5, "BBB": -0.5, "CCC": 0.0},
            {"AAA": 200.0, "CCC": 100.0},
            capital=1_000.0,
        )
        by_symbol = {order.symbol: order for order in orders}
        self.assertEqual(by_symbol["AAA"].side, "buy")
        self.assertEqual(by_symbol["AAA"].notional, 300.0)
        self.assertEqual(by_symbol["BBB"].side, "sell")
        self.assertEqual(by_symbol["CCC"].side, "sell")

    def test_stale_signal_is_rejected_for_submission(self):
        with self.assertRaises(ValueError):
            validate_signal_freshness(date(2024, 1, 1), date(2024, 1, 10), 4)

    def test_signal_loader_rejects_excess_gross_weight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "signal.csv"
            pd.DataFrame(
                {
                    "signal_date": ["2026-08-29", "2026-08-29"],
                    "symbol": ["AAA", "BBB"],
                    "target_weight": [0.7, -0.7],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_target_weights(path)

    def test_signal_loader_rejects_infinite_weight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "signal.csv"
            pd.DataFrame(
                {
                    "signal_date": ["2026-08-29"],
                    "symbol": ["AAA"],
                    "target_weight": [float("inf")],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_target_weights(path)


if __name__ == "__main__":
    unittest.main()
