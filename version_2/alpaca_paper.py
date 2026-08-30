"""Dry-run-first Alpaca paper-trading adapter for Version 2 target weights."""

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    side: str
    notional: float
    current_exposure: float
    target_exposure: float


def load_target_weights(signal_path: Path) -> tuple[date, dict[str, float]]:
    signal = pd.read_csv(signal_path)
    required_columns = {"signal_date", "symbol", "target_weight"}
    if not required_columns.issubset(signal.columns):
        raise ValueError(f"Signal file must contain {sorted(required_columns)}")
    if signal["symbol"].duplicated().any():
        raise ValueError("Signal file contains duplicate symbols")
    weights = pd.to_numeric(signal["target_weight"], errors="raise")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Target weights must be finite")
    gross_weight = weights.abs().sum()
    if gross_weight > 1.000001:
        raise ValueError("Absolute target weights cannot exceed 100%")
    signal_dates = pd.to_datetime(signal["signal_date"], errors="raise").dt.date.unique()
    if len(signal_dates) != 1:
        raise ValueError("Signal file must contain exactly one signal date")
    targets = dict(zip(signal["symbol"].str.upper(), weights.astype(float)))
    return signal_dates[0], targets


def validate_signal_freshness(
    signal_date: date, current_date: date, maximum_age_days: int = 4
) -> None:
    age = (current_date - signal_date).days
    if age < 0:
        raise ValueError("Signal date cannot be in the future")
    if age > maximum_age_days:
        raise ValueError(
            f"Signal is {age} calendar days old; refresh the market data first"
        )


def build_order_plan(
    target_weights: dict[str, float],
    current_exposure: dict[str, float],
    capital: float,
    minimum_notional: float = 1.0,
) -> list[PlannedOrder]:
    if capital <= 0:
        raise ValueError("capital must be positive")
    planned_orders = []
    for symbol in sorted(set(target_weights) | set(current_exposure)):
        current = float(current_exposure.get(symbol, 0.0))
        target = capital * float(target_weights.get(symbol, 0.0))
        difference = target - current
        if abs(difference) < minimum_notional:
            continue
        planned_orders.append(
            PlannedOrder(
                symbol=symbol,
                side="buy" if difference > 0 else "sell",
                notional=abs(difference),
                current_exposure=current,
                target_exposure=target,
            )
        )
    return planned_orders


def submit_paper_orders(
    signal_date: date,
    target_weights: dict[str, float],
    capital: float,
    confirmation: str,
) -> list[str]:
    if confirmation != "PAPER":
        raise ValueError("Paper submission requires --confirm PAPER")
    validate_signal_freshness(signal_date, date.today())
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to paper credentials")

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

    client = TradingClient(api_key, secret_key, paper=True)
    clock = client.get_clock()
    if not clock.is_open:
        raise RuntimeError("Paper orders are only submitted while the market is open")
    open_orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
    )
    if open_orders:
        raise RuntimeError("Resolve existing open paper orders before rebalancing")
    account = client.get_account()
    if account.trading_blocked:
        raise RuntimeError("The Alpaca paper account is blocked from trading")
    if capital > float(account.equity):
        raise ValueError("Configured capital exceeds current paper-account equity")

    current_exposure = {
        position.symbol: float(position.market_value)
        for position in client.get_all_positions()
        if position.symbol in target_weights
    }
    planned_orders = build_order_plan(target_weights, current_exposure, capital)
    submitted_ids = []
    for order in sorted(planned_orders, key=lambda planned: planned.side != "sell"):
        asset = client.get_asset(order.symbol)
        if not asset.tradable:
            raise RuntimeError(f"{order.symbol} is not tradable on Alpaca")
        if not asset.fractionable:
            raise RuntimeError(f"{order.symbol} does not support notional orders")
        if target_weights.get(order.symbol, 0.0) < 0 and not asset.shortable:
            raise RuntimeError(f"{order.symbol} is not currently shortable")
        request = MarketOrderRequest(
            symbol=order.symbol,
            notional=round(order.notional, 2),
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        submitted = client.submit_order(order_data=request)
        submitted_ids.append(str(submitted.id))
    return submitted_ids


def format_plan(planned_orders: list[PlannedOrder]) -> str:
    if not planned_orders:
        return "No orders required."
    return pd.DataFrame(
        [
            {
                "symbol": order.symbol,
                "side": order.side,
                "notional": round(order.notional, 2),
                "current_exposure": round(order.current_exposure, 2),
                "target_exposure": round(order.target_exposure, 2),
            }
            for order in planned_orders
        ]
    ).to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("signal_file", type=Path)
    parser.add_argument("--capital", type=float, required=True)
    parser.add_argument("--submit-paper-orders", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    signal_date, target_weights = load_target_weights(args.signal_file)
    if not args.submit_paper_orders:
        plan = build_order_plan(target_weights, {}, args.capital)
        print(f"Dry run from a flat portfolio for signal date {signal_date}:")
        print(format_plan(plan))
        return
    submitted_ids = submit_paper_orders(
        signal_date, target_weights, args.capital, args.confirm
    )
    print(f"Submitted {len(submitted_ids)} paper orders: {submitted_ids}")


if __name__ == "__main__":
    main()
