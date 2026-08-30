"""Isolated Alpaca market-data refresh and past-only PCA residual construction."""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "version_2" / "data"
UNIVERSE = (
    "AAPL",
    "ADBE",
    "AMD",
    "AMZN",
    "CRM",
    "CSCO",
    "GOOGL",
    "IBM",
    "INTC",
    "META",
    "MSFT",
    "NVDA",
    "ORCL",
    "QCOM",
    "TXN",
)


def calculate_walk_forward_residuals(
    returns: pd.DataFrame,
    training_window: int = 60,
    refit_interval: int = 5,
    components: int = 3,
) -> pd.DataFrame:
    """Fit PCA on earlier returns and calculate residuals for each next block."""
    if len(returns) <= training_window:
        raise ValueError("Not enough returns for the PCA training window")
    residual_rows = []
    residual_dates = []
    for start in range(training_window, len(returns), refit_interval):
        training = returns.iloc[start - training_window : start]
        training_values = training.to_numpy()
        training_mean = training_values.mean(axis=0)
        training_standard_deviation = training_values.std(axis=0)
        if (training_standard_deviation == 0).any():
            raise ValueError("PCA training window contains a zero-volatility stock")
        standardized_training = (
            training_values - training_mean
        ) / training_standard_deviation
        _, _, component_loadings = np.linalg.svd(
            standardized_training, full_matrices=False
        )
        component_loadings = component_loadings[:components]
        factor_returns = standardized_training @ component_loadings.T
        design = np.column_stack([np.ones(training_window), factor_returns])
        coefficients = np.linalg.lstsq(
            design, standardized_training, rcond=None
        )[0]
        for row_number in range(start, min(start + refit_interval, len(returns))):
            current = (
                returns.iloc[row_number].to_numpy() - training_mean
            ) / training_standard_deviation
            current_factors = current @ component_loadings.T
            predicted = np.concatenate([[1.0], current_factors]) @ coefficients
            residual_rows.append(current - predicted)
            residual_dates.append(returns.index[row_number])
    return pd.DataFrame(
        residual_rows, index=residual_dates, columns=returns.columns
    )


def close_prices_from_bar_frame(bar_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpaca's symbol/timestamp bars into a complete close-price panel."""
    if "close" not in bar_frame.columns:
        raise ValueError("Alpaca bar data does not contain close prices")
    if not isinstance(bar_frame.index, pd.MultiIndex):
        raise ValueError("Expected Alpaca bars indexed by symbol and timestamp")
    prices = bar_frame["close"].unstack(level="symbol")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    return prices.reindex(columns=UNIVERSE).dropna()


def refresh_version_two_data(
    output_dir: Path = DEFAULT_OUTPUT_DIR, lookback_days: int = 1_600
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download adjusted daily bars into Version 2's isolated data directory."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to paper credentials")

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=list(UNIVERSE),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )
    prices = close_prices_from_bar_frame(client.get_stock_bars(request).df)
    returns = prices.pct_change().dropna()
    returns.index.name = "Date"
    prices.index.name = "Date"
    residuals = calculate_walk_forward_residuals(returns)
    output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_dir / "prices.csv")
    returns.to_csv(output_dir / "returns.csv")
    residuals.to_csv(output_dir / "residuals.csv")
    return prices, returns, residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback-days", type=int, default=1_600)
    args = parser.parse_args()
    prices, returns, residuals = refresh_version_two_data(
        args.output_dir, args.lookback_days
    )
    print(
        f"Saved {len(prices)} prices, {len(returns)} returns, and "
        f"{len(residuals)} residual rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
