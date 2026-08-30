"""Export the latest local Version 2 target weights without contacting a broker."""

import argparse
from pathlib import Path

import pandas as pd

from version_2.strategy import build_final_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "version_2" / "outputs" / "latest_signal.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "version_2" / "data"


def generate_signal(
    output_path: Path = DEFAULT_OUTPUT, data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    if not (data_dir / "residuals.csv").exists() or not (data_dir / "returns.csv").exists():
        raise FileNotFoundError(
            f"Version 2 data not found in {data_dir}; run version_2.market_data first"
        )
    residuals = pd.read_csv(
        data_dir / "residuals.csv", index_col=0, parse_dates=True
    )
    returns = pd.read_csv(
        data_dir / "returns.csv", index_col="Date", parse_dates=True
    )
    positions, _, zscores = build_final_strategy(residuals, returns)
    latest_date = positions.index[-1]
    latest_positions = positions.loc[latest_date]
    gross_position = latest_positions.abs().sum()
    target_weights = (
        latest_positions / gross_position
        if gross_position > 0
        else latest_positions.astype(float)
    )
    signal = pd.DataFrame(
        {
            "signal_date": latest_date.date().isoformat(),
            "symbol": target_weights.index,
            "target_weight": target_weights.to_numpy(),
            "zscore": zscores.loc[latest_date].to_numpy(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal.to_csv(output_path, index=False)
    return signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    signal = generate_signal(args.output, args.data_dir)
    print(signal.to_string(index=False))


if __name__ == "__main__":
    main()
