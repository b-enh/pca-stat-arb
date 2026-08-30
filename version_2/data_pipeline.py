"""Download the frozen public UK validation panel used by Version 2."""

from pathlib import Path

import yfinance as yf


VERSION_DIR = Path(__file__).resolve().parent
UK_TICKERS = [
    "AZN.L",
    "HSBA.L",
    "BP.L",
    "ULVR.L",
    "GSK.L",
    "DGE.L",
    "RIO.L",
    "BATS.L",
    "LSEG.L",
    "REL.L",
    "NG.L",
    "VOD.L",
    "BARC.L",
    "LLOY.L",
    "TSCO.L",
]


def download_frozen_uk_data() -> None:
    """Download the public price panel fixed before external evaluation."""
    prices = yf.download(
        UK_TICKERS,
        start="2017-01-01",
        end="2022-01-01",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )["Close"]
    prices = prices.reindex(columns=UK_TICKERS).dropna()
    if len(prices) < 1_000:
        raise RuntimeError(f"Incomplete frozen UK panel: shape={prices.shape}")

    output_dir = VERSION_DIR / "data" / "external_uk_2017_2021"
    output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_dir / "prices.csv")
    prices.pct_change(fill_method=None).dropna().to_csv(output_dir / "returns.csv")


def main() -> None:
    download_frozen_uk_data()


if __name__ == "__main__":
    main()
