import pandas as pd
import yfinance as yf
from pathlib import Path

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD",
           "INTC", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "QCOM", "TXN"]

# Downloads historical data only; it is not part of the evaluation pipeline.
data = yf.download(tickers, period="4y", interval="1d")["Close"]


data = data.dropna()


returns = data.pct_change().dropna()

print(data.shape)      
print(returns.head())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data.to_csv(PROJECT_ROOT / "data" / "prices.csv")
returns.to_csv(PROJECT_ROOT / "data" / "returns.csv")
