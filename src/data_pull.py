import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD",
           "INTC", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "QCOM", "TXN"]

# Downloadingg 2 years of daily price data for all tickers
data = yf.download(tickers, period="4y", interval="1d")["Close"]


data = data.dropna()


returns = data.pct_change().dropna()

print(data.shape)      
print(returns.head())

# needs updating
data.to_csv("../data/prices.csv")
returns.to_csv("../data/returns.csv")