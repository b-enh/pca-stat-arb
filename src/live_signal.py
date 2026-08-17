import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

with open("../data/universe_config.txt") as f:
    config = dict(line.strip().split("=") for line in f)
selected_stocks = config["selected_stocks"].split(",")

WINDOW = 60
ZSCORE_WINDOW = 20
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5
N_COMPONENTS = 3

ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD",
               "INTC", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "QCOM", "TXN"]

print("Fetching recent data...")
data = yf.download(ALL_TICKERS, period="6mo", interval="1d")["Close"]
returns = data.pct_change().dropna()

recent_window = returns.iloc[-(WINDOW + ZSCORE_WINDOW):]

scaler = StandardScaler()
window_scaled = scaler.fit_transform(recent_window.iloc[:WINDOW])
pca = PCA(n_components=N_COMPONENTS)
factor_returns_train = pca.fit_transform(window_scaled)

X = np.column_stack([np.ones(WINDOW), factor_returns_train])
betas = np.linalg.lstsq(X, window_scaled, rcond=None)[0]

residuals_list = []
for i in range(WINDOW, len(recent_window)):
    actual_scaled = scaler.transform(recent_window.iloc[[i]])[0]
    factor_today = pca.transform(actual_scaled.reshape(1, -1))[0]
    X_today = np.concatenate([[1], factor_today])
    predicted = X_today @ betas
    residuals_list.append(actual_scaled - predicted)

residuals = pd.DataFrame(residuals_list, columns=ALL_TICKERS,
                          index=recent_window.index[WINDOW:])
residuals = residuals[selected_stocks]

cum_residuals = residuals.cumsum()
zscore_today = (cum_residuals.iloc[-1] - cum_residuals.mean()) / cum_residuals.std()

print("\nToday's z-scores (selected universe):")
print(zscore_today.sort_values())

target_positions = {}
for ticker in selected_stocks:
    z = zscore_today[ticker]
    if z > ENTRY_THRESHOLD:
        target_positions[ticker] = -1
    elif z < -ENTRY_THRESHOLD:
        target_positions[ticker] = 1
    else:
        target_positions[ticker] = 0

print("\nTarget positions:")
print(target_positions)


load_dotenv()
client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

CAPITAL_FRACTION = 0.25

account = client.get_account()
buying_power = float(account.buying_power)
capital_to_use = buying_power * CAPITAL_FRACTION

open_targets = {t: p for t, p in target_positions.items() if p != 0}
capital_per_position = capital_to_use / len(open_targets) if open_targets else 0

print(f"\nBuying power: ${buying_power:,.2f}")
print(f"Capital allocated to strategy: ${capital_to_use:,.2f}")
print(f"Open target positions: {len(open_targets)}")
print(f"Capital per position: ${capital_per_position:,.2f}")

# was unsure of alpaca api so had to research how to do the rest
current_positions = {p.symbol: int(p.qty) if p.side == "long" else -int(p.qty)
                      for p in client.get_all_positions()}
print(f"\nCurrent Alpaca positions: {current_positions}")


print("\n--- Planned actions (dry run, nothing submitted yet) ---")
for ticker in selected_stocks:
    target = target_positions[ticker]
    current_qty = current_positions.get(ticker, 0)
    current_side = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)

    if target != current_side:
        print(f"{ticker}: currently {current_side}, target {target} -> WOULD TRADE")
    else:
        print(f"{ticker}: currently {current_side}, target {target} -> no change")

print("\n--- Executing trades ---")
for ticker in selected_stocks:
    target = target_positions[ticker]
    current_qty = current_positions.get(ticker, 0)
    current_side = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)

    if target == current_side:
        continue

    
    if current_side != 0:
        close_side = OrderSide.SELL if current_side > 0 else OrderSide.BUY
        order = MarketOrderRequest(
            symbol=ticker,
            qty=abs(current_qty),
            side=close_side,
            time_in_force=TimeInForce.DAY
        )
        client.submit_order(order)
        print(f"{ticker}: closed existing position ({abs(current_qty)} shares)")

   
    if target != 0:
        latest_price = data[ticker].iloc[-1]
        shares = int(capital_per_position // latest_price)
        if shares > 0:
            side = OrderSide.BUY if target == 1 else OrderSide.SELL
            order = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            print(f"{ticker}: opened {'long' if target == 1 else 'short'}, {shares} shares (~${shares * latest_price:,.2f})")
        else:
            print(f"{ticker}: target {target}, but capital per position too small for even 1 share")