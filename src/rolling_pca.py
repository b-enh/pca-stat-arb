import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

returns = pd.read_csv("../data/returns.csv", index_col="Date", parse_dates=True)

WINDOW = 60
STEP = 5
N_COMPONENTS = 3

tickers = returns.columns
dates = returns.index

residuals_list = []
residual_dates = []

for i in range(WINDOW, len(returns) - STEP, STEP):
    window_data = returns.iloc[i - WINDOW : i].values

    scaler = StandardScaler()
    window_scaled = scaler.fit_transform(window_data)

    pca = PCA(n_components=N_COMPONENTS)
    factor_returns_window = pca.fit_transform(window_scaled) 

    X = np.column_stack([np.ones(WINDOW), factor_returns_window])
    betas = np.linalg.lstsq(X, window_scaled, rcond=None)[0] 

   
    for j in range(i, min(i + STEP, len(returns))):
        actual_raw = returns.iloc[j].values
        actual_scaled = scaler.transform(actual_raw.reshape(1, -1))[0]

        factor_return_today = pca.transform(actual_scaled.reshape(1, -1))[0]

        X_today = np.concatenate([[1], factor_return_today])
        predicted = X_today @ betas
        residual = actual_scaled - predicted

        residuals_list.append(residual)
        residual_dates.append(dates[j])

residuals_df = pd.DataFrame(residuals_list, index=residual_dates, columns=tickers)
residuals_df.to_csv("../data/residuals.csv")

print(residuals_df.shape)
print(residuals_df.head())
