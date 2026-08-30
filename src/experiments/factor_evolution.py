import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "experiments" / "factor_evolution" / "factor_evolution.csv"

returns = pd.read_csv(DATA_DIR / "returns.csv", index_col="Date", parse_dates=True)

WINDOW = 60
STEP = 5

SOFTWARE_GROUP = ["ADBE", "CRM", "IBM", "MSFT"]
SEMIS_GROUP = ["TXN", "INTC", "AMD", "QCOM"]

dates = returns.index
split_strength = []
split_dates = []

for i in range(WINDOW, len(returns), STEP):
    window_data = returns.iloc[i - WINDOW : i].values

    scaler = StandardScaler()
    window_scaled = scaler.fit_transform(window_data)

    pca = PCA(n_components=3)
    pca.fit(window_scaled)

    pc2_loadings = pd.Series(pca.components_[1], index=returns.columns)

    software_avg = pc2_loadings[SOFTWARE_GROUP].mean()
    semis_avg = pc2_loadings[SEMIS_GROUP].mean()
    strength = software_avg - semis_avg

    split_strength.append(strength)
    split_dates.append(dates[i])

split_series = pd.Series(split_strength, index=split_dates, name="pc2_split_strength")
split_series.to_csv(OUTPUT_PATH)

print(split_series.describe())
print("\nStrongest split (largest divergence):")
print(split_series.sort_values(ascending=False).head())
print("\nWeakest split (most blurred):")
print(split_series.sort_values().head())
