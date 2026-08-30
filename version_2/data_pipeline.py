"""Past-only rolling PCA residual construction inherited from Version 1."""

import numpy as np
import pandas as pd


def residuals_from_returns(
    returns: pd.DataFrame, window: int = 60, components: int = 3, step: int = 5
) -> pd.DataFrame:
    """Fit PCA on prior returns and derive contemporaneous cross-sectional residuals."""
    rows, dates = [], []
    for start in range(window, len(returns), step):
        training = returns.iloc[start - window : start]
        mean = training.mean()
        standard_deviation = training.std(ddof=0)
        if (standard_deviation == 0).any():
            raise ValueError("PCA training window contains a zero-volatility stock")
        standardized = (training - mean) / standard_deviation
        _, _, loadings = np.linalg.svd(standardized.to_numpy(), full_matrices=False)
        loadings = loadings[:components]
        factors = standardized.to_numpy() @ loadings.T
        coefficients = np.linalg.lstsq(
            np.column_stack([np.ones(len(training)), factors]),
            standardized.to_numpy(),
            rcond=None,
        )[0]
        for row in range(start, min(start + step, len(returns))):
            current = (returns.iloc[row] - mean) / standard_deviation
            predicted = np.concatenate([[1.0], current.to_numpy() @ loadings.T]) @ coefficients
            rows.append(current.to_numpy() - predicted)
            dates.append(returns.index[row])
    return pd.DataFrame(rows, index=dates, columns=returns.columns)
