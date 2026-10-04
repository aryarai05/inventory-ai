"""
Baseline Forecasting Models
Implements statistical benchmarks: Naive Lag Baseline and Moving Average.
"""

import numpy as np
import pandas as pd
from typing import Optional


class NaiveBaselineForecaster:
    """
    Naive baseline forecasting using same-day-last-week (lag_7) or prior-day (lag_1).
    """
    def __init__(self, lag: int = 7):
        self.lag = lag

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NaiveBaselineForecaster":
        # Baselines are non-parametric heuristics
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag_col = f"lag_{self.lag}"
        if lag_col in X.columns:
            return np.clip(X[lag_col].values, 0, None)
        elif "lag_1" in X.columns:
            return np.clip(X["lag_1"].values, 0, None)
        else:
            return np.zeros(len(X))


class MovingAverageForecaster:
    """
    Moving average baseline utilizing shifted rolling mean features.
    """
    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MovingAverageForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        roll_col = f"rolling_mean_{self.window}"
        if roll_col in X.columns:
            return np.clip(X[roll_col].values, 0, None)
        elif "rolling_mean_7" in X.columns:
            return np.clip(X["rolling_mean_7"].values, 0, None)
        else:
            return np.zeros(len(X))
