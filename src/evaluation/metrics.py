"""
Model Evaluation Metrics Module
Implements standard time-series forecasting metrics: MAE, RMSE, MAPE, WAPE, and Forecast Bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Uses epsilon stabilization to safely handle zero demand days without division by zero.
    """
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100.0)


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Weighted Absolute Percentage Error (WAPE).
    Formula: sum(|y_true - y_pred|) / sum(y_true) * 100
    Standard industry metric for retail/inventory demand planning.
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Forecast Bias / Mean Error: sum(y_pred - y_true) / sum(y_true) * 100.
    Positive indicates over-forecasting, negative indicates under-forecasting.
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(y_pred - y_true) / total_actual * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates complete suite of forecasting metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Predictions cannot be negative in physical inventory sales
    y_pred_clipped = np.clip(y_pred, 0, None)

    return {
        "MAE": round(mean_absolute_error(y_true, y_pred_clipped), 3),
        "RMSE": round(root_mean_squared_error(y_true, y_pred_clipped), 3),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred_clipped), 2),
        "WAPE": round(weighted_absolute_percentage_error(y_true, y_pred_clipped), 2),
        "Bias_Pct": round(forecast_bias(y_true, y_pred_clipped), 2),
    }
