"""
Unit Tests for Model Evaluation Metrics
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
    forecast_bias,
    evaluate_forecast,
)


def test_mae_calculation():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |10-12| = 2, |20-18| = 2, |30-33| = 3. Mean = 7/3 = 2.333...
    assert pytest.approx(mean_absolute_error(y_true, y_pred), 0.001) == 2.333


def test_rmse_calculation():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 24.0, 27.0])
    # 0^2 + 4^2 + (-3)^2 = 0 + 16 + 9 = 25. Mean = 25/3 = 8.333. sqrt = 2.8867
    assert pytest.approx(root_mean_squared_error(y_true, y_pred), 0.001) == 2.887


def test_wape_calculation():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # sum(|err|) = 10 + 20 = 30. sum(actual) = 300. WAPE = 30/300 * 100 = 10.0%
    assert pytest.approx(weighted_absolute_percentage_error(y_true, y_pred), 0.001) == 10.0


def test_forecast_bias():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 120.0])
    # Overforecasting by 30 units on 200 total actual = +15.0%
    assert pytest.approx(forecast_bias(y_true, y_pred), 0.001) == 15.0


def test_evaluate_forecast_suite():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    metrics = evaluate_forecast(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["WAPE"] == 0.0
    assert metrics["Bias_Pct"] == 0.0
