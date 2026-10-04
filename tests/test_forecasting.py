"""
Unit Tests for Forecasting Models and Predictor
"""

import pytest
import numpy as np
import pandas as pd
from src.models.baseline import NaiveBaselineForecaster, MovingAverageForecaster
from src.forecasting.predictor import DemandPredictor


def test_baseline_forecasters():
    df = pd.DataFrame({
        "lag_7": [15, 25, 35],
        "rolling_mean_7": [14.0, 24.0, 34.0],
    })

    naive = NaiveBaselineForecaster(lag=7)
    naive_preds = naive.predict(df)
    assert np.array_equal(naive_preds, np.array([15, 25, 35]))

    ma = MovingAverageForecaster(window=7)
    ma_preds = ma.predict(df)
    assert np.array_equal(ma_preds, np.array([14.0, 24.0, 34.0]))


def test_trained_predictor():
    predictor = DemandPredictor(model_path="models/best_forecasting_model.joblib")
    assert predictor.model is not None

    # Load small sample of featured dataset
    sample_df = pd.read_csv("data/processed/featured_dataset.csv").tail(200)
    forecast_df = predictor.predict_horizon(recent_data=sample_df, horizon_days=7)

    assert not forecast_df.empty
    assert "predicted_demand" in forecast_df.columns
    assert "lower_bound" in forecast_df.columns
    assert "upper_bound" in forecast_df.columns

    # Check non-negativity and bounds
    assert (forecast_df["predicted_demand"] >= 0).all()
    assert (forecast_df["lower_bound"] <= forecast_df["predicted_demand"]).all()
    assert (forecast_df["upper_bound"] >= forecast_df["predicted_demand"]).all()
