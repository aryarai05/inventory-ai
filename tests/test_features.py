"""
Unit Tests for Feature Engineering & Lookahead Leakage Prevention
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer


def test_feature_generation_and_leakage_absence():
    # Construct a synthetic sequential sequence
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    sales = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df = pd.DataFrame({
        "date": dates,
        "product_id": ["P1"] * 10,
        "sales_quantity": sales,
        "ending_inventory": [500 - s for s in sales],
        "discount_rate": [0.0] * 10,
        "unit_price": [10.0] * 10,
        "lead_time_days": [5] * 10,
    })

    fe = FeatureEngineer(
        lag_days=[1, 7],
        rolling_windows=[3],
        target_column="sales_quantity",
        group_column="product_id",
        date_column="date",
    )
    res = fe.generate_features(df)

    # 1. Check lag_1 for day index 1 (date 2024-01-02, sales=20). lag_1 MUST be 10 (day 0 sales)
    assert res.loc[1, "lag_1"] == 10
    assert res.loc[2, "lag_1"] == 20

    # 2. Check rolling_mean_3 for day index 3 (date 2024-01-04, sales=40)
    # Since it is shifted by 1 to prevent leakage, the window covers days 0, 1, 2: sales = [10, 20, 30]
    # Mean of (10+20+30)/3 = 20.0. NOT including today's 40!
    assert res.loc[3, "rolling_mean_3"] == 20.0

    # 3. Check calendar features exist
    assert "day_of_week" in res.columns
    assert "month" in res.columns
    assert "is_weekend" in res.columns


def test_chronological_split():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "product_id": ["P1"] * 100,
        "sales_quantity": np.random.randint(10, 50, size=100),
    })

    fe = FeatureEngineer(date_column="date")
    train, val, test = fe.split_time_aware(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Verify strict temporal ordering
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
