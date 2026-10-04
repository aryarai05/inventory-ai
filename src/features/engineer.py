"""
Feature Engineering Module
Constructs leakage-free time-series lags, shifted rolling statistics, calendar indicators,
and operational inventory signals for machine learning demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


class FeatureEngineer:
    def __init__(
        self,
        lag_days: List[int] = [1, 7, 14, 28],
        rolling_windows: List[int] = [7, 14, 28],
        target_column: str = "sales_quantity",
        group_column: str = "product_id",
        date_column: str = "date",
    ):
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        self.target_column = target_column
        self.group_column = group_column
        self.date_column = date_column

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates all time-series features strictly avoiding lookahead bias / data leakage.
        """
        data = df.copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.sort_values(by=[self.group_column, self.date_column]).reset_index(drop=True)

        grouped = data.groupby(self.group_column)

        # 1. Historical Lag Features
        for lag in self.lag_days:
            data[f"lag_{lag}"] = grouped[self.target_column].shift(lag)

        # 2. Shifted Rolling Statistics (Shift by 1 first to guarantee ZERO lookahead leakage!)
        shifted_target = grouped[self.target_column].shift(1)
        for window in self.rolling_windows:
            data[f"rolling_mean_{window}"] = grouped[self.target_column].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            data[f"rolling_std_{window}"] = grouped[self.target_column].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )

        # Rolling min/max to capture volatility range
        data["rolling_min_7"] = grouped[self.target_column].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).min()
        )
        data["rolling_max_7"] = grouped[self.target_column].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).max()
        )

        # 3. Calendar & Cyclical Features
        if "day_of_week" not in data.columns:
            data["day_of_week"] = data[self.date_column].dt.dayofweek
        if "week_of_year" not in data.columns:
            data["week_of_year"] = data[self.date_column].dt.isocalendar().week.astype(int)
        if "month" not in data.columns:
            data["month"] = data[self.date_column].dt.month
        if "quarter" not in data.columns:
            data["quarter"] = data[self.date_column].dt.quarter
        if "year" not in data.columns:
            data["year"] = data[self.date_column].dt.year
        if "is_weekend" not in data.columns:
            data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical transformations for month and day of week
        data["sin_day_of_week"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
        data["cos_day_of_week"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
        data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12.0)
        data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12.0)

        # 4. Economic & Price Signal Features
        # Price change relative to baseline unit price
        if "unit_price" in data.columns:
            data["effective_price"] = data["unit_price"] * (1.0 - data["discount_rate"])
            data["price_change"] = grouped["effective_price"].diff().fillna(0.0)
        else:
            data["price_change"] = 0.0

        # 5. Inventory & Operational Features
        # Previous inventory level (yesterday's ending inventory)
        if "ending_inventory" in data.columns:
            data["previous_inventory"] = grouped["ending_inventory"].shift(1).fillna(0)
        elif "beginning_inventory" in data.columns:
            data["previous_inventory"] = grouped["beginning_inventory"].shift(1).fillna(0)
        else:
            data["previous_inventory"] = 0

        # Supplier Lead Time
        if "lead_time_days" not in data.columns:
            data["supplier_lead_time"] = 10
        else:
            data["supplier_lead_time"] = data["lead_time_days"]

        # 6. Fill initial lag NaNs using first available value or 0
        feature_cols = [c for c in data.columns if c.startswith("lag_") or c.startswith("rolling_")]
        for col in feature_cols:
            data[col] = data[col].bfill().fillna(0)

        return data

    def split_time_aware(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs strict chronological splitting across all series.
        Train: First 70% of dates
        Val: Next 15% of dates
        Test: Final 15% of dates
        """
        unique_dates = np.sort(df[self.date_column].unique())
        n_dates = len(unique_dates)

        train_cutoff_idx = int(n_dates * train_ratio)
        val_cutoff_idx = int(n_dates * (train_ratio + val_ratio))

        train_end_date = unique_dates[train_cutoff_idx - 1]
        val_end_date = unique_dates[val_cutoff_idx - 1]

        train_df = df[df[self.date_column] <= train_end_date].copy()
        val_df = df[(df[self.date_column] > train_end_date) & (df[self.date_column] <= val_end_date)].copy()
        test_df = df[df[self.date_column] > val_end_date].copy()

        return train_df, val_df, test_df
