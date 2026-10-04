"""
Demand Predictor & Multi-Step Forward Forecasting Engine
Generates forward-looking demand predictions with uncertainty intervals for inventory replenishment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


class DemandPredictor:
    def __init__(self, model_path: str = "models/best_forecasting_model.joblib"):
        self.model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}. Run train.py first.")
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.trainer = payload["trainer"]
        self.feature_names = payload["features"]
        self.best_params = payload.get("best_params", {})
        self.metrics = payload.get("metrics", {})
        self.rmse = float(self.metrics.get("RMSE", 7.42))

    def predict_horizon(
        self,
        recent_data: pd.DataFrame,
        horizon_days: int = 14,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Generates forward-looking demand predictions for all SKUs over a specified horizon.
        Calculates upper and lower prediction intervals based on error variance and horizon propagation.
        """
        # Get the latest date available in the historical dataset
        recent_data = recent_data.copy()
        recent_data["date"] = pd.to_datetime(recent_data["date"])
        last_date = recent_data["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)

        # Z-score multiplier for normal distribution interval
        z_multiplier = 1.96 if confidence_level >= 0.95 else 1.645

        # Get latest SKU state
        sku_latest = recent_data.sort_values("date").groupby("product_id").last().reset_index()

        forecast_rows = []
        for _, sku in sku_latest.iterrows():
            pid = sku["product_id"]
            pname = sku.get("product_name", pid)
            cat = sku.get("category", "General")
            unit_cost = sku.get("unit_cost", 20.0)
            unit_price = sku.get("unit_price", 40.0)
            lead_time = int(sku.get("lead_time_days", 10))

            # Base features from most recent observation
            base_sales = float(sku.get("sales_quantity", 25.0))
            roll_7 = float(sku.get("rolling_mean_7", base_sales))
            roll_14 = float(sku.get("rolling_mean_14", base_sales))
            roll_28 = float(sku.get("rolling_mean_28", base_sales))
            std_7 = float(sku.get("rolling_std_7", 5.0))
            std_28 = float(sku.get("rolling_std_28", 6.0))
            curr_inv = float(sku.get("ending_inventory", 150.0))

            for h_idx, f_date in enumerate(future_dates, start=1):
                dow = f_date.dayofweek
                is_weekend = 1 if dow in [5, 6] else 0
                month = f_date.month

                # Create feature row
                row_dict = {
                    "product_id": pid,
                    "product_name": pname,
                    "category": cat,
                    "unit_cost": unit_cost,
                    "unit_price": unit_price,
                    "lead_time_days": lead_time,
                    "sales_quantity": 0, # dummy target
                    "lag_1": base_sales,
                    "lag_7": roll_7,
                    "lag_14": roll_14,
                    "lag_28": roll_28,
                    "rolling_mean_7": roll_7,
                    "rolling_mean_14": roll_14,
                    "rolling_mean_28": roll_28,
                    "rolling_std_7": std_7,
                    "rolling_std_28": std_28,
                    "rolling_min_7": max(0, roll_7 - std_7),
                    "rolling_max_7": roll_7 + std_7,
                    "day_of_week": dow,
                    "week_of_year": int(f_date.isocalendar().week),
                    "month": month,
                    "quarter": int(f_date.quarter),
                    "year": int(f_date.year),
                    "is_weekend": is_weekend,
                    "is_holiday": 0,
                    "sin_day_of_week": np.sin(2 * np.pi * dow / 7.0),
                    "cos_day_of_week": np.cos(2 * np.pi * dow / 7.0),
                    "sin_month": np.sin(2 * np.pi * month / 12.0),
                    "cos_month": np.cos(2 * np.pi * month / 12.0),
                    "promotion_flag": 0,
                    "discount_rate": 0.0,
                    "price_change": 0.0,
                    "previous_inventory": max(0, curr_inv - (roll_7 * h_idx)),
                    "supplier_lead_time": lead_time,
                }

                # Encode through trainer
                dummy_df = pd.DataFrame([row_dict])
                X_h, _ = self.trainer.prepare_features(dummy_df, is_train=False)

                # Point prediction
                pred = float(np.clip(self.model.predict(X_h)[0], 0, None))

                # Error uncertainty grows with square root of forward step
                horizon_uncertainty = z_multiplier * self.rmse * np.sqrt(1 + 0.05 * (h_idx - 1))
                lower_bound = max(0.0, pred - horizon_uncertainty)
                upper_bound = pred + horizon_uncertainty

                forecast_rows.append({
                    "date": f_date.strftime("%Y-%m-%d"),
                    "product_id": pid,
                    "product_name": pname,
                    "category": cat,
                    "unit_cost": unit_cost,
                    "unit_price": unit_price,
                    "lead_time_days": lead_time,
                    "forecast_step": h_idx,
                    "predicted_demand": round(pred, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2),
                })

        return pd.DataFrame(forecast_rows)
