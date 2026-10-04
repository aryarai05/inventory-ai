"""
Machine Learning Forecasting Models
Implements Random Forest, Ridge Regression, and LightGBM models with hyperparameter tuning.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from src.evaluation.metrics import evaluate_forecast


FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
    "rolling_min_7", "rolling_max_7",
    "day_of_week", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_holiday",
    "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month",
    "promotion_flag", "discount_rate", "price_change",
    "previous_inventory", "supplier_lead_time",
    "unit_cost", "unit_price",
]


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.category_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.feature_cols = list(FEATURE_COLUMNS)

    def prepare_features(
        self,
        df: pd.DataFrame,
        is_train: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encodes categoricals and extracts feature matrix and target vector."""
        data = df.copy()

        # Target column
        y = data["sales_quantity"].values if "sales_quantity" in data.columns else np.zeros(len(data))

        # Categorical Encodings
        if is_train:
            data["category_code"] = self.category_encoder.fit_transform(data["category"].astype(str))
            data["product_code"] = self.product_encoder.fit_transform(data["product_id"].astype(str))
        else:
            # Handle unseen categories gracefully
            cat_classes = set(self.category_encoder.classes_)
            data["category_clean"] = data["category"].astype(str).apply(lambda x: x if x in cat_classes else self.category_encoder.classes_[0])
            data["category_code"] = self.category_encoder.transform(data["category_clean"])

            prod_classes = set(self.product_encoder.classes_)
            data["product_clean"] = data["product_id"].astype(str).apply(lambda x: x if x in prod_classes else self.product_encoder.classes_[0])
            data["product_code"] = self.product_encoder.transform(data["product_clean"])

        active_features = self.feature_cols + ["category_code", "product_code"]
        X = data[active_features].fillna(0)

        return X, y

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 12,
    ) -> RandomForestRegressor:
        """Trains a Random Forest ensemble."""
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_train, y_train)
        return rf

    def train_ridge(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        alpha: float = 1.0,
    ) -> Ridge:
        """Trains a regularized Ridge linear regression model."""
        ridge = Ridge(alpha=alpha, random_state=self.random_state)
        ridge.fit(X_train, y_train)
        return ridge

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> lgb.LGBMRegressor:
        """Trains a gradient boosted decision tree model with LightGBM."""
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        model = lgb.LGBMRegressor(**default_params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        return model

    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> Tuple[lgb.LGBMRegressor, Dict[str, Any], float]:
        """
        Performs time-aware validation grid search for LightGBM.
        Evaluates parameter combinations against the chronological validation set.
        """
        param_grid = [
            {"n_estimators": 100, "learning_rate": 0.08, "num_leaves": 20, "max_depth": 6},
            {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8},
            {"n_estimators": 200, "learning_rate": 0.04, "num_leaves": 40, "max_depth": 8},
            {"n_estimators": 200, "learning_rate": 0.06, "num_leaves": 31, "max_depth": 10},
        ]

        best_score = float("inf")
        best_params = {}
        best_model = None

        for p in param_grid:
            model = self.train_lightgbm(X_train, y_train, X_val, y_val, params=p)
            val_preds = model.predict(X_val)
            val_metrics = evaluate_forecast(y_val, val_preds)
            val_mae = val_metrics["MAE"]

            if val_mae < best_score:
                best_score = val_mae
                best_params = p
                best_model = model

        return best_model, best_params, best_score

    def save_model(self, model: Any, filepath: str) -> None:
        """Serializes trained model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)

    def load_model(self, filepath: str) -> Any:
        """Loads serialized model from disk."""
        return joblib.load(filepath)
