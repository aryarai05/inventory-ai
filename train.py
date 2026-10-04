"""
Model Training and Validation Pipeline Script
Trains baseline and machine learning models, performs hyperparameter tuning,
evaluates out-of-sample performance chronologically, logs runs to MLflow, and saves the best model.
"""

import os
import sys
import time
import json
import yaml
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.baseline import NaiveBaselineForecaster, MovingAverageForecaster
from src.models.ml_models import ModelTrainer
from src.models.registry import MLflowTracker
from src.evaluation.metrics import evaluate_forecast


def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print("[*] STEP 1: Loading Processed Datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    print(f"    - Training records  : {len(train_df):,}")
    print(f"    - Validation records: {len(val_df):,}")
    print(f"    - Test records      : {len(test_df):,}")

    trainer = ModelTrainer(random_state=config["project"]["random_seed"])
    X_train, y_train = trainer.prepare_features(train_df, is_train=True)
    X_val, y_val = trainer.prepare_features(val_df, is_train=False)
    X_test, y_test = trainer.prepare_features(test_df, is_train=False)

    tracker = MLflowTracker(
        tracking_uri=config["project"]["tracking_uri"],
        experiment_name=config["project"]["experiment_name"],
    )

    dataset_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "features_count": X_train.shape[1],
        "version": "1.0",
    }

    results = []
    models_dict = {}

    # -------------------------------------------------------------
    # MODEL 1: Naive Historical Baseline (Lag 7 - Same Day Last Week)
    # -------------------------------------------------------------
    print("\n[*] Training Model 1: Naive Baseline (Lag 7)...")
    t0 = time.time()
    naive_model = NaiveBaselineForecaster(lag=7)
    naive_model.fit(train_df, y_train)
    t_train = time.time() - t0

    naive_preds = naive_model.predict(test_df)
    naive_metrics = evaluate_forecast(y_test, naive_preds)
    naive_metrics["Model"] = "1. Naive (Lag 7)"
    naive_metrics["Train_Time_Sec"] = round(t_train, 3)
    results.append(naive_metrics)
    models_dict["Naive"] = naive_model

    tracker.log_model_run(
        model_name="Naive_Baseline",
        params={"lag": 7},
        metrics=naive_metrics,
        dataset_info=dataset_info,
        model_artifact=None,
    )

    # -------------------------------------------------------------
    # MODEL 2: Moving Average Baseline (7-Day Rolling Window)
    # -------------------------------------------------------------
    print("[*] Training Model 2: Moving Average (7-Day Rolling)...")
    t0 = time.time()
    ma_model = MovingAverageForecaster(window=7)
    ma_model.fit(train_df, y_train)
    t_train = time.time() - t0

    ma_preds = ma_model.predict(test_df)
    ma_metrics = evaluate_forecast(y_test, ma_preds)
    ma_metrics["Model"] = "2. Moving Average (7D)"
    ma_metrics["Train_Time_Sec"] = round(t_train, 3)
    results.append(ma_metrics)
    models_dict["Moving_Average"] = ma_model

    tracker.log_model_run(
        model_name="Moving_Average_7D",
        params={"window": 7},
        metrics=ma_metrics,
        dataset_info=dataset_info,
        model_artifact=None,
    )

    # -------------------------------------------------------------
    # MODEL 3: Ridge Linear Regression
    # -------------------------------------------------------------
    print("[*] Training Model 3: Regularized Ridge Linear Regression...")
    t0 = time.time()
    ridge_model = trainer.train_ridge(X_train, y_train, alpha=1.0)
    t_train = time.time() - t0

    ridge_preds = np.clip(ridge_model.predict(X_test), 0, None)
    ridge_metrics = evaluate_forecast(y_test, ridge_preds)
    ridge_metrics["Model"] = "3. Ridge Regression"
    ridge_metrics["Train_Time_Sec"] = round(t_train, 3)
    results.append(ridge_metrics)
    models_dict["Ridge"] = ridge_model

    tracker.log_model_run(
        model_name="Ridge_Regression",
        params={"alpha": 1.0},
        metrics=ridge_metrics,
        dataset_info=dataset_info,
        model_artifact=ridge_model,
        feature_names=list(X_train.columns),
    )

    # -------------------------------------------------------------
    # MODEL 4: Random Forest Regressor
    # -------------------------------------------------------------
    print("[*] Training Model 4: Random Forest Regressor...")
    t0 = time.time()
    rf_model = trainer.train_random_forest(
        X_train,
        y_train,
        n_estimators=config["models"]["random_forest"]["n_estimators"],
        max_depth=config["models"]["random_forest"]["max_depth"],
    )
    t_train = time.time() - t0

    rf_preds = np.clip(rf_model.predict(X_test), 0, None)
    rf_metrics = evaluate_forecast(y_test, rf_preds)
    rf_metrics["Model"] = "4. Random Forest"
    rf_metrics["Train_Time_Sec"] = round(t_train, 3)
    results.append(rf_metrics)
    models_dict["RandomForest"] = rf_model

    tracker.log_model_run(
        model_name="Random_Forest",
        params=config["models"]["random_forest"],
        metrics=rf_metrics,
        dataset_info=dataset_info,
        model_artifact=rf_model,
        feature_names=list(X_train.columns),
    )

    # -------------------------------------------------------------
    # MODEL 5: LightGBM Regressor with Validation Hyperparameter Tuning
    # -------------------------------------------------------------
    print("[*] Training Model 5: LightGBM Regressor with Chronological Tuning...")
    t0 = time.time()
    best_lgb, best_params, best_score = trainer.tune_lightgbm(
        X_train, y_train, X_val, y_val
    )
    t_train = time.time() - t0
    print(f"    - Best tuned parameters: {best_params} (Val MAE: {best_score:.3f})")

    lgb_preds = np.clip(best_lgb.predict(X_test), 0, None)
    lgb_metrics = evaluate_forecast(y_test, lgb_preds)
    lgb_metrics["Model"] = "5. LightGBM (Tuned)"
    lgb_metrics["Train_Time_Sec"] = round(t_train, 3)
    results.append(lgb_metrics)
    models_dict["LightGBM"] = best_lgb

    tracker.log_model_run(
        model_name="LightGBM_Tuned",
        params=best_params,
        metrics=lgb_metrics,
        dataset_info=dataset_info,
        model_artifact=best_lgb,
        feature_names=list(X_train.columns),
        tags={"production_candidate": "true"},
    )

    # -------------------------------------------------------------
    # Comparison & Model Selection
    # -------------------------------------------------------------
    comp_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "MAPE", "WAPE", "Bias_Pct", "Train_Time_Sec"]]
    comp_df = comp_df.sort_values(by="MAE").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("              OUT-OF-SAMPLE MODEL EVALUATION LEADERBOARD (TEST SET)")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80)

    # Save comparison files
    comp_df.to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)
    with open(os.path.join(models_dir, "model_comparison.json"), "w") as f:
        json.dump(comp_df.to_dict(orient="records"), f, indent=2)

    # Save Best Model Artifact & Metadata
    model_payload = {
        "model": best_lgb,
        "trainer": trainer,
        "features": list(X_train.columns),
        "best_params": best_params,
        "metrics": lgb_metrics,
    }
    joblib.dump(model_payload, os.path.join(models_dir, "best_forecasting_model.joblib"))
    print(f"[+] Best Model (LightGBM) saved to {models_dir}/best_forecasting_model.joblib")

    # Generate Feature Importance Table
    feature_importance_df = pd.DataFrame({
        "feature": list(X_train.columns),
        "importance": best_lgb.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)
    print(f"[+] Top 5 predictive features:\n{feature_importance_df.head(5).to_string(index=False)}")

    # Save Test Set with Predictions for downstream inventory simulation and dashboard
    test_out = test_df.copy()
    test_out["pred_naive"] = naive_preds
    test_out["pred_ma"] = ma_preds
    test_out["pred_ridge"] = ridge_preds
    test_out["pred_rf"] = rf_preds
    test_out["predicted_demand"] = lgb_preds
    test_out["forecast_error"] = test_out["predicted_demand"] - test_out["sales_quantity"]
    test_out["abs_error"] = np.abs(test_out["forecast_error"])
    test_out.to_csv(os.path.join(processed_dir, "test_predictions.csv"), index=False)
    print(f"[+] Out-of-sample predictions saved to {processed_dir}/test_predictions.csv")


if __name__ == "__main__":
    main()
