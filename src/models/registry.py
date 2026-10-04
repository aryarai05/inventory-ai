"""
MLflow Model Registry & Experiment Tracking Module
Logs hyperparameter configurations, metrics (MAE, RMSE, MAPE, WAPE),
feature lists, dataset signatures, and model artifacts to a local MLflow repository.
"""

import os
import json
import numbers
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import pandas as pd
from typing import Dict, Any, Optional


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "Inventory_Demand_Forecasting",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set local tracking uri
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_model_run(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_info: Dict[str, Any],
        model_artifact: Optional[Any] = None,
        feature_names: Optional[list] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Executes an MLflow experiment run and records all parameters, metrics, and metadata.
        Returns the unique run_id.
        """
        run_name = f"{model_name}_{dataset_info.get('version', 'v1')}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # 1. Log Tags
            mlflow.set_tag("model_family", model_name)
            mlflow.set_tag("pipeline", "demand-forecasting")
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))

            # 2. Log Parameters
            for p_key, p_val in params.items():
                mlflow.log_param(p_key, p_val)
            for d_key, d_val in dataset_info.items():
                mlflow.log_param(f"data_{d_key}", d_val)

            # 3. Log Performance Metrics (Filter for numeric types only)
            for m_key, m_val in metrics.items():
                if isinstance(m_val, (int, float, numbers.Number)) and not isinstance(m_val, bool):
                    mlflow.log_metric(m_key, float(m_val))
                elif isinstance(m_val, str):
                    mlflow.log_param(f"meta_{m_key}", m_val)

            # 4. Log Feature Names as text artifact
            if feature_names:
                tmp_feat_file = f"features_{model_name}.json"
                with open(tmp_feat_file, "w") as f:
                    json.dump(feature_names, f, indent=2)
                mlflow.log_artifact(tmp_feat_file)
                if os.path.exists(tmp_feat_file):
                    os.remove(tmp_feat_file)

            # 5. Log Model Artifact
            if model_artifact is not None:
                try:
                    if "lightgbm" in model_name.lower():
                        mlflow.lightgbm.log_model(model_artifact, artifact_path="model")
                    else:
                        mlflow.sklearn.log_model(model_artifact, artifact_path="model")
                except Exception as e:
                    # Non-fatal notification
                    print(f"[*] Note: Logged model metadata (MLflow flavor notice: {e})")

            return run_id

    def get_experiment_summary(self) -> pd.DataFrame:
        """Retrieves history of runs from MLflow experiment."""
        current_exp = mlflow.get_experiment_by_name(self.experiment_name)
        if current_exp is None:
            return pd.DataFrame()
        runs_df = mlflow.search_runs(experiment_ids=[current_exp.experiment_id])
        return runs_df
