"""
Inventory Optimization and Forward Planning Pipeline
Generates ML demand forecasts, evaluates inventory risk classifications,
runs PuLP mathematical replenishment optimization, and exports Power BI datasets.
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.forecasting.predictor import DemandPredictor
from src.inventory.risk_engine import InventoryRiskEngine
from src.optimization.pulp_optimizer import InventoryPuLPOptimizer
from src.data.database import DatabaseManager


def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    powerbi_dir = config["data"]["powerbi_dir"]
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(powerbi_dir, exist_ok=True)

    print("[*] STEP 1: Generating 14-Day Forward Demand Forecasts with Best ML Model (LightGBM)...")
    predictor = DemandPredictor(model_path="models/best_forecasting_model.joblib")
    
    # Load recent history from processed features or database
    featured_df = pd.read_csv(os.path.join(processed_dir, "featured_dataset.csv"))
    products_df = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    inventory_df = pd.read_csv(os.path.join(raw_dir, "inventory.csv"))
    sales_df = pd.read_csv(os.path.join(raw_dir, "sales.csv"))

    forecast_df = predictor.predict_horizon(
        recent_data=featured_df,
        horizon_days=config["inventory"]["forecast_horizon_days"],
        confidence_level=0.95,
    )
    forecast_path = os.path.join(processed_dir, "forward_demand_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"[+] 14-Day forward forecasts generated for {forecast_df['product_id'].nunique()} SKUs. Saved to {forecast_path}")

    print("[*] STEP 2: Running Inventory Risk Classification Engine...")
    risk_engine = InventoryRiskEngine(
        service_level_z=config["inventory"]["service_level_z"],
        fixed_order_cost=config["inventory"]["fixed_order_cost"],
        holding_cost_rate=config["inventory"]["default_holding_cost_rate"],
    )
    risk_summary_df = risk_engine.evaluate_inventory_risk(
        inventory_df=inventory_df,
        forecast_df=forecast_df,
        products_df=products_df,
    )
    risk_path = os.path.join(processed_dir, "inventory_risk_report.csv")
    risk_summary_df.to_csv(risk_path, index=False)
    print(f"[+] Inventory risk evaluated across all SKUs. Saved to {risk_path}")

    # Risk breakdown count
    risk_counts = risk_summary_df["risk_level"].value_counts().to_dict()
    print("[+] SKU Risk Breakdown:")
    for r_lvl, count in risk_counts.items():
        print(f"    - {r_lvl:25s}: {count:2d} products")

    print("[*] STEP 3: Running PuLP Mixed-Integer Linear Programming Optimization...")
    optimizer = InventoryPuLPOptimizer(
        holding_cost_rate=config["inventory"]["default_holding_cost_rate"],
        fixed_order_cost=config["inventory"]["fixed_order_cost"],
        stockout_penalty_multiplier=config["inventory"]["stockout_penalty_multiplier"],
        warehouse_capacity=config["inventory"]["max_warehouse_capacity_units"],
    )

    opt_recommendations_df, opt_summary = optimizer.optimize_portfolio(
        risk_summary_df=risk_summary_df,
        forecast_df=forecast_df,
        horizon_days=config["inventory"]["forecast_horizon_days"],
    )
    opt_path = os.path.join(processed_dir, "optimization_recommendations.csv")
    opt_recommendations_df.to_csv(opt_path, index=False)

    summary_file = os.path.join(reports_dir, "inventory_optimization_summary.json")
    with open(summary_file, "w") as f:
        json.dump(opt_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("               PORTFOLIO OPTIMIZATION FINANCIAL SUMMARY")
    print("=" * 70)
    for k, v in opt_summary.items():
        print(f"  {k:30s}: ${v:,.2f}" if "Cost" in k or "Penalty" in k else f"  {k:30s}: {v}")
    print("=" * 70)

    print("[*] STEP 4: Exporting Clean Datasets for Power BI Integration...")
    # 1. inventory_dashboard_data.csv
    risk_summary_df.to_csv(os.path.join(powerbi_dir, "inventory_dashboard_data.csv"), index=False)

    # 2. forecast_dashboard_data.csv
    forecast_df.to_csv(os.path.join(powerbi_dir, "forecast_dashboard_data.csv"), index=False)

    # 3. product_metrics.csv
    prod_metrics = risk_summary_df.merge(
        opt_recommendations_df[["product_id", "optimal_reorder_qty", "total_cost"]],
        on="product_id",
        how="left"
    )
    prod_metrics.to_csv(os.path.join(powerbi_dir, "product_metrics.csv"), index=False)

    # 4. regional_metrics.csv
    db = DatabaseManager(db_path=config["data"]["database_path"])
    reg_sql = """
    SELECT 
        r.region_id,
        r.region_name,
        r.warehouse_id,
        COUNT(DISTINCT s.product_id) AS active_products,
        SUM(s.sales_quantity) AS total_units_sold,
        ROUND(SUM(s.revenue), 2) AS total_revenue,
        SUM(s.unmet_demand) AS total_unmet_units,
        ROUND(100.0 * SUM(s.unmet_demand) / NULLIF(SUM(s.sales_quantity + s.unmet_demand), 0), 2) AS stockout_rate_pct
    FROM regions r
    JOIN sales s ON r.region_id = s.region_id
    GROUP BY r.region_id, r.region_name, r.warehouse_id
    ORDER BY total_revenue DESC;
    """
    reg_df = db.execute_query(reg_sql)
    reg_df.to_csv(os.path.join(powerbi_dir, "regional_metrics.csv"), index=False)
    print(f"[+] Power BI datasets exported to {powerbi_dir}/")
    print("[*] Inventory optimization pipeline completed successfully.")


if __name__ == "__main__":
    main()
