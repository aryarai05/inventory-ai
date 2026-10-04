"""
Unit Tests for Inventory Risk Classification Engine
"""

import pytest
import pandas as pd
from src.inventory.risk_engine import InventoryRiskEngine


def test_risk_classification_rules():
    engine = InventoryRiskEngine(
        service_level_z=1.645,
        fixed_order_cost=100.0,
        holding_cost_rate=0.20,
    )

    # Mock products
    products_df = pd.DataFrame([{
        "product_id": "P_RISK",
        "product_name": "Test Risk Product",
        "category": "Electronics",
        "unit_cost": 50.0,
        "unit_price": 100.0,
        "lead_time_days": 10,
    }])

    # Case 1: Low inventory (below safety stock) -> should be CRITICAL
    inv_df_critical = pd.DataFrame([{
        "date": "2024-01-01",
        "product_id": "P_RISK",
        "ending_inventory": 5, # critically low
        "sales_quantity": 20,
    }])
    forecast_df = pd.DataFrame([
        {"product_id": "P_RISK", "predicted_demand": 25.0} for _ in range(14)
    ])

    crit_res = engine.evaluate_inventory_risk(inv_df_critical, forecast_df, products_df)
    assert len(crit_res) == 1
    assert crit_res.iloc[0]["risk_level"] == "CRITICAL STOCKOUT RISK"
    assert crit_res.iloc[0]["reorder_needed"] == True
    assert crit_res.iloc[0]["recommended_reorder_qty"] > 0
    assert "CRITICAL" in crit_res.iloc[0]["diagnostic_reason"]

    # Case 2: Huge excess inventory -> should be EXCESS INVENTORY
    inv_df_excess = pd.DataFrame([{
        "date": "2024-01-01",
        "product_id": "P_RISK",
        "ending_inventory": 5000, # 5000 / 25 = 200 days of supply
        "sales_quantity": 25,
    }])
    excess_res = engine.evaluate_inventory_risk(inv_df_excess, forecast_df, products_df)
    assert excess_res.iloc[0]["risk_level"] == "EXCESS INVENTORY"
    assert excess_res.iloc[0]["days_of_supply"] > 60.0
