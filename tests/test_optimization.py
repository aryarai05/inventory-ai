"""
Unit Tests for PuLP Inventory Replenishment Optimization
"""

import pytest
from src.optimization.pulp_optimizer import InventoryPuLPOptimizer


def test_pulp_optimization_feasibility():
    optimizer = InventoryPuLPOptimizer(
        holding_cost_rate=0.20,
        fixed_order_cost=50.0,
        stockout_penalty_multiplier=1.5,
    )

    # 7-day horizon with initial inventory of 20 and daily demand of 10
    forecast_demands = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    result = optimizer.optimize_sku_horizon(
        product_id="TEST_SKU",
        initial_inventory=20,
        forecast_demands=forecast_demands,
        lead_time_days=2,
        unit_cost=15.0,
        unit_price=30.0,
        safety_stock_target=10,
        min_order_qty=10,
    )

    assert result["status"] == "Optimal"
    assert result["horizon_days"] == 7
    assert result["total_optimal_cost"] > 0
    assert "daily_plan" in result
    assert len(result["daily_plan"]) == 7

    # Verify order quantities are non-negative
    for _, row in result["daily_plan"].iterrows():
        assert row["recommended_order"] >= 0
        assert row["expected_inventory"] >= 0
        assert row["stockout_units"] >= 0
