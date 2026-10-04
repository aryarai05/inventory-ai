"""
Unit Tests for FastAPI Backend REST Endpoints & PuLP API
"""

import pytest
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_api_status_endpoint():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert "LightGBM" in data["forecasting_engine"]
    assert "PuLP" in data["optimization_solver"]


def test_api_kpis_endpoint():
    res = client.get("/api/kpis")
    assert res.status_code == 200
    data = res.json()
    assert "total_revenue" in data
    assert "total_inventory_value" in data
    assert data["total_units_sold"] > 0


def test_api_forecast_endpoint():
    res = client.get("/api/forecast?product_id=PROD_001&horizon_days=14")
    assert res.status_code == 200
    data = res.json()
    assert data["product_id"] == "PROD_001"
    assert len(data["forecast"]) == 14
    assert len(data["historical"]) > 0


def test_api_pulp_optimization_endpoint():
    payload = {
        "product_id": "PROD_001",
        "horizon_days": 14,
        "holding_cost_rate": 0.20,
        "fixed_order_cost": 150.0,
        "stockout_penalty_multiplier": 1.5,
    }
    res = client.post("/api/optimize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "Optimal"
    assert "total_optimal_cost" in data
    assert len(data["daily_plan"]) == 14


def test_api_sql_execute_endpoint():
    sql = "SELECT product_id, product_name, category FROM products LIMIT 5;"
    res = client.post("/api/sql/execute", json={"query": sql})
    assert res.status_code == 200
    data = res.json()
    assert len(data["rows"]) == 5
    assert "product_id" in data["columns"]
