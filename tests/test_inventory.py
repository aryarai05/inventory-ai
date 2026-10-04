"""
Unit Tests for Inventory Formulas and Policy
"""

import pytest
import math
from src.inventory.policy import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_days_of_supply,
    calculate_economic_order_quantity,
)


def test_safety_stock_calculation():
    # L = 9 days, sigma_d = 4, d_bar = 20, sigma_L = 1.0, Z = 1.645
    # var = (9 * 16) + (400 * 1) = 144 + 400 = 544
    # sqrt(544) = 23.3238
    # SS = 1.645 * 23.3238 = 38.36 -> ceil = 39
    ss = calculate_safety_stock(
        lead_time_days=9.0,
        demand_std=4.0,
        mean_daily_demand=20.0,
        lead_time_std=1.0,
        service_level_z=1.645,
    )
    assert ss == 39


def test_reorder_point_calculation():
    # Lead time = 10, mean daily = 15, safety stock = 40
    # ROP = (10 * 15) + 40 = 190
    rop = calculate_reorder_point(
        lead_time_days=10.0,
        mean_daily_demand=15.0,
        safety_stock=40,
    )
    assert rop == 190


def test_days_of_supply():
    # 300 units on hand / 15 units/day = 20.0 days
    dos = calculate_days_of_supply(current_inventory=300.0, mean_daily_demand=15.0)
    assert dos == 20.0

    # Zero demand edge case
    dos_zero = calculate_days_of_supply(current_inventory=100.0, mean_daily_demand=0.0)
    assert dos_zero == 999.0


def test_economic_order_quantity():
    # Annual demand = 3650 units (10/day)
    # Order cost = $100
    # Unit cost = $50, holding cost rate = 0.20 -> H = $10/unit/yr
    # EOQ = sqrt( (2 * 3650 * 100) / 10 ) = sqrt(73000) = 270.18 -> ceil = 271
    eoq = calculate_economic_order_quantity(
        annual_demand=3650.0,
        fixed_order_cost=100.0,
        unit_cost=50.0,
        holding_cost_rate=0.20,
    )
    assert eoq == 271
