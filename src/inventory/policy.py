"""
Inventory Policy & Formula Calculations
Computes Safety Stock, Reorder Point, Days of Supply, and Economic Order Quantity (EOQ).
"""

import math
import numpy as np
from typing import Dict, Any


def calculate_safety_stock(
    lead_time_days: float,
    demand_std: float,
    mean_daily_demand: float,
    lead_time_std: float = 1.5,
    service_level_z: float = 1.645, # 95% service level
) -> int:
    """
    Computes Safety Stock accounting for both demand variance and lead-time variance.
    Formula: SS = Z * sqrt( L * sigma_d^2 + d_bar^2 * sigma_L^2 )
    """
    variance_term = (lead_time_days * (demand_std ** 2)) + ((mean_daily_demand ** 2) * (lead_time_std ** 2))
    ss = service_level_z * math.sqrt(max(0.1, variance_term))
    return int(math.ceil(ss))


def calculate_reorder_point(
    lead_time_days: float,
    mean_daily_demand: float,
    safety_stock: int,
) -> int:
    """
    Computes Reorder Point (ROP).
    Formula: ROP = (Lead Time * Mean Daily Demand) + Safety Stock
    """
    lead_time_demand = lead_time_days * mean_daily_demand
    return int(math.ceil(lead_time_demand + safety_stock))


def calculate_days_of_supply(
    current_inventory: float,
    mean_daily_demand: float,
) -> float:
    """Computes Days of Supply (DOS): Current Inventory / Mean Daily Demand."""
    if mean_daily_demand <= 0:
        return 999.0 if current_inventory > 0 else 0.0
    return round(current_inventory / mean_daily_demand, 1)


def calculate_economic_order_quantity(
    annual_demand: float,
    fixed_order_cost: float,
    unit_cost: float,
    holding_cost_rate: float = 0.20,
) -> int:
    """
    Computes Wilson Economic Order Quantity (EOQ).
    Formula: EOQ = sqrt( (2 * D * S) / H )
    """
    holding_cost_per_unit = unit_cost * holding_cost_rate
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 10
    eoq = math.sqrt((2 * annual_demand * fixed_order_cost) / holding_cost_per_unit)
    return int(math.ceil(eoq))
