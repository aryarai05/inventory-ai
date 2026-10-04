"""
Inventory Risk Classification Engine
Evaluates SKU health, computes safety stock buffers, identifies stockout and excess risks,
and generates auditable operational diagnostic explanations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.inventory.policy import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_days_of_supply,
    calculate_economic_order_quantity,
)


class InventoryRiskEngine:
    def __init__(
        self,
        service_level_z: float = 1.645, # 95% service level
        fixed_order_cost: float = 150.0,
        holding_cost_rate: float = 0.20,
    ):
        self.service_level_z = service_level_z
        self.fixed_order_cost = fixed_order_cost
        self.holding_cost_rate = holding_cost_rate

    def evaluate_inventory_risk(
        self,
        inventory_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges current on-hand inventory with forward forecasted demand to classify risks and recommend reorders.
        """
        # 1. Get latest inventory snapshot per product
        inv_copy = inventory_df.copy()
        inv_copy["date"] = pd.to_datetime(inv_copy["date"])
        latest_inv = inv_copy.sort_values("date").groupby("product_id").last().reset_index()

        # 2. Get forecasted demand statistics per product over horizon
        f_stats = forecast_df.groupby("product_id").agg(
            total_forecasted_demand=("predicted_demand", "sum"),
            mean_forecasted_daily_demand=("predicted_demand", "mean"),
            forecast_horizon_days=("predicted_demand", "count"),
        ).reset_index()

        # 3. Merge metadata
        merged = latest_inv.merge(f_stats, on="product_id", how="left")
        merged = merged.merge(
            products_df[["product_id", "product_name", "category", "unit_cost", "unit_price", "lead_time_days"]],
            on="product_id",
            how="left",
            suffixes=("", "_meta")
        )

        # Standard deviation of demand estimated from recent sales variance
        recent_std = inv_copy.groupby("product_id")["sales_quantity"].std().fillna(3.0).to_dict()

        risk_rows = []
        for _, row in merged.iterrows():
            pid = row["product_id"]
            pname = row.get("product_name", pid)
            cat = row.get("category", "General")
            unit_cost = float(row.get("unit_cost", 20.0))
            unit_price = float(row.get("unit_price", 40.0))
            lead_time = int(row.get("lead_time_days", 10))
            curr_inv = int(row.get("ending_inventory", 0))

            mean_demand = float(row.get("mean_forecasted_daily_demand", 20.0))
            if np.isnan(mean_demand) or mean_demand <= 0:
                mean_demand = 15.0

            d_std = recent_std.get(pid, mean_demand * 0.25)
            if np.isnan(d_std) or d_std <= 0:
                d_std = max(1.0, mean_demand * 0.25)

            # Safety Stock and Reorder Point
            ss = calculate_safety_stock(
                lead_time_days=lead_time,
                demand_std=d_std,
                mean_daily_demand=mean_demand,
                lead_time_std=1.2,
                service_level_z=self.service_level_z,
            )
            rop = calculate_reorder_point(
                lead_time_days=lead_time,
                mean_daily_demand=mean_demand,
                safety_stock=ss,
            )
            dos = calculate_days_of_supply(curr_inv, mean_demand)

            # Annualized demand for EOQ
            annual_demand = mean_demand * 365.0
            eoq = calculate_economic_order_quantity(
                annual_demand=annual_demand,
                fixed_order_cost=self.fixed_order_cost,
                unit_cost=unit_cost,
                holding_cost_rate=self.holding_cost_rate,
            )

            # Risk Classification Logic
            reorder_needed = False
            rec_reorder_qty = 0

            if curr_inv < ss or dos < lead_time:
                risk_level = "CRITICAL STOCKOUT RISK"
                reorder_needed = True
                rec_reorder_qty = max(eoq, rop - curr_inv + ss)
                reason = (
                    f"CRITICAL: Current inventory ({curr_inv} units) is below safety stock ({ss} units). "
                    f"Days of supply ({dos:.1f}d) is less than supplier lead time ({lead_time}d). "
                    f"Stockout is imminent before replenishment can arrive. Immediate reorder required."
                )
            elif curr_inv <= rop:
                risk_level = "HIGH RISK"
                reorder_needed = True
                rec_reorder_qty = max(eoq, rop - curr_inv)
                reason = (
                    f"HIGH RISK: Current inventory ({curr_inv} units) has breached reorder point ({rop} units). "
                    f"Replenishment of {rec_reorder_qty} units must be placed within {lead_time} days to avoid stockout."
                )
            elif curr_inv <= rop + (0.5 * ss):
                risk_level = "MEDIUM RISK"
                reorder_needed = False
                rec_reorder_qty = 0
                reason = (
                    f"MEDIUM RISK: Inventory ({curr_inv} units) is approaching reorder threshold ({rop} units). "
                    f"Supply buffer is currently {dos:.1f} days. Monitor closely for demand spikes."
                )
            elif dos > 60.0:
                risk_level = "EXCESS INVENTORY"
                reorder_needed = False
                rec_reorder_qty = 0
                excess_units = max(0, curr_inv - int(mean_demand * 45))
                capital_tied = excess_units * unit_cost
                reason = (
                    f"EXCESS: Days of supply ({dos:.1f}d) exceeds 60-day operational limit. "
                    f"Estimated {excess_units} surplus units tying up ${capital_tied:,.2f} in working capital."
                )
            else:
                risk_level = "HEALTHY"
                reorder_needed = False
                rec_reorder_qty = 0
                reason = (
                    f"HEALTHY: Inventory ({curr_inv} units) is optimal. Days of supply is {dos:.1f} days, "
                    f"comfortably covering {lead_time}-day lead time plus safety buffer ({ss} units)."
                )

            # Financial implications
            holding_cost_daily = (curr_inv * unit_cost * (self.holding_cost_rate / 365.0))
            potential_stockout_loss = max(0, (mean_demand * lead_time) - curr_inv) * unit_price

            risk_rows.append({
                "product_id": pid,
                "product_name": pname,
                "category": cat,
                "unit_cost": unit_cost,
                "unit_price": unit_price,
                "lead_time_days": lead_time,
                "current_inventory": curr_inv,
                "mean_daily_demand": round(mean_demand, 1),
                "safety_stock": ss,
                "reorder_point": rop,
                "days_of_supply": dos,
                "risk_level": risk_level,
                "reorder_needed": reorder_needed,
                "recommended_reorder_qty": int(rec_reorder_qty),
                "eoq": eoq,
                "daily_holding_cost": round(holding_cost_daily, 2),
                "potential_stockout_loss": round(potential_stockout_loss, 2),
                "diagnostic_reason": reason,
            })

        return pd.DataFrame(risk_rows).sort_values(
            by=["reorder_needed", "days_of_supply"], ascending=[False, True]
        ).reset_index(drop=True)
