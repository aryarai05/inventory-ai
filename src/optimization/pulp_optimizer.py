"""
Mathematical Inventory Optimization Module using PuLP
Formulates and solves mixed-integer linear programming (MILP) replenishment problems
balancing Holding Costs, Fixed Ordering Costs, and Stockout Penalties across a planning horizon.
"""

import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class InventoryPuLPOptimizer:
    def __init__(
        self,
        holding_cost_rate: float = 0.20,
        fixed_order_cost: float = 150.0,
        stockout_penalty_multiplier: float = 1.5,
        warehouse_capacity: float = 50000.0,
    ):
        self.holding_cost_rate = holding_cost_rate
        self.fixed_order_cost = fixed_order_cost
        self.stockout_penalty_multiplier = stockout_penalty_multiplier
        self.warehouse_capacity = warehouse_capacity

    def optimize_sku_horizon(
        self,
        product_id: str,
        initial_inventory: int,
        forecast_demands: List[float],
        lead_time_days: int,
        unit_cost: float,
        unit_price: float,
        safety_stock_target: int,
        min_order_qty: int = 10,
    ) -> Dict[str, Any]:
        """
        Solves single-SKU dynamic multi-period lot sizing with lead times and shortages.
        
        Decision Variables:
        - Q[t] >= 0: Order quantity dispatched on day t
        - Y[t] in {0, 1}: Binary order placement indicator
        - I[t] >= 0: Ending inventory on day t
        - S[t] >= 0: Stockout (unmet demand) on day t
        """
        H = len(forecast_demands) # Horizon length (e.g. 14 or 30 days)
        daily_holding_cost = unit_cost * (self.holding_cost_rate / 365.0)
        stockout_penalty_per_unit = unit_price * self.stockout_penalty_multiplier
        big_M = max(10000.0, initial_inventory + sum(forecast_demands) * 2.0)

        # Create PuLP Model
        model = pulp.LpProblem(f"Inventory_Optimization_{product_id}", pulp.LpMinimize)

        # Variables
        Q = {t: pulp.LpVariable(f"OrderQty_{t}", lowBound=0, cat=pulp.LpContinuous) for t in range(H)}
        Y = {t: pulp.LpVariable(f"OrderTrigger_{t}", cat=pulp.LpBinary) for t in range(H)}
        I = {t: pulp.LpVariable(f"EndingInv_{t}", lowBound=0, cat=pulp.LpContinuous) for t in range(H)}
        S = {t: pulp.LpVariable(f"Stockout_{t}", lowBound=0, cat=pulp.LpContinuous) for t in range(H)}

        # Objective Function: Minimize Total Cost
        # Total Cost = Holding Cost + Fixed Order Cost + Stockout Penalty
        model += pulp.lpSum([
            (daily_holding_cost * I[t]) +
            (self.fixed_order_cost * Y[t]) +
            (stockout_penalty_per_unit * S[t])
            for t in range(H)
        ])

        # Constraints
        for t in range(H):
            # 1. Big-M Order Linking: Q[t] <= M * Y[t]
            model += Q[t] <= big_M * Y[t], f"BigM_Link_{t}"
            model += Q[t] >= min_order_qty * Y[t], f"MinOrder_{t}"

            # 2. Inventory Balance Equation:
            # I[t] = I[t-1] + Inflow[t] - Demand[t] + S[t]
            prev_inv = initial_inventory if t == 0 else I[t - 1]
            
            # Orders arrive after lead_time_days
            inflow = Q[t - lead_time_days] if (t - lead_time_days) >= 0 else 0

            model += (
                I[t] == prev_inv + inflow - forecast_demands[t] + S[t],
                f"InvBalance_{t}"
            )

        # Solve using default CBC solver
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = model.solve(solver)
        status_str = pulp.LpStatus[status]

        # Extract optimal trajectories
        plan_rows = []
        total_holding = 0.0
        total_ordering = 0.0
        total_shortage = 0.0
        total_orders_placed = 0

        for t in range(H):
            q_val = float(Q[t].varValue or 0.0)
            y_val = int(round(Y[t].varValue or 0.0))
            i_val = float(I[t].varValue or 0.0)
            s_val = float(S[t].varValue or 0.0)

            h_cost = i_val * daily_holding_cost
            o_cost = y_val * self.fixed_order_cost
            p_cost = s_val * stockout_penalty_per_unit

            total_holding += h_cost
            total_ordering += o_cost
            total_shortage += p_cost
            if y_val == 1:
                total_orders_placed += 1

            plan_rows.append({
                "day_step": t + 1,
                "forecast_demand": round(forecast_demands[t], 1),
                "recommended_order": int(round(q_val)),
                "order_triggered": y_val,
                "expected_inventory": int(round(i_val)),
                "stockout_units": int(round(s_val)),
                "holding_cost": round(h_cost, 2),
                "ordering_cost": round(o_cost, 2),
                "stockout_cost": round(p_cost, 2),
            })

        total_cost = total_holding + total_ordering + total_shortage

        return {
            "product_id": product_id,
            "status": status_str,
            "horizon_days": H,
            "initial_inventory": initial_inventory,
            "safety_stock_target": safety_stock_target,
            "total_optimal_cost": round(total_cost, 2),
            "total_holding_cost": round(total_holding, 2),
            "total_ordering_cost": round(total_ordering, 2),
            "total_stockout_penalty": round(total_shortage, 2),
            "total_orders_placed": total_orders_placed,
            "immediate_order_qty": plan_rows[0]["recommended_order"] if H > 0 else 0,
            "daily_plan": pd.DataFrame(plan_rows),
        }

    def optimize_portfolio(
        self,
        risk_summary_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        horizon_days: int = 14,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Runs PuLP optimization across all products in portfolio.
        Returns portfolio recommendations and aggregate cost breakdown.
        """
        results = []
        tot_holding = 0.0
        tot_ordering = 0.0
        tot_stockout = 0.0

        for _, item in risk_summary_df.iterrows():
            pid = item["product_id"]
            init_inv = int(item["current_inventory"])
            unit_cost = float(item["unit_cost"])
            unit_price = float(item["unit_price"])
            lead_time = int(item["lead_time_days"])
            ss = int(item["safety_stock"])

            # Extract horizon forecasts
            sku_f = forecast_df[forecast_df["product_id"] == pid].sort_values("forecast_step")
            if len(sku_f) >= horizon_days:
                demands = sku_f["predicted_demand"].values[:horizon_days].tolist()
            else:
                demands = [float(item["mean_daily_demand"])] * horizon_days

            opt_res = self.optimize_sku_horizon(
                product_id=pid,
                initial_inventory=init_inv,
                forecast_demands=demands,
                lead_time_days=lead_time,
                unit_cost=unit_cost,
                unit_price=unit_price,
                safety_stock_target=ss,
            )

            tot_holding += opt_res["total_holding_cost"]
            tot_ordering += opt_res["total_ordering_cost"]
            tot_stockout += opt_res["total_stockout_penalty"]

            results.append({
                "product_id": pid,
                "product_name": item["product_name"],
                "category": item["category"],
                "current_inventory": init_inv,
                "safety_stock": ss,
                "optimal_reorder_qty": opt_res["immediate_order_qty"],
                "planned_orders_count": opt_res["total_orders_placed"],
                "total_holding_cost": opt_res["total_holding_cost"],
                "total_ordering_cost": opt_res["total_ordering_cost"],
                "total_stockout_cost": opt_res["total_stockout_penalty"],
                "total_cost": opt_res["total_optimal_cost"],
                "optimization_status": opt_res["status"],
            })

        summary_metrics = {
            "Total_Portfolio_Cost": round(tot_holding + tot_ordering + tot_stockout, 2),
            "Total_Holding_Cost": round(tot_holding, 2),
            "Total_Ordering_Cost": round(tot_ordering, 2),
            "Total_Stockout_Penalty": round(tot_stockout, 2),
            "SKUs_Optimized": len(results),
        }

        return pd.DataFrame(results), summary_metrics
