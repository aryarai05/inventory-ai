"""
Supply Chain & Demand Visualization Suite
Provides interactive Plotly charts for exploratory data analysis, forecasting, and inventory risk.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List


def plot_overall_sales_trend(sales_df: pd.DataFrame) -> go.Figure:
    """1. Overall daily enterprise sales trend with 30-day moving average."""
    daily = sales_df.groupby("date")["sales_quantity"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["ma_30"] = daily["sales_quantity"].rolling(30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales_quantity"],
        mode="lines", name="Daily Sales",
        line=dict(color="rgba(41, 128, 185, 0.4)", width=1.2)
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma_30"],
        mode="lines", name="30-Day Moving Average",
        line=dict(color="#1f77b4", width=3)
    ))
    fig.update_layout(
        title="Enterprise Daily Sales Volume & Trend (3 Years)",
        xaxis_title="Date", yaxis_title="Units Sold",
        template="plotly_white", hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_monthly_sales_trend(sales_df: pd.DataFrame) -> go.Figure:
    """2. Monthly sales trend showing year-over-year growth."""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["year", "month"])["sales_quantity"].sum().reset_index()
    monthly["year_str"] = monthly["year"].astype(str)

    fig = px.line(
        monthly, x="month", y="sales_quantity", color="year_str",
        markers=True, title="Monthly Sales Volume by Year (YoY Comparison)",
        labels={"month": "Calendar Month", "sales_quantity": "Units Sold", "year_str": "Year"},
        template="plotly_white"
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    return fig


def plot_weekly_sales_trend(sales_df: pd.DataFrame) -> go.Figure:
    """3. Weekly aggregate sales trend across the 3-year timeline."""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_week"] = df["date"].dt.to_period("W").dt.to_timestamp()
    weekly = df.groupby("year_week")["sales_quantity"].sum().reset_index()

    fig = px.area(
        weekly, x="year_week", y="sales_quantity",
        title="Weekly Aggregated Sales Trajectory",
        labels={"year_week": "Week", "sales_quantity": "Total Units Sold"},
        color_discrete_sequence=["#2ecc71"],
        template="plotly_white"
    )
    return fig


def plot_product_demand_comparison(sales_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """4. Product demand comparison (units vs revenue)."""
    prod_agg = sales_df.groupby(["product_id", "product_name"]) if "product_name" in sales_df.columns else sales_df.groupby("product_id")
    summary = prod_agg.agg(
        total_units=("sales_quantity", "sum"),
        total_revenue=("revenue", "sum") if "revenue" in sales_df.columns else ("sales_quantity", "count")
    ).reset_index().sort_values("total_units", ascending=False).head(top_n)

    name_col = "product_name" if "product_name" in summary.columns else "product_id"
    fig = px.bar(
        summary, x="total_units", y=name_col, orientation="h",
        color="total_revenue", title=f"Top {top_n} Products by Demand Volume & Revenue",
        labels={"total_units": "Units Sold", name_col: "Product", "total_revenue": "Revenue ($)"},
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_category_demand(sales_df: pd.DataFrame) -> go.Figure:
    """5. Category demand distribution."""
    if "category" not in sales_df.columns:
        return go.Figure()
    cat_summary = sales_df.groupby("category")["sales_quantity"].sum().reset_index().sort_values("sales_quantity", ascending=False)
    fig = px.pie(
        cat_summary, names="category", values="sales_quantity",
        title="Total Demand Share by Merchandise Category",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_regional_demand(sales_df: pd.DataFrame) -> go.Figure:
    """6. Regional demand volume comparison."""
    if "region_id" not in sales_df.columns:
        return go.Figure()
    reg = sales_df.groupby("region_id")["sales_quantity"].sum().reset_index().sort_values("sales_quantity", ascending=False)
    fig = px.bar(
        reg, x="region_id", y="sales_quantity",
        title="Regional Fulfillment Demand Distribution",
        labels={"region_id": "Region", "sales_quantity": "Units Sold"},
        color="sales_quantity", color_continuous_scale="Blues",
        template="plotly_white"
    )
    return fig


def plot_inventory_level_trend(inventory_df: pd.DataFrame) -> go.Figure:
    """7. Enterprise inventory level and safety stock buffer over time."""
    inv = inventory_df.copy()
    inv["date"] = pd.to_datetime(inv["date"])
    daily_inv = inv.groupby("date").agg(
        ending_inv=("ending_inventory", "sum"),
        safety_stock=("safety_stock", "sum") if "safety_stock" in inv.columns else ("ending_inventory", "mean")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_inv["date"], y=daily_inv["ending_inv"],
        name="Total Ending Inventory", line=dict(color="#34495e", width=2)
    ))
    if "safety_stock" in daily_inv.columns:
        fig.add_trace(go.Scatter(
            x=daily_inv["date"], y=daily_inv["safety_stock"],
            name="Aggregate Safety Stock Target", line=dict(color="#e74c3c", width=1.5, dash="dash")
        ))
    fig.update_layout(
        title="Enterprise Inventory Level vs. Safety Stock Threshold",
        xaxis_title="Date", yaxis_title="Units On-Hand",
        template="plotly_white", hovermode="x unified"
    )
    return fig


def plot_stockout_frequency(inventory_df: pd.DataFrame) -> go.Figure:
    """8. Stockout incidence frequency by product category or SKU."""
    if "stockout_flag" not in inventory_df.columns:
        return go.Figure()
    inv = inventory_df.copy()
    stockouts = inv.groupby("product_id")["stockout_flag"].sum().reset_index()
    stockouts = stockouts.sort_values("stockout_flag", ascending=False).head(15)

    fig = px.bar(
        stockouts, x="product_id", y="stockout_flag",
        title="Top 15 SKUs with Highest Stockout Frequency (Days Short)",
        labels={"product_id": "Product ID", "stockout_flag": "Days with Stockout"},
        color="stockout_flag", color_continuous_scale="Reds",
        template="plotly_white"
    )
    return fig


def plot_promotion_vs_sales(sales_df: pd.DataFrame) -> go.Figure:
    """9. Promotional lift comparison: Baseline vs Promotional Days."""
    if "promotion_flag" not in sales_df.columns:
        return go.Figure()
    daily_promo = sales_df.groupby(["date", "promotion_flag"])["sales_quantity"].sum().reset_index()
    daily_promo["Promotion_Status"] = daily_promo["promotion_flag"].map({0: "Regular Operations", 1: "Active Promotion Campaign"})

    fig = px.box(
        daily_promo, x="Promotion_Status", y="sales_quantity",
        color="Promotion_Status",
        title="Demand Distribution: Regular Operations vs. Promotional Periods",
        labels={"sales_quantity": "Daily Units Demanded", "Promotion_Status": "Operating State"},
        color_discrete_map={"Regular Operations": "#95a5a6", "Active Promotion Campaign": "#e67e22"},
        template="plotly_white"
    )
    return fig


def plot_seasonality_patterns(sales_df: pd.DataFrame) -> go.Figure:
    """10. Weekly & monthly seasonality heatmap matrix."""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_name"] = df["date"].dt.day_name()
    df["month_name"] = df["date"].dt.month_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    heat = df.groupby(["day_name", "month_name"])["sales_quantity"].mean().reset_index()
    pivot = heat.pivot(index="day_name", columns="month_name", values="sales_quantity")
    pivot = pivot.reindex(index=day_order, columns=[m for m in month_order if m in pivot.columns])

    fig = px.imshow(
        pivot, labels=dict(x="Month", y="Day of Week", color="Avg Units"),
        title="Demand Heatmap Matrix: Day of Week vs. Calendar Month",
        color_continuous_scale="Blues", aspect="auto",
        template="plotly_white"
    )
    return fig


def plot_top10_products(sales_df: pd.DataFrame) -> go.Figure:
    """11. Top 10 revenue generating products (ABC Category A)."""
    group_col = "product_name" if "product_name" in sales_df.columns else "product_id"
    agg = sales_df.groupby(group_col).agg(
        revenue=("revenue", "sum") if "revenue" in sales_df.columns else ("sales_quantity", "sum"),
        units=("sales_quantity", "sum")
    ).reset_index().sort_values("revenue", ascending=False).head(10)

    fig = px.bar(
        agg, x="revenue", y=group_col, orientation="h",
        title="Top 10 SKUs by Enterprise Revenue Realization",
        labels={"revenue": "Total Revenue ($)", group_col: "Product"},
        color="units", color_continuous_scale="Teal",
        template="plotly_white"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_slow_moving_products(sales_df: pd.DataFrame) -> go.Figure:
    """12. Slow-moving products (ABC Category C / Long Tail)."""
    group_col = "product_name" if "product_name" in sales_df.columns else "product_id"
    agg = sales_df.groupby(group_col).agg(
        units=("sales_quantity", "sum"),
        avg_daily=("sales_quantity", "mean")
    ).reset_index().sort_values("units", ascending=True).head(10)

    fig = px.bar(
        agg, x="units", y=group_col, orientation="h",
        title="Bottom 10 Slow-Moving Products (Long Inventory Tail)",
        labels={"units": "Total Units Sold (3 Years)", group_col: "Product"},
        color="avg_daily", color_continuous_scale="Oranges",
        template="plotly_white"
    )
    return fig
