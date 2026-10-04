"""
Data Ingestion and Cleaning Pipeline
Handles loading raw data, cleaning, schema validation, outlier filtering, and database synchronization.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from src.data.database import DatabaseManager


class DataLoader:
    def __init__(self, raw_dir: str = "data/raw", db_path: str = "data/inventory.db"):
        self.raw_dir = raw_dir
        self.db_manager = DatabaseManager(db_path=db_path)

    def load_raw_files(self) -> Dict[str, pd.DataFrame]:
        """Loads all raw CSV files into pandas DataFrames."""
        tables = ["suppliers", "regions", "products", "calendar", "promotions", "sales", "inventory"]
        data = {}
        for table in tables:
            path = os.path.join(self.raw_dir, f"{table}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Raw data file missing: {path}. Run generate_data.py first.")
            df = pd.read_csv(path)
            data[table] = df
        return data

    def populate_database(self, schema_file: str = "sql/schema.sql") -> None:
        """Initializes SQLite schema and loads raw datasets into tables."""
        self.db_manager.initialize_schema(schema_file)
        data = self.load_raw_files()
        for table_name, df in data.items():
            self.db_manager.load_table(table_name, df, if_exists="append")

    def clean_and_merge_master(self) -> pd.DataFrame:
        """
        Cleans data and produces an integrated analytical master table.
        Joins sales, calendar, products, suppliers, and daily inventory.
        Handles missing values, duplicate checks, and date conversions.
        """
        data = self.load_raw_files()
        sales = data["sales"].copy()
        products = data["products"].copy()
        calendar = data["calendar"].copy()
        inventory = data["inventory"].copy()

        # 1. Clean Dates
        sales["date"] = pd.to_datetime(sales["date"])
        calendar["date"] = pd.to_datetime(calendar["date"])
        inventory["date"] = pd.to_datetime(inventory["date"])

        # 2. Check and remove duplicates
        sales = sales.drop_duplicates(subset=["date", "product_id", "region_id"])
        inventory = inventory.drop_duplicates(subset=["date", "product_id", "warehouse_id"])

        # 3. Aggregate sales to product-date level for forecasting demand
        daily_sales = sales.groupby(["date", "product_id"]).agg(
            sales_quantity=("sales_quantity", "sum"),
            unmet_demand=("unmet_demand", "sum"),
            revenue=("revenue", "sum"),
            promotion_flag=("promotion_flag", "max"),
            discount_rate=("discount_rate", "max"),
        ).reset_index()

        # True unconstrained customer demand = fulfilled sales + unmet demand
        daily_sales["total_demand"] = daily_sales["sales_quantity"] + daily_sales["unmet_demand"]

        # 4. Merge product metadata
        merged = daily_sales.merge(
            products[[
                "product_id", "product_name", "category", "unit_cost", 
                "unit_price", "holding_cost_rate", "supplier_id", "lead_time_days"
            ]],
            on="product_id",
            how="left"
        )

        # 5. Merge calendar features
        merged = merged.merge(
            calendar[[
                "date", "day_of_week", "day_name", "week_of_year", 
                "month", "month_name", "quarter", "year", "is_weekend", "is_holiday"
            ]],
            on="date",
            how="left"
        )

        # 6. Merge inventory beginning / ending stock
        # Select key inventory metrics
        inv_subset = inventory[["date", "product_id", "beginning_inventory", "ending_inventory", "stockout_flag"]].copy()
        merged = merged.merge(inv_subset, on=["date", "product_id"], how="left")

        # 7. Handle missing values
        merged["beginning_inventory"] = merged["beginning_inventory"].fillna(0)
        merged["ending_inventory"] = merged["ending_inventory"].fillna(0)
        merged["stockout_flag"] = merged["stockout_flag"].fillna(0).astype(int)
        merged["promotion_flag"] = merged["promotion_flag"].fillna(0).astype(int)
        merged["discount_rate"] = merged["discount_rate"].fillna(0.0)

        # 8. Sort chronologically by product and date
        merged = merged.sort_values(by=["product_id", "date"]).reset_index(drop=True)

        return merged
