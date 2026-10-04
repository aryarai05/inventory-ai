"""
Data Preprocessing & Database Pipeline Script
Initializes SQLite database, executes schema & analytical SQL queries,
generates time-series features without data leakage, and exports train/val/test splits.
"""

import os
import sys
import yaml
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer


def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    db_path = config["data"]["database_path"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[*] STEP 1: Initializing SQLite Database and Populating Tables...")
    loader = DataLoader(raw_dir=raw_dir, db_path=db_path)
    loader.populate_database(schema_file="sql/schema.sql")
    print(f"[+] SQLite database initialized and loaded at: {db_path}")

    print("[*] STEP 2: Executing Verification SQL Queries...")
    test_turnover_query = """
    SELECT 
        p.product_id,
        p.product_name,
        p.category,
        ROUND(SUM(s.sales_quantity * p.unit_cost), 2) AS total_cogs
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_id
    ORDER BY total_cogs DESC
    LIMIT 5;
    """
    turnover_df = loader.db_manager.execute_query(test_turnover_query)
    print("[+] SQL Query Executed Successfully (Top 5 SKUs by COGS):")
    print(turnover_df.to_string(index=False))

    print("[*] STEP 3: Cleaning & Merging Analytical Master Dataset...")
    master_df = loader.clean_and_merge_master()
    print(f"[+] Master dataset compiled with {len(master_df):,} rows and {master_df.shape[1]} columns.")

    print("[*] STEP 4: Engineering Time-Series Features (Strictly Zero Lookahead Bias)...")
    fe = FeatureEngineer(
        lag_days=config["features"]["lag_days"],
        rolling_windows=config["features"]["rolling_windows"],
        target_column=config["features"]["target_column"],
        group_column=config["features"]["group_column"],
        date_column=config["features"]["date_column"],
    )
    featured_df = fe.generate_features(master_df)
    print(f"[+] Features engineered successfully. Total columns: {featured_df.shape[1]}")

    print("[*] STEP 5: Chronological Train (70%), Validation (15%), Test (15%) Splitting...")
    train_df, val_df, test_df = fe.split_time_aware(
        featured_df,
        train_ratio=config["preprocessing"]["train_ratio"],
        val_ratio=config["preprocessing"]["val_ratio"],
        test_ratio=config["preprocessing"]["test_ratio"],
    )

    print(f"    - Train set: {len(train_df):,} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"    - Val set  : {len(val_df):,} records ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"    - Test set : {len(test_df):,} records ({test_df['date'].min()} to {test_df['date'].max()})")

    # Export processed datasets
    featured_df.to_csv(os.path.join(processed_dir, "featured_dataset.csv"), index=False)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    print(f"[+] Processed datasets saved to {processed_dir}/")
    print("[*] Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
