"""
Unit Tests for Data Preprocessing and Database Integration
"""

import pytest
import os
import pandas as pd
from src.data.loader import DataLoader
from src.data.database import DatabaseManager


def test_database_initialization_and_query(tmp_path):
    test_db = os.path.join(tmp_path, "test_inv.db")
    db = DatabaseManager(db_path=test_db)
    db.initialize_schema("sql/schema.sql")

    # Verify tables created
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    conn.close()

    assert "products" in tables
    assert "sales" in tables
    assert "inventory" in tables
    assert "suppliers" in tables


def test_dataloader_clean_and_merge():
    loader = DataLoader(raw_dir="data/raw", db_path="data/inventory.db")
    df = loader.clean_and_merge_master()

    assert not df.empty
    assert "sales_quantity" in df.columns
    assert "total_demand" in df.columns
    assert "product_id" in df.columns
    assert "unit_cost" in df.columns
    assert "lead_time_days" in df.columns
    assert df["sales_quantity"].isna().sum() == 0
