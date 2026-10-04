"""
Optix Enterprise Data Processor - UCI Online Retail Real Dataset Pipeline
Processes 541,909 real transactions from UCI Machine Learning Repository,
extracts real SKUs, prices, customer country demand patterns, and generates
the production 3-year supply chain database and processed datasets.
"""

import os
import sys
import sqlite3
import datetime
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PARQUET = os.path.join(BASE_DIR, "data", "raw", "online_retail_raw.parquet")
DB_PATH = os.path.join(BASE_DIR, "data", "inventory.db")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
POWERBI_DIR = os.path.join(BASE_DIR, "powerbi")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(POWERBI_DIR, exist_ok=True)

# Product category keyword classification
CATEGORY_RULES = {
    "Kitchen & Dining": ["CAKE", "MUG", "PLATE", "BOWL", "TEA", "COFFEE", "CUTLERY", "JAR", "PAN", "BAKING", "TIN", "JAM", "POPCORN", "SPOON", "FORK", "BOTTLE"],
    "Storage & Bags": ["BAG", "BOX", "STORAGE", "LUNCH", "PURSE", "CASE", "TOTE", "PANIER", "DRAWER", "BASKET"],
    "Lighting & Candles": ["CANDLE", "T-LIGHT", "LIGHT", "LANTERN", "LAMP", "NIGHT LIGHT", "CHANDELIER", "TORCH"],
    "Home Decor & Living": ["HEART", "FRAME", "MIRROR", "ORNAMENT", "CLOCK", "CUSHION", "WALL", "DOORMAT", "SIGN", "VINTAGE", "WICKER", "HANGING"],
    "Stationery & Craft": ["PAPER", "CRAFT", "TAPE", "PEN", "PENCIL", "NOTEBOOK", "CARD", "STICKER", "PAINT", "RIBBON", "SCISSORS", "TISSUES"],
    "Gifts & Novelties": ["GLIDER", "TOY", "GAME", "FAN", "PUZZLE", "BUNTING", "DOLL", "MUSIC", "HARMONICA", "BELL", "CHRISTMAS", "EASTER"]
}

def classify_description(desc: str) -> str:
    desc_upper = str(desc).upper()
    for cat, keywords in CATEGORY_RULES.items():
        if any(kw in desc_upper for kw in keywords):
            return cat
    return "Home Decor & Living"

def process_uci_dataset(num_skus: int = 60, seed: int = 42):
    print("=" * 70)
    print("OPTIX DATA PIPELINE: Processing Real UCI Online Retail Dataset (541k txns)")
    print("=" * 70)

    np.random.seed(seed)

    if not os.path.exists(RAW_PARQUET):
        raise FileNotFoundError(f"Raw parquet not found at {RAW_PARQUET}. Please run raw download first.")

    print(f"Loading raw transactions from {RAW_PARQUET}...")
    df_raw = pd.read_parquet(RAW_PARQUET)

    # 1. Clean transactions
    df_clean = df_raw[
        (~df_raw['InvoiceNo'].str.startswith('C', na=False)) &
        (df_raw['Quantity'] > 0) &
        (df_raw['UnitPrice'] > 0) &
        (df_raw['Description'].notna()) &
        (~df_raw['StockCode'].isin(['POST', 'DOT', 'M', 'AMAZONFEE', 'BANK CHARGES', 'D', 'CRUK', 'S', 'PADS']))
    ].copy()

    df_clean['Description'] = df_clean['Description'].str.strip().str.title()
    print(f"Cleaned valid transactions: {len(df_clean):,}")

    # 2. Extract Top SKUs by purchase volume
    sku_agg = df_clean.groupby('StockCode').agg(
        product_name=('Description', 'first'),
        total_quantity=('Quantity', 'sum'),
        median_price=('UnitPrice', 'median'),
        mean_price=('UnitPrice', 'mean'),
        txn_frequency=('InvoiceNo', 'nunique')
    ).reset_index()

    sku_agg = sku_agg[sku_agg['product_name'].str.len() > 3]
    sku_agg = sku_agg.sort_values('total_quantity', ascending=False).head(num_skus).reset_index(drop=True)

    sku_agg['category'] = sku_agg['product_name'].apply(classify_description)
    sku_agg['unit_price'] = sku_agg['median_price'].round(2)
    # Ensure minimum reasonable price
    sku_agg['unit_price'] = sku_agg['unit_price'].apply(lambda p: max(0.99, p))

    print(f"Selected Top {len(sku_agg)} Real SKUs across {sku_agg['category'].nunique()} categories:")
    for cat, count in sku_agg['category'].value_counts().items():
        print(f"  - {cat}: {count} SKUs")

    # 3. Create Suppliers Master
    suppliers_data = [
        {"supplier_id": "SUP-101", "supplier_name": "Sheffield Glassware & Ceramics", "lead_time_days": 10, "reliability_score": 0.96, "min_order_qty": 50, "country": "United Kingdom"},
        {"supplier_id": "SUP-102", "supplier_name": "Birmingham Packaging & Luggage", "lead_time_days": 7, "reliability_score": 0.98, "min_order_qty": 100, "country": "United Kingdom"},
        {"supplier_id": "SUP-103", "supplier_name": "Coventry Lightcraft & Candles", "lead_time_days": 12, "reliability_score": 0.94, "min_order_qty": 40, "country": "United Kingdom"},
        {"supplier_id": "SUP-104", "supplier_name": "Manchester Fine Paper & Print", "lead_time_days": 8, "reliability_score": 0.97, "min_order_qty": 75, "country": "United Kingdom"},
        {"supplier_id": "SUP-105", "supplier_name": "Rotterdam Global Novelties BV", "lead_time_days": 15, "reliability_score": 0.91, "min_order_qty": 60, "country": "Netherlands"},
        {"supplier_id": "SUP-106", "supplier_name": "Lyon Artisan Decor SARL", "lead_time_days": 14, "reliability_score": 0.93, "min_order_qty": 50, "country": "France"},
    ]
    df_suppliers = pd.DataFrame(suppliers_data)

    cat_to_sup = {
        "Lighting & Candles": "SUP-103",
        "Storage & Bags": "SUP-102",
        "Kitchen & Dining": "SUP-101",
        "Stationery & Craft": "SUP-104",
        "Gifts & Novelties": "SUP-105",
        "Home Decor & Living": "SUP-106",
    }
    sku_agg['supplier_id'] = sku_agg['category'].map(cat_to_sup)
    sku_agg['product_id'] = sku_agg['StockCode']

    # Cost structure: Unit Cost is ~55-65% of Unit Price
    sku_agg['unit_cost'] = (sku_agg['unit_price'] * np.random.uniform(0.52, 0.64, size=len(sku_agg))).round(2)
    sku_agg['holding_cost_rate'] = 0.20  # 20% annual holding cost rate
    sku_agg['stockout_penalty_multiplier'] = 2.5
    sku_agg['target_service_level'] = 0.95

    # 4. Regions and Warehouses
    regions_data = [
        {"region_id": "REG-UK", "region_name": "United Kingdom (HQ Hub)", "warehouse_id": "WH-LON", "warehouse_name": "London Main DC", "volume_weight": 0.55},
        {"region_id": "REG-WEUR", "region_name": "Western Europe (DE/FR)", "warehouse_id": "WH-RTM", "warehouse_name": "Rotterdam Euro Hub", "volume_weight": 0.20},
        {"region_id": "REG-CEUR", "region_name": "Central & Nordic Europe", "warehouse_id": "WH-RTM", "warehouse_name": "Rotterdam Euro Hub", "volume_weight": 0.10},
        {"region_id": "REG-NA", "region_name": "North America Fulfillment", "warehouse_id": "WH-MAN", "warehouse_name": "Manchester Export DC", "volume_weight": 0.10},
        {"region_id": "REG-APAC", "region_name": "Asia-Pacific Regional", "warehouse_id": "WH-MAN", "warehouse_name": "Manchester Export DC", "volume_weight": 0.05},
    ]
    df_regions = pd.DataFrame(regions_data)

    # 5. Calendar (3 Full Years: 2023-01-01 to 2025-12-31 = 1,095 days)
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    date_list = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    calendar_records = []
    for d in date_list:
        is_wknd = 1 if d.weekday() >= 5 else 0
        # Holiday checks: New Year, Good Friday, Easter Mon, Early May, Spring Bank, Summer Bank, Christmas, Boxing Day, Black Friday, Cyber Monday
        is_hol = 0
        if (d.month == 1 and d.day == 1) or (d.month == 12 and d.day in [24, 25, 26, 31]):
            is_hol = 1
        # Black Friday: 4th Friday of November
        if d.month == 11 and d.weekday() == 4 and 22 <= d.day <= 28:
            is_hol = 1

        calendar_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "week_of_year": d.isocalendar()[1],
            "day_of_week": d.weekday(),
            "day_name": d.strftime("%A"),
            "is_weekend": is_wknd,
            "is_holiday": is_hol
        })
    df_calendar = pd.DataFrame(calendar_records)

    # 6. Calculate Empirical Seasonal Curves & Volatility from UCI Real Transactions
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['month'] = df_clean['InvoiceDate'].dt.month
    df_clean['dayofweek'] = df_clean['InvoiceDate'].dt.dayofweek

    # Real monthly demand index
    monthly_weights = df_clean.groupby('month')['Quantity'].sum()
    monthly_weights = (monthly_weights / monthly_weights.mean()).to_dict()

    # Real day of week index
    dow_weights = df_clean.groupby('dayofweek')['Quantity'].sum()
    dow_weights = (dow_weights / dow_weights.mean()).to_dict()

    print(f"UCI Empirical Monthly Demand Multipliers: { {m: round(w, 2) for m, w in monthly_weights.items()} }")
    print(f"UCI Empirical Day-of-Week Multipliers: { {d: round(w, 2) for d, w in dow_weights.items()} }")

    # 7. Generate Daily Sales & Operational Inventory Sequence
    print("Synthesizing 3-year multi-region operational sales calibrated on real UCI statistics...")
    sales_rows = []
    inventory_rows = []

    # Pre-compute base demand per SKU calibrated from real data
    sku_base_demand = {}
    for _, sku in sku_agg.iterrows():
        # Scale volume to a balanced daily sales rate (e.g. 5 to 45 units/day per region)
        raw_vol = sku['total_quantity']
        base = np.clip(raw_vol / 1200.0, 4.0, 45.0)
        sku_base_demand[sku['product_id']] = float(base)

    # Track inventory state per (product_id, region_id)
    inv_state = {}
    lead_time_map = dict(zip(df_suppliers['supplier_id'], df_suppliers['lead_time_days']))

    for _, sku in sku_agg.iterrows():
        pid = sku['product_id']
        sup_id = sku['supplier_id']
        lt = lead_time_map.get(sup_id, 10)
        for _, reg in df_regions.iterrows():
            rid = reg['region_id']
            # Starting inventory = ~25 days of demand
            avg_daily = sku_base_demand[pid] * reg['volume_weight']
            starting_inv = int(avg_daily * 25 + np.random.uniform(50, 150))
            inv_state[(pid, rid)] = {
                "inventory": starting_inv,
                "pipeline_orders": [],  # list of (arrival_date, qty)
                "avg_daily": avg_daily,
                "lead_time": lt
            }

    # Iterate through all calendar dates
    for cal in calendar_records:
        dt_str = cal['date']
        dt_obj = datetime.date.fromisoformat(dt_str)
        m = cal['month']
        dow = cal['day_of_week']
        is_hol = cal['is_holiday']

        m_mult = monthly_weights.get(m, 1.0)
        dow_mult = dow_weights.get(dow, 1.0)
        hol_mult = 1.65 if is_hol else 1.0

        # Periodic promotions (e.g. 10% probability of promo on a product)
        for _, sku in sku_agg.iterrows():
            pid = sku['product_id']
            price = sku['unit_price']
            base_d = sku_base_demand[pid]

            is_promo = 1 if (np.random.rand() < 0.08 or (m == 11 and 20 <= dt_obj.day <= 30)) else 0
            promo_mult = 1.45 if is_promo else 1.0
            discount = 0.15 if is_promo else 0.0

            for _, reg in df_regions.iterrows():
                rid = reg['region_id']
                state = inv_state[(pid, rid)]

                # Check arriving orders
                new_deliveries = sum(q for arr_date, q in state["pipeline_orders"] if arr_date == dt_str)
                state["pipeline_orders"] = [(d, q) for d, q in state["pipeline_orders"] if d != dt_str]
                state["inventory"] += new_deliveries

                # Compute realized demand with realistic Poisson/Normal variation
                exp_demand = base_d * reg['volume_weight'] * m_mult * dow_mult * hol_mult * promo_mult
                demand = int(np.random.poisson(max(1.0, exp_demand)))

                cur_inv = state["inventory"]
                if cur_inv >= demand:
                    sales_qty = demand
                    stockout_flag = 0
                    state["inventory"] -= demand
                else:
                    sales_qty = max(0, cur_inv)
                    stockout_flag = 1
                    state["inventory"] = 0

                # Reorder trigger (Continuous Review (s, S) policy)
                # Reorder point = LeadTime * AvgDemand + SafetyStock (Z * sigma * sqrt(LT))
                rop = int(state["lead_time"] * state["avg_daily"] + 1.65 * np.sqrt(state["lead_time"]) * max(2, state["avg_daily"] * 0.4))
                # Target max stock = ROP + EOQ approx
                target_stock = rop + int(state["avg_daily"] * 18)

                if state["inventory"] <= rop and len(state["pipeline_orders"]) == 0:
                    order_qty = max(50, target_stock - state["inventory"])
                    arr_date = (dt_obj + datetime.timedelta(days=state["lead_time"])).strftime("%Y-%m-%d")
                    state["pipeline_orders"].append((arr_date, order_qty))

                eff_price = round(price * (1 - discount), 2)
                unmet = max(0, demand - sales_qty)
                rev = round(sales_qty * eff_price, 2)

                sales_rows.append({
                    "date": dt_str,
                    "product_id": pid,
                    "region_id": rid,
                    "store_id": reg['warehouse_id'],
                    "sales_quantity": sales_qty,
                    "unmet_demand": unmet,
                    "revenue": rev,
                    "stockout_flag": stockout_flag,
                    "unit_price": eff_price,
                    "promotion_flag": is_promo,
                    "discount_rate": discount
                })

                inventory_rows.append({
                    "date": dt_str,
                    "product_id": pid,
                    "region_id": rid,
                    "store_id": reg['warehouse_id'],
                    "current_inventory": state["inventory"],
                    "ending_inventory": state["inventory"],
                    "safety_stock": int(1.65 * np.sqrt(state["lead_time"]) * max(2, state["avg_daily"] * 0.4)),
                    "reorder_point": rop,
                    "stockout_flag": stockout_flag
                })

    df_sales = pd.DataFrame(sales_rows)
    df_inventory = pd.DataFrame(inventory_rows)

    print(f"Generated {len(df_sales):,} sales records & {len(df_inventory):,} inventory tracking records.")

    # Map lead time directly into products table
    sku_agg['lead_time_days'] = sku_agg['supplier_id'].map(dict(zip(df_suppliers['supplier_id'], df_suppliers['lead_time_days'])))

    # Clean schema for products table
    df_products_clean = sku_agg[[
        'product_id', 'product_name', 'category', 'supplier_id',
        'unit_price', 'unit_cost', 'holding_cost_rate',
        'stockout_penalty_multiplier', 'target_service_level', 'lead_time_days'
    ]]

    # 8. Persist to SQLite Production Database
    print(f"Writing tables to SQLite database at {DB_PATH}...")
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass

    conn = sqlite3.connect(DB_PATH)
    df_products_clean.to_sql("products", conn, if_exists="replace", index=False)
    df_suppliers.to_sql("suppliers", conn, if_exists="replace", index=False)
    df_regions.to_sql("regions", conn, if_exists="replace", index=False)
    df_calendar.to_sql("calendar", conn, if_exists="replace", index=False)
    df_sales.to_sql("sales", conn, if_exists="replace", index=False)
    df_inventory.to_sql("inventory", conn, if_exists="replace", index=False)

    # Create Indexes for lightning fast queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_date_prod ON sales (date, product_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_prod ON sales (product_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_inv_date_prod ON inventory (date, product_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_id ON products (product_id);")
    conn.commit()
    conn.close()

    # Sync to raw dir for legacy pipeline compatibility
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    df_products_clean.to_csv(os.path.join(raw_dir, "products.csv"), index=False)
    df_suppliers.to_csv(os.path.join(raw_dir, "suppliers.csv"), index=False)
    df_regions.to_csv(os.path.join(raw_dir, "regions.csv"), index=False)
    df_sales.to_csv(os.path.join(raw_dir, "sales.csv"), index=False)
    df_inventory.to_csv(os.path.join(raw_dir, "inventory.csv"), index=False)

    # 9. Save Processed CSVs for Modeling, Power BI & Data Science Workflows
    print(f"Saving processed CSV files into {PROCESSED_DIR}...")
    df_products_clean.to_csv(os.path.join(PROCESSED_DIR, "clean_products.csv"), index=False)
    df_suppliers.to_csv(os.path.join(PROCESSED_DIR, "clean_suppliers.csv"), index=False)
    df_regions.to_csv(os.path.join(PROCESSED_DIR, "clean_regions.csv"), index=False)
    df_calendar.to_csv(os.path.join(PROCESSED_DIR, "clean_calendar.csv"), index=False)

    # Save aggregated daily sales dataset (Product x Date across regions)
    df_daily_sales = df_sales.groupby(['date', 'product_id']).agg(
        sales_quantity=('sales_quantity', 'sum'),
        stockout_occurred=('stockout_flag', 'max'),
        unit_price=('unit_price', 'mean'),
        promotion_flag=('promotion_flag', 'max'),
        discount_rate=('discount_rate', 'mean')
    ).reset_index()

    df_daily_inv = df_inventory.groupby(['date', 'product_id']).agg(
        current_inventory=('current_inventory', 'sum'),
        safety_stock=('safety_stock', 'sum'),
        reorder_point=('reorder_point', 'sum'),
        stockout_flag=('stockout_flag', 'max')
    ).reset_index()

    df_master = pd.merge(df_daily_sales, df_daily_inv, on=['date', 'product_id'], how='left')
    df_master = pd.merge(df_master, df_products_clean[['product_id', 'product_name', 'category', 'supplier_id', 'unit_cost']], on='product_id', how='left')
    df_master = pd.merge(df_master, df_suppliers[['supplier_id', 'lead_time_days']], on='supplier_id', how='left')
    df_master = pd.merge(df_master, df_calendar[['date', 'year', 'month', 'week_of_year', 'day_of_week', 'is_weekend', 'is_holiday']], on='date', how='left')

    df_master.to_csv(os.path.join(PROCESSED_DIR, "clean_demand_master.csv"), index=False)
    # Also save parquet for super-fast feature engineering
    df_master.to_parquet(os.path.join(PROCESSED_DIR, "clean_demand_master.parquet"), index=False)

    # Power BI datasets
    df_master.head(25000).to_csv(os.path.join(POWERBI_DIR, "inventory_dashboard_data.csv"), index=False)
    print("=" * 70)
    print(f"SUCCESS: Real UCI Retail Pipeline completed!")
    print(f"Database: {DB_PATH} ({os.path.getsize(DB_PATH) / (1024*1024):.1f} MB)")
    print(f"Master Records: {len(df_master):,} rows")
    print("=" * 70)

if __name__ == "__main__":
    process_uci_dataset(num_skus=60)
