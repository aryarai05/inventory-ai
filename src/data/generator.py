"""
Realistic Supply Chain Data Generator
Generates multi-year historical time-series data with true trends, seasonality,
holiday effects, promotional elasticity, supplier lead times, and inventory depletion/stockouts.
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class SyntheticInventoryDataGenerator:
    def __init__(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2025-12-31",
        random_seed: int = 42,
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Master Catalog Definitions
        self.categories = [
            "Electronics",
            "Apparel & Fashion",
            "Home & Kitchen",
            "Health & Personal Care",
            "Office & Industrial",
        ]

        self.regions = [
            {"region_id": "REG_NE", "region_name": "North America - East", "warehouse_id": "WH_NORTHEAST", "demand_weight": 1.15},
            {"region_id": "REG_WC", "region_name": "North America - West", "warehouse_id": "WH_WESTCOAST", "demand_weight": 1.20},
            {"region_id": "REG_MW", "region_name": "Midwest Hub", "warehouse_id": "WH_CENTRAL", "demand_weight": 0.95},
            {"region_id": "REG_SE", "region_name": "Southeast Distribution", "warehouse_id": "WH_SOUTHEAST", "demand_weight": 1.05},
            {"region_id": "REG_NW", "region_name": "Pacific Northwest", "warehouse_id": "WH_NORTHWEST", "demand_weight": 0.85},
        ]

        self.suppliers = [
            {"supplier_id": "SUPP_01", "supplier_name": "Apex Micro Technologies", "lead_time_mean": 12.0, "lead_time_std": 2.1, "reliability_score": 0.96},
            {"supplier_id": "SUPP_02", "supplier_name": "Global Fabric & Textiles", "lead_time_mean": 18.0, "lead_time_std": 3.4, "reliability_score": 0.91},
            {"supplier_id": "SUPP_03", "supplier_name": "HomeCraft Industries", "lead_time_mean": 9.0, "lead_time_std": 1.5, "reliability_score": 0.98},
            {"supplier_id": "SUPP_04", "supplier_name": "BioHealth Formulations", "lead_time_mean": 7.0, "lead_time_std": 1.2, "reliability_score": 0.99},
            {"supplier_id": "SUPP_05", "supplier_name": "Vanguard Workspace Systems", "lead_time_mean": 14.0, "lead_time_std": 2.5, "reliability_score": 0.93},
        ]

    def _generate_products(self) -> pd.DataFrame:
        """Generates 50 distinct realistic SKUs with realistic cost, margin, and supplier links."""
        product_templates = {
            "Electronics": [
                ("ProBook 15.6-inch Laptop", 650.0, 999.0, "SUPP_01", 14),
                ("Noise-Canceling Wireless Headphones", 110.0, 229.0, "SUPP_01", 10),
                ("Ultra-HD 27-inch Curved Monitor", 175.0, 299.0, "SUPP_01", 12),
                ("Mechanical RGB Gaming Keyboard", 45.0, 89.0, "SUPP_01", 8),
                ("Smart Fitness Tracker Band", 35.0, 79.0, "SUPP_01", 9),
                ("4K Streaming Media Player", 22.0, 49.0, "SUPP_01", 6),
                ("USB-C Dual Docking Station", 50.0, 119.0, "SUPP_01", 8),
                ("External 2TB SSD Drive", 65.0, 139.0, "SUPP_01", 11),
                ("Smart Home Security Camera", 40.0, 89.0, "SUPP_01", 7),
                ("Bluetooth Portable Speaker", 30.0, 65.0, "SUPP_01", 8),
            ],
            "Apparel & Fashion": [
                ("Men's All-Weather Parka", 55.0, 149.0, "SUPP_02", 20),
                ("Women's Thermal Puffer Jacket", 60.0, 159.0, "SUPP_02", 21),
                ("Classic Straight-Fit Denim Jeans", 22.0, 68.0, "SUPP_02", 15),
                ("Breathable Running Shoes", 35.0, 95.0, "SUPP_02", 18),
                ("Organic Cotton Everyday T-Shirt", 7.0, 24.0, "SUPP_02", 12),
                ("Merino Wool Crewneck Sweater", 38.0, 98.0, "SUPP_02", 19),
                ("Seamless High-Rise Yoga Leggings", 16.0, 48.0, "SUPP_02", 14),
                ("Waterproof Trail Hiking Boots", 48.0, 129.0, "SUPP_02", 17),
                ("Fleece Zip-Up Hoodie", 18.0, 52.0, "SUPP_02", 14),
                ("Cushioned Athletic Socks (6-Pack)", 6.0, 19.0, "SUPP_02", 10),
            ],
            "Home & Kitchen": [
                ("Barista Espresso Coffee Machine", 140.0, 299.0, "SUPP_03", 10),
                ("XL Digital Air Fryer 6Qt", 42.0, 99.0, "SUPP_03", 8),
                ("High-Speed Blender 1200W", 35.0, 85.0, "SUPP_03", 9),
                ("10-Piece Non-Stick Cookware Set", 75.0, 189.0, "SUPP_03", 12),
                ("Laser-Navigating Robot Vacuum", 160.0, 349.0, "SUPP_03", 14),
                ("HEPA Filter Air Purifier", 52.0, 129.0, "SUPP_03", 10),
                ("Bamboo Luxury Bed Sheets (Queen)", 28.0, 75.0, "SUPP_03", 8),
                ("Dimmable LED Architect Desk Lamp", 18.0, 45.0, "SUPP_03", 7),
                ("Countertop Convection Microwave", 80.0, 179.0, "SUPP_03", 11),
                ("German Steel Kitchen Knife Set", 45.0, 119.0, "SUPP_03", 9),
            ],
            "Health & Personal Care": [
                ("Sonic Electric Toothbrush", 25.0, 69.0, "SUPP_04", 7),
                ("Deep Tissue Muscle Massage Gun", 40.0, 99.0, "SUPP_04", 8),
                ("Anti-Aging Vitamin C Serum 30ml", 9.0, 32.0, "SUPP_04", 6),
                ("Daily Multivitamin Tablets (180ct)", 8.0, 24.0, "SUPP_04", 5),
                ("Digital Arm Blood Pressure Monitor", 22.0, 54.0, "SUPP_04", 7),
                ("Ionic Ceramic Hair Dryer", 28.0, 72.0, "SUPP_04", 7),
                ("Mineral SPF 50 Broad Sunscreen", 6.5, 21.0, "SUPP_04", 5),
                ("High-Density Eco Yoga Mat", 14.0, 39.0, "SUPP_04", 6),
                ("Comprehensive First Aid Kit", 12.0, 32.0, "SUPP_04", 6),
                ("Hydrating Facial Moisturizer Cream", 11.0, 29.0, "SUPP_04", 5),
            ],
            "Office & Industrial": [
                ("Ergonomic Mesh Executive Chair", 110.0, 249.0, "SUPP_05", 15),
                ("Electric Dual-Motor Standing Desk", 195.0, 429.0, "SUPP_05", 16),
                ("Cross-Cut Heavy Duty Paper Shredder", 45.0, 109.0, "SUPP_05", 12),
                ("Thermal Shipping Label Printer", 55.0, 129.0, "SUPP_05", 11),
                ("Magnetic Dry Erase Whiteboard 48x36", 32.0, 79.0, "SUPP_05", 10),
                ("Under-Desk 3-Drawer Steel Cabinet", 65.0, 149.0, "SUPP_05", 14),
                ("Surge Protector Power Strip (12-Outlet)", 12.0, 29.0, "SUPP_05", 7),
                ("Heavy-Duty Packaging Tape (6-Pack)", 8.0, 22.0, "SUPP_05", 8),
                ("Dual Monitor Adjustable Arm Mount", 26.0, 64.0, "SUPP_05", 9),
                ("Adjustable Footrest with Massage Rollers", 15.0, 38.0, "SUPP_05", 8),
            ],
        }

        records = []
        prod_idx = 1
        for category, items in product_templates.items():
            for name, cost, price, supp_id, lead_time in items:
                prod_id = f"PROD_{prod_idx:03d}"
                # Base volume parameters
                base_demand = np.random.uniform(12.0, 45.0)
                reorder_point = int(base_demand * (lead_time + 4))
                min_order_qty = int(base_demand * 10)
                records.append({
                    "product_id": prod_id,
                    "product_name": name,
                    "category": category,
                    "unit_cost": cost,
                    "unit_price": price,
                    "holding_cost_rate": 0.20,
                    "supplier_id": supp_id,
                    "lead_time_days": lead_time,
                    "reorder_point_base": reorder_point,
                    "min_order_qty": min_order_qty,
                    "base_daily_demand": base_demand,
                    "trend_slope": np.random.uniform(-0.005, 0.015), # realistic mild growth/decline
                    "seasonal_amplitude": np.random.uniform(0.15, 0.45),
                    "price_elasticity": np.random.uniform(1.2, 2.4),
                })
                prod_idx += 1

        return pd.DataFrame(records)

    def _generate_calendar(self) -> pd.DataFrame:
        """Generates continuous date records with rich calendar & holiday annotations."""
        dates = pd.date_range(self.start_date, self.end_date, freq="D")
        cal_records = []

        us_holidays = {
            # 2023
            "2023-01-01": "New Year's Day",
            "2023-05-29": "Memorial Day",
            "2023-07-04": "Independence Day",
            "2023-07-11": "Prime Day Event",
            "2023-07-12": "Prime Day Event",
            "2023-09-04": "Labor Day",
            "2023-11-23": "Thanksgiving Day",
            "2023-11-24": "Black Friday",
            "2023-11-27": "Cyber Monday",
            "2023-12-24": "Christmas Eve",
            "2023-12-25": "Christmas Day",
            "2023-12-31": "New Year's Eve",
            # 2024
            "2024-01-01": "New Year's Day",
            "2024-05-27": "Memorial Day",
            "2024-07-04": "Independence Day",
            "2024-07-16": "Prime Day Event",
            "2024-07-17": "Prime Day Event",
            "2024-09-02": "Labor Day",
            "2024-11-28": "Thanksgiving Day",
            "2024-11-29": "Black Friday",
            "2024-12-02": "Cyber Monday",
            "2024-12-24": "Christmas Eve",
            "2024-12-25": "Christmas Day",
            "2024-12-31": "New Year's Eve",
            # 2025
            "2025-01-01": "New Year's Day",
            "2025-05-26": "Memorial Day",
            "2025-07-04": "Independence Day",
            "2025-07-15": "Prime Day Event",
            "2025-07-16": "Prime Day Event",
            "2025-09-01": "Labor Day",
            "2025-11-27": "Thanksgiving Day",
            "2025-11-28": "Black Friday",
            "2025-12-01": "Cyber Monday",
            "2025-12-24": "Christmas Eve",
            "2025-12-25": "Christmas Day",
            "2025-12-31": "New Year's Eve",
        }

        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            is_hol = 1 if d_str in us_holidays else 0
            hol_name = us_holidays.get(d_str, None)
            is_weekend = 1 if d.dayofweek in [5, 6] else 0

            cal_records.append({
                "date": d_str,
                "day_of_week": d.dayofweek, # 0 = Monday, 6 = Sunday
                "day_name": d.strftime("%A"),
                "week_of_year": d.isocalendar().week,
                "month": d.month,
                "month_name": d.strftime("%B"),
                "quarter": d.quarter,
                "year": d.year,
                "is_weekend": is_weekend,
                "is_holiday": is_hol,
                "holiday_name": hol_name,
            })

        return pd.DataFrame(cal_records)

    def _generate_promotions(self, products_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """Generates realistic marketing promotional campaigns."""
        promos = []
        promo_id = 1
        all_dates = calendar_df["date"].tolist()

        # Strategic promotional windows
        promo_periods = [
            ("Spring Refresh", "03-15", "03-22", 0.15),
            ("Summer Kickoff", "05-20", "05-29", 0.20),
            ("Back to School", "08-10", "08-25", 0.18),
            ("Fall Clearance", "10-05", "10-14", 0.15),
            ("Cyber Week Surge", "11-22", "12-02", 0.25),
            ("Holiday Rush", "12-15", "12-24", 0.20),
        ]

        for year in [2023, 2024, 2025]:
            for p_name, s_md, e_md, discount in promo_periods:
                s_date = f"{year}-{s_md}"
                e_date = f"{year}-{e_md}"
                if s_date in all_dates and e_date in all_dates:
                    # Select 35% of products to be on discount in this campaign
                    selected_prods = products_df.sample(frac=0.35, random_state=promo_id)["product_id"].tolist()
                    for pid in selected_prods:
                        promos.append({
                            "promo_id": f"PRM_{promo_id:05d}",
                            "product_id": pid,
                            "start_date": s_date,
                            "end_date": e_date,
                            "discount_pct": discount,
                            "promo_channel": np.random.choice(["Digital Ads", "Email Blast", "In-Store Feature", "Omnichannel"]),
                        })
                        promo_id += 1

        return pd.DataFrame(promos)

    def generate_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Executes the full simulation generating:
        - suppliers, products, regions, calendar, promotions
        - realistic sales time-series with causal components
        - daily inventory balances with realistic stockouts and lead-time deliveries
        """
        suppliers_df = pd.DataFrame(self.suppliers)
        regions_df = pd.DataFrame(self.regions)
        products_df = self._generate_products()
        calendar_df = self._generate_calendar()
        promotions_df = self._generate_promotions(products_df, calendar_df)

        # Quick lookup for promotions: (product_id, date) -> discount_pct
        promo_lookup = {}
        for _, row in promotions_df.iterrows():
            p_dates = pd.date_range(row["start_date"], row["end_date"])
            for d in p_dates:
                promo_lookup[(row["product_id"], d.strftime("%Y-%m-%d"))] = row["discount_pct"]

        # Simulate Sales and Inventory
        dates_list = calendar_df["date"].tolist()
        num_days = len(dates_list)

        sales_records = []
        inventory_records = []

        # Weekly profile factors by category: [Mon, Tue, Wed, Thu, Fri, Sat, Sun]
        weekly_profiles = {
            "Electronics": [0.85, 0.90, 0.95, 1.05, 1.15, 1.35, 1.25],
            "Apparel & Fashion": [0.80, 0.85, 0.90, 1.00, 1.20, 1.50, 1.40],
            "Home & Kitchen": [0.85, 0.90, 0.95, 1.00, 1.10, 1.40, 1.30],
            "Health & Personal Care": [1.05, 1.02, 1.00, 1.00, 1.05, 1.15, 1.10],
            "Office & Industrial": [1.30, 1.35, 1.25, 1.20, 1.05, 0.50, 0.40],
        }

        # Simulate product-by-product
        for _, prod in products_df.iterrows():
            pid = prod["product_id"]
            cat = prod["category"]
            base_vol = prod["base_daily_demand"]
            slope = prod["trend_slope"]
            amp = prod["seasonal_amplitude"]
            elasticity = prod["price_elasticity"]
            unit_cost = prod["unit_cost"]
            unit_price = prod["unit_price"]
            lead_time = int(prod["lead_time_days"])
            reorder_point = int(prod["reorder_point_base"])
            min_order_qty = int(prod["min_order_qty"])
            weekly_profile = weekly_profiles[cat]

            # Track inventory state per product
            safety_stock = int(base_vol * 3.5) # ~3.5 days demand buffer
            current_inv = int(reorder_point * 1.8) # healthy initial inventory
            pending_orders = [] # list of (arrival_day_idx, quantity)

            for t_idx, d_str in enumerate(dates_list):
                cal_row = calendar_df.iloc[t_idx]
                day_of_week = cal_row["day_of_week"]
                is_hol = cal_row["is_holiday"]
                hol_name = cal_row["holiday_name"]

                # Process arriving shipments
                arrived_qty = sum(qty for arr_day, qty in pending_orders if arr_day == t_idx)
                current_inv += arrived_qty
                beginning_inv = current_inv

                # Calculate True Demand Signal
                # 1. Base + Trend
                trend_component = 1.0 + (slope * (t_idx / 365.0))
                
                # 2. Annual Seasonality (Winter/Q4 peak for retail, summer for travel/apparel)
                day_of_year = t_idx % 365
                if cat in ["Electronics", "Apparel & Fashion", "Home & Kitchen"]:
                    # Strong Q4 holiday seasonality peaking in late Nov / Dec
                    seasonal_factor = 1.0 + amp * math.sin(2 * math.pi * (day_of_year - 80) / 365.0)
                    if cal_row["month"] in [11, 12]:
                        seasonal_factor *= 1.35
                else:
                    # Steady with modest cycle
                    seasonal_factor = 1.0 + amp * math.sin(2 * math.pi * day_of_year / 365.0)

                # 3. Day of week factor
                dow_factor = weekly_profile[day_of_week]

                # 4. Holiday Boost
                holiday_multiplier = 1.0
                if is_hol:
                    if hol_name in ["Black Friday", "Cyber Monday"]:
                        holiday_multiplier = 2.4
                    elif hol_name in ["Prime Day Event"]:
                        holiday_multiplier = 2.1
                    elif hol_name in ["Christmas Eve", "Thanksgiving Day"]:
                        holiday_multiplier = 1.7
                    else:
                        holiday_multiplier = 1.3

                # 5. Promotional Elasticity
                discount = promo_lookup.get((pid, d_str), 0.0)
                promo_multiplier = 1.0 + (elasticity * discount)
                promo_flag = 1 if discount > 0 else 0

                # 6. Random Error (log-normal to ensure positive skew)
                noise = np.random.lognormal(mean=0.0, sigma=0.18)

                # Total unconstrained consumer demand for this product today
                total_demanded = base_vol * trend_component * seasonal_factor * dow_factor * holiday_multiplier * promo_multiplier * noise
                total_demanded = max(0, int(round(total_demanded)))

                # Distribute demand across regions
                reg_fractions = [r["demand_weight"] for r in self.regions]
                total_weight = sum(reg_fractions)
                reg_probs = [w / total_weight for w in reg_fractions]
                regional_demands = np.random.multinomial(total_demanded, reg_probs)

                # Total fulfilled sales capped by available inventory
                available_for_sale = current_inv
                actual_sales_total = min(available_for_sale, total_demanded)
                unmet_demand_total = max(0, total_demanded - actual_sales_total)
                stockout_flag = 1 if unmet_demand_total > 0 else 0

                # Update ending inventory
                ending_inv = current_inv - actual_sales_total
                current_inv = ending_inv

                # Place replenishment order if inventory falls below Reorder Point
                # and no order is already in pipeline
                has_pending_order = any(arr_day > t_idx for arr_day, _ in pending_orders)
                reorder_triggered = 0
                if ending_inv <= reorder_point and not has_pending_order:
                    # Realistic lead time with supplier variation
                    actual_lead_time = max(3, int(np.random.normal(lead_time, 1.2)))
                    order_qty = max(min_order_qty, int(reorder_point * 1.5))
                    pending_orders.append((t_idx + actual_lead_time, order_qty))
                    reorder_triggered = 1

                # Record inventory snapshot
                inv_val = ending_inv * unit_cost
                inventory_records.append({
                    "date": d_str,
                    "product_id": pid,
                    "warehouse_id": "WH_CENTRAL",
                    "beginning_inventory": beginning_inv,
                    "sales_quantity": actual_sales_total,
                    "replenishment_received": arrived_qty,
                    "ending_inventory": ending_inv,
                    "stockout_flag": stockout_flag,
                    "reorder_point": reorder_point,
                    "safety_stock": safety_stock,
                    "inventory_value": inv_val,
                })

                # Record sales across regions (proportional allocation)
                actual_allocations = (
                    np.random.multinomial(actual_sales_total, reg_probs)
                    if actual_sales_total > 0
                    else [0] * len(self.regions)
                )
                effective_price = unit_price * (1.0 - discount)

                for r_idx, reg in enumerate(self.regions):
                    reg_units = int(actual_allocations[r_idx])
                    reg_unmet = int(round(unmet_demand_total * reg_probs[r_idx]))
                    rev = reg_units * effective_price
                    sales_records.append({
                        "date": d_str,
                        "product_id": pid,
                        "region_id": reg["region_id"],
                        "sales_quantity": reg_units,
                        "unmet_demand": reg_unmet,
                        "unit_price": unit_price,
                        "revenue": rev,
                        "promotion_flag": promo_flag,
                        "discount_rate": discount,
                    })

        sales_df = pd.DataFrame(sales_records)
        inventory_df = pd.DataFrame(inventory_records)

        return {
            "suppliers": suppliers_df,
            "regions": regions_df,
            "products": products_df,
            "calendar": calendar_df,
            "promotions": promotions_df,
            "sales": sales_df,
            "inventory": inventory_df,
        }

    def save_to_disk(self, raw_dir: str = "data/raw") -> Dict[str, str]:
        """Generates and writes all datasets to disk as CSV files."""
        os.makedirs(raw_dir, exist_ok=True)
        data = self.generate_full_dataset()
        saved_paths = {}

        for name, df in data.items():
            path = os.path.join(raw_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            saved_paths[name] = path

        # Create Data Dictionary
        dict_path = os.path.join(raw_dir, "DATA_DICTIONARY.md")
        self._generate_data_dictionary(dict_path)
        saved_paths["data_dictionary"] = dict_path

        return saved_paths

    def _generate_data_dictionary(self, filepath: str) -> None:
        """Generates formal Markdown documentation explaining all fields."""
        doc = """# Data Dictionary: AI-Powered Inventory Demand Forecasting & Optimization System

This dataset represents a comprehensive, multi-year supply chain transactional system across 50 products, 5 categories, 5 geographic regions, and central fulfillment facilities from 2023-01-01 through 2025-12-31 (1,095 days).

---

## 1. Products Table (`products.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `product_id` | String (PK) | Unique SKU identifier (e.g. `PROD_001` - `PROD_050`) |
| `product_name` | String | Commercial product catalog name |
| `category` | String | Major retail merchandise category |
| `unit_cost` | Float | Unit acquisition cost / COGS ($) |
| `unit_price` | Float | Regular retail selling price ($) |
| `holding_cost_rate` | Float | Annual inventory holding cost percentage rate (default 0.20 = 20%) |
| `supplier_id` | String (FK) | Primary contracted vendor ID |
| `lead_time_days` | Integer | Contractual vendor lead time in calendar days |
| `reorder_point_base` | Integer | Baseline reorder trigger inventory level |
| `min_order_qty` | Integer | Minimum order quantity (MOQ) for replenishment |
| `base_daily_demand` | Float | Intrinsic unconstrained daily demand volume |
| `trend_slope` | Float | Long-term macro growth or decay trajectory |
| `seasonal_amplitude` | Float | Amplitude coefficient for annual seasonality waves |
| `price_elasticity` | Float | Demand elasticity response to promotional price reductions |

---

## 2. Suppliers Table (`suppliers.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `supplier_id` | String (PK) | Vendor identifier |
| `supplier_name` | String | Legal entity name of supplier |
| `lead_time_mean` | Float | Expected average shipment delivery transit days |
| `lead_time_std` | Float | Standard deviation of transit delivery days (lead time variance) |
| `reliability_score` | Float | Historical on-time delivery rate (0.0 to 1.0) |

---

## 3. Regions Table (`regions.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `region_id` | String (PK) | Regional market identifier (`REG_NE`, `REG_WC`, etc.) |
| `region_name` | String | Regional territory description |
| `warehouse_id` | String | Primary regional fulfillment center |
| `demand_weight` | Float | Relative demand share multiplier |

---

## 4. Calendar Table (`calendar.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `date` | Date (PK) | Calendar date (YYYY-MM-DD) |
| `day_of_week` | Integer | Day of week index (0=Monday, 6=Sunday) |
| `day_name` | String | Full day name |
| `week_of_year` | Integer | ISO calendar week number (1-53) |
| `month` | Integer | Calendar month index (1-12) |
| `month_name` | String | Full month name |
| `quarter` | Integer | Fiscal/calendar quarter (1-4) |
| `year` | Integer | Calendar year |
| `is_weekend` | Integer | Binary flag: 1 for Saturday/Sunday, else 0 |
| `is_holiday` | Integer | Binary flag: 1 for major retail holiday, else 0 |
| `holiday_name` | String | Official holiday name (e.g. Black Friday, Cyber Monday, Christmas) |

---

## 5. Promotions Table (`promotions.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `promo_id` | String (PK) | Marketing campaign identifier |
| `product_id` | String (FK) | Targeted SKU |
| `start_date` | Date | Campaign commencement date |
| `end_date` | Date | Campaign termination date |
| `discount_pct` | Float | Fractional price discount (e.g. 0.20 = 20% off) |
| `promo_channel` | String | Marketing acquisition channel |

---

## 6. Daily Sales Table (`sales.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `date` | Date (FK) | Transaction date |
| `product_id` | String (FK) | SKU sold |
| `region_id` | String (FK) | Market region of fulfillment |
| `sales_quantity` | Integer | Realized fulfilled units sold to consumers |
| `unmet_demand` | Integer | Lost sales volume due to on-hand stockout |
| `unit_price` | Float | Base product retail price ($) |
| `revenue` | Float | Total dollar revenue realized: `sales_quantity * unit_price * (1 - discount)` |
| `promotion_flag` | Integer | Binary flag: 1 if active promotion applied on date |
| `discount_rate` | Float | Applied promotional discount percentage |

---

## 7. Daily Inventory Snapshots Table (`inventory.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `date` | Date (FK) | Snapshot observation date |
| `product_id` | String (FK) | SKU tracked |
| `warehouse_id` | String | Central distribution center |
| `beginning_inventory` | Integer | On-hand physical units available at start of day |
| `sales_quantity` | Integer | Units fulfilled and dispatched during the day |
| `replenishment_received`| Integer | Inbound purchase order units received from supplier |
| `ending_inventory` | Integer | Physical inventory at end of business day |
| `stockout_flag` | Integer | Binary flag: 1 if demand exceeded inventory |
| `reorder_point` | Integer | Dynamic policy threshold triggering purchase order |
| `safety_stock` | Integer | Calculated buffer stock to prevent stockouts during lead time |
| `inventory_value` | Float | Capital tied up in ending inventory (`ending_inventory * unit_cost`) |
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc)
