# Data Dictionary: AI-Powered Inventory Demand Forecasting & Optimization System

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
