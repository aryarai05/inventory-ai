-- ==========================================================
-- AI-Powered Inventory Demand Forecasting & Optimization System
-- SQLite Database Schema Definition
-- ==========================================================

DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS promotions;
DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS regions;

-- 1. Suppliers Table
CREATE TABLE suppliers (
    supplier_id VARCHAR(20) PRIMARY KEY,
    supplier_name VARCHAR(100) NOT NULL,
    lead_time_mean REAL NOT NULL,
    lead_time_std REAL NOT NULL,
    reliability_score REAL NOT NULL,
    country VARCHAR(50) DEFAULT 'USA'
);

-- 2. Products Table
CREATE TABLE products (
    product_id VARCHAR(20) PRIMARY KEY,
    product_name VARCHAR(120) NOT NULL,
    category VARCHAR(50) NOT NULL,
    unit_cost REAL NOT NULL,
    unit_price REAL NOT NULL,
    holding_cost_rate REAL DEFAULT 0.20,
    supplier_id VARCHAR(20) NOT NULL,
    lead_time_days INTEGER NOT NULL,
    reorder_point_base INTEGER DEFAULT 50,
    min_order_qty INTEGER DEFAULT 10,
    base_daily_demand REAL,
    trend_slope REAL,
    seasonal_amplitude REAL,
    price_elasticity REAL,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

-- 3. Regions / Warehouses Table
CREATE TABLE regions (
    region_id VARCHAR(20) PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL,
    warehouse_id VARCHAR(30) NOT NULL,
    demand_weight REAL DEFAULT 1.0,
    warehouse_capacity INTEGER DEFAULT 50000
);

-- 4. Calendar Table
CREATE TABLE calendar (
    date DATE PRIMARY KEY,
    day_of_week INTEGER NOT NULL,
    day_name VARCHAR(15) NOT NULL,
    week_of_year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(15) NOT NULL,
    quarter INTEGER NOT NULL,
    year INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL,
    is_holiday INTEGER NOT NULL,
    holiday_name VARCHAR(50)
);

-- 5. Promotions Table
CREATE TABLE promotions (
    promo_id VARCHAR(20) PRIMARY KEY,
    product_id VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    discount_pct REAL NOT NULL,
    promo_channel VARCHAR(50),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- 6. Daily Sales Transactions Table
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    product_id VARCHAR(20) NOT NULL,
    region_id VARCHAR(20) NOT NULL,
    sales_quantity INTEGER NOT NULL,
    unmet_demand INTEGER DEFAULT 0,
    unit_price REAL NOT NULL,
    revenue REAL NOT NULL,
    promotion_flag INTEGER DEFAULT 0,
    discount_rate REAL DEFAULT 0.0,
    FOREIGN KEY (date) REFERENCES calendar(date),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

-- 7. Daily Inventory Snapshots Table
CREATE TABLE inventory (
    inv_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    product_id VARCHAR(20) NOT NULL,
    warehouse_id VARCHAR(30) NOT NULL,
    beginning_inventory INTEGER NOT NULL,
    sales_quantity INTEGER NOT NULL,
    replenishment_received INTEGER DEFAULT 0,
    ending_inventory INTEGER NOT NULL,
    stockout_flag INTEGER DEFAULT 0,
    reorder_point INTEGER NOT NULL,
    safety_stock INTEGER NOT NULL,
    inventory_value REAL NOT NULL,
    FOREIGN KEY (date) REFERENCES calendar(date),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Indexes for high-performance aggregations and time-series joins
CREATE INDEX idx_sales_date_prod ON sales (date, product_id);
CREATE INDEX idx_sales_prod ON sales (product_id);
CREATE INDEX idx_sales_region ON sales (region_id);
CREATE INDEX idx_inv_date_prod ON inventory (date, product_id);
CREATE INDEX idx_inv_stockout ON inventory (stockout_flag);
