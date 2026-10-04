-- ==========================================================
-- AI-Powered Inventory Demand Forecasting & Optimization System
-- Time-Series & Cross-Sectional Aggregation Queries
-- ==========================================================

-- 1. Daily Sales Aggregation across Enterprise
-- Used for macro executive trend tracking
SELECT 
    s.date,
    c.day_name,
    c.is_weekend,
    c.is_holiday,
    COUNT(DISTINCT s.product_id) AS active_products_sold,
    SUM(s.sales_quantity) AS total_units_sold,
    SUM(s.unmet_demand) AS total_lost_sales_units,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    ROUND(AVG(s.unit_price), 2) AS avg_unit_price
FROM sales s
JOIN calendar c ON s.date = c.date
GROUP BY s.date
ORDER BY s.date ASC;

-- 2. Weekly Sales Aggregation
-- Smooths weekday-weekend variance for mid-term trend analysis
SELECT 
    c.year,
    c.week_of_year,
    MIN(s.date) AS week_start_date,
    SUM(s.sales_quantity) AS weekly_units_sold,
    ROUND(SUM(s.revenue), 2) AS weekly_revenue,
    SUM(s.unmet_demand) AS weekly_unmet_demand
FROM sales s
JOIN calendar c ON s.date = c.date
GROUP BY c.year, c.week_of_year
ORDER BY c.year ASC, c.week_of_year ASC;

-- 3. Monthly Sales Aggregation
-- High-level finance and S&OP planning
SELECT 
    c.year,
    c.month,
    c.month_name,
    COUNT(DISTINCT s.sale_id) AS total_transactions,
    SUM(s.sales_quantity) AS monthly_units_sold,
    ROUND(SUM(s.revenue), 2) AS monthly_revenue,
    ROUND(AVG(s.sales_quantity), 2) AS avg_units_per_order
FROM sales s
JOIN calendar c ON s.date = c.date
GROUP BY c.year, c.month, c.month_name
ORDER BY c.year ASC, c.month ASC;

-- 4. Product-Level Demand Summary
-- Aggregates overall performance, volatility, and volume per SKU
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.unit_cost,
    p.unit_price,
    ROUND(p.unit_price - p.unit_cost, 2) AS unit_margin,
    SUM(s.sales_quantity) AS total_sales_volume,
    ROUND(AVG(s.sales_quantity), 2) AS avg_daily_demand,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    SUM(s.unmet_demand) AS total_unmet_demand,
    ROUND(100.0 * SUM(s.unmet_demand) / NULLIF(SUM(s.sales_quantity + s.unmet_demand), 0), 2) AS stockout_impact_pct
FROM products p
LEFT JOIN sales s ON p.product_id = s.product_id
GROUP BY p.product_id
ORDER BY total_revenue DESC;

-- 5. Category-Level Demand Aggregation
-- Assesses portfolio distribution
SELECT 
    p.category,
    COUNT(DISTINCT p.product_id) AS product_count,
    SUM(s.sales_quantity) AS category_units_sold,
    ROUND(SUM(s.revenue), 2) AS category_revenue,
    ROUND(100.0 * SUM(s.revenue) / (SELECT SUM(revenue) FROM sales), 2) AS revenue_share_pct
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY category_revenue DESC;

-- 6. Regional Demand Aggregation
-- Geographic sales distribution and logistics hubs
SELECT 
    r.region_id,
    r.region_name,
    r.warehouse_id,
    SUM(s.sales_quantity) AS total_regional_units,
    ROUND(SUM(s.revenue), 2) AS total_regional_revenue,
    ROUND(AVG(s.sales_quantity), 2) AS avg_units_per_day
FROM regions r
JOIN sales s ON r.region_id = s.region_id
GROUP BY r.region_id, r.region_name, r.warehouse_id
ORDER BY total_regional_revenue DESC;
