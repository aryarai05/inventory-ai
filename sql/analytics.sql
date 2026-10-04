-- ==========================================================
-- AI-Powered Inventory Demand Forecasting & Optimization System
-- Advanced Supply Chain & Inventory Analytics Queries
-- ==========================================================

-- 1. Inventory Turnover Ratio by Product
-- Formula: Annualized COGS / Average Inventory Value
-- High turnover = efficient capital; Low turnover = stagnation / risk of obsolescence
WITH product_cogs AS (
    SELECT 
        s.product_id,
        SUM(s.sales_quantity * p.unit_cost) AS total_cogs
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY s.product_id
),
product_avg_inv AS (
    SELECT 
        i.product_id,
        AVG(i.ending_inventory * p.unit_cost) AS avg_inventory_value
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    GROUP BY i.product_id
)
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    ROUND(c.total_cogs, 2) AS total_cogs,
    ROUND(a.avg_inventory_value, 2) AS avg_inventory_value,
    ROUND(c.total_cogs / NULLIF(a.avg_inventory_value, 0), 2) AS inventory_turnover_ratio
FROM products p
JOIN product_cogs c ON p.product_id = c.product_id
JOIN product_avg_inv a ON p.product_id = a.product_id
ORDER BY inventory_turnover_ratio DESC;

-- 2. Stockout Frequency & Lost Revenue Analysis
-- Pinpoints SKUs suffering most from stockouts and calculates lost revenue
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    COUNT(i.inv_id) AS total_inventory_snapshots,
    SUM(i.stockout_flag) AS stockout_days,
    ROUND(100.0 * SUM(i.stockout_flag) / COUNT(i.inv_id), 2) AS stockout_frequency_pct,
    SUM(s.unmet_demand) AS total_unmet_demand_units,
    ROUND(SUM(s.unmet_demand * p.unit_price), 2) AS estimated_lost_revenue
FROM products p
JOIN inventory i ON p.product_id = i.product_id
JOIN sales s ON p.product_id = s.product_id AND i.date = s.date
GROUP BY p.product_id
HAVING stockout_days > 0
ORDER BY estimated_lost_revenue DESC;

-- 3. Top-Selling Products (Fast Movers / ABC Analysis Category A)
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.sales_quantity) AS units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    ROUND(AVG(s.sales_quantity), 1) AS avg_daily_units,
    ROUND(100.0 * SUM(s.revenue) / (SELECT SUM(revenue) FROM sales), 2) AS pct_of_total_sales
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY p.product_id
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Slow-Moving Products (Long Inventory Tail / ABC Analysis Category C)
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.sales_quantity) AS units_sold,
    ROUND(AVG(s.sales_quantity), 2) AS avg_daily_units,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    (SELECT ending_inventory FROM inventory WHERE product_id = p.product_id ORDER BY date DESC LIMIT 1) AS latest_inventory,
    ROUND((SELECT ending_inventory FROM inventory WHERE product_id = p.product_id ORDER BY date DESC LIMIT 1) / NULLIF(AVG(s.sales_quantity), 0), 1) AS estimated_days_of_supply
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY p.product_id
ORDER BY units_sold ASC
LIMIT 10;

-- 5. Inventory Valuation Trend by Category (Working Capital Analysis)
SELECT 
    p.category,
    ROUND(SUM(i.ending_inventory * p.unit_cost), 2) AS total_inventory_cost_value,
    ROUND(SUM(i.ending_inventory * p.unit_price), 2) AS potential_retail_value,
    ROUND(AVG(i.ending_inventory), 0) AS avg_units_on_hand,
    ROUND(SUM(i.ending_inventory * p.unit_cost * (p.holding_cost_rate / 365.0)), 2) AS estimated_daily_holding_cost
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE i.date = (SELECT MAX(date) FROM inventory)
GROUP BY p.category
ORDER BY total_inventory_cost_value DESC;

-- 6. Supplier Lead-Time and Operational Performance
SELECT 
    s.supplier_id,
    s.supplier_name,
    s.reliability_score,
    s.lead_time_mean,
    s.lead_time_std,
    COUNT(DISTINCT p.product_id) AS products_supplied,
    ROUND(AVG(p.lead_time_days), 1) AS avg_contract_lead_time
FROM suppliers s
LEFT JOIN products p ON s.supplier_id = p.supplier_id
GROUP BY s.supplier_id
ORDER BY s.reliability_score DESC;
