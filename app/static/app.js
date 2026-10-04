/**
 * Optix Enterprise Inventory AI - Core Application Logic
 * Self-contained, robust asynchronous JavaScript client.
 * Manages reactive UI state, REST API calls, charting engines, and live PuLP optimization.
 */

// Global State
const state = {
  activeTab: 'executive',
  products: [],
  risks: [],
  currentRiskFilter: 'All',
  charts: {
    executiveTrend: null,
    categoryShare: null,
    regionalDemand: null,
    skuForecast: null,
    pulpTrajectory: null,
    featureImportance: null,
    residuals: null
  },
  lastSQLResults: null
};

// Application Bootstrap
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadSystemStatus();
    await loadExecutiveDashboard();
    await loadProductCatalog();
    await loadRiskReport();
    await loadMLOpsLeaderboard();
  } catch (err) {
    console.error('Bootstrap error:', err);
  }
});

// -----------------------------------------------------------------------------
// TAB SWITCHING
// -----------------------------------------------------------------------------
function switchTab(tabId) {
  const tabs = ['executive', 'forecasting', 'risks', 'products', 'optimization', 'mlops', 'sql', 'highlights'];
  tabs.forEach(t => {
    const viewEl = document.getElementById(`view-${t}`);
    const btnEl = document.getElementById(`tab-btn-${t}`);
    if (viewEl) viewEl.classList.add('hidden');
    if (btnEl) btnEl.classList.remove('active');
  });

  const targetView = document.getElementById(`view-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  if (targetView) targetView.classList.remove('hidden');
  if (targetBtn) targetBtn.classList.add('active');

  state.activeTab = tabId;

  // Lazy render charts when their tab becomes visible
  if (tabId === 'forecasting' && !state.charts.skuForecast) {
    loadSKUForecast();
  } else if (tabId === 'optimization' && !state.charts.pulpTrajectory) {
    executePuLPOptimization();
  }
}

// -----------------------------------------------------------------------------
// SYSTEM STATUS
// -----------------------------------------------------------------------------
async function loadSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const dbPill = document.getElementById('status-db');
    if (dbPill && data.database) {
      dbPill.innerText = `SQLite Connected (${data.database.total_records.toLocaleString()} rows)`;
    }
  } catch (err) {
    console.warn('Status error:', err);
  }
}

// -----------------------------------------------------------------------------
// EXECUTIVE DASHBOARD
// -----------------------------------------------------------------------------
async function loadExecutiveDashboard() {
  try {
    // 1. Fetch Executive KPIs
    const kpiRes = await fetch('/api/kpis');
    const kpi = await kpiRes.json();

    document.getElementById('kpi-revenue').innerText = `$${(kpi.total_revenue / 1e6).toFixed(2)}M`;
    document.getElementById('kpi-units-sold').innerText = `${kpi.total_units_sold.toLocaleString()} units fulfilled`;
    document.getElementById('kpi-inv-value').innerText = `$${(kpi.total_inventory_value / 1e3).toFixed(1)}K`;
    document.getElementById('kpi-inv-units').innerText = `${kpi.current_inventory_units.toLocaleString()} units in DC`;
    document.getElementById('kpi-stockout-rate').innerText = `${kpi.stockout_rate_pct.toFixed(2)}%`;
    document.getElementById('kpi-accuracy').innerText = `${kpi.forecast_accuracy_pct.toFixed(1)}%`;
    document.getElementById('kpi-critical-count').innerText = `${kpi.critical_stockouts_count} SKUs`;
    document.getElementById('kpi-reorder-count').innerText = `${kpi.reorders_needed_count} SKUs`;

    // 2. Fetch Executive Charts
    const chartRes = await fetch('/api/charts/executive');
    const chartsData = await chartRes.json();

    renderExecutiveTrendChart(chartsData.daily_trend);
    renderCategoryShareChart(chartsData.category_share);
    renderRegionalDemandChart(chartsData.regional_demand);

  } catch (err) {
    console.error('Executive dashboard load error:', err);
  }
}

function renderExecutiveTrendChart(trendData) {
  if (typeof ApexCharts === 'undefined') return;

  const step = 4;
  const sampledDates = trendData.dates.filter((_, i) => i % step === 0);
  const sampledSales = trendData.sales_quantity.filter((_, i) => i % step === 0);
  const sampledMA = trendData.ma_30.filter((_, i) => i % step === 0);

  const options = {
    series: [
      { name: 'Daily Units Demanded', type: 'area', data: sampledSales },
      { name: '30-Day Moving Average', type: 'line', data: sampledMA }
    ],
    chart: { height: 340, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#3b82f6', '#10b981'],
    stroke: { width: [1.2, 3], curve: 'smooth' },
    fill: {
      type: ['gradient', 'solid'],
      gradient: { shade: 'dark', type: 'vertical', shadeIntensity: 0.5, opacityFrom: 0.25, opacityTo: 0.0 }
    },
    labels: sampledDates,
    xaxis: {
      type: 'datetime',
      labels: { style: { colors: '#64748b', fontSize: '11px' } },
      axisBorder: { show: false },
      axisTicks: { show: false }
    },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { labels: { colors: '#94a3b8' }, position: 'top', horizontalAlign: 'right' },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-executive-trend");
  if (el) {
    el.innerHTML = '';
    state.charts.executiveTrend = new ApexCharts(el, options);
    state.charts.executiveTrend.render();
  }
}

function renderCategoryShareChart(catData) {
  if (typeof ApexCharts === 'undefined') return;

  const labels = catData.map(c => c.category);
  const series = catData.map(c => c.units_sold);

  const options = {
    series: series,
    labels: labels,
    chart: { type: 'donut', height: 320, background: 'transparent' },
    colors: ['#6366f1', '#10b981', '#3b82f6', '#f59e0b', '#ec4899'],
    stroke: { colors: ['#0f172a'], width: 2 },
    dataLabels: { enabled: false },
    legend: { position: 'bottom', labels: { colors: '#94a3b8' }, fontSize: '11px' },
    plotOptions: {
      pie: {
        donut: {
          size: '72%',
          labels: {
            show: true,
            total: {
              show: true,
              label: 'Total Units',
              color: '#94a3b8',
              formatter: () => series.reduce((a, b) => a + b, 0).toLocaleString()
            }
          }
        }
      }
    },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-category-share");
  if (el) {
    el.innerHTML = '';
    state.charts.categoryShare = new ApexCharts(el, options);
    state.charts.categoryShare.render();
  }
}

function renderRegionalDemandChart(regData) {
  if (typeof ApexCharts === 'undefined') return;

  const options = {
    series: [{ name: 'Units Sold', data: regData.map(r => r.units_sold) }],
    chart: { type: 'bar', height: 260, background: 'transparent', toolbar: { show: false } },
    colors: ['#6366f1'],
    plotOptions: { bar: { borderRadius: 4, horizontal: true, barHeight: '55%' } },
    dataLabels: { enabled: false },
    xaxis: {
      categories: regData.map(r => r.region_name),
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    yaxis: { labels: { style: { colors: '#94a3b8', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-regional-demand");
  if (el) {
    el.innerHTML = '';
    state.charts.regionalDemand = new ApexCharts(el, options);
    state.charts.regionalDemand.render();
  }
}

// -----------------------------------------------------------------------------
// PRODUCT CATALOG & DROPDOWNS
// -----------------------------------------------------------------------------
async function loadProductCatalog() {
  try {
    const res = await fetch('/api/products');
    state.products = await res.json();

    // Populate category dropdown
    const categories = ['All', ...new Set(state.products.map(p => p.category))];
    const catSelect = document.getElementById('forecast-cat-filter');
    if (catSelect) {
      catSelect.innerHTML = categories.map(c => `<option value="${c}">${c === 'All' ? 'All Categories' : c}</option>`).join('');
    }

    // Populate SKU dropdowns
    populateProductDropdowns('All');

    // Populate Catalog Table in Tab 4
    renderCatalogTable(state.products);

  } catch (err) {
    console.error('Catalog error:', err);
  }
}

function populateProductDropdowns(selectedCategory) {
  let filtered = state.products;
  if (selectedCategory && selectedCategory !== 'All') {
    filtered = state.products.filter(p => p.category === selectedCategory);
  }

  const skuSelect = document.getElementById('forecast-sku-filter');
  const optSelect = document.getElementById('opt-sku-select');

  const optionsHtml = filtered.map(p => `<option value="${p.product_id}">${p.product_id} - ${p.product_name}</option>`).join('');

  if (skuSelect) skuSelect.innerHTML = optionsHtml;
  if (optSelect) optSelect.innerHTML = state.products.map(p => `<option value="${p.product_id}">${p.product_id} - ${p.product_name}</option>`).join('');
}

function onForecastCategoryChange() {
  const cat = document.getElementById('forecast-cat-filter').value;
  populateProductDropdowns(cat);
  loadSKUForecast();
}

function renderCatalogTable(products) {
  const tbody = document.querySelector('#table-catalog tbody');
  if (!tbody) return;

  tbody.innerHTML = products.map(p => `
    <tr>
      <td style="font-family: monospace; font-weight: 700; color: #818cf8;">${p.product_id}</td>
      <td style="font-weight: 700; color: white;">${p.product_name}</td>
      <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted);">${p.category}</span></td>
      <td style="font-family: monospace; color: var(--text-muted);">$${p.unit_cost.toFixed(2)}</td>
      <td style="font-family: monospace; font-weight: 700; color: white;">$${p.unit_price.toFixed(2)}</td>
      <td style="font-weight: 700; color: #34d399;">$${p.unit_margin.toFixed(2)}</td>
      <td>${p.lead_time_days} days</td>
      <td style="color: var(--text-muted);">${p.supplier_name || 'Tier-1 Supplier'}</td>
      <td>
        <span class="badge badge-healthy">
          ${((p.reliability_score || 0.95) * 100).toFixed(0)}%
        </span>
      </td>
      <td>
        <button onclick="inspectSKU('${p.product_id}')" class="btn btn-outline" style="font-size: 0.72rem; padding: 0.3rem 0.6rem;">
          Forecast
        </button>
      </td>
    </tr>
  `).join('');
}

function filterCatalogTable() {
  const q = document.getElementById('catalog-search').value.toLowerCase();
  const filtered = state.products.filter(p => 
    p.product_id.toLowerCase().includes(q) || 
    p.product_name.toLowerCase().includes(q) ||
    p.category.toLowerCase().includes(q)
  );
  renderCatalogTable(filtered);
}

function inspectSKU(pid) {
  const skuSelect = document.getElementById('forecast-sku-filter');
  if (skuSelect) skuSelect.value = pid;
  switchTab('forecasting');
  loadSKUForecast();
}

// -----------------------------------------------------------------------------
// DEMAND FORECASTING STUDIO
// -----------------------------------------------------------------------------
async function loadSKUForecast() {
  const skuSelect = document.getElementById('forecast-sku-filter');
  if (!skuSelect || !skuSelect.value) return;

  const pid = skuSelect.value;
  const horizon = document.getElementById('forecast-horizon-slider').value || 14;

  try {
    const res = await fetch(`/api/forecast?product_id=${pid}&horizon_days=${horizon}`);
    const data = await res.json();

    document.getElementById('f-sku-id').innerText = data.product_id;
    document.getElementById('f-sku-name').innerText = data.product_name;
    document.getElementById('f-sku-price').innerText = `$${data.unit_price.toFixed(2)} / $${data.unit_cost.toFixed(2)}`;
    const margin = (((data.unit_price - data.unit_cost) / data.unit_price) * 100).toFixed(1);
    document.getElementById('f-sku-margin').innerText = `${margin}% Gross Margin`;
    document.getElementById('f-sku-leadtime').innerText = `${data.lead_time_days} Days`;
    document.getElementById('f-sku-runrate').innerText = `${data.mean_daily_demand.toFixed(1)} units/day`;
    document.getElementById('f-sku-total').innerText = `${Math.round(data.total_forecasted_units)} units`;
    document.getElementById('f-sku-horizon-desc').innerText = `Across ${data.horizon_days} planning days`;

    renderForecastChart(data);
    renderForecastScheduleTable(data.forecast);

  } catch (err) {
    console.error('Forecast error:', err);
  }
}

function renderForecastChart(data) {
  if (typeof ApexCharts === 'undefined') return;

  const histSlice = data.historical.slice(-45);
  const histDates = histSlice.map(h => h.date);
  const histSales = histSlice.map(h => h.actual_sales);

  const fDates = data.forecast.map(f => f.date);
  const fPreds = data.forecast.map(f => f.predicted_demand);
  const fLower = data.forecast.map(f => f.lower_bound);
  const fUpper = data.forecast.map(f => f.upper_bound);

  const allDates = [...histDates, ...fDates];
  const histSeriesData = [...histSales, ...Array(fDates.length).fill(null)];
  
  const lastHistVal = histSales[histSales.length - 1];
  const fSeriesData = [...Array(histDates.length - 1).fill(null), lastHistVal, ...fPreds];
  const fLowerData = [...Array(histDates.length - 1).fill(null), lastHistVal, ...fLower];
  const fUpperData = [...Array(histDates.length - 1).fill(null), lastHistVal, ...fUpper];

  const options = {
    series: [
      { name: 'Historical Actual Demand', type: 'line', data: histSeriesData },
      { name: 'LightGBM Point Forecast', type: 'line', data: fSeriesData },
      { name: '95% Upper Bound', type: 'line', data: fUpperData },
      { name: '95% Lower Bound', type: 'area', data: fLowerData }
    ],
    chart: { height: 380, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#94a3b8', '#10b981', 'transparent', 'rgba(16, 185, 129, 0.15)'],
    stroke: { width: [2, 3, 0, 0], curve: 'smooth', dashArray: [0, 5, 0, 0] },
    fill: {
      type: ['solid', 'solid', 'solid', 'solid'],
      colors: ['transparent', 'transparent', 'transparent', 'rgba(16, 185, 129, 0.18)']
    },
    labels: allDates,
    xaxis: {
      type: 'datetime',
      labels: { style: { colors: '#64748b', fontSize: '11px' } },
      axisBorder: { show: false },
      axisTicks: { show: false }
    },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { show: false },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-sku-forecast");
  if (el) {
    el.innerHTML = '';
    state.charts.skuForecast = new ApexCharts(el, options);
    state.charts.skuForecast.render();
  }
}

function renderForecastScheduleTable(forecast) {
  const tbody = document.querySelector('#table-forecast-schedule tbody');
  if (!tbody) return;

  tbody.innerHTML = forecast.map(f => {
    const spread = (f.upper_bound - f.lower_bound).toFixed(1);
    return `
      <tr>
        <td style="font-family: monospace; color: var(--text-muted);">${f.date}</td>
        <td style="font-weight: 700; color: #818cf8;">Day +${f.forecast_step}</td>
        <td style="font-weight: 800; color: white;">${f.predicted_demand.toFixed(1)} units</td>
        <td style="color: var(--text-muted);">${f.lower_bound.toFixed(1)}</td>
        <td style="color: var(--text-muted);">${f.upper_bound.toFixed(1)}</td>
        <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted);">±${(spread / 2).toFixed(1)}</span></td>
      </tr>
    `;
  }).join('');
}

// -----------------------------------------------------------------------------
// INVENTORY RISK ENGINE
// -----------------------------------------------------------------------------
async function loadRiskReport() {
  try {
    const res = await fetch('/api/inventory/risks');
    state.risks = await res.json();
    renderRiskTable(state.risks);
  } catch (err) {
    console.error('Risk error:', err);
  }
}

function setRiskFilter(level) {
  state.currentRiskFilter = level;
  filterRiskTable();
}

function filterRiskTable() {
  const searchQ = (document.getElementById('risk-search-input')?.value || '').toLowerCase();
  let filtered = state.risks;

  if (state.currentRiskFilter && state.currentRiskFilter !== 'All') {
    filtered = filtered.filter(r => r.risk_level === state.currentRiskFilter);
  }
  if (searchQ) {
    filtered = filtered.filter(r => 
      r.product_id.toLowerCase().includes(searchQ) ||
      r.product_name.toLowerCase().includes(searchQ) ||
      r.category.toLowerCase().includes(searchQ)
    );
  }

  renderRiskTable(filtered);
}

function renderRiskTable(risks) {
  const tbody = document.querySelector('#table-inventory-risks tbody');
  const countEl = document.getElementById('risk-table-count');
  if (countEl) countEl.innerText = `Showing ${risks.length} SKUs`;
  if (!tbody) return;

  const badgeMap = {
    'CRITICAL STOCKOUT RISK': 'badge-critical',
    'HIGH RISK': 'badge-high',
    'MEDIUM RISK': 'badge-medium',
    'HEALTHY': 'badge-healthy',
    'EXCESS INVENTORY': 'badge-excess'
  };

  tbody.innerHTML = risks.map(r => {
    // Calculate stock meter percentage
    const maxReference = Math.max(r.reorder_point * 1.5, r.current_inventory, 100);
    const fillPct = Math.min(100, Math.max(5, (r.current_inventory / maxReference) * 100));
    const fillColor = r.current_inventory < r.safety_stock ? '#f43f5e' : (r.current_inventory <= r.reorder_point ? '#f59e0b' : '#10b981');

    return `
      <tr>
        <td style="font-family: monospace; font-weight: 700; color: #818cf8;">${r.product_id}</td>
        <td style="font-weight: 700; color: white; white-space: nowrap;">${r.product_name}</td>
        <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted);">${r.category}</span></td>
        <td style="font-family: monospace; font-weight: 800; color: ${r.current_inventory <= r.safety_stock ? '#f43f5e' : 'white'};">${r.current_inventory}</td>
        <td style="min-width: 90px;">
          <div class="stock-meter-track" title="On-Hand: ${r.current_inventory} / Target: ${r.reorder_point}">
            <div class="stock-meter-fill" style="width: ${fillPct}%; background: ${fillColor};"></div>
          </div>
        </td>
        <td style="color: var(--text-muted);">${r.mean_daily_demand}</td>
        <td style="color: var(--text-muted);">${r.safety_stock}</td>
        <td style="color: var(--text-muted);">${r.reorder_point}</td>
        <td style="font-weight: 700; color: ${r.days_of_supply < r.lead_time_days ? '#f43f5e' : '#34d399'};">${r.days_of_supply.toFixed(1)}d</td>
        <td><span class="${badgeMap[r.risk_level] || 'badge-healthy'}">${r.risk_level}</span></td>
        <td style="font-weight: 700; color: ${r.recommended_reorder_qty > 0 ? '#f59e0b' : 'var(--text-dim)'};">
          ${r.recommended_reorder_qty > 0 ? r.recommended_reorder_qty + ' units' : '—'}
        </td>
        <td style="font-size: 0.72rem; color: var(--text-muted); max-width: 320px;" class="truncate" title="${r.diagnostic_reason}">${r.diagnostic_reason}</td>
      </tr>
    `;
  }).join('');
}

// -----------------------------------------------------------------------------
// PuLP MATHEMATICAL REPLENISHMENT OPTIMIZATION
// -----------------------------------------------------------------------------
async function executePuLPOptimization() {
  const skuSelect = document.getElementById('opt-sku-select');
  if (!skuSelect || !skuSelect.value) return;

  const btn = document.getElementById('btn-run-pulp');
  if (btn) btn.innerText = 'Solving Mixed-Integer Program...';

  const payload = {
    product_id: skuSelect.value,
    horizon_days: 14,
    holding_cost_rate: parseFloat(document.getElementById('opt-holding-slider').value) / 100.0,
    fixed_order_cost: parseFloat(document.getElementById('opt-order-slider').value),
    stockout_penalty_multiplier: parseFloat(document.getElementById('opt-penalty-slider').value)
  };

  try {
    const res = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    document.getElementById('opt-status').innerText = result.status;
    document.getElementById('opt-total-cost').innerText = `$${result.total_optimal_cost.toLocaleString()}`;
    document.getElementById('opt-holding-cost').innerText = `$${result.total_holding_cost.toLocaleString()}`;
    document.getElementById('opt-ordering-cost').innerText = `$${result.total_ordering_cost.toLocaleString()}`;
    document.getElementById('opt-po-count').innerText = `${result.total_orders_placed} POs Dispatched`;
    document.getElementById('opt-stockout-penalty').innerText = `$${result.total_stockout_penalty.toLocaleString()}`;

    renderPuLPTrajectoryChart(result.daily_plan);
    renderPuLPDailyTable(result.daily_plan);

  } catch (err) {
    console.error('Optimization error:', err);
  } finally {
    if (btn) btn.innerText = 'Run PuLP MILP Solver';
  }
}

function renderPuLPTrajectoryChart(dailyPlan) {
  if (typeof ApexCharts === 'undefined') return;

  const steps = dailyPlan.map(d => `Day +${d.day_step}`);
  const orders = dailyPlan.map(d => d.recommended_order);
  const demand = dailyPlan.map(d => d.forecast_demand);
  const endingStock = dailyPlan.map(d => d.expected_inventory);

  const options = {
    series: [
      { name: 'Optimal PO Order Dispatch', type: 'column', data: orders },
      { name: 'Forecasted Demand', type: 'line', data: demand },
      { name: 'Ending On-Hand Stock', type: 'line', data: endingStock }
    ],
    chart: { height: 360, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#10b981', '#f59e0b', '#3b82f6'],
    stroke: { width: [0, 2.5, 2.5], curve: 'smooth' },
    plotOptions: { bar: { columnWidth: '35%', borderRadius: 4 } },
    xaxis: {
      categories: steps,
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { labels: { colors: '#94a3b8' }, position: 'top', horizontalAlign: 'right' },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-pulp-trajectory");
  if (el) {
    el.innerHTML = '';
    state.charts.pulpTrajectory = new ApexCharts(el, options);
    state.charts.pulpTrajectory.render();
  }
}

function renderPuLPDailyTable(dailyPlan) {
  const tbody = document.querySelector('#table-opt-daily tbody');
  if (!tbody) return;

  tbody.innerHTML = dailyPlan.map(d => `
    <tr>
      <td style="font-weight: 700; color: #818cf8;">Step ${d.day_step}</td>
      <td style="color: var(--text-muted);">${d.forecast_demand} units</td>
      <td style="font-weight: 800; color: ${d.recommended_order > 0 ? '#10b981' : 'var(--text-dim)'};">
        ${d.recommended_order > 0 ? d.recommended_order + ' units' : '—'}
      </td>
      <td>
        <span class="badge ${d.order_triggered === 1 ? 'badge-healthy' : ''}" style="${d.order_triggered === 0 ? 'background: rgba(255,255,255,0.06); color: var(--text-dim);' : ''}">
          ${d.order_triggered === 1 ? 'TRIGGERED (Y=1)' : 'IDLE (Y=0)'}
        </span>
      </td>
      <td style="font-family: monospace; font-weight: 700; color: white;">${d.expected_inventory}</td>
      <td style="color: ${d.stockout_units > 0 ? '#f43f5e' : 'var(--text-dim)'};">${d.stockout_units > 0 ? d.stockout_units : '0'}</td>
      <td style="font-family: monospace; color: var(--text-muted);">$${d.holding_cost.toFixed(2)}</td>
      <td style="font-family: monospace; color: var(--text-muted);">$${d.ordering_cost.toFixed(2)}</td>
    </tr>
  `).join('');
}

// -----------------------------------------------------------------------------
// MLOPS LEADERBOARD & BENCHMARK
// -----------------------------------------------------------------------------
async function loadMLOpsLeaderboard() {
  try {
    const res = await fetch('/api/models/leaderboard');
    const data = await res.json();

    const tbody = document.querySelector('#table-mlops-leaderboard tbody');
    if (tbody && data.leaderboard) {
      tbody.innerHTML = data.leaderboard.map((m, idx) => `
        <tr style="${idx === 0 ? 'background: rgba(16, 185, 129, 0.05);' : ''}">
          <td style="font-weight: 700; color: ${idx === 0 ? '#34d399' : 'white'};">${m.Model}</td>
          <td style="font-family: monospace; color: white;">${m.MAE.toFixed(3)}</td>
          <td style="font-family: monospace; color: white;">${m.RMSE.toFixed(3)}</td>
          <td style="font-family: monospace; color: var(--text-muted);">${m.MAPE > 9999 ? '>10,000%' : m.MAPE.toFixed(1) + '%'}</td>
          <td style="font-family: monospace; font-weight: 800; color: ${idx === 0 ? '#34d399' : 'white'};">${m.WAPE.toFixed(2)}%</td>
          <td style="font-family: monospace; color: ${m.Bias_Pct >= 0 ? '#34d399' : '#f43f5e'};">${m.Bias_Pct > 0 ? '+' : ''}${m.Bias_Pct.toFixed(2)}%</td>
          <td style="color: var(--text-muted);">${m.Train_Time_Sec.toFixed(3)}s</td>
          <td>
            <span class="badge ${idx === 0 ? 'badge-healthy' : ''}" style="${idx !== 0 ? 'background: rgba(255,255,255,0.06); color: var(--text-dim);' : ''}">
              ${idx === 0 ? '★ Production Champion' : 'Benchmark Candidate'}
            </span>
          </td>
        </tr>
      `).join('');
    }

    if (data.top_features && data.top_features.length > 0) {
      renderFeatureImportanceChart(data.top_features);
    }

    if (data.sample_residuals && data.sample_residuals.length > 0) {
      renderResidualsChart(data.sample_residuals);
    }

  } catch (err) {
    console.error('MLOps error:', err);
  }
}

function renderFeatureImportanceChart(features) {
  if (typeof ApexCharts === 'undefined') return;

  const options = {
    series: [{ name: 'Split Importance', data: features.map(f => f.importance) }],
    chart: { type: 'bar', height: 340, background: 'transparent', toolbar: { show: false } },
    colors: ['#10b981'],
    plotOptions: { bar: { horizontal: true, borderRadius: 4, barHeight: '60%' } },
    dataLabels: { enabled: false },
    xaxis: {
      categories: features.map(f => f.feature),
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    yaxis: { labels: { style: { colors: '#94a3b8', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-feature-importance");
  if (el) {
    el.innerHTML = '';
    state.charts.featureImportance = new ApexCharts(el, options);
    state.charts.featureImportance.render();
  }
}

function renderResidualsChart(residuals) {
  if (typeof ApexCharts === 'undefined') return;

  const minVal = -20;
  const maxVal = 20;
  const binWidth = 2;
  const bins = {};
  for (let b = minVal; b <= maxVal; b += binWidth) {
    bins[b] = 0;
  }
  residuals.forEach(r => {
    const rounded = Math.round(r / binWidth) * binWidth;
    if (rounded >= minVal && rounded <= maxVal) {
      bins[rounded] = (bins[rounded] || 0) + 1;
    }
  });

  const categories = Object.keys(bins).map(k => `${k} err`);
  const counts = Object.values(bins);

  const options = {
    series: [{ name: 'Residual Frequency', data: counts }],
    chart: { type: 'bar', height: 340, background: 'transparent', toolbar: { show: false } },
    colors: ['#3b82f6'],
    plotOptions: { bar: { borderRadius: 4, columnWidth: '70%' } },
    dataLabels: { enabled: false },
    xaxis: {
      categories: categories,
      labels: { style: { colors: '#64748b', fontSize: '10px' } }
    },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-residuals");
  if (el) {
    el.innerHTML = '';
    state.charts.residuals = new ApexCharts(el, options);
    state.charts.residuals.render();
  }
}

// -----------------------------------------------------------------------------
// LIVE SQL QUERY STUDIO
// -----------------------------------------------------------------------------
const sampleSQLQueries = {
  turnover: `SELECT 
    p.product_id,
    p.product_name,
    p.category,
    ROUND(SUM(s.sales_quantity * p.unit_cost), 2) AS total_cogs,
    ROUND(AVG(i.ending_inventory * p.unit_cost), 2) AS avg_inv_val,
    ROUND(SUM(s.sales_quantity * p.unit_cost) / NULLIF(AVG(i.ending_inventory * p.unit_cost), 0), 2) AS turnover_ratio
FROM products p
JOIN sales s ON p.product_id = s.product_id
JOIN inventory i ON p.product_id = i.product_id
GROUP BY p.product_id
ORDER BY turnover_ratio DESC
LIMIT 10;`,
  stockout: `SELECT 
    p.product_id,
    p.product_name,
    SUM(i.stockout_flag) AS stockout_days,
    COUNT(i.inv_id) AS total_days,
    ROUND(100.0 * SUM(i.stockout_flag)/COUNT(i.inv_id), 2) AS stockout_pct
FROM products p
JOIN inventory i ON p.product_id = i.product_id
GROUP BY p.product_id
HAVING stockout_days > 0
ORDER BY stockout_days DESC
LIMIT 10;`,
  top10: `SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.sales_quantity) AS units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY p.product_id
ORDER BY total_revenue DESC
LIMIT 10;`,
  regions: `SELECT 
    r.region_name,
    r.warehouse_id,
    SUM(s.sales_quantity) AS total_units,
    ROUND(SUM(s.revenue), 2) AS regional_revenue
FROM regions r
JOIN sales s ON r.region_id = s.region_id
GROUP BY r.region_id
ORDER BY regional_revenue DESC;`
};

function loadSampleSQL(queryKey) {
  const editor = document.getElementById('sql-query-editor');
  if (editor && sampleSQLQueries[queryKey]) {
    editor.value = sampleSQLQueries[queryKey];
    runCustomSQL();
  }
}

async function runCustomSQL() {
  const query = document.getElementById('sql-query-editor').value;
  const metricsEl = document.getElementById('sql-execution-metrics');
  const exportBtn = document.getElementById('btn-export-sql');
  const t0 = performance.now();

  try {
    const res = await fetch('/api/sql/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const result = await res.json();
    const elapsed = (performance.now() - t0).toFixed(1);

    if (res.status !== 200) {
      metricsEl.innerHTML = `<span style="color: #f43f5e; font-weight: 700;">${result.detail || 'Execution error'}</span>`;
      return;
    }

    metricsEl.innerHTML = `<span style="color: #34d399; font-weight: 700;">✓ Query executed in ${elapsed} ms. Returned ${result.total_rows_returned} rows.</span>`;
    state.lastSQLResults = result;

    if (exportBtn) exportBtn.classList.remove('hidden');
    renderSQLResultsTable(result);

  } catch (err) {
    metricsEl.innerHTML = `<span style="color: #f43f5e; font-weight: 700;">Error: ${err.message}</span>`;
  }
}

function renderSQLResultsTable(result) {
  const thead = document.querySelector('#table-sql-results thead');
  const tbody = document.querySelector('#table-sql-results tbody');

  thead.innerHTML = `<tr>${result.columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
  tbody.innerHTML = result.rows.map(row => `
    <tr>
      ${result.columns.map(c => `<td style="font-family: monospace; font-size: 0.75rem;">${row[c] !== null ? row[c] : '<span style="color: #475569;">NULL</span>'}</td>`).join('')}
    </tr>
  `).join('');
}

function downloadSQLResultsAsCSV() {
  if (!state.lastSQLResults) return;
  const { columns, rows } = state.lastSQLResults;

  let csvContent = "data:text/csv;charset=utf-8," + columns.join(",") + "\n";
  rows.forEach(r => {
    csvContent += columns.map(c => `"${r[c] !== null ? r[c] : ''}"`).join(",") + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "sql_query_result.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
