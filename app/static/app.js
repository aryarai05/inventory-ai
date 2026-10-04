/**
 * Optix Enterprise Inventory AI - Core Application Logic
 * Manages reactive UI state, REST API calls, ApexCharts rendering, and live PuLP optimization.
 */

// Global state cache
let state = {
  activeTab: 'executive',
  products: [],
  risks: [],
  currentRiskFilter: 'All',
  executiveCharts: {},
  forecastChart: null,
  pulpChart: null,
  featureChart: null,
  residualChart: null,
  lastSQLResults: null
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
  lucide.createIcons();
  await loadSystemStatus();
  await loadExecutiveDashboard();
  await loadProductCatalog();
  await loadRiskReport();
  await loadMLOpsLeaderboard();
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
  lucide.createIcons();

  // Lazy trigger renders for charts in hidden tabs
  if (tabId === 'forecasting' && !state.forecastChart) {
    loadSKUForecast();
  } else if (tabId === 'optimization' && !state.pulpChart) {
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
    console.error('Failed to load status:', err);
  }
}

// -----------------------------------------------------------------------------
// EXECUTIVE DASHBOARD
// -----------------------------------------------------------------------------
async function loadExecutiveDashboard() {
  try {
    // 1. Fetch KPIs
    const kpiRes = await fetch('/api/kpis');
    const kpi = await kpiRes.json();

    document.getElementById('kpi-revenue').innerText = `$${(kpi.total_revenue / 1e6).toFixed(2)}M`;
    document.getElementById('kpi-units-sold').innerHTML = `<i data-lucide="package" class="w-3 h-3"></i> ${kpi.total_units_sold.toLocaleString()} units fulfilled`;
    document.getElementById('kpi-inv-value').innerText = `$${(kpi.total_inventory_value / 1e3).toFixed(1)}K`;
    document.getElementById('kpi-inv-units').innerText = `${kpi.current_inventory_units.toLocaleString()} units on-hand`;
    document.getElementById('kpi-stockout-rate').innerText = `${kpi.stockout_rate_pct.toFixed(2)}%`;
    document.getElementById('kpi-accuracy').innerText = `${kpi.forecast_accuracy_pct.toFixed(1)}%`;
    document.getElementById('kpi-critical-count').innerText = `${kpi.critical_stockouts_count} SKUs`;
    document.getElementById('kpi-reorder-count').innerText = `${kpi.reorders_needed_count} SKUs`;
    lucide.createIcons();

    // 2. Fetch Executive Charts Data
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
  // Subsample dates for smoother UI rendering (every 4th day)
  const step = 4;
  const sampledDates = trendData.dates.filter((_, i) => i % step === 0);
  const sampledSales = trendData.sales_quantity.filter((_, i) => i % step === 0);
  const sampledMA = trendData.ma_30.filter((_, i) => i % step === 0);

  const options = {
    series: [
      { name: 'Daily Units Demanded', type: 'area', data: sampledSales },
      { name: '30-Day Moving Average', type: 'line', data: sampledMA }
    ],
    chart: {
      height: 340,
      type: 'line',
      background: 'transparent',
      toolbar: { show: false },
      zoom: { enabled: false }
    },
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
    yaxis: {
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { labels: { colors: '#94a3b8' }, position: 'top', horizontalAlign: 'right' },
    tooltip: { theme: 'dark' }
  };

  const chartEl = document.querySelector("#chart-executive-trend");
  if (chartEl) {
    chartEl.innerHTML = '';
    const chart = new ApexCharts(chartEl, options);
    chart.render();
  }
}

function renderCategoryShareChart(catData) {
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

  const chartEl = document.querySelector("#chart-category-share");
  if (chartEl) {
    chartEl.innerHTML = '';
    const chart = new ApexCharts(chartEl, options);
    chart.render();
  }
}

function renderRegionalDemandChart(regData) {
  const options = {
    series: [{ name: 'Units Sold', data: regData.map(r => r.units_sold) }],
    chart: { type: 'bar', height: 280, background: 'transparent', toolbar: { show: false } },
    colors: ['#6366f1'],
    plotOptions: {
      bar: { borderRadius: 6, horizontal: true, distributed: false, barHeight: '55%' }
    },
    dataLabels: { enabled: false },
    xaxis: {
      categories: regData.map(r => r.region_name),
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    yaxis: {
      labels: { style: { colors: '#94a3b8', fontSize: '11px' } }
    },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    tooltip: { theme: 'dark' }
  };

  const chartEl = document.querySelector("#chart-regional-demand");
  if (chartEl) {
    chartEl.innerHTML = '';
    const chart = new ApexCharts(chartEl, options);
    chart.render();
  }
}

// -----------------------------------------------------------------------------
// PRODUCT CATALOG & DROPDOWNS
// -----------------------------------------------------------------------------
async function loadProductCatalog() {
  try {
    const res = await fetch('/api/products');
    state.products = await res.json();

    // Populate Category Filter
    const categories = ['All', ...new Set(state.products.map(p => p.category))];
    const catSelect = document.getElementById('forecast-cat-filter');
    if (catSelect) {
      catSelect.innerHTML = categories.map(c => `<option value="${c}">${c === 'All' ? 'All Categories' : c}</option>`).join('');
    }

    // Populate Product Dropdowns
    populateProductDropdowns('All');

    // Populate Catalog Table in SKUs Tab
    renderCatalogTable(state.products);

  } catch (err) {
    console.error('Catalog load error:', err);
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
      <td class="font-mono text-xs font-bold text-indigo-400">${p.product_id}</td>
      <td class="font-semibold text-white">${p.product_name}</td>
      <td><span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">${p.category}</span></td>
      <td class="font-mono text-slate-300">$${p.unit_cost.toFixed(2)}</td>
      <td class="font-mono text-slate-200 font-semibold">$${p.unit_price.toFixed(2)}</td>
      <td class="text-emerald-400 font-semibold">$${p.unit_margin.toFixed(2)}</td>
      <td>${p.lead_time_days} days</td>
      <td class="text-slate-300">${p.supplier_name || 'Tier-1 Partner'}</td>
      <td>
        <span class="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          ${((p.reliability_score || 0.95) * 100).toFixed(0)}%
        </span>
      </td>
      <td>
        <button onclick="inspectSKU('${p.product_id}')" class="px-2.5 py-1 text-xs bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 rounded hover:bg-indigo-600/30 transition">
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
  if (skuSelect) {
    skuSelect.value = pid;
  }
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

    // Set KPI header metrics
    document.getElementById('f-sku-id').innerText = data.product_id;
    document.getElementById('f-sku-name').innerText = data.product_name;
    document.getElementById('f-sku-price').innerText = `$${data.unit_price.toFixed(2)} / $${data.unit_cost.toFixed(2)}`;
    const margin = (((data.unit_price - data.unit_cost) / data.unit_price) * 100).toFixed(1);
    document.getElementById('f-sku-margin').innerText = `${margin}% gross margin`;
    document.getElementById('f-sku-leadtime').innerText = `${data.lead_time_days} Days`;
    document.getElementById('f-sku-runrate').innerText = `${data.mean_daily_demand.toFixed(1)} units/day`;
    document.getElementById('f-sku-total').innerText = `${Math.round(data.total_forecasted_units)} units`;
    document.getElementById('f-sku-horizon-desc').innerText = `Across ${data.horizon_days} planning days`;

    // Render Canvas Forecast Chart
    renderForecastChart(data);

    // Populate Daily Table
    renderForecastScheduleTable(data.forecast);

  } catch (err) {
    console.error('Forecast load error:', err);
  }
}

function renderForecastChart(data) {
  // Historical points (last 45 days for clear visual focus)
  const histSlice = data.historical.slice(-45);
  const histDates = histSlice.map(h => h.date);
  const histSales = histSlice.map(h => h.actual_sales);

  // Forecast points
  const fDates = data.forecast.map(f => f.date);
  const fPreds = data.forecast.map(f => f.predicted_demand);
  const fLower = data.forecast.map(f => f.lower_bound);
  const fUpper = data.forecast.map(f => f.upper_bound);

  const allDates = [...histDates, ...fDates];
  const histSeriesData = [...histSales, ...Array(fDates.length).fill(null)];
  
  // Bridge the gap between historical and forecast
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
    yaxis: {
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { show: false },
    tooltip: { theme: 'dark' }
  };

  const chartEl = document.querySelector("#chart-sku-forecast");
  if (chartEl) {
    chartEl.innerHTML = '';
    state.forecastChart = new ApexCharts(chartEl, options);
    state.forecastChart.render();
  }
}

function renderForecastScheduleTable(forecast) {
  const tbody = document.querySelector('#table-forecast-schedule tbody');
  if (!tbody) return;

  tbody.innerHTML = forecast.map(f => {
    const spread = (f.upper_bound - f.lower_bound).toFixed(1);
    return `
      <tr>
        <td class="font-mono text-xs text-slate-300">${f.date}</td>
        <td class="font-semibold text-indigo-400">Day +${f.forecast_step}</td>
        <td class="font-bold text-white">${f.predicted_demand.toFixed(1)} units</td>
        <td class="text-slate-400">${f.lower_bound.toFixed(1)}</td>
        <td class="text-slate-400">${f.upper_bound.toFixed(1)}</td>
        <td><span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">±${(spread / 2).toFixed(1)}</span></td>
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
    console.error('Risk report load error:', err);
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

  tbody.innerHTML = risks.map(r => `
    <tr>
      <td class="font-mono text-xs font-bold text-indigo-400">${r.product_id}</td>
      <td class="font-semibold text-white whitespace-nowrap">${r.product_name}</td>
      <td><span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">${r.category}</span></td>
      <td class="font-mono font-bold ${r.current_inventory <= r.safety_stock ? 'text-rose-400' : 'text-slate-200'}">${r.current_inventory}</td>
      <td class="text-slate-300">${r.mean_daily_demand}</td>
      <td class="text-slate-400">${r.safety_stock}</td>
      <td class="text-slate-400">${r.reorder_point}</td>
      <td class="font-semibold ${r.days_of_supply < r.lead_time_days ? 'text-rose-400' : 'text-emerald-400'}">${r.days_of_supply.toFixed(1)}d</td>
      <td><span class="${badgeMap[r.risk_level] || 'badge-healthy'}">${r.risk_level}</span></td>
      <td class="font-bold ${r.recommended_reorder_qty > 0 ? 'text-amber-400' : 'text-slate-500'}">
        ${r.recommended_reorder_qty > 0 ? r.recommended_reorder_qty + ' units' : '—'}
      </td>
      <td class="text-xs text-slate-400 max-w-[340px] truncate" title="${r.diagnostic_reason}">${r.diagnostic_reason}</td>
    </tr>
  `).join('');
}

// -----------------------------------------------------------------------------
// PuLP MATHEMATICAL REPLENISHMENT OPTIMIZATION
// -----------------------------------------------------------------------------
async function executePuLPOptimization() {
  const skuSelect = document.getElementById('opt-sku-select');
  if (!skuSelect || !skuSelect.value) return;

  const btn = document.getElementById('btn-run-pulp');
  if (btn) btn.innerHTML = '<span class="inline-block animate-spin mr-1">⏳</span> Solving MILP...';

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

    // Update KPI cards
    document.getElementById('opt-status').innerText = result.status;
    document.getElementById('opt-total-cost').innerText = `$${result.total_optimal_cost.toLocaleString()}`;
    document.getElementById('opt-holding-cost').innerText = `$${result.total_holding_cost.toLocaleString()}`;
    document.getElementById('opt-ordering-cost').innerText = `$${result.total_ordering_cost.toLocaleString()}`;
    document.getElementById('opt-po-count').innerText = `${result.total_orders_placed} POs dispatched`;
    document.getElementById('opt-stockout-penalty').innerText = `$${result.total_stockout_penalty.toLocaleString()}`;

    // Render Trajectory Chart
    renderPuLPTrajectoryChart(result.daily_plan);

    // Populate Daily Schedule Table
    renderPuLPDailyTable(result.daily_plan);

  } catch (err) {
    console.error('Optimization error:', err);
  } finally {
    if (btn) btn.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i> Run PuLP MILP Solver';
    lucide.createIcons();
  }
}

function renderPuLPTrajectoryChart(dailyPlan) {
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
    yaxis: {
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { labels: { colors: '#94a3b8' }, position: 'top', horizontalAlign: 'right' },
    tooltip: { theme: 'dark' }
  };

  const chartEl = document.querySelector("#chart-pulp-trajectory");
  if (chartEl) {
    chartEl.innerHTML = '';
    state.pulpChart = new ApexCharts(chartEl, options);
    state.pulpChart.render();
  }
}

function renderPuLPDailyTable(dailyPlan) {
  const tbody = document.querySelector('#table-opt-daily tbody');
  if (!tbody) return;

  tbody.innerHTML = dailyPlan.map(d => `
    <tr>
      <td class="font-bold text-indigo-400">Step ${d.day_step}</td>
      <td class="text-slate-300">${d.forecast_demand} units</td>
      <td class="font-bold ${d.recommended_order > 0 ? 'text-emerald-400' : 'text-slate-500'}">
        ${d.recommended_order > 0 ? d.recommended_order + ' units' : '—'}
      </td>
      <td>
        <span class="px-2 py-0.5 rounded text-xs font-semibold ${d.order_triggered === 1 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-500'}">
          ${d.order_triggered === 1 ? 'TRIGGERED (Y=1)' : 'IDLE (Y=0)'}
        </span>
      </td>
      <td class="font-mono text-slate-200">${d.expected_inventory}</td>
      <td class="text-rose-400">${d.stockout_units > 0 ? d.stockout_units : '0'}</td>
      <td class="text-slate-400 font-mono">$${d.holding_cost.toFixed(2)}</td>
      <td class="text-slate-400 font-mono">$${d.ordering_cost.toFixed(2)}</td>
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

    // 1. Leaderboard Table
    const tbody = document.querySelector('#table-mlops-leaderboard tbody');
    if (tbody && data.leaderboard) {
      tbody.innerHTML = data.leaderboard.map((m, idx) => `
        <tr class="${idx === 0 ? 'bg-emerald-500/5' : ''}">
          <td class="font-bold ${idx === 0 ? 'text-emerald-400' : 'text-white'}">${m.Model}</td>
          <td class="font-mono text-slate-200">${m.MAE.toFixed(3)}</td>
          <td class="font-mono text-slate-200">${m.RMSE.toFixed(3)}</td>
          <td class="font-mono text-slate-400">${m.MAPE > 9999 ? '>10,000%' : m.MAPE.toFixed(1) + '%'}</td>
          <td class="font-mono font-bold ${idx === 0 ? 'text-emerald-400' : 'text-slate-200'}">${m.WAPE.toFixed(2)}%</td>
          <td class="font-mono ${m.Bias_Pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${m.Bias_Pct > 0 ? '+' : ''}${m.Bias_Pct.toFixed(2)}%</td>
          <td class="text-slate-400">${m.Train_Time_Sec.toFixed(3)}s</td>
          <td>
            <span class="px-2 py-0.5 rounded-full text-xs font-semibold ${idx === 0 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-500'}">
              ${idx === 0 ? '★ Production Champion' : 'Benchmark Candidate'}
            </span>
          </td>
        </tr>
      `).join('');
    }

    // 2. Feature Importance Chart
    if (data.top_features && data.top_features.length > 0) {
      renderFeatureImportanceChart(data.top_features);
    }

    // 3. Residual Error Histogram
    if (data.sample_residuals && data.sample_residuals.length > 0) {
      renderResidualsChart(data.sample_residuals);
    }

  } catch (err) {
    console.error('MLOps leaderboard load error:', err);
  }
}

function renderFeatureImportanceChart(features) {
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
    yaxis: {
      labels: { style: { colors: '#94a3b8', fontSize: '11px' } }
    },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    tooltip: { theme: 'dark' }
  };

  const chartEl = document.querySelector("#chart-feature-importance");
  if (chartEl) {
    chartEl.innerHTML = '';
    state.featureChart = new ApexCharts(chartEl, options);
    state.featureChart.render();
  }
}

function renderResidualsChart(residuals) {
  // Compute histogram bins manually for ApexCharts
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
    yaxis: {
      labels: { style: { colors: '#64748b', fontSize: '11px' } }
    },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    tooltip: { theme: 'dark' }
  };

  const chartEl = document.querySelector("#chart-residuals");
  if (chartEl) {
    chartEl.innerHTML = '';
    state.residualChart = new ApexCharts(chartEl, options);
    state.residualChart.render();
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
      metricsEl.innerHTML = `<span class="text-rose-400 font-semibold">${result.detail || 'Execution error'}</span>`;
      return;
    }

    metricsEl.innerHTML = `<span class="text-emerald-400 font-semibold">✓ Query executed in ${elapsed} ms. Returned ${result.total_rows_returned} rows.</span>`;
    state.lastSQLResults = result;

    if (exportBtn) exportBtn.classList.remove('hidden');

    // Render Table
    renderSQLResultsTable(result);

  } catch (err) {
    metricsEl.innerHTML = `<span class="text-rose-400 font-semibold">Error: ${err.message}</span>`;
  }
}

function renderSQLResultsTable(result) {
  const thead = document.querySelector('#table-sql-results thead');
  const tbody = document.querySelector('#table-sql-results tbody');

  thead.innerHTML = `<tr>${result.columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
  tbody.innerHTML = result.rows.map(row => `
    <tr>
      ${result.columns.map(c => `<td class="font-mono text-xs">${row[c] !== null ? row[c] : '<span class="text-slate-600">NULL</span>'}</td>`).join('')}
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
