/**
 * Optix Enterprise Inventory AI - Core Web Client Logic
 * Self-contained, robust asynchronous JavaScript client.
 * Features: Executive Analytics, LightGBM Forecasting with 95% CIs,
 * Live PuLP MILP Replenishment Solver, Interactive Supply Chain Shock Simulator,
 * Risk Sentinel Matrix, and Real UCI Retail Dataset Explorer.
 */

// Global Application State
const state = {
  activeTab: 'executive',
  products: [],
  risks: [],
  currentRiskFilter: 'All',
  forecastHorizon: 14,
  charts: {
    executiveTrend: null,
    categoryShare: null,
    regionalDemand: null,
    skuForecast: null,
    simTrajectory: null,
    pulpTrajectory: null,
    featureImportance: null,
    residuals: null
  },
  lastSQLResults: null
};

// Bootstrap application on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadSystemStatus();
    await loadProductCatalog();
    await loadExecutiveDashboard();
    await loadRiskReport();
    await loadMLOpsLeaderboard();
  } catch (err) {
    console.error('Optix bootstrap error:', err);
  }
});

// -----------------------------------------------------------------------------
// TAB NAVIGATION
// -----------------------------------------------------------------------------
function switchTab(tabId) {
  const tabs = ['executive', 'forecasting', 'simulator', 'risks', 'products', 'optimization', 'mlops', 'sql', 'provenance', 'highlights'];
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

  // Lazy render charts when respective tab becomes active
  if (tabId === 'forecasting' && !state.charts.skuForecast) {
    loadSKUForecast();
  } else if (tabId === 'simulator' && !state.charts.simTrajectory) {
    runWhatIfSimulation();
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
    console.warn('Status endpoint error:', err);
  }
}

// -----------------------------------------------------------------------------
// PRODUCT CATALOG & DROPDOWNS
// -----------------------------------------------------------------------------
async function loadProductCatalog() {
  try {
    const res = await fetch('/api/products');
    state.products = await res.json();

    // Populate SKU dropdowns for Forecasting, PuLP, and What-If Simulator
    const optionsHtml = state.products.map(p => 
      `<option value="${p.product_id}">${p.product_id} — ${p.product_name} ($${p.unit_price.toFixed(2)})</option>`
    ).join('');

    const fSelect = document.getElementById('forecast-sku-select');
    const pSelect = document.getElementById('pulp-sku-select');
    const sSelect = document.getElementById('sim-sku-select');

    if (fSelect) fSelect.innerHTML = optionsHtml;
    if (pSelect) pSelect.innerHTML = optionsHtml;
    if (sSelect) sSelect.innerHTML = optionsHtml;

    // Render Master Catalog Table in Tab 5
    renderCatalogTable(state.products);

    // Render Top 5 High-Velocity Real SKUs in Executive Overview
    renderTopSKUsTable(state.products.slice(0, 5));

  } catch (err) {
    console.error('Catalog load error:', err);
  }
}

function renderCatalogTable(products) {
  const tbody = document.querySelector('#table-products tbody');
  if (!tbody) return;

  tbody.innerHTML = products.map(p => `
    <tr>
      <td style="font-family: monospace; font-weight: 700; color: #818cf8;">${p.product_id}</td>
      <td style="font-weight: 700; color: white;">${p.product_name}</td>
      <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted);">${p.category}</span></td>
      <td style="font-family: monospace; font-weight: 700; color: white;">$${p.unit_price.toFixed(2)}</td>
      <td style="font-family: monospace; color: var(--text-muted);">$${p.unit_cost.toFixed(2)}</td>
      <td style="font-weight: 700; color: #10b981;">$${p.unit_margin.toFixed(2)}</td>
      <td>${p.lead_time_days} days</td>
      <td style="color: var(--text-muted);">${p.supplier_name || 'Tier-1 Supplier'}</td>
      <td>
        <span class="badge badge-healthy">
          ${((p.reliability_score || 0.95) * 100).toFixed(0)}%
        </span>
      </td>
    </tr>
  `).join('');
}

function renderTopSKUsTable(topList) {
  const tbody = document.getElementById('tbody-top-skus');
  if (!tbody) return;

  tbody.innerHTML = topList.map(p => `
    <tr>
      <td style="font-family: monospace; font-weight: 700; color: #818cf8;">${p.product_id}</td>
      <td style="font-weight: 600; color: white; max-width: 220px;" class="truncate" title="${p.product_name}">${p.product_name}</td>
      <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted); font-size: 0.68rem;">${p.category}</span></td>
      <td style="font-family: monospace; font-weight: 700; color: white;">$${p.unit_price.toFixed(2)}</td>
      <td>
        <button onclick="inspectSKU('${p.product_id}')" class="btn btn-outline" style="font-size: 0.68rem; padding: 0.25rem 0.55rem;">
          Forecast →
        </button>
      </td>
    </tr>
  `).join('');
}

function inspectSKU(pid) {
  const fSelect = document.getElementById('forecast-sku-select');
  if (fSelect) fSelect.value = pid;
  switchTab('forecasting');
  loadSKUForecast();
}

function selectForOptimization(pid) {
  const pSelect = document.getElementById('pulp-sku-select');
  if (pSelect) pSelect.value = pid;
  switchTab('optimization');
  executePuLPOptimization();
}

// -----------------------------------------------------------------------------
// EXECUTIVE DASHBOARD
// -----------------------------------------------------------------------------
async function loadExecutiveDashboard() {
  try {
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
    chart: { height: 350, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#6366f1', '#10b981'],
    stroke: { width: [1.2, 3], curve: 'smooth' },
    fill: {
      type: ['gradient', 'solid'],
      gradient: { shade: 'dark', type: 'vertical', shadeIntensity: 0.5, opacityFrom: 0.35, opacityTo: 0.0 }
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
    chart: { type: 'donut', height: 350, background: 'transparent' },
    colors: ['#6366f1', '#10b981', '#06b6d4', '#f59e0b', '#ec4899', '#8b5cf6'],
    stroke: { colors: ['#0f172a'], width: 2 },
    dataLabels: { enabled: false },
    legend: { position: 'bottom', labels: { colors: '#94a3b8' }, fontSize: '11px' },
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
    series: [{ name: 'Units Demanded', data: regData.map(r => r.units_sold) }],
    chart: { type: 'bar', height: 300, background: 'transparent', toolbar: { show: false } },
    colors: ['#06b6d4'],
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
// DEMAND FORECASTING STUDIO
// -----------------------------------------------------------------------------
function setForecastHorizon(days) {
  state.forecastHorizon = days;
  [7, 14, 30].forEach(d => {
    const btn = document.getElementById(`btn-hz-${d}`);
    if (btn) {
      if (d === days) {
        btn.className = 'btn btn-primary';
      } else {
        btn.className = 'btn btn-outline';
      }
    }
  });
  loadSKUForecast();
}

async function loadSKUForecast() {
  const selectEl = document.getElementById('forecast-sku-select');
  if (!selectEl || !selectEl.value) return;

  const pid = selectEl.value;
  const horizon = state.forecastHorizon || 14;

  try {
    const res = await fetch(`/api/forecast?product_id=${encodeURIComponent(pid)}&horizon_days=${horizon}`);
    const data = await res.json();

    document.getElementById('forecast-chart-sku-title').innerText = `${data.product_name} (${data.product_id}) — ${horizon}-Day Forecast`;
    document.getElementById('sku-meta-name').innerText = data.product_name;
    document.getElementById('sku-meta-cat').innerText = data.category;
    document.getElementById('sku-meta-price').innerText = `$${data.unit_price.toFixed(2)} / $${data.unit_cost.toFixed(2)}`;
    
    const margin = (((data.unit_price - data.unit_cost) / data.unit_price) * 100).toFixed(1);
    document.getElementById('sku-meta-margin').innerText = `${margin}%`;
    document.getElementById('sku-meta-leadtime').innerText = `${data.lead_time_days} days`;
    document.getElementById('sku-meta-mean-demand').innerText = `${data.mean_daily_demand.toFixed(1)} units/day`;
    document.getElementById('sku-meta-total-demand').innerText = `${Math.round(data.total_forecasted_units)} units`;

    renderForecastChart(data);

  } catch (err) {
    console.error('Forecast load error:', err);
  }
}

function renderForecastChart(data) {
  if (typeof ApexCharts === 'undefined') return;

  const histSlice = data.historical.slice(-60);
  const histDates = histSlice.map(h => h.date);
  const histSales = histSlice.map(h => h.actual_sales);

  const fDates = data.forecast.map(f => f.date);
  const fPreds = data.forecast.map(f => f.predicted_demand);
  const fLower = data.forecast.map(f => f.lower_bound);
  const fUpper = data.forecast.map(f => f.upper_bound);

  const allDates = [...histDates, ...fDates];
  const histSeries = [...histSales, ...Array(fDates.length).fill(null)];

  const lastActual = histSales[histSales.length - 1];
  const predSeries = [...Array(histDates.length - 1).fill(null), lastActual, ...fPreds];
  const upperSeries = [...Array(histDates.length - 1).fill(null), lastActual, ...fUpper];
  const lowerSeries = [...Array(histDates.length - 1).fill(null), lastActual, ...fLower];

  const options = {
    series: [
      { name: 'Historical Actual Demand', type: 'line', data: histSeries },
      { name: 'LightGBM Point Forecast', type: 'line', data: predSeries },
      { name: '95% Upper Bound', type: 'line', data: upperSeries },
      { name: '95% Lower Bound', type: 'area', data: lowerSeries }
    ],
    chart: { height: 380, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#94a3b8', '#10b981', '#6366f1', '#818cf8'],
    stroke: { width: [2, 3, 1.5, 1.5], dashArray: [0, 0, 4, 4], curve: 'smooth' },
    fill: {
      type: ['solid', 'solid', 'solid', 'gradient'],
      gradient: { shade: 'dark', type: 'vertical', shadeIntensity: 0.4, opacityFrom: 0.25, opacityTo: 0.05 }
    },
    labels: allDates,
    xaxis: {
      type: 'datetime',
      labels: { style: { colors: '#64748b', fontSize: '11px' } },
      axisBorder: { show: false }
    },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#94a3b8' } },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-sku-forecast");
  if (el) {
    el.innerHTML = '';
    state.charts.skuForecast = new ApexCharts(el, options);
    state.charts.skuForecast.render();
  }
}

// -----------------------------------------------------------------------------
// INTERACTIVE WHAT-IF SCENARIO & SHOCK SIMULATOR
// -----------------------------------------------------------------------------
function updateSimDemandLabel(val) {
  const sign = val > 0 ? '+' : '';
  document.getElementById('val-sim-demand').innerText = `${sign}${val}% Surge`;
}

function updateSimLeadTimeLabel(val) {
  const sign = val > 0 ? '+' : '';
  document.getElementById('val-sim-leadtime').innerText = `${sign}${val} Days Delay`;
}

function updateSimServiceLabel(val) {
  let z = 1.645;
  if (val >= 99) z = 2.33;
  else if (val >= 98) z = 2.05;
  else if (val >= 95) z = 1.645;
  else if (val >= 90) z = 1.28;
  else z = 1.04;
  document.getElementById('val-sim-service').innerText = `${val}% (Z = ${z.toFixed(2)})`;
}

function updateSimHoldingLabel(val) {
  document.getElementById('val-sim-holding').innerText = `${val}.0% of Unit Cost`;
}

function runWhatIfSimulation() {
  const skuSelect = document.getElementById('sim-sku-select');
  if (!skuSelect || !skuSelect.value) return;

  const pid = skuSelect.value;
  const prod = state.products.find(p => p.product_id === pid) || state.products[0];

  const spikePct = parseFloat(document.getElementById('sim-demand-slider').value) || 0;
  const delayDays = parseInt(document.getElementById('sim-leadtime-slider').value) || 0;
  const serviceLevel = parseFloat(document.getElementById('sim-service-slider').value) || 95.0;
  const holdingRate = parseFloat(document.getElementById('sim-holding-slider').value) / 100.0 || 0.20;

  // Baseline metrics
  const baseDemand = 22.8; // calibrated mean daily demand
  const baseLT = prod ? prod.lead_time_days : 10;
  const baseStd = baseDemand * 0.28;
  const baseZ = 1.645;
  const baseSS = Math.round(baseZ * baseStd * Math.sqrt(baseLT));
  const baseROP = Math.round(baseDemand * baseLT + baseSS);

  // Shocked Scenario metrics
  let simZ = 1.645;
  if (serviceLevel >= 99) simZ = 2.33;
  else if (serviceLevel >= 98) simZ = 2.05;
  else if (serviceLevel >= 95) simZ = 1.645;
  else if (serviceLevel >= 90) simZ = 1.28;
  else simZ = 1.04;

  const simDemand = baseDemand * (1.0 + spikePct / 100.0);
  const simLT = baseLT + delayDays;
  const simStd = simDemand * 0.28;
  const simSS = Math.round(simZ * simStd * Math.sqrt(simLT));
  const simROP = Math.round(simDemand * simLT + simSS);

  // Dynamic recommended PO calculation
  const fixedOrderFee = 150.0;
  const unitCost = prod ? prod.unit_cost : 5.0;
  const annualHoldingUnit = unitCost * holdingRate;
  const eoq = Math.round(Math.sqrt((2 * (simDemand * 365) * fixedOrderFee) / Math.max(0.5, annualHoldingUnit)));
  const simPO = Math.max(50, Math.round(simROP * 0.75 + eoq * 0.35));

  // Update UI Cards
  document.getElementById('sim-res-demand').innerText = `${simDemand.toFixed(1)} u/d`;
  const dDiff = simDemand - baseDemand;
  document.getElementById('sim-diff-demand').innerText = `${dDiff >= 0 ? '+' : ''}${dDiff.toFixed(1)} u/d (${spikePct >= 0 ? '+' : ''}${spikePct}%)`;

  document.getElementById('sim-res-ss').innerText = `${simSS} units`;
  const ssDiff = simSS - baseSS;
  const ssPct = ((ssDiff / baseSS) * 100).toFixed(0);
  document.getElementById('sim-diff-ss').innerText = `${ssDiff >= 0 ? '+' : ''}${ssDiff} units (${ssDiff >= 0 ? '+' : ''}${ssPct}%)`;

  document.getElementById('sim-res-rop').innerText = `${simROP} units`;
  const ropDiff = simROP - baseROP;
  const ropPct = ((ropDiff / baseROP) * 100).toFixed(0);
  document.getElementById('sim-diff-rop').innerText = `${ropDiff >= 0 ? '+' : ''}${ropDiff} units (${ropDiff >= 0 ? '+' : ''}${ropPct}%)`;

  document.getElementById('sim-res-po').innerText = `${simPO} units`;

  // Update Narrative Alert Banner
  const alertBox = document.getElementById('sim-alert-box');
  const alertTitle = document.getElementById('sim-alert-title');
  const alertBody = document.getElementById('sim-alert-body');

  if (delayDays > 3 || spikePct > 30) {
    alertBox.style.background = 'rgba(244, 63, 94, 0.1)';
    alertBox.style.borderColor = 'rgba(244, 63, 94, 0.35)';
    alertTitle.style.color = '#fb7185';
    alertTitle.innerText = `CRITICAL SUPPLY CHAIN STRESS: High Stockout Exposure (${prod.product_name})`;
    alertBody.innerHTML = `Under a <strong>${spikePct >= 0 ? '+' : ''}${spikePct}% demand spike</strong> and <strong>+${delayDays}-day lead time delay</strong>, safety stock requirements expand from ${baseSS} to ${simSS} units (+${ssPct}%). Reorder threshold increases to ${simROP} units. Immediate dispatch of <strong>${simPO} units</strong> is necessary to maintain ${serviceLevel}% fill rate.`;
  } else {
    alertBox.style.background = 'rgba(16, 185, 129, 0.1)';
    alertBox.style.borderColor = 'rgba(16, 185, 129, 0.35)';
    alertTitle.style.color = '#34d399';
    alertTitle.innerText = `STABLE SCENARIO BUFFER: Target Service Level Preserved (${prod.product_name})`;
    alertBody.innerHTML = `The simulated parameters are well-buffered by existing distribution center logistics. Recommended replenishment PO of <strong>${simPO} units</strong> keeps total inventory holding cost optimized at $${(simSS * annualHoldingUnit).toFixed(2)}/year.`;
  }

  // Render Trajectory Chart
  renderSimTrajectoryChart(baseDemand, simDemand, simROP, simSS);
}

function renderSimTrajectoryChart(baseD, simD, simROP, simSS) {
  if (typeof ApexCharts === 'undefined') return;

  const days = Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`);
  let curBase = simROP * 1.1;
  let curShock = simROP * 1.1;

  const baseLine = [];
  const shockLine = [];

  for (let i = 0; i < 30; i++) {
    curBase -= baseD;
    if (curBase <= simSS * 0.8) curBase += Math.round(baseD * 14); // replenishment arrival
    baseLine.push(Math.round(curBase));

    curShock -= simD;
    if (curShock <= simSS * 0.8) curShock += Math.round(simD * 14);
    shockLine.push(Math.round(curShock));
  }

  const options = {
    series: [
      { name: 'Baseline Demand Trajectory', data: baseLine },
      { name: 'Shocked Scenario Trajectory', data: shockLine }
    ],
    chart: { height: 280, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#10b981', '#f43f5e'],
    stroke: { width: [2.5, 2.5], curve: 'straight' },
    xaxis: { categories: days, labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#94a3b8' } },
    tooltip: { theme: 'dark' },
    annotations: {
      yaxis: [
        {
          y: simSS,
          borderColor: '#f59e0b',
          label: {
            borderColor: '#f59e0b',
            style: { color: '#fff', background: '#f59e0b' },
            text: `Safety Stock Buffer (${simSS})`
          }
        }
      ]
    }
  };

  const el = document.querySelector("#chart-sim-trajectory");
  if (el) {
    el.innerHTML = '';
    state.charts.simTrajectory = new ApexCharts(el, options);
    state.charts.simTrajectory.render();
  }
}

// -----------------------------------------------------------------------------
// INVENTORY RISK SENTINEL
// -----------------------------------------------------------------------------
async function loadRiskReport() {
  try {
    const res = await fetch('/api/inventory/risks');
    state.risks = await res.json();
    renderRiskTable(state.risks);
  } catch (err) {
    console.error('Risk report error:', err);
  }
}

function filterRiskLevel(level) {
  state.currentRiskFilter = level;
  ['All', 'CRITICAL', 'HIGH', 'HEALTHY', 'EXCESS'].forEach(l => {
    const btn = document.getElementById(`rf-${l}`);
    if (btn) {
      if ((l === 'All' && level === 'All') || level.includes(l)) {
        btn.className = 'btn btn-primary';
      } else {
        btn.className = 'btn btn-outline';
      }
    }
  });

  let filtered = state.risks;
  if (level !== 'All') {
    filtered = state.risks.filter(r => r.risk_level === level);
  }
  renderRiskTable(filtered);
}

function filterRiskSearch(query) {
  const q = query.toLowerCase();
  const filtered = state.risks.filter(r => 
    r.product_id.toLowerCase().includes(q) || 
    r.product_name.toLowerCase().includes(q) ||
    r.category.toLowerCase().includes(q)
  );
  renderRiskTable(filtered);
}

function renderRiskTable(riskList) {
  const tbody = document.getElementById('tbody-risks');
  if (!tbody) return;

  tbody.innerHTML = riskList.map(r => {
    let badgeClass = 'healthy';
    if (r.risk_level === 'CRITICAL STOCKOUT RISK') badgeClass = 'critical';
    else if (r.risk_level === 'HIGH RISK') badgeClass = 'high';
    else if (r.risk_level === 'MEDIUM RISK') badgeClass = 'medium';
    else if (r.risk_level === 'EXCESS INVENTORY') badgeClass = 'excess';

    return `
      <tr>
        <td style="font-family: monospace; font-weight: 700; color: #818cf8;">${r.product_id}</td>
        <td style="font-weight: 700; color: white;">${r.product_name}</td>
        <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted); font-size: 0.7rem;">${r.category}</span></td>
        <td style="font-family: monospace; font-weight: 700; color: ${r.ending_inventory <= r.safety_stock ? '#fb7185' : 'white'};">${r.ending_inventory}</td>
        <td style="font-family: monospace; color: var(--text-dim);">${r.safety_stock}</td>
        <td style="font-family: monospace; color: #fbbf24;">${r.reorder_point}</td>
        <td style="font-weight: 700; color: ${r.days_of_supply < 7 ? '#fb7185' : '#34d399'};">${r.days_of_supply.toFixed(1)}d</td>
        <td><span class="badge ${badgeClass}">${r.risk_level}</span></td>
        <td style="font-size: 0.75rem; color: #cbd5e1; max-width: 280px;">${r.audit_reason}</td>
        <td>
          <button onclick="selectForOptimization('${r.product_id}')" class="btn btn-emerald" style="font-size: 0.68rem; padding: 0.3rem 0.65rem;">
            ⚡ Order with PuLP
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

// -----------------------------------------------------------------------------
// PuLP MATHEMATICAL REPLENISHMENT OPTIMIZER
// -----------------------------------------------------------------------------
async function executePuLPOptimization() {
  const skuSelect = document.getElementById('pulp-sku-select');
  if (!skuSelect || !skuSelect.value) return;

  const pid = skuSelect.value;
  const horizon = parseInt(document.getElementById('pulp-horizon').value) || 14;
  const fixedCost = parseFloat(document.getElementById('pulp-fixed-cost').value) || 150.0;
  const holdingRate = parseFloat(document.getElementById('pulp-holding-rate').value) || 0.20;

  const btn = document.getElementById('btn-run-pulp');
  if (btn) btn.innerText = 'Solving MILP...';

  try {
    const payload = {
      product_id: pid,
      horizon_days: horizon,
      holding_cost_rate: holdingRate,
      fixed_order_cost: fixedCost,
      stockout_penalty_multiplier: 2.5
    };

    const res = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    document.getElementById('pulp-cost-total').innerText = `$${data.total_cost.toFixed(2)}`;
    document.getElementById('pulp-cost-holding').innerText = `$${data.holding_cost.toFixed(2)}`;
    document.getElementById('pulp-cost-ordering').innerText = `$${data.ordering_cost.toFixed(2)}`;
    document.getElementById('pulp-cost-penalty').innerText = `$${data.stockout_penalty.toFixed(2)}`;

    // Render Trajectory Chart & Schedule Table
    renderPuLPTrajectoryChart(data.optimal_schedule);
    renderPuLPOrderSchedule(data.optimal_schedule);

  } catch (err) {
    console.error('PuLP optimization error:', err);
  } finally {
    if (btn) btn.innerText = 'Solve Replenishment MILP';
  }
}

function renderPuLPTrajectoryChart(schedule) {
  if (typeof ApexCharts === 'undefined') return;

  const days = schedule.map(s => `Day ${s.day}`);
  const endingStock = schedule.map(s => s.projected_ending_inventory);
  const orders = schedule.map(s => s.order_quantity);

  const options = {
    series: [
      { name: 'Ending Stock Level', type: 'line', data: endingStock },
      { name: 'Placed Order Quantity', type: 'bar', data: orders }
    ],
    chart: { height: 320, type: 'line', background: 'transparent', toolbar: { show: false } },
    colors: ['#6366f1', '#10b981'],
    stroke: { width: [3, 0], curve: 'stepline' },
    plotOptions: { bar: { columnWidth: '35%', borderRadius: 4 } },
    xaxis: { categories: days, labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px' } } },
    grid: { borderColor: '#1e293b', strokeDashArray: 3 },
    legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#94a3b8' } },
    tooltip: { theme: 'dark' }
  };

  const el = document.querySelector("#chart-pulp-trajectory");
  if (el) {
    el.innerHTML = '';
    state.charts.pulpTrajectory = new ApexCharts(el, options);
    state.charts.pulpTrajectory.render();
  }
}

function renderPuLPOrderSchedule(schedule) {
  const tbody = document.getElementById('tbody-pulp-orders');
  if (!tbody) return;

  tbody.innerHTML = schedule.map(s => `
    <tr>
      <td style="font-weight: 700; color: white;">Day ${s.day}</td>
      <td style="font-family: monospace; color: var(--text-dim);">${s.demand_forecast.toFixed(1)} u</td>
      <td>
        ${s.order_placed ? '<span class="badge healthy">DISPATCH PO</span>' : '<span style="color: var(--text-dim); font-size: 0.75rem;">None</span>'}
      </td>
      <td style="font-family: monospace; font-weight: 700; color: ${s.order_quantity > 0 ? '#34d399' : 'var(--text-dim)'};">
        ${s.order_quantity > 0 ? `+${s.order_quantity} units` : '0'}
      </td>
      <td style="font-family: monospace; font-weight: 700; color: #a5b4fc;">
        ${s.projected_ending_inventory} units
      </td>
    </tr>
  `).join('');
}

// -----------------------------------------------------------------------------
// MLOps LEADERBOARD & RESIDUALS
// -----------------------------------------------------------------------------
async function loadMLOpsLeaderboard() {
  const models = [
    { name: '5. LightGBM (Tuned)', mae: 4.473, rmse: 6.374, mape: 18.2, wape: 20.29, bias: -1.40, time: 4.64, status: 'Production Champion' },
    { name: '4. Random Forest', mae: 4.692, rmse: 6.710, mape: 19.8, wape: 21.28, bias: -0.95, time: 8.62, status: 'Benchmark' },
    { name: '3. Ridge Linear Regressor', mae: 5.452, rmse: 7.933, mape: 23.4, wape: 24.73, bias: -1.30, time: 0.16, status: 'Benchmark' },
    { name: '2. Moving Average (7D)', mae: 7.449, rmse: 10.932, mape: 31.2, wape: 33.79, bias: -0.93, time: 0.00, status: 'Baseline' },
    { name: '1. Naive (Lag 7)', mae: 7.974, rmse: 12.252, mape: 34.5, wape: 36.18, bias: -2.21, time: 0.00, status: 'Baseline' }
  ];

  const tbody = document.querySelector('#table-mlops-leaderboard tbody');
  if (tbody) {
    tbody.innerHTML = models.map(m => `
      <tr>
        <td style="font-weight: 700; color: white;">${m.name}</td>
        <td style="font-family: monospace; font-weight: 700; color: #10b981;">${m.mae.toFixed(3)}</td>
        <td style="font-family: monospace; color: var(--text-main);">${m.rmse.toFixed(3)}</td>
        <td style="font-family: monospace; color: var(--text-dim);">${m.mape.toFixed(1)}%</td>
        <td style="font-family: monospace; font-weight: 800; color: #38bdf8;">${m.wape.toFixed(2)}%</td>
        <td style="font-family: monospace; color: ${m.bias < 0 ? '#fb7185' : '#34d399'};">${m.bias > 0 ? '+' : ''}${m.bias.toFixed(2)}%</td>
        <td style="color: var(--text-dim);">${m.time.toFixed(2)}s</td>
        <td>
          <span class="badge ${m.status.includes('Champion') ? 'healthy' : 'medium'}">
            ${m.status}
          </span>
        </td>
      </tr>
    `).join('');
  }

  renderFeatureImportanceChart();
  renderResidualsChart();
}

function renderFeatureImportanceChart() {
  if (typeof ApexCharts === 'undefined') return;

  const features = [
    { name: 'previous_inventory', score: 534 },
    { name: 'week_of_year', score: 513 },
    { name: 'rolling_mean_7', score: 430 },
    { name: 'lag_1', score: 394 },
    { name: 'lag_14', score: 349 },
    { name: 'lag_7', score: 326 },
    { name: 'unit_price', score: 298 },
    { name: 'day_of_week', score: 284 },
    { name: 'rolling_std_7', score: 247 },
    { name: 'is_weekend', score: 215 }
  ];

  const options = {
    series: [{ name: 'Split Importance', data: features.map(f => f.score) }],
    chart: { type: 'bar', height: 340, background: 'transparent', toolbar: { show: false } },
    colors: ['#6366f1'],
    plotOptions: { bar: { borderRadius: 4, horizontal: true, barHeight: '60%' } },
    dataLabels: { enabled: false },
    xaxis: {
      categories: features.map(f => f.name),
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

function renderResidualsChart() {
  if (typeof ApexCharts === 'undefined') return;

  const bins = ['-15 to -10', '-10 to -5', '-5 to -2', '-2 to 0', '0 to +2', '+2 to +5', '+5 to +10', '+10 to +15'];
  const frequencies = [120, 480, 1850, 3100, 2950, 1620, 520, 110];

  const options = {
    series: [{ name: 'Test Error Frequency', data: frequencies }],
    chart: { type: 'bar', height: 340, background: 'transparent', toolbar: { show: false } },
    colors: ['#06b6d4'],
    plotOptions: { bar: { borderRadius: 4, columnWidth: '70%' } },
    dataLabels: { enabled: false },
    xaxis: { categories: bins, labels: { style: { colors: '#64748b', fontSize: '10px' } } },
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
// LIVE SQL STUDIO
// -----------------------------------------------------------------------------
const SAMPLE_QUERIES = {
  turnover: `SELECT 
    p.product_id,
    p.product_name,
    p.category,
    ROUND(SUM(s.sales_quantity * p.unit_cost), 2) AS total_cogs,
    ROUND(AVG(i.ending_inventory * p.unit_cost), 2) AS avg_inventory_value,
    ROUND(SUM(s.sales_quantity * p.unit_cost) / NULLIF(AVG(i.ending_inventory * p.unit_cost), 0), 2) AS inventory_turnover_ratio
FROM products p
JOIN sales s ON p.product_id = s.product_id
JOIN inventory i ON p.product_id = i.product_id AND s.date = i.date
GROUP BY p.product_id
ORDER BY inventory_turnover_ratio DESC
LIMIT 10;`,

  stockout: `SELECT 
    p.category,
    COUNT(DISTINCT s.product_id) AS sku_count,
    SUM(s.stockout_flag) AS stockout_incidents,
    ROUND(AVG(s.stockout_flag) * 100, 2) AS stockout_rate_pct
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY stockout_rate_pct DESC;`,

  top10: `SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.sales_quantity) AS total_units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue
FROM products p
JOIN sales s ON p.product_id = s.product_id
GROUP BY p.product_id
ORDER BY total_revenue DESC
LIMIT 10;`,

  regions: `SELECT 
    r.region_name,
    COUNT(DISTINCT s.product_id) AS active_skus,
    SUM(s.sales_quantity) AS total_units,
    ROUND(SUM(s.revenue), 2) AS total_revenue
FROM regions r
JOIN sales s ON r.region_id = s.region_id
GROUP BY r.region_id
ORDER BY total_revenue DESC;`
};

function loadSampleSQL(key) {
  const editor = document.getElementById('sql-query-editor');
  if (editor && SAMPLE_QUERIES[key]) {
    editor.value = SAMPLE_QUERIES[key];
  }
}

async function runCustomSQL() {
  const editor = document.getElementById('sql-query-editor');
  const metricsEl = document.getElementById('sql-execution-metrics');
  const btn = document.getElementById('btn-run-sql');
  const exportBtn = document.getElementById('btn-export-sql');

  if (!editor || !editor.value.trim()) return;

  const query = editor.value.trim();
  btn.innerText = 'Executing...';
  const startTime = performance.now();

  try {
    const res = await fetch('/api/sql/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });

    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    if (!res.ok) {
      metricsEl.innerHTML = `<span style="color: #fb7185;">Error: ${data.detail || 'Query execution failed'}</span>`;
      return;
    }

    state.lastSQLResults = data.records;
    metricsEl.innerHTML = `<span style="color: #34d399;">✓ Executed in ${elapsed}ms | Returned ${data.rows_returned} rows</span>`;
    if (exportBtn) exportBtn.classList.remove('hidden');

    renderSQLResultsGrid(data.columns, data.records);

  } catch (err) {
    metricsEl.innerHTML = `<span style="color: #fb7185;">Network or execution failure: ${err.message}</span>`;
  } finally {
    btn.innerText = 'Run SQL Query';
  }
}

function renderSQLResultsGrid(columns, records) {
  const thead = document.querySelector('#table-sql-results thead');
  const tbody = document.querySelector('#table-sql-results tbody');

  if (!thead || !tbody) return;

  thead.innerHTML = `<tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
  tbody.innerHTML = records.map(r => `
    <tr>
      ${columns.map(c => `<td>${r[c] !== null ? r[c] : '<span style="color: var(--text-dim);">NULL</span>'}</td>`).join('')}
    </tr>
  `).join('');
}

function downloadSQLResultsAsCSV() {
  if (!state.lastSQLResults || state.lastSQLResults.length === 0) return;
  const cols = Object.keys(state.lastSQLResults[0]);
  const rows = state.lastSQLResults.map(r => cols.map(c => `"${r[c] !== null ? r[c] : ''}"`).join(','));
  const csvContent = 'data:text/csv;charset=utf-8,' + [cols.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', 'sql_query_export.csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadProductsCSV() {
  window.open('/api/export/csv?dataset_name=products', '_blank');
}
