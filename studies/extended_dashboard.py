"""Extended evaluation dashboard with:
1. Quantile comparison (consumer c/kWh) with timeline slider
2. Contour map (spectrogram) with Actual / Forecast / Difference toggle
3. Predicted vs Actual scatter plots (consumer c/kWh)
4. Calibration chart
"""
import pandas as pd, numpy as np, yaml, sys, json, math, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from src.features import build_features

prices = pd.read_parquet("output/fi_prices.parquet")["price_eur_mwh"]
neighbor = pd.read_parquet("output/fi_neighbor_prices.parquet")
weather = pd.read_parquet("output/fi_weather.parquet")
grid_df = pd.read_parquet("output/fi_grid_data.parquet")
with open("config/regions/finland.yaml") as f:
    config = yaml.safe_load(f)

df, feature_cols, _ = build_features(
    prices, weather, config,
    neighbor_prices={col: neighbor[col] for col in neighbor.columns},
    grid_data={col: grid_df[col] for col in grid_df.columns},
)

fi = df["price_eur_mwh"].values
local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
hours = local.hour.values
dates = local.date
dow = local.dayofweek.values
months = local.month.values

se3 = neighbor["se3"].reindex(df.index).ffill().values
se1 = neighbor["se1"].reindex(df.index).ffill().values
wind = df["wind_speed_weighted"].values
solar = df["solar_irradiance_weighted"].values
temp = weather["temperature_weighted"].reindex(df.index).ffill().values
hdd = np.maximum(0, 17 - temp)

nuc = grid_df.get("nuclear_mw")
nuc_vals = nuc.reindex(df.index).ffill().bfill().fillna(0).values if nuc is not None else np.zeros(len(fi))
nuclear_deficit = np.maximum(0, 1.0 - nuc_vals)

SEGMENTS = {
    "night":   list(range(22, 24)) + list(range(0, 6)),
    "morning": list(range(6, 10)),
    "midday":  list(range(10, 16)),
    "evening": list(range(16, 22)),
}
SEG_HOURS = {"night": 8, "morning": 4, "midday": 6, "evening": 6}
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

feature_names = ["wind_mean", "solar_mean", "hdd_mean", "se3_mean", "se1_mean",
                  "nuclear_deficit", "is_workday", "month_sin", "month_cos",
                  "wind_log_scarcity"]

# Build segment data
print("Building segment data...")
unique_dates = sorted(set(dates))
segment_data = {seg: [] for seg in SEGMENTS}

for d in unique_dates:
    d_mask = np.array([dd == d for dd in dates])
    if d_mask.sum() < 20:
        continue
    is_wd = 1.0 if dow[d_mask][0] < 5 else 0.0
    mo = months[d_mask][0]

    for seg_name, seg_hours_list in SEGMENTS.items():
        seg_mask = d_mask & np.isin(hours, seg_hours_list)
        if seg_mask.sum() < 2:
            continue
        segment_data[seg_name].append({
            "features": {
                "wind_mean": float(wind[seg_mask].mean()),
                "solar_mean": float(solar[seg_mask].mean()),
                "hdd_mean": float(hdd[seg_mask].mean()),
                "se3_mean": float(se3[seg_mask].mean()),
                "se1_mean": float(se1[seg_mask].mean()),
                "nuclear_deficit": float(nuclear_deficit[seg_mask].mean()),
                "is_workday": is_wd,
                "month_sin": math.sin(2 * math.pi * mo / 12),
                "month_cos": math.cos(2 * math.pi * mo / 12),
                "wind_log_scarcity": float(np.log1p(np.maximum(0, 8 - wind[seg_mask])).mean()),
            },
            "targets": {
                "prices": fi[seg_mask].tolist(),
                "mean": float(fi[seg_mask].mean()),
                "date": str(d),
            },
        })

# Train QRidge models (best extrapolation)
print("Training QRidge models...")
split_frac = 0.85
qr_models = {}

for seg_name in SEGMENTS:
    data = segment_data[seg_name]
    split = int(len(data) * split_frac)
    qr_models[seg_name] = {}

    X_list, y_list = [], []
    for d in data[:split]:
        x = [d["features"][f] for f in feature_names]
        for price in d["targets"]["prices"]:
            X_list.append(x)
            y_list.append(price)
    X_tr = np.array(X_list)
    y_tr = np.array(y_list)

    for q in QUANTILES:
        model = QuantileRegressor(quantile=q, alpha=1.0, solver="highs")
        model.fit(X_tr, y_tr)
        qr_models[seg_name][q] = model

print("Models trained.")

# Predict on full test period
print("Generating predictions for test period...")

test_daily = []
for d in unique_dates:
    d_mask = np.array([dd == d for dd in dates])
    if d_mask.sum() < 20:
        continue

    # Check if this is in test set
    d_idx = unique_dates.index(d)
    split_idx = int(len(unique_dates) * split_frac)
    is_test = d_idx >= split_idx

    # Predict all segments
    daily_pred_quantiles = {q: 0.0 for q in QUANTILES}
    daily_actual_prices = []
    total_h = 0

    for seg_name in SEGMENTS:
        seg_data_list = [s for s in segment_data[seg_name] if s["targets"]["date"] == str(d)]
        if not seg_data_list:
            continue
        sd = seg_data_list[0]
        x = np.array([[sd["features"][f] for f in feature_names]])
        nh = SEG_HOURS[seg_name]

        for q in QUANTILES:
            pred_q = float(qr_models[seg_name][q].predict(x)[0])
            daily_pred_quantiles[q] += pred_q * nh

        daily_actual_prices.extend(sd["targets"]["prices"])
        total_h += nh

    if total_h < 20:
        continue

    # Normalize by hours
    for q in QUANTILES:
        daily_pred_quantiles[q] /= total_h

    # Actual quantiles
    ap = np.array(daily_actual_prices)
    actual_quantiles = {q: float(np.percentile(ap, q * 100)) for q in QUANTILES}

    # Consumer price conversion
    def to_cons(spot):
        return (max(0, spot) / 1000 + 0.0361 + 0.02325) * 1.255 * 100

    test_daily.append({
        "date": str(d),
        "is_test": is_test,
        "pred": {("p%02d" % int(q*100)): daily_pred_quantiles[q] for q in QUANTILES},
        "actual": {("p%02d" % int(q*100)): actual_quantiles[q] for q in QUANTILES},
        "actual_mean": float(ap.mean()),
        "pred_mean": daily_pred_quantiles[0.50],
        "actual_prices": ap.tolist(),
        "cons_pred": {("p%02d" % int(q*100)): to_cons(daily_pred_quantiles[q]) for q in QUANTILES},
        "cons_actual": {("p%02d" % int(q*100)): to_cons(actual_quantiles[q]) for q in QUANTILES},
    })

print("Generated %d daily records (%d test)" %
      (len(test_daily), sum(1 for d in test_daily if d["is_test"])))

# Build contour data: price density over time
# For each day, compute histogram at fine resolution
contour_bins = np.arange(-5, 65, 1)  # 1 EUR/MWh resolution
contour_actual = []
for d in test_daily:
    hist, _ = np.histogram(d["actual_prices"], bins=contour_bins, density=True)
    contour_actual.append([round(float(x), 4) for x in hist])

# Build PREDICTED contour data from predicted quantiles
# Reconstruct density from 7 quantiles by spreading probability mass
contour_pred = []
q_levels = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
q_mass = [0.05, 0.05, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05]  # mass between adjacent quantiles
bin_edges = contour_bins.astype(float)

for d in test_daily:
    pred = d["pred"]
    # Quantile values, extend with extrapolated tails
    p05, p95 = pred["p05"], pred["p95"]
    tail_low = max(bin_edges[0], p05 - (pred["p25"] - p05))
    tail_high = min(bin_edges[-1], p95 + (p95 - pred["p75"]))
    q_vals = [tail_low, p05, pred["p10"], pred["p25"], pred["p50"],
              pred["p75"], pred["p90"], p95, tail_high]

    hist = np.zeros(len(bin_edges) - 1)
    for seg_i in range(len(q_mass)):
        lo, hi = q_vals[seg_i], q_vals[seg_i + 1]
        if hi <= lo:
            hi = lo + 0.5
        mass = q_mass[seg_i]
        # Find bins overlapping [lo, hi] and distribute mass uniformly
        for b in range(len(hist)):
            b_lo, b_hi = bin_edges[b], bin_edges[b + 1]
            overlap = max(0, min(hi, b_hi) - max(lo, b_lo))
            span = hi - lo
            if span > 0:
                hist[b] += mass * overlap / span
    # Normalize to density
    total = hist.sum()
    if total > 0:
        hist = hist / total * len(hist) / (bin_edges[-1] - bin_edges[0])
    contour_pred.append([round(float(x), 4) for x in hist])

# Weekly rolling average for smoother contour
def smooth_contour(raw, window=7):
    result = []
    for i in range(len(raw)):
        s = max(0, i - window // 2)
        e = min(len(raw), i + window // 2 + 1)
        avg = np.mean(raw[s:e], axis=0)
        result.append([round(float(x), 4) for x in avg])
    return result

contour_actual_smooth = smooth_contour(contour_actual)
contour_pred_smooth = smooth_contour(contour_pred)

# Difference: forecast - actual
contour_diff_smooth = []
for i in range(len(contour_actual_smooth)):
    diff = [round(float(x), 4) for x in (np.array(contour_pred_smooth[i]) - np.array(contour_actual_smooth[i]))]
    contour_diff_smooth.append(diff)

# ================================================================
# Compute per-quantile MAE on test set for summary
# ================================================================
test_only = [d for d in test_daily if d["is_test"]]
print("\nQuantile MAE (test set, consumer c/kWh):")
def to_cons_py(spot):
    return (max(0, spot) / 1000 + 0.0361 + 0.02325) * 1.255 * 100
for ql in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]:
    mae_q = np.mean([abs(to_cons_py(d["pred"][ql]) - to_cons_py(d["actual"][ql])) for d in test_only])
    print("  %s: %.3f c/kWh" % (ql.upper(), mae_q))
# P90 bias check
p90_bias = np.mean([to_cons_py(d["pred"]["p90"]) - to_cons_py(d["actual"]["p90"]) for d in test_only])
print("  P90 bias: %+.3f c/kWh (positive = over-predicting)" % p90_bias)

# ================================================================
# BUILD HTML DASHBOARD
# ================================================================
print("\nBuilding HTML dashboard...")

# Strip actual_prices from JSON to reduce size (contour is pre-computed)
daily_slim = []
for d in test_daily[-220:]:
    slim = {k: v for k, v in d.items() if k != "actual_prices"}
    daily_slim.append(slim)

chart_json = {
    "daily": daily_slim,
    "contour_bins": contour_bins[:-1].tolist(),
    "contour_actual": contour_actual_smooth[-220:],
    "contour_pred": contour_pred_smooth[-220:],
    "contour_diff": contour_diff_smooth[-220:],
    "quantile_labels": ["p%02d" % int(q*100) for q in QUANTILES],
}

html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Distributional Price Forecast — Extended Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
h1 { color: #f8fafc; margin-bottom: 5px; font-size: 22px; }
h2 { color: #94a3b8; margin: 25px 0 12px; font-size: 16px; border-bottom: 1px solid #334155; padding-bottom: 6px; }
.subtitle { color: #64748b; font-size: 12px; margin-bottom: 15px; }
.chart-box { background: #1e293b; border-radius: 8px; padding: 15px; margin: 12px 0; position: relative; }
canvas { width: 100% !important; }
.contour-wrap { overflow-x: auto; }
.grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }
.metric-row { display: flex; gap: 12px; margin: 10px 0; flex-wrap: wrap; }
.metric { background: #1e293b; border-radius: 8px; padding: 10px 14px; text-align: center; min-width: 100px; flex: 1; }
.metric-val { font-size: 18px; font-weight: 700; }
.metric-lbl { font-size: 10px; color: #94a3b8; margin-top: 2px; }
.v-cyan { color: #22d3ee; }
.v-yellow { color: #facc15; }
.v-orange { color: #f97316; }
.v-red { color: #ef4444; }
.v-green { color: #4ade80; }
.v-blue { color: #60a5fa; }
.slider-controls { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.slider-controls button, .contour-controls button {
    background: #334155; color: #94a3b8; border: none; border-radius: 4px;
    padding: 4px 12px; font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.slider-controls button:hover, .contour-controls button:hover { background: #475569; color: #e2e8f0; }
.slider-controls button.active, .contour-controls button.active { background: #60a5fa; color: #0f172a; font-weight: 600; }
.slider-controls input[type="range"] { flex: 1; min-width: 120px; accent-color: #60a5fa; }
.slider-label { color: #64748b; font-size: 11px; min-width: 140px; text-align: right; }
.toggle-row { display: flex; gap: 12px; margin: 6px 0 2px; flex-wrap: wrap; }
.toggle-btn {
    font-size: 11px; padding: 3px 10px; border-radius: 3px; cursor: pointer;
    border: 1px solid; transition: all 0.15s; background: transparent;
}
.toggle-btn.on { opacity: 1; }
.toggle-btn.off { opacity: 0.35; }
.contour-controls { display: flex; gap: 8px; margin-bottom: 10px; }
footer { margin-top: 30px; padding: 12px 0; text-align: center; color: #475569; font-size: 11px; border-top: 1px solid #1e293b; }
footer a { color: #60a5fa; text-decoration: none; }
</style>
</head>
<body>
<h1>Distributional Price Forecast — Extended Evaluation</h1>
<p class="subtitle">Segment-hierarchical quantile regression (night/morning/midday/evening) with SE3/SE1 coupling. Full test period.</p>

<!-- Dynamic metrics (consumer c/kWh) -->
<div class="metric-row" id="qMetrics"></div>

<!-- Main comparison: Actual vs Predicted quantiles in consumer c/kWh -->
<h2>Quantile Comparison: Actual vs Predicted (Consumer c/kWh)</h2>
<p class="subtitle">Solid lines = actual daily quantiles, dashed lines = predicted. Each color is one quantile level.</p>
<div class="chart-box">
  <div class="slider-controls">
    <button onclick="setWindow(7)" id="btn7">7d</button>
    <button onclick="setWindow(30)" id="btn30">30d</button>
    <button onclick="setWindow(90)" id="btn90">90d</button>
    <button onclick="setWindow(0)" id="btnAll">All</button>
    <input type="range" id="timeSlider" min="0" max="100" value="100">
    <span class="slider-label" id="sliderLabel"></span>
  </div>
  <div class="toggle-row" id="qToggles"></div>
  <canvas id="qCompare" height="300"></canvas>
</div>

<!-- Contour map with Actual / Forecast / Difference toggle -->
<h2>Price Density Spectrogram (EUR/MWh)</h2>
<p class="subtitle">Color intensity = relative density of hours at each price level per day. 7-day rolling average.</p>
<div class="chart-box contour-wrap">
  <div class="contour-controls">
    <button onclick="setContour('actual')" id="cBtn-actual" class="active">Actual</button>
    <button onclick="setContour('pred')" id="cBtn-pred">Forecast</button>
    <button onclick="setContour('diff')" id="cBtn-diff">Difference (F-A)</button>
  </div>
  <canvas id="contour" height="350"></canvas>
</div>

<!-- 4-panel scatter (consumer c/kWh) -->
<h2>Predicted vs Actual Scatter (Consumer c/kWh)</h2>
<div class="grid-4">
  <div class="chart-box"><div style="font-size:12px;color:#e2e8f0;margin-bottom:4px;font-weight:600">P25</div><canvas id="scat-p25" height="200"></canvas></div>
  <div class="chart-box"><div style="font-size:12px;color:#e2e8f0;margin-bottom:4px;font-weight:600">P50 (Median)</div><canvas id="scat-p50" height="200"></canvas></div>
  <div class="chart-box"><div style="font-size:12px;color:#e2e8f0;margin-bottom:4px;font-weight:600">P75</div><canvas id="scat-p75" height="200"></canvas></div>
  <div class="chart-box"><div style="font-size:12px;color:#e2e8f0;margin-bottom:4px;font-weight:600">P90</div><canvas id="scat-p90" height="200"></canvas></div>
</div>

<!-- Calibration -->
<h2>Calibration: Actual Coverage per Quantile</h2>
<div class="chart-box"><canvas id="calibration" height="220"></canvas></div>

<footer>
  HA-spot-price-predictor | Segment-hierarchical QRidge |
  <a href="https://github.com/watti-matti/HA-spot-price-predictor">GitHub</a>
</footer>

<script>
const D = ''' + json.dumps(chart_json) + ''';

const test = D.daily.filter(d => d.is_test);
const N = test.length;

function toC(s) { return (Math.max(0,s)/1000+0.0361+0.02325)*1.255*100; }

// ── Quantile colors ──
const Q_CFG = {
    p05: { color: '#64748b', label: 'P05' },
    p10: { color: '#818cf8', label: 'P10' },
    p25: { color: '#22d3ee', label: 'P25' },
    p50: { color: '#facc15', label: 'P50' },
    p75: { color: '#f97316', label: 'P75' },
    p90: { color: '#ef4444', label: 'P90' },
    p95: { color: '#f472b6', label: 'P95' },
};
const Q_KEYS = Object.keys(Q_CFG);
const qVisible = { p05:false, p10:false, p25:true, p50:true, p75:true, p90:true, p95:false };

// ── Slider state ──
let windowDays = 30;
let sliderPos = 100;

function getWindowSlice() {
    if (windowDays === 0) return { start: 0, end: N };
    const winSize = Math.min(windowDays, N);
    const maxStart = N - winSize;
    const start = Math.round(maxStart * sliderPos / 100);
    return { start, end: start + winSize };
}

// ── Metrics (consumer c/kWh) ──
function updateMetrics(start, end) {
    const slice = test.slice(start, end);
    const n = slice.length;
    let html = '';
    for (const qk of ['p25','p50','p75','p90']) {
        let sumAE = 0;
        slice.forEach(d => { sumAE += Math.abs(toC(d.pred[qk]) - toC(d.actual[qk])); });
        const mae = n > 0 ? (sumAE / n).toFixed(3) : '—';
        const cls = {p25:'v-cyan',p50:'v-yellow',p75:'v-orange',p90:'v-red'}[qk];
        html += '<div class="metric"><div class="metric-val ' + cls + '">' + mae +
                '</div><div class="metric-lbl">MAE ' + qk.toUpperCase() + ' (c/kWh)</div></div>';
    }
    // P90 bias
    let sumBias = 0;
    slice.forEach(d => { sumBias += toC(d.pred.p90) - toC(d.actual.p90); });
    const bias = n > 0 ? (sumBias / n) : 0;
    const biasSign = bias >= 0 ? '+' : '';
    html += '<div class="metric"><div class="metric-val v-red">' + biasSign + bias.toFixed(3) +
            '</div><div class="metric-lbl">P90 Bias (c/kWh)</div></div>';
    // Spearman on median
    const predMed = slice.map(d => d.pred.p50);
    const actMed = slice.map(d => d.actual.p50);
    const rho = spearman(predMed, actMed);
    html += '<div class="metric"><div class="metric-val v-blue">' + rho.toFixed(3) +
            '</div><div class="metric-lbl">Spearman P50</div></div>';
    html += '<div class="metric"><div class="metric-val v-green">' + n +
            '</div><div class="metric-lbl">Days</div></div>';
    document.getElementById('qMetrics').innerHTML = html;
}

function spearman(a, b) {
    const n = a.length;
    if (n < 3) return 0;
    function ranks(arr) {
        const sorted = arr.map((v,i) => ({v,i})).sort((a,b) => a.v - b.v);
        const r = new Array(n);
        for (let i = 0; i < n; i++) r[sorted[i].i] = i + 1;
        return r;
    }
    const ra = ranks(a), rb = ranks(b);
    let sumD2 = 0;
    for (let i = 0; i < n; i++) sumD2 += (ra[i] - rb[i]) ** 2;
    return 1 - 6 * sumD2 / (n * (n*n - 1));
}

// ── Quantile comparison chart (consumer c/kWh) ──
let qChart = null;

function buildQCompareChart() {
    const sl = getWindowSlice();
    const slice = test.slice(sl.start, sl.end);
    const labs = slice.map(d => d.date.substring(5));

    if (slice.length > 0) {
        document.getElementById('sliderLabel').textContent =
            slice[0].date + ' .. ' + slice[slice.length-1].date;
    }
    updateMetrics(sl.start, sl.end);

    const datasets = [];
    for (const qk of Q_KEYS) {
        if (!qVisible[qk]) continue;
        const cfg = Q_CFG[qk];
        datasets.push({
            label: 'A ' + cfg.label,
            data: slice.map(d => toC(d.actual[qk])),
            borderColor: cfg.color,
            borderWidth: 1.8, pointRadius: 0, fill: false, tension: 0.3, borderDash: [],
        });
        datasets.push({
            label: 'F ' + cfg.label,
            data: slice.map(d => toC(d.pred[qk])),
            borderColor: cfg.color,
            borderWidth: 1.8, pointRadius: 0, fill: false, tension: 0.3, borderDash: [6, 3],
        });
    }

    const data = { labels: labs, datasets };

    if (qChart) {
        qChart.data = data;
        qChart.update();
    } else {
        qChart = new Chart(document.getElementById('qCompare').getContext('2d'), {
            type: 'line', data,
            options: {
                responsive: true,
                animation: { duration: 200 },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: true, labels: { color: '#e2e8f0', font: { size: 10 }, boxWidth: 16, padding: 8 } },
                    tooltip: {
                        mode: 'index', intersect: false,
                        backgroundColor: '#1e2433', borderColor: '#374151', borderWidth: 1,
                        titleColor: '#e2e8f0', bodyColor: '#e2e8f0',
                        callbacks: {
                            label: function(ctx) {
                                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + ' c/kWh';
                            }
                        }
                    },
                },
                scales: {
                    x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 20, font: { size: 9 } } },
                    y: { title: { display: true, text: 'Consumer (c/kWh)', color: '#e2e8f0' }, grid: { color: '#334155' }, ticks: { color: '#e2e8f0' } },
                },
            }
        });
    }

    // Sync contour and scatter to same time window
    drawContour();
    buildScatterPlots(sl.start, sl.end);
}

// ── Slider controls ──
function setWindow(d) {
    windowDays = d;
    document.querySelectorAll('.slider-controls button').forEach(b => b.classList.remove('active'));
    const id = d === 0 ? 'btnAll' : 'btn' + d;
    document.getElementById(id).classList.add('active');
    buildQCompareChart();
}

document.getElementById('timeSlider').addEventListener('input', function() {
    sliderPos = parseInt(this.value);
    buildQCompareChart();
});

// ── Quantile toggles ──
(function() {
    const row = document.getElementById('qToggles');
    for (const qk of Q_KEYS) {
        const cfg = Q_CFG[qk];
        const btn = document.createElement('button');
        btn.className = 'toggle-btn ' + (qVisible[qk] ? 'on' : 'off');
        btn.style.borderColor = cfg.color;
        btn.style.color = cfg.color;
        btn.textContent = cfg.label;
        btn.addEventListener('click', () => {
            qVisible[qk] = !qVisible[qk];
            btn.className = 'toggle-btn ' + (qVisible[qk] ? 'on' : 'off');
            buildQCompareChart();
        });
        row.appendChild(btn);
    }
})();

// ── Contour state (must be before setWindow) ──
let contourMode = 'actual';
let scatterCharts = {};

setWindow(30);

// ── Contour map with Actual / Forecast / Difference toggle ──

function drawContour() {
    const sl = getWindowSlice();
    const canvas = document.getElementById('contour');
    const ctx = canvas.getContext('2d');
    const bins = D.contour_bins;

    // Slice contour data to visible window (contour arrays align with test indices)
    let fullData, isDiff = false;
    if (contourMode === 'actual') fullData = D.contour_actual;
    else if (contourMode === 'pred') fullData = D.contour_pred;
    else { fullData = D.contour_diff; isDiff = true; }

    // Map test indices to contour indices (contour covers all daily, test is subset)
    // Both arrays are last 220 entries of test_daily; test filters is_test
    // We need to map sl.start/end (test indices) to the contour array
    const allDays = D.daily;
    const testIndices = [];
    for (let i = 0; i < allDays.length; i++) {
        if (allDays[i].is_test) testIndices.push(i);
    }
    const cStart = testIndices[sl.start] || 0;
    const cEnd = testIndices[Math.min(sl.end - 1, testIndices.length - 1)] + 1 || fullData.length;
    const data = fullData.slice(cStart, cEnd);
    const sliceDates = allDays.slice(cStart, cEnd);

    const nDays = data.length;
    const nBins = bins.length;

    const minW = 800;
    const pixPerDay = Math.max(3, Math.min(12, Math.floor(minW / Math.max(nDays, 1))));
    canvas.width = Math.max(nDays * pixPerDay + 60, minW);
    canvas.height = 350;
    const plotW = canvas.width - 60;
    const plotH = canvas.height - 40;
    const cellW = plotW / Math.max(nDays, 1);
    const cellH = plotH / nBins;

    let maxVal = 0;
    if (isDiff) {
        data.forEach(row => row.forEach(v => { if(Math.abs(v)>maxVal) maxVal=Math.abs(v); }));
    } else {
        data.forEach(row => row.forEach(v => { if(v>maxVal) maxVal=v; }));
    }

    function densityColor(v) {
        const t = Math.pow(Math.min(v / (maxVal * 0.6), 1), 0.5);
        if (t < 0.25) {
            const s = t / 0.25;
            return 'rgb(' + Math.round(15+s*10) + ',' + Math.round(23+s*40) + ',' + Math.round(42+s*80) + ')';
        } else if (t < 0.5) {
            const s = (t-0.25) / 0.25;
            return 'rgb(' + Math.round(25+s*10) + ',' + Math.round(63+s*130) + ',' + Math.round(122+s*60) + ')';
        } else if (t < 0.75) {
            const s = (t-0.5) / 0.25;
            return 'rgb(' + Math.round(35+s*215) + ',' + Math.round(193+s*10) + ',' + Math.round(182-s*160) + ')';
        } else {
            const s = (t-0.75) / 0.25;
            return 'rgb(' + Math.round(250) + ',' + Math.round(203-s*150) + ',' + Math.round(22-s*20) + ')';
        }
    }

    function diffColor(v) {
        const scale = maxVal * 0.5 || 1;
        const t = Math.min(Math.abs(v) / scale, 1);
        const intensity = Math.pow(t, 0.6);
        if (v > 0) {
            return 'rgb(' + Math.round(40+215*intensity) + ',' + Math.round(30+20*intensity) + ',' + Math.round(40) + ')';
        } else {
            return 'rgb(' + Math.round(30) + ',' + Math.round(40+60*intensity) + ',' + Math.round(50+200*intensity) + ')';
        }
    }

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let d = 0; d < nDays; d++) {
        for (let b = 0; b < nBins; b++) {
            const v = data[d][b] || 0;
            const threshold = isDiff ? (maxVal * 0.02) : (maxVal * 0.01);
            if (Math.abs(v) > threshold) {
                ctx.fillStyle = isDiff ? diffColor(v) : densityColor(v);
                const x = 50 + d * cellW;
                const y = 10 + (nBins - 1 - b) * cellH;
                ctx.fillRect(x, y, Math.max(cellW, 1.5), cellH + 0.5);
            }
        }
    }

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let p = 0; p <= 60; p += 10) {
        const bIdx = bins.indexOf(p);
        if (bIdx >= 0) {
            const y = 10 + (nBins - 1 - bIdx) * cellH;
            ctx.fillText(p + '', 45, y + 4);
            ctx.strokeStyle = '#33415544';
            ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }
    }

    ctx.textAlign = 'center';
    ctx.fillStyle = '#e2e8f0';
    let lastLabel = '';
    for (let d = 0; d < nDays; d++) {
        const dd = sliceDates[d] ? sliceDates[d].date : '';
        // For short windows show more date labels
        const labelKey = nDays <= 14 ? dd.substring(5) : dd.substring(0, 7);
        if (labelKey !== lastLabel && dd) {
            const x = 50 + d * cellW;
            ctx.fillText(nDays <= 14 ? dd.substring(5) : dd.substring(0, 7), x, canvas.height - 5);
            lastLabel = labelKey;
        }
    }

    ctx.fillStyle = '#e2e8f0';
    ctx.save();
    ctx.translate(12, plotH / 2 + 10);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('EUR/MWh', 0, 0);
    ctx.restore();

    if (isDiff) {
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#ef4444'; ctx.fillRect(canvas.width - 180, 15, 12, 10);
        ctx.fillStyle = '#e2e8f0'; ctx.fillText('Forecast > Actual', canvas.width - 164, 24);
        ctx.fillStyle = '#3b82f6'; ctx.fillRect(canvas.width - 180, 30, 12, 10);
        ctx.fillStyle = '#e2e8f0'; ctx.fillText('Forecast < Actual', canvas.width - 164, 39);
    }
}

function setContour(mode) {
    contourMode = mode;
    document.querySelectorAll('.contour-controls button').forEach(b => b.classList.remove('active'));
    document.getElementById('cBtn-' + mode).classList.add('active');
    drawContour();
}

// ── 4-panel scatter plots (consumer c/kWh, slider-synced) ──
function buildScatterPlots(start, end) {
    const slice = test.slice(start, end);
    scatterPlot('scat-p25', 'p25', '#22d3ee', slice);
    scatterPlot('scat-p50', 'p50', '#facc15', slice);
    scatterPlot('scat-p75', 'p75', '#f97316', slice);
    scatterPlot('scat-p90', 'p90', '#ef4444', slice);
}

function scatterPlot(canvasId, qKey, color, slice) {
    const pts = slice.map(d => ({x: toC(d.pred[qKey]), y: toC(d.actual[qKey])}));
    const vals = pts.flatMap(p => [p.x, p.y]);
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);

    const n = pts.length;
    let sumAE = 0, sumSE = 0, yMean = 0, sumBias = 0;
    pts.forEach(p => { sumAE += Math.abs(p.x - p.y); yMean += p.y; sumBias += (p.x - p.y); });
    yMean /= n;
    let ssTot = 0;
    pts.forEach(p => { sumSE += (p.x - p.y)**2; ssTot += (p.y - yMean)**2; });
    const r2 = 1 - sumSE / (ssTot || 1);
    const mae = sumAE / n;
    const bias = sumBias / n;
    const biasStr = (bias >= 0 ? '+' : '') + bias.toFixed(2);
    const titleText = 'MAE=' + mae.toFixed(2) + '  R\\u00b2=' + r2.toFixed(3) + '  Bias=' + biasStr;

    const chartData = {
        datasets: [
            { data: pts, backgroundColor: color + '66', borderColor: color, pointRadius: 2.5, borderWidth: 0.5 },
            { data: [{x:minV,y:minV},{x:maxV,y:maxV}], borderColor: '#94a3b8', borderDash: [4,4],
              pointRadius: 0, showLine: true, fill: false }
        ]
    };

    if (scatterCharts[canvasId]) {
        scatterCharts[canvasId].data = chartData;
        scatterCharts[canvasId].options.plugins.title.text = titleText;
        scatterCharts[canvasId].update();
    } else {
        scatterCharts[canvasId] = new Chart(document.getElementById(canvasId).getContext('2d'), {
            type: 'scatter',
            data: chartData,
            options: {
                responsive: true, animation: false, aspectRatio: 1,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: titleText,
                             color: '#e2e8f0', font: { size: 10, weight: 'normal' } },
                },
                scales: {
                    x: { title:{display:true,text:'Predicted (c/kWh)',color:'#e2e8f0',font:{size:9}}, grid:{color:'#334155'}, ticks:{color:'#e2e8f0',font:{size:9}} },
                    y: { title:{display:true,text:'Actual (c/kWh)',color:'#e2e8f0',font:{size:9}}, grid:{color:'#334155'}, ticks:{color:'#e2e8f0',font:{size:9}} }
                }
            }
        });
    }
}

// ── Calibration chart ──
(function() {
    const ctx = document.getElementById('calibration').getContext('2d');
    const qLabels = D.quantile_labels;
    const ideal = qLabels.map(q => parseInt(q.substring(1)));

    const cov_mean = qLabels.map(ql => {
        let covered = 0;
        test.forEach(d => { if (d.actual_mean <= d.pred[ql]) covered++; });
        return (covered / test.length * 100);
    });
    const cov_median = qLabels.map(ql => {
        let covered = 0;
        test.forEach(d => { if (d.actual.p50 <= d.pred[ql]) covered++; });
        return (covered / test.length * 100);
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: qLabels.map(q => q.toUpperCase()),
            datasets: [
                { label: 'Ideal', data: ideal, backgroundColor: '#64748b33', borderColor: '#64748b', borderWidth: 1 },
                { label: 'Coverage (mean)', data: cov_mean, backgroundColor: '#60a5fa55', borderColor: '#60a5fa', borderWidth: 1 },
                { label: 'Coverage (median)', data: cov_median, backgroundColor: '#facc1555', borderColor: '#facc15', borderWidth: 1 },
            ]
        },
        options: {
            responsive: true, animation: false,
            plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 10 } } } },
            scales: {
                y: { title:{display:true,text:'Coverage %',color:'#e2e8f0'}, grid:{color:'#334155'}, ticks:{color:'#e2e8f0'}, max: 100 },
                x: { grid:{color:'#334155'}, ticks:{color:'#e2e8f0'} }
            }
        }
    });
})();
</script>
</body>
</html>'''

with open("output/extended_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Dashboard saved to: output/extended_dashboard.html")
