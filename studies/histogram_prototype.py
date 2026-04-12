"""Build comprehensive histogram-based forecast analysis and HTML prototype.

Generates an interactive HTML dashboard showing:
1. Ridgeline plot: 7-day price distribution evolution
2. Heatmap: price bin x day matrix with color intensity
3. Fan chart: prediction intervals widening over forecast horizon
4. Daily summary cards: cheapest block, mean, max, cheap hours count
5. Model performance analysis with histogram statistics
"""
import pandas as pd, numpy as np, yaml, sys, json, math
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, gaussian_kde
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
se3 = neighbor["se3"].reindex(df.index).ffill().values
se1 = neighbor["se1"].reindex(df.index).ffill().values
ee = neighbor["ee"].reindex(df.index).ffill().values
non_ar = [n for n in feature_cols if not n.startswith("ar_")]
X_base = df[non_ar].values.astype(np.float64)

local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
dates = local.date
hours = local.hour.values

half_life = 120; alpha = 1.0
decay = np.log(2.0) / (half_life * 24.0)

def train_ridge(Xtr, ytr, w):
    ws = w.sum()
    fm = (w[:, None] * Xtr).sum(0) / ws
    fv = (w[:, None] * (Xtr ** 2)).sum(0) / ws - fm ** 2
    fs = np.sqrt(np.maximum(0, fv)); fs[fs < 1e-10] = 1.0
    ym = (w * ytr).sum() / ws
    Xs = (Xtr - fm) / fs; ys = ytr - ym
    Xw = Xs * w[:, None]
    cs = np.linalg.solve(Xw.T @ Xs + alpha * np.eye(Xtr.shape[1]), Xw.T @ ys)
    coefs = cs / fs; ic = ym - (fm / fs) @ cs
    return coefs, ic

# Build daily feature/target data
unique_dates = sorted(set(dates))
price_bins = [-10, 0, 2, 5, 8, 12, 18, 25, 40, 60, 100, 500]
bin_labels = []
for i in range(len(price_bins) - 1):
    if price_bins[i+1] >= 100:
        bin_labels.append("%d+" % price_bins[i])
    else:
        bin_labels.append("%d-%d" % (price_bins[i], price_bins[i+1]))

# Consumer price bins
cons_bins = [0, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 14, 17, 50]
cons_labels = []
for i in range(len(cons_bins) - 1):
    if cons_bins[i+1] >= 50:
        cons_labels.append(">%.0f" % cons_bins[i])
    else:
        cons_labels.append("%.1f-%.1f" % (cons_bins[i], cons_bins[i+1]))

daily_records = []
for d in unique_dates:
    mask = np.array([dd == d for dd in dates])
    if mask.sum() < 20:
        continue
    fi_day = fi[mask]
    se3_day = se3[mask]
    se1_day = se1[mask]
    ee_day = ee[mask]
    x_day = X_base[mask]

    # Histogram
    hist, _ = np.histogram(fi_day, bins=price_bins)
    hist_frac = (hist / hist.sum()).tolist()

    # Consumer price histogram
    cons_prices = (fi_day / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    cons_hist, _ = np.histogram(cons_prices, bins=cons_bins)
    cons_hist_frac = (cons_hist / cons_hist.sum()).tolist()

    # Block statistics
    n = len(fi_day)
    blocks_4h = [fi_day[i:i+4].mean() for i in range(n - 3)]
    blocks_2h = [fi_day[i:i+2].mean() for i in range(n - 1)]

    daily_records.append({
        "date": str(d),
        "weekday": d.strftime("%a") if hasattr(d, "strftime") else "?",
        "mean": float(fi_day.mean()),
        "median": float(np.median(fi_day)),
        "p25": float(np.percentile(fi_day, 25)),
        "p75": float(np.percentile(fi_day, 75)),
        "p90": float(np.percentile(fi_day, 90)),
        "max": float(fi_day.max()),
        "min": float(fi_day.min()),
        "cheapest_4h": float(min(blocks_4h)),
        "expensive_4h": float(max(blocks_4h)),
        "cheapest_2h": float(min(blocks_2h)),
        "cheap_hours": int((fi_day < 5).sum()),
        "expensive_hours": int((fi_day > 15).sum()),
        "hist": hist_frac,
        "cons_hist": cons_hist_frac,
        "se3_mean": float(se3_day.mean()),
        "se1_mean": float(se1_day.mean()),
        "x_mean": x_day.mean(axis=0).tolist(),
    })

print("Built %d daily records" % len(daily_records))

# Train models for each histogram statistic
X_daily = np.column_stack([
    np.array([r["x_mean"] for r in daily_records]),
    np.array([[r["se3_mean"]/100, r["se1_mean"]/100]
              for r in daily_records]),
])

split_d = int(len(X_daily) * 0.85)
age_d = np.arange(split_d - 1, -1, -1, dtype=np.float64)
w_d = np.exp(-decay * age_d * 24)
w_d *= split_d / w_d.sum()

# Train prediction models for key statistics
models = {}
for target_name in ["mean", "median", "p25", "p75", "p90", "max",
                      "cheapest_4h", "expensive_4h", "cheap_hours"]:
    y = np.array([r[target_name] for r in daily_records])
    c, ic = train_ridge(X_daily[:split_d], y[:split_d], w_d)
    pred = X_daily[split_d:] @ c + ic
    mae = mean_absolute_error(y[split_d:], pred)
    r2 = r2_score(y[split_d:], pred)
    rho, _ = spearmanr(pred, y[split_d:])
    models[target_name] = {"coefs": c.tolist(), "intercept": float(ic),
                            "mae": float(mae), "r2": float(r2), "spearman": float(rho)}

# Train histogram bin models
bin_models = []
Y_hist = np.array([r["hist"] for r in daily_records])
for i in range(len(bin_labels)):
    y = Y_hist[:, i]
    c, ic = train_ridge(X_daily[:split_d], y[:split_d], w_d)
    bin_models.append({"coefs": c.tolist(), "intercept": float(ic)})

# Consumer histogram models
Y_cons = np.array([r["cons_hist"] for r in daily_records])
cons_bin_models = []
for i in range(len(cons_labels)):
    y = Y_cons[:, i]
    c, ic = train_ridge(X_daily[:split_d], y[:split_d], w_d)
    cons_bin_models.append({"coefs": c.tolist(), "intercept": float(ic)})

# Use last 7 days of test data as example forecast
last_7 = daily_records[-7:]

# Generate predictions for last 7 days
pred_7 = []
for r in last_7:
    x = np.array(r["x_mean"] + [r["se3_mean"]/100, r["se1_mean"]/100])
    preds = {}
    for name, m in models.items():
        preds[name] = float(x @ np.array(m["coefs"]) + m["intercept"])
    # Predicted histogram
    pred_hist = []
    for bm in bin_models:
        v = float(x @ np.array(bm["coefs"]) + bm["intercept"])
        pred_hist.append(max(0, v))
    total = sum(pred_hist) or 1
    pred_hist = [v / total for v in pred_hist]
    # Consumer histogram
    pred_cons = []
    for cm in cons_bin_models:
        v = float(x @ np.array(cm["coefs"]) + cm["intercept"])
        pred_cons.append(max(0, v))
    total_c = sum(pred_cons) or 1
    pred_cons = [v / total_c for v in pred_cons]

    pred_7.append({
        "date": r["date"],
        "weekday": r["weekday"],
        "actual": r,
        "predicted": preds,
        "pred_hist": pred_hist,
        "actual_hist": r["hist"],
        "pred_cons_hist": pred_cons,
        "actual_cons_hist": r["cons_hist"],
    })

# Performance summary
print("\nModel Performance:")
print("%-15s %6s %6s %6s" % ("Target", "MAE", "R2", "Spear"))
for name, m in models.items():
    print("%-15s %6.2f %6.4f %6.4f" % (name, m["mae"], m["r2"], m["spearman"]))

# ================================================================
# BUILD HTML PROTOTYPE
# ================================================================

# Color scales
def price_color(price):
    """Green for cheap, yellow for moderate, red for expensive."""
    if price < 3: return "#22c55e"
    if price < 8: return "#84cc16"
    if price < 15: return "#eab308"
    if price < 25: return "#f97316"
    if price < 40: return "#ef4444"
    return "#dc2626"

def bin_color(i, n):
    """Color scale from green to red for histogram bins."""
    colors = ["#22c55e", "#4ade80", "#86efac", "#a3e635", "#facc15",
              "#fbbf24", "#f97316", "#ef4444", "#dc2626", "#b91c1c", "#7f1d1d"]
    return colors[min(i, len(colors) - 1)]

# Prepare JSON data for the HTML
chart_data = {
    "predictions": pred_7,
    "bin_labels": bin_labels,
    "cons_labels": cons_labels,
    "cons_bins": cons_bins,
    "price_bins": price_bins,
    "models": {k: {"mae": v["mae"], "r2": v["r2"], "spearman": v["spearman"]}
               for k, v in models.items()},
}

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>7-Day Price Histogram Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0f172a; color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 20px;
}
h1 { color: #f8fafc; margin-bottom: 5px; font-size: 24px; }
h2 { color: #94a3b8; margin: 30px 0 15px; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
h3 { color: #cbd5e1; margin: 15px 0 10px; font-size: 14px; }
.subtitle { color: #64748b; font-size: 13px; margin-bottom: 20px; }

.grid-7 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; margin: 15px 0; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0; }

.day-card {
    background: #1e293b; border-radius: 8px; padding: 12px;
    border: 1px solid #334155; text-align: center;
}
.day-card.expensive { border-color: #ef4444; }
.day-card.cheap { border-color: #22c55e; }
.day-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; }
.day-date { font-size: 12px; color: #64748b; }
.day-price { font-size: 20px; font-weight: 700; margin: 6px 0; }
.day-detail { font-size: 11px; color: #94a3b8; line-height: 1.6; }
.day-detail .cheap { color: #22c55e; }
.day-detail .expensive { color: #ef4444; }

.heatmap-container { overflow-x: auto; }
.heatmap { width: 100%; border-collapse: collapse; margin: 10px 0; }
.heatmap th { padding: 6px 8px; font-size: 11px; color: #94a3b8; text-align: center; }
.heatmap td {
    padding: 4px 6px; text-align: center; font-size: 11px; font-weight: 600;
    border: 1px solid #0f172a; min-width: 50px;
}
.heatmap .bin-label { text-align: right; color: #94a3b8; font-weight: 400; border: none; }

.chart-container { background: #1e293b; border-radius: 8px; padding: 15px; margin: 10px 0; }
canvas { max-height: 350px; }

.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; }
.metric { background: #1e293b; border-radius: 8px; padding: 12px; text-align: center; }
.metric-value { font-size: 22px; font-weight: 700; color: #f8fafc; }
.metric-label { font-size: 11px; color: #64748b; margin-top: 4px; }
.metric-sub { font-size: 11px; color: #94a3b8; }

.ridgeline-row { display: flex; align-items: center; margin: 2px 0; }
.ridgeline-label { width: 80px; text-align: right; padding-right: 10px; font-size: 12px; color: #94a3b8; }
.ridgeline-bar { height: 30px; display: flex; }
.ridgeline-segment { height: 100%; transition: width 0.3s; }

.legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #94a3b8; }
.legend-swatch { width: 12px; height: 12px; border-radius: 2px; }
</style>
</head>
<body>
<h1>7-Day Electricity Price Distribution Forecast</h1>
<p class="subtitle">Histogram-based analysis showing predicted price distributions for each day. Colors represent price ranges — green is cheap, red is expensive.</p>

<h2>Daily Overview</h2>
<div class="grid-7" id="day-cards"></div>

<h2>Price Distribution Heatmap (Consumer c/kWh)</h2>
<p class="subtitle">Each cell shows the fraction of hours predicted in that price range. Darker = more hours.</p>
<div class="heatmap-container">
<table class="heatmap" id="heatmap-cons"></table>
</div>

<h2>Price Distribution Heatmap (Spot EUR/MWh)</h2>
<div class="heatmap-container">
<table class="heatmap" id="heatmap-spot"></table>
</div>

<h2>Ridgeline: Predicted vs Actual Distribution</h2>
<p class="subtitle">Top bar: predicted, Bottom bar: actual. Width shows fraction of hours in each price range.</p>
<div id="ridgeline"></div>

<div class="grid-2">
<div>
<h2>Fan Chart: Daily Price Range</h2>
<div class="chart-container"><canvas id="fan-chart"></canvas></div>
</div>
<div>
<h2>Daily Statistics Comparison</h2>
<div class="chart-container"><canvas id="stats-chart"></canvas></div>
</div>
</div>

<h2>Model Performance</h2>
<div class="metric-grid" id="metrics"></div>

<script>
const DATA = """ + json.dumps(chart_data) + """;

const COLORS = ['#22c55e','#4ade80','#86efac','#a3e635','#facc15',
                '#fbbf24','#f97316','#ef4444','#dc2626','#b91c1c','#7f1d1d'];
const CONS_COLORS = ['#22c55e','#4ade80','#86efac','#a3e635','#facc15',
                     '#fbbf24','#f97316','#ef4444','#dc2626','#b91c1c','#7f1d1d'];

function toConsumer(spot) {
    return (spot / 1000 + 0.0361 + 0.02325) * 1.255 * 100;
}

// Day cards
const cardsEl = document.getElementById('day-cards');
DATA.predictions.forEach(p => {
    const mean = p.predicted.mean;
    const cheapest = Math.max(0, p.predicted.cheapest_4h);
    const expens = Math.max(0, p.predicted.expensive_4h);
    const cheapHrs = Math.round(Math.max(0, p.predicted.cheap_hours));
    const consMean = toConsumer(mean).toFixed(1);
    const consCheap = toConsumer(cheapest).toFixed(1);
    const consExp = toConsumer(expens).toFixed(1);
    const cls = mean > 15 ? 'expensive' : mean < 3 ? 'cheap' : '';

    cardsEl.innerHTML += `
    <div class="day-card ${cls}">
        <div class="day-label">${p.weekday}</div>
        <div class="day-date">${p.date.substring(5)}</div>
        <div class="day-price" style="color:${mean<5?'#22c55e':mean<12?'#facc15':'#ef4444'}">${consMean}c</div>
        <div class="day-detail">
            <span class="cheap">Best 4h: ${consCheap}c</span><br>
            <span class="expensive">Peak 4h: ${consExp}c</span><br>
            Cheap hrs: ${cheapHrs}/24
        </div>
    </div>`;
});

// Consumer heatmap
function buildHeatmap(tableId, labels, dataKey, colors) {
    const table = document.getElementById(tableId);
    let html = '<tr><th></th>';
    DATA.predictions.forEach(p => { html += '<th>' + p.weekday + '<br>' + p.date.substring(5) + '</th>'; });
    html += '</tr>';
    labels.forEach((label, i) => {
        html += '<tr><td class="bin-label">' + label + '</td>';
        DATA.predictions.forEach(p => {
            const frac = p[dataKey][i] || 0;
            const pct = (frac * 100).toFixed(0);
            const alpha = Math.min(1, frac * 3 + 0.05);
            const color = colors[Math.min(i, colors.length - 1)];
            html += '<td style="background:' + color + ';opacity:' + alpha.toFixed(2) + '">' +
                    (frac > 0.03 ? pct + '%' : '') + '</td>';
        });
        html += '</tr>';
    });
    table.innerHTML = html;
}

buildHeatmap('heatmap-cons', DATA.cons_labels, 'pred_cons_hist', CONS_COLORS);
buildHeatmap('heatmap-spot', DATA.bin_labels, 'pred_hist', COLORS);

// Ridgeline (stacked horizontal bars)
const ridgeEl = document.getElementById('ridgeline');
DATA.predictions.forEach(p => {
    function makeBar(hist, label) {
        let html = '<div class="ridgeline-row"><div class="ridgeline-label">' + label + '</div><div class="ridgeline-bar" style="width:calc(100% - 90px)">';
        hist.forEach((frac, i) => {
            if (frac > 0.01) {
                html += '<div class="ridgeline-segment" style="width:' + (frac*100).toFixed(1) +
                        '%;background:' + COLORS[Math.min(i, COLORS.length-1)] +
                        '" title="' + DATA.bin_labels[i] + ': ' + (frac*100).toFixed(0) + '%"></div>';
            }
        });
        html += '</div></div>';
        return html;
    }
    ridgeEl.innerHTML += '<h3>' + p.weekday + ' ' + p.date.substring(5) + '</h3>';
    ridgeEl.innerHTML += makeBar(p.pred_hist, 'Predicted');
    ridgeEl.innerHTML += makeBar(p.actual_hist, 'Actual');
});

// Legend
let legendHtml = '<div class="legend">';
DATA.bin_labels.forEach((label, i) => {
    legendHtml += '<div class="legend-item"><div class="legend-swatch" style="background:' +
                  COLORS[Math.min(i, COLORS.length-1)] + '"></div>' + label + '</div>';
});
legendHtml += '</div>';
ridgeEl.innerHTML = legendHtml + ridgeEl.innerHTML;

// Fan chart
const fanCtx = document.getElementById('fan-chart').getContext('2d');
const days = DATA.predictions.map(p => p.weekday + ' ' + p.date.substring(5));
new Chart(fanCtx, {
    type: 'line',
    data: {
        labels: days,
        datasets: [
            { label: 'p90', data: DATA.predictions.map(p => toConsumer(p.actual.p90)),
              borderColor: '#ef444466', backgroundColor: '#ef444422', fill: '+1', tension: 0.3, pointRadius: 0 },
            { label: 'p75', data: DATA.predictions.map(p => toConsumer(p.actual.p75)),
              borderColor: '#f9731666', backgroundColor: '#f9731622', fill: '+1', tension: 0.3, pointRadius: 0 },
            { label: 'Mean', data: DATA.predictions.map(p => toConsumer(p.actual.mean)),
              borderColor: '#facc15', backgroundColor: '#facc1533', fill: '+1', tension: 0.3, borderWidth: 2 },
            { label: 'p25', data: DATA.predictions.map(p => toConsumer(p.actual.p25)),
              borderColor: '#22c55e66', backgroundColor: '#22c55e22', fill: false, tension: 0.3, pointRadius: 0 },
            { label: 'Min', data: DATA.predictions.map(p => toConsumer(p.actual.min)),
              borderColor: '#22c55e33', fill: false, tension: 0.3, pointRadius: 0, borderDash: [3,3] },
            { label: 'Pred Mean', data: DATA.predictions.map(p => toConsumer(p.predicted.mean)),
              borderColor: '#60a5fa', borderWidth: 2, borderDash: [5,3], fill: false, tension: 0.3 },
        ]
    },
    options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
        scales: {
            y: { title: { display: true, text: 'c/kWh', color: '#94a3b8' },
                 grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
            x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
        }
    }
});

// Stats comparison chart
const statsCtx = document.getElementById('stats-chart').getContext('2d');
new Chart(statsCtx, {
    type: 'bar',
    data: {
        labels: days,
        datasets: [
            { label: 'Cheapest 4h (actual)', data: DATA.predictions.map(p => toConsumer(p.actual.cheapest_4h)),
              backgroundColor: '#22c55e88', borderColor: '#22c55e', borderWidth: 1 },
            { label: 'Cheapest 4h (predicted)', data: DATA.predictions.map(p => toConsumer(Math.max(0, p.predicted.cheapest_4h))),
              backgroundColor: '#22c55e33', borderColor: '#22c55e', borderWidth: 1, borderDash: [3,3] },
            { label: 'Peak 4h (actual)', data: DATA.predictions.map(p => toConsumer(p.actual.expensive_4h)),
              backgroundColor: '#ef444488', borderColor: '#ef4444', borderWidth: 1 },
            { label: 'Peak 4h (predicted)', data: DATA.predictions.map(p => toConsumer(Math.max(0, p.predicted.expensive_4h))),
              backgroundColor: '#ef444433', borderColor: '#ef4444', borderWidth: 1, borderDash: [3,3] },
        ]
    },
    options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
        scales: {
            y: { title: { display: true, text: 'c/kWh', color: '#94a3b8' },
                 grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
            x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
        }
    }
});

// Metrics
const metricsEl = document.getElementById('metrics');
[['Cheapest 4h', 'cheapest_4h'], ['Daily Mean', 'mean'],
 ['Peak 4h', 'expensive_4h'], ['Cheap Hours', 'cheap_hours']].forEach(([label, key]) => {
    const m = DATA.models[key];
    metricsEl.innerHTML += `
    <div class="metric">
        <div class="metric-value">${m.spearman ? (m.spearman * 100).toFixed(0) : '?'}%</div>
        <div class="metric-label">${label}</div>
        <div class="metric-sub">Rank correlation</div>
        <div class="metric-sub">MAE: ${m.mae.toFixed(2)} | R²: ${m.r2.toFixed(3)}</div>
    </div>`;
});
</script>
</body>
</html>"""

# Save HTML
output_path = "output/histogram_forecast.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)
print("\nHTML prototype saved to: %s" % output_path)
print("Open: file:///C:/Users/matti/OneDrive/Documents/GitHub/watti-matti/HA-spot-price-predictor/output/histogram_forecast.html")
