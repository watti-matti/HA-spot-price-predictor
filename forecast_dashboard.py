"""Forecast Dashboard — 7-day electricity price forecast visualization.

Fetches live weather forecasts, runs both hourly and duration models,
and generates an interactive HTML dashboard showing:
  1. D(k) forecast line chart: daily cost for cheapest k hours
  2. Hourly price forecast with day/night shading
  3. Daily summary cards with cheapest block info

Usage:
    python forecast_dashboard.py [--region finland]
    Output: output/forecast.html

Designed for Home Assistant iframe embedding (dark theme, responsive).
"""
import json
import math
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
import yaml

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
region_name = "finland"
if "--region" in sys.argv:
    idx = sys.argv.index("--region")
    if idx + 1 < len(sys.argv):
        region_name = sys.argv[idx + 1]

config_path = Path("config/regions") / f"{region_name}.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

region = config.get("region", {})
TZ = ZoneInfo(region.get("timezone", "Europe/Helsinki"))
FORECAST_DAYS = 7
FORECAST_HOURS = FORECAST_DAYS * 24

# Consumer pricing
cons_cfg = config.get("consumer_pricing", {})
_vat = cons_cfg.get("vat_multiplier", 1.255)
_tax = cons_cfg.get("energy_tax_eur_kwh", 0.02325)
_seller = cons_cfg.get("seller_margin_eur_kwh", 0.0)
_default_op = cons_cfg.get("default_operator", "Elenia")
_operators = {op["name"]: op for op in cons_cfg.get("operators", [])}
_day_transfer = _operators.get(_default_op, {}).get("day_rate_eur_kwh", 0.0361)
_night_transfer = _operators.get(_default_op, {}).get("night_rate_eur_kwh", _day_transfer)


def to_cons(spot_eur_mwh, is_night=False):
    """Convert spot EUR/MWh to consumer c/kWh with day/night transfer rate."""
    transfer = _night_transfer if is_night else _day_transfer
    return (max(0.0, spot_eur_mwh) / 1000 + transfer + _tax + _seller) * _vat * 100


# Weather locations from config
locations = config.get("weather_source", {}).get("locations", [])

# ================================================================
# FETCH WEATHER FORECAST (Open-Meteo, sync)
# ================================================================
print("Fetching weather forecast from Open-Meteo...")
forecast_url = config.get("weather_source", {}).get(
    "forecast_url", "https://api.open-meteo.com/v1/forecast")

location_data = []
for loc in locations:
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "hourly": "wind_speed_120m,global_tilted_irradiance_instant,temperature_2m",
        "wind_speed_unit": "ms",
        "tilt": config.get("weather_source", {}).get("solar_tilt_deg", 45),
        "forecast_days": FORECAST_DAYS + 1,
        "timezone": "UTC",
    }
    try:
        r = requests.get(forecast_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        location_data.append({
            "wind": hourly.get("wind_speed_120m", []),
            "solar": hourly.get("global_tilted_irradiance_instant", []),
            "temp": hourly.get("temperature_2m", []),
            "times": hourly.get("time", []),
            "loc": loc,
        })
        print(f"  {loc['name']}: {len(hourly.get('time', []))} hours")
    except Exception as e:
        print(f"  WARNING: {loc['name']} failed: {e}")

if not location_data:
    print("ERROR: No weather data fetched")
    sys.exit(1)

# Aggregate: capacity-weighted average
wind_w_sum = sum(ld["loc"].get("wind_weight", 0) for ld in location_data)
solar_w_sum = sum(ld["loc"].get("solar_weight", 0) for ld in location_data)
temp_w_sum = sum(ld["loc"].get("temp_weight", 0) for ld in location_data)

n_hours = min(len(ld["wind"]) for ld in location_data)
n_hours = min(n_hours, FORECAST_HOURS + 24)

weather_hours = []
utc_times = []
for i in range(n_hours):
    wind_val = sum(
        (ld["wind"][i] or 0) * ld["loc"].get("wind_weight", 0)
        for ld in location_data) / max(wind_w_sum, 1e-6)
    solar_val = sum(
        (ld["solar"][i] or 0) * ld["loc"].get("solar_weight", 0)
        for ld in location_data) / max(solar_w_sum, 1e-6)
    temp_val = sum(
        (ld["temp"][i] or 0) * ld["loc"].get("temp_weight", 0)
        for ld in location_data) / max(temp_w_sum, 1e-6)
    weather_hours.append({
        "wind": wind_val,
        "solar": solar_val,
        "temp": temp_val,
    })
    utc_times.append(location_data[0]["times"][i])

print(f"  Aggregated: {len(weather_hours)} hours from {len(location_data)} locations")

# ================================================================
# FETCH NEIGHBOR PRICES (Elpriset for SE1/SE3, Elering for EE)
# ================================================================
print("Fetching neighbor prices...")
neighbor_latest = {}

# Elpriset SE1/SE3 — fetch today + tomorrow
for zone in ["SE1", "SE3"]:
    prefix = zone.lower()
    prices = []
    for day_offset in range(-2, 2):
        dt = datetime.now(tz=timezone.utc) + timedelta(days=day_offset)
        url = f"https://www.elprisetjustnu.se/api/v1/prices/{dt.year}/{dt.strftime('%m-%d')}_{zone}.json"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for entry in r.json():
                    prices.append({
                        "time": entry["time_start"],
                        "price": entry["EUR_per_kWh"] * 1000,
                    })
        except Exception:
            pass
    if prices:
        neighbor_latest[prefix] = prices
        print(f"  {zone}: {len(prices)} hours")

# Elering EE
try:
    start = (datetime.now(tz=timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00.000Z")
    end = (datetime.now(tz=timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT00:00:00.000Z")
    r = requests.get("https://dashboard.elering.ee/api/nps/price",
                      params={"start": start, "end": end}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        ee_data = data.get("data", {}).get("ee", [])
        prices = [{"time": e["timestamp"], "price": e["price"]} for e in ee_data if e.get("price") is not None]
        if prices:
            neighbor_latest["ee"] = prices
            print(f"  EE: {len(prices)} hours")
except Exception as e:
    print(f"  EE failed: {e}")

# ================================================================
# FETCH ACTUAL SPOT PRICES (Sahkotin, 72h) & COMPUTE ACTUAL D(k)
# ================================================================
print("Fetching actual spot prices from Sahkotin...")
actual_dk = []
try:
    # Fetch past 3 days + today (Nordpool day-ahead prices are known)
    # Extra day needed because UTC->local timezone shift can drop first day
    _hist_start = (datetime.now(tz=timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00.000Z")
    _hist_end = (datetime.now(tz=timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z")
    r = requests.get("https://sahkotin.fi/prices",
                      params={"start": _hist_start, "end": _hist_end}, timeout=15)
    r.raise_for_status()
    raw_prices = r.json()
    if isinstance(raw_prices, dict) and "prices" in raw_prices:
        raw_prices = raw_prices["prices"]

    # Group by local date
    actual_by_date = {}
    for entry in raw_prices:
        ts_str = entry.get("date") or entry.get("timestamp", "")
        price_eur_mwh = float(entry.get("value", 0.0))
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            local_dt = ts.astimezone(TZ)
            date_str = local_dt.strftime("%Y-%m-%d")
            if date_str not in actual_by_date:
                actual_by_date[date_str] = {}
            actual_by_date[date_str][local_dt.hour] = price_eur_mwh
        except Exception:
            pass

    # Compute D(k) for each complete day (24 hours)
    # Include today — Nordpool day-ahead prices are published by 14:00 yesterday
    tomorrow_str = (datetime.now(tz=TZ) + timedelta(days=1)).strftime("%Y-%m-%d")
    for date_str in sorted(actual_by_date.keys()):
        hours_map = actual_by_date[date_str]
        if len(hours_map) < 24:
            continue
        # Skip future days (tomorrow onward — those are forecasts)
        if date_str >= tomorrow_str:
            continue

        # Sort 24 hourly prices ascending
        sorted_prices = sorted(hours_map[h] for h in range(24))
        running = 0.0
        dk_curve = []
        for i, p in enumerate(sorted_prices):
            running += p
            dk_curve.append(running / (i + 1))

        # Compute consumer D(k) directly from hourly consumer prices (with day/night tariff)
        consumer_by_hour = [to_cons(hours_map[h], is_night=(h < 7 or h >= 22)) for h in range(24)]
        sorted_cons = sorted(consumer_by_hour)
        running_cons = 0.0
        dk_cons_curve = []
        for i, cp in enumerate(sorted_cons):
            running_cons += cp
            dk_cons_curve.append(running_cons / (i + 1))

        dow = datetime.strptime(date_str, "%Y-%m-%d").weekday()
        actual_dk.append({
            "date": date_str,
            "dow": dow,
            "source": "actual",
            "dk": [round(v, 2) for v in dk_curve],
            "dk_cons": [round(v, 2) for v in dk_cons_curve],
            "hourly_sorted_cons": [round(v, 2) for v in sorted_cons],
            "d1": round(dk_cons_curve[0], 2),
            "d4": round(dk_cons_curve[3], 2) if len(dk_cons_curve) > 3 else 0,
            "d8": round(dk_cons_curve[7], 2) if len(dk_cons_curve) > 7 else 0,
            "d24": round(dk_cons_curve[23], 2) if len(dk_cons_curve) > 23 else 0,
        })
    print(f"  {len(actual_dk)} actual D(k) days: {[d['date'] for d in actual_dk]}")
except Exception as e:
    print(f"  Sahkotin fetch failed: {e}")

# ================================================================
# LOAD MODEL & BUILD AR FORECASTS
# ================================================================
print("Loading model...")
coefs_path = Path("output/model_coefs.json")
with open(coefs_path) as f:
    model_coefs = json.load(f)

# Build holiday set early (needed by AR loop and feature computation)
from src.holidays import build_holiday_set
now = datetime.now(tz=timezone.utc)
holidays = build_holiday_set(config, now.year - 1, now.year + 2)

# Build proper AR(2) neighbor forecasts using hourly profiles + deviation tracking
ar_models = model_coefs.get("ar_models", {})
ar_hourly = {}  # prefix -> list of forecasted prices (one per forecast hour)

for prefix in ["se1", "se3", "ee"]:
    if prefix not in ar_models:
        continue
    m = ar_models[prefix]
    profile_wd = m["profile_wd"]  # 24 hourly baseline values for workdays
    profile_we = m["profile_we"]  # 24 hourly baseline values for weekends
    ar_c = m["ar_coefs"]          # [phi1, phi2] for AR(2)

    # Get recent actual prices to initialize AR deviation
    actual_prices = neighbor_latest.get(prefix, [])
    # Build time-indexed lookup of actual prices
    actual_by_hour = {}
    for p in actual_prices:
        t = p["time"]
        # Parse various timestamp formats
        try:
            if isinstance(t, (int, float)):
                dt = datetime.fromtimestamp(t, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            local_dt = dt.astimezone(TZ)
            key = local_dt.strftime("%Y-%m-%d-%H")
            actual_by_hour[key] = p["price"]
        except Exception:
            pass

    # Compute deviations from last 2 known hours
    dev_t1 = 0.0  # deviation at t-1
    dev_t2 = 0.0  # deviation at t-2
    now_local = datetime.now(tz=TZ)
    for lookback in range(48, 0, -1):
        check_dt = now_local - timedelta(hours=lookback)
        key = check_dt.strftime("%Y-%m-%d-%H")
        if key in actual_by_hour:
            h = check_dt.hour
            is_wd = 1 if check_dt.weekday() < 5 and check_dt.strftime("%Y-%m-%d") not in holidays else 0
            profile_val = profile_wd[h] if is_wd else profile_we[h]
            dev = actual_by_hour[key] - profile_val
            dev_t2 = dev_t1
            dev_t1 = dev

    # Generate AR forecast for each hour
    forecasts = []
    ar_start_utc = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    for i in range(FORECAST_HOURS):
        utc_dt = ar_start_utc + timedelta(hours=i)
        local_dt = utc_dt.astimezone(TZ)
        h = local_dt.hour
        dow = local_dt.weekday()
        date_str = local_dt.strftime("%Y-%m-%d")
        is_wd = 1 if dow < 5 and date_str not in holidays else 0

        profile_val = profile_wd[h] if is_wd else profile_we[h]
        ar_dev = ar_c[0] * dev_t1 + ar_c[1] * dev_t2
        predicted = profile_val + ar_dev
        forecasts.append(predicted)

        # Shift deviations forward
        dev_t2 = dev_t1
        dev_t1 = ar_dev

    ar_hourly[prefix] = forecasts
    print(f"  AR {prefix.upper()}: profile+deviation, range {min(forecasts):.1f}-{max(forecasts):.1f} EUR/MWh")

# Also compute export potential from recent FI vs SE3 spread
fi_mean = 0.0  # We don't have FI actual here, approximate as 0
export_potential_se3 = 0.0  # Will be overridden per hour if data available

# Feature computation (pure Python, same as HA features.py)
demand = config.get("demand", {})
feat_cfg = config.get("features", {})
hdd_threshold = demand.get("hdd_threshold", 17.0)
wind_scarcity_base = feat_cfg.get("wind_log_scarcity_base", 8.0)
wind_calm_thresh = feat_cfg.get("wind_calm_threshold", 6.0)
wind_low_thresh = feat_cfg.get("wind_low_threshold", 5.0)
am_center = demand.get("peak_am_center", 9)
am_sigma = demand.get("peak_am_sigma", 1.8)
pm_center = demand.get("peak_pm_center", 19)
pm_sigma = demand.get("peak_pm_sigma", 2.0)
ar_divisor = feat_cfg.get("ar_normalize_divisor", 100)

def gauss_bump(h, center, sigma):
    return math.exp(-0.5 * ((h - center) / sigma) ** 2)


def compute_features(utc_dt, wind, solar, temp, hour_idx=0):
    """Compute feature dict for one forecast hour."""
    local_dt = utc_dt.astimezone(TZ)
    h = local_dt.hour
    mo = local_dt.month
    dow = local_dt.weekday()
    date_str = local_dt.strftime("%Y-%m-%d")

    is_holiday_val = 1.0 if date_str in holidays else 0.0
    is_workday = 1.0 if (dow < 5 and is_holiday_val == 0.0) else 0.0

    pi = math.pi
    hour_sin = math.sin(2 * pi * h / 24)
    hour_cos = math.cos(2 * pi * h / 24)
    month_sin = math.sin(2 * pi * mo / 12)
    month_cos = math.cos(2 * pi * mo / 12)

    hdd = max(0.0, hdd_threshold - temp)
    hdd_sq = hdd ** 2

    raw_am = gauss_bump(h, am_center, am_sigma)
    raw_pm = gauss_bump(h, pm_center, pm_sigma)
    double_peak_am = raw_am * is_workday
    double_peak_pm = raw_pm * is_workday

    wind_log_scarcity = math.log1p(max(0.0, wind_scarcity_base - wind))
    wind_calm = max(0.0, wind_calm_thresh - wind)

    features = {
        "wind_speed_weighted": wind,
        "solar_irradiance_weighted": solar,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_holiday": is_holiday_val,
        "hdd_sq": hdd_sq,
        "wind_log_scarcity": wind_log_scarcity,
        "wind_calm_x_peak_am": wind_calm * double_peak_am,
        "wind_calm_x_peak_pm": wind_calm * double_peak_pm,
    }

    # Cross-border: AR neighbor prices (proper hourly AR(2) forecast)
    for prefix in ["se1", "se3", "ee"]:
        if prefix in ar_hourly and hour_idx < len(ar_hourly[prefix]):
            features[f"ar_{prefix}"] = ar_hourly[prefix][hour_idx] / ar_divisor
        else:
            features[f"ar_{prefix}"] = 0.4  # fallback ~40 EUR/MWh

    # Export potential SE3 (from recent spread)
    features["export_potential_se3"] = export_potential_se3

    # Nuclear: use reasonable default if no Fingrid key
    features["nuclear_deficit"] = 0.05  # ~95% capacity typical
    low_wind = max(0.0, wind_low_thresh - wind)
    peak_full = max(raw_am, raw_pm)
    features["nuclear_x_scarcity"] = 0.05 * low_wind * hdd * peak_full

    return features


# ================================================================
# RUN HOURLY MODEL
# ================================================================
print("Running hourly model...")
feature_names = model_coefs["feature_names"]
feature_coefs = np.array([f["coef"] for f in model_coefs["features"]])
intercept = model_coefs["intercept"]
log_offset = model_coefs.get("log_offset", 55)
power_scale = model_coefs.get("power_scale", 1.0)
power_exp = model_coefs.get("power_exp", 1.0)

hourly_forecast = []
start_utc = datetime.fromisoformat(utc_times[0].replace("Z", "+00:00"))
if start_utc.tzinfo is None:
    start_utc = start_utc.replace(tzinfo=timezone.utc)

for i in range(min(FORECAST_HOURS, len(weather_hours))):
    utc_dt = start_utc + timedelta(hours=i)
    w = weather_hours[i]
    feat = compute_features(utc_dt, w["wind"], w["solar"], w["temp"], hour_idx=i)

    # Predict
    x = np.array([feat.get(fn, 0.0) for fn in feature_names])
    linear = float(x @ feature_coefs + intercept)
    raw = max(0.0, math.exp(min(linear, 20.0)) - log_offset)
    price = power_scale * raw ** power_exp if raw > 0 else 0.0

    local_dt = utc_dt.astimezone(TZ)
    hourly_forecast.append({
        "utc": utc_dt.isoformat(),
        "local": local_dt.strftime("%Y-%m-%d %H:%M"),
        "local_date": local_dt.strftime("%Y-%m-%d"),
        "local_hour": local_dt.hour,
        "dow": local_dt.weekday(),
        "price_eur_mwh": round(price, 2),
        "price_cons": round(to_cons(price, is_night=(local_dt.hour < 7 or local_dt.hour >= 22)), 2),
        "wind": round(w["wind"], 1),
        "solar": round(w["solar"], 0),
        "temp": round(w["temp"], 1),
    })

# Mark forecast entries
for hf in hourly_forecast:
    hf["source"] = "forecast"

# Prepend actual hourly prices from Sahkotin (already fetched in actual_by_date)
actual_hourly = []
if actual_by_date:
    for date_str in sorted(actual_by_date.keys()):
        hours_map = actual_by_date[date_str]
        if len(hours_map) < 24:
            continue
        today_str = datetime.now(tz=TZ).strftime("%Y-%m-%d")
        if date_str >= today_str:
            continue
        for h in range(24):
            spot = hours_map.get(h, 0.0)
            local_dt = datetime.strptime(f"{date_str} {h:02d}:00", "%Y-%m-%d %H:%M")
            local_dt = local_dt.replace(tzinfo=TZ)
            utc_dt = local_dt.astimezone(timezone.utc)
            actual_hourly.append({
                "utc": utc_dt.isoformat(),
                "local": local_dt.strftime("%Y-%m-%d %H:%M"),
                "local_date": date_str,
                "local_hour": h,
                "dow": local_dt.weekday(),
                "price_eur_mwh": round(spot, 2),
                "price_cons": round(to_cons(spot, is_night=(h < 7 or h >= 22)), 2),
                "wind": 0,
                "solar": 0,
                "temp": 0,
                "source": "actual",
            })

if actual_hourly:
    # Remove any forecast entries that overlap with actual dates
    actual_dates = {h["local_date"] for h in actual_hourly}
    hourly_forecast = [h for h in hourly_forecast if h["local_date"] not in actual_dates]
    hourly_forecast = actual_hourly + hourly_forecast

print(f"  {len(hourly_forecast)} total hourly entries ({len(actual_hourly)} actual + {len(hourly_forecast) - len(actual_hourly)} forecast)")

# ================================================================
# BUILD DAILY D(k) DURATION CURVES
# ================================================================
print("Computing duration curves...")
dur_cfg = config.get("duration_model", {})
dur_data = model_coefs.get("duration_model")

daily_dk = []
if dur_data:
    segments = dur_data.get("segments", {})
    dur_features = dur_data.get("feature_names", [])
    dur_log_offset = dur_data.get("log_offset", 55)
    dur_exp_cap = dur_data.get("exp_cap", 20.0)
    seg_defs = dur_cfg.get("segments", {
        "night":   [22, 23, 0, 1, 2, 3, 4, 5, 6],  # aligned with night tariff 22-07
        "morning": [7, 8, 9, 10, 11],
        "midday":  [12, 13, 14, 15, 16, 17],
        "evening": [18, 19, 20, 21],
    })

    # Group forecast hours by date
    by_date = {}
    for fh in hourly_forecast:
        d = fh["local_date"]
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(fh)

    # Dates with actual D(k) — skip these in forecast D(k) computation
    actual_dk_dates = {d["date"] for d in actual_dk}

    for date_str in sorted(by_date.keys()):
        # Skip dates that have actual D(k) from Sahkotin
        if date_str in actual_dk_dates:
            continue

        day_hours = by_date[date_str]
        if len(day_hours) < 20:
            continue

        hour_lookup = {h["local_hour"]: h for h in day_hours}
        is_wd = 1.0 if day_hours[0]["dow"] < 5 else 0.0
        mo = int(date_str.split("-")[1])

        all_pred_prices = []

        for seg_name, seg_hours_list in seg_defs.items():
            if seg_name not in segments:
                continue
            seg_models = segments[seg_name]["models"]
            n_levels = len(seg_models)

            # Compute segment features
            seg_hrs = [hour_lookup.get(h) for h in seg_hours_list if h in hour_lookup]
            if len(seg_hrs) < 2:
                continue

            wind_mean = np.mean([h["wind"] for h in seg_hrs])
            solar_mean = np.mean([h["solar"] for h in seg_hrs])
            temp_mean = np.mean([h["temp"] for h in seg_hrs])
            hdd_mean = max(0.0, hdd_threshold - temp_mean)

            # Compute neighbor means for this segment from AR hourly forecasts
            seg_hour_indices = []
            for sh in seg_hours_list:
                for fh in day_hours:
                    if fh["local_hour"] == sh:
                        # Find the index in hourly_forecast
                        idx_in_fc = hourly_forecast.index(fh)
                        seg_hour_indices.append(idx_in_fc)
                        break

            def ar_seg_mean(prefix, fallback=40.0):
                if prefix not in ar_hourly or not seg_hour_indices:
                    return fallback
                vals = [ar_hourly[prefix][j] for j in seg_hour_indices if j < len(ar_hourly[prefix])]
                return np.mean(vals) if vals else fallback

            seg_features = {
                "wind_mean": wind_mean,
                "solar_mean": solar_mean,
                "hdd_mean": hdd_mean,
                "se3_mean": ar_seg_mean("se3"),
                "se1_mean": ar_seg_mean("se1"),
                "nuclear_deficit": 0.05,
                "is_workday": is_wd,
                "month_sin": math.sin(2 * math.pi * mo / 12),
                "month_cos": math.cos(2 * math.pi * mo / 12),
                "wind_log_scarcity": math.log1p(max(0.0, wind_scarcity_base - wind_mean)),
            }

            # Predict D(k) for each level
            raw_dk = []
            for model in seg_models:
                linear = model["intercept"]
                for j, fname in enumerate(dur_features):
                    linear += model["coefs"][j] * seg_features.get(fname, 0.0)
                dk = max(0.0, math.exp(min(linear, dur_exp_cap)) - dur_log_offset)
                raw_dk.append(dk)

            # PAVA: enforce monotonicity
            blocks = [[v, 1] for v in raw_dk]
            merged = True
            while merged:
                merged = False
                new_blocks = [blocks[0]]
                for j in range(1, len(blocks)):
                    if new_blocks[-1][0] / new_blocks[-1][1] > blocks[j][0] / blocks[j][1]:
                        new_blocks[-1][0] += blocks[j][0]
                        new_blocks[-1][1] += blocks[j][1]
                        merged = True
                    else:
                        new_blocks.append(blocks[j])
                blocks = new_blocks
            pava_dk = []
            for bs, bc in blocks:
                pava_dk.extend([bs / bc] * int(bc))

            # Extract sorted prices from D(k)
            for k in range(len(pava_dk)):
                if k == 0:
                    all_pred_prices.append(pava_dk[0])
                else:
                    p = (k + 1) * pava_dk[k] - k * pava_dk[k - 1]
                    all_pred_prices.append(max(0.0, p))

        if not all_pred_prices:
            continue

        # Full-day D(k) from merged sorted prices
        all_pred_prices.sort()
        n_h = len(all_pred_prices)
        running = 0.0
        dk_curve = []
        for i, p in enumerate(all_pred_prices):
            running += p
            dk_curve.append(running / (i + 1))

        # Compute consumer D(k) from hourly forecast prices with day/night tariff
        hourly_cons = sorted(
            to_cons(h["price_eur_mwh"], is_night=(h["local_hour"] < 7 or h["local_hour"] >= 22))
            for h in day_hours
        )
        running_cons = 0.0
        dk_cons_curve = []
        for i, cp in enumerate(hourly_cons):
            running_cons += cp
            dk_cons_curve.append(running_cons / (i + 1))

        daily_dk.append({
            "date": date_str,
            "dow": day_hours[0]["dow"],
            "source": "forecast",
            "dk": [round(v, 2) for v in dk_curve],
            "dk_cons": [round(v, 2) for v in dk_cons_curve],
            "hourly_sorted_cons": [round(v, 2) for v in hourly_cons],
            "d1": round(dk_cons_curve[0], 2) if dk_cons_curve else 0,
            "d4": round(dk_cons_curve[3], 2) if len(dk_cons_curve) > 3 else 0,
            "d8": round(dk_cons_curve[7], 2) if len(dk_cons_curve) > 7 else 0,
            "d24": round(dk_cons_curve[min(23, len(dk_cons_curve) - 1)], 2) if dk_cons_curve else 0,
        })

    print(f"  {len(daily_dk)} forecast D(k) curves")
else:
    print("  WARNING: No duration model in model_coefs.json")

# Prepend actual D(k) before forecast, avoiding duplicate dates
forecast_dates = {d["date"] for d in daily_dk}
actual_to_prepend = [d for d in actual_dk if d["date"] not in forecast_dates]
if actual_to_prepend:
    daily_dk = actual_to_prepend + daily_dk
    print(f"  Prepended {len(actual_to_prepend)} actual D(k) days -> {len(daily_dk)} total")

# ================================================================
# BUILD HTML
# ================================================================
print("Building forecast HTML...")

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

forecast_json = json.dumps({
    "hourly": hourly_forecast,
    "daily_dk": daily_dk,
    "day_names": day_names,
    "generated": datetime.now(tz=TZ).strftime("%Y-%m-%d %H:%M"),
    "consumer": {
        "vat": _vat,
        "tax_eur_kwh": _tax,
        "transfer_day_eur_kwh": _day_transfer,
        "transfer_night_eur_kwh": _night_transfer,
        "seller_eur_kwh": _seller,
        "operator": _default_op,
    },
})

html = '''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Electricity Price Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0f172a; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:16px; }
h1 { color:#f8fafc; font-size:20px; margin-bottom:4px; }
h2 { color:#94a3b8; margin:20px 0 8px; font-size:15px; border-bottom:1px solid #334155; padding-bottom:5px; }
.sub { color:#64748b; font-size:11px; margin-bottom:10px; }
.box { background:#1e293b; border-radius:8px; padding:14px; margin:10px 0; }
canvas { width:100%!important; }

/* Daily cards */
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:10px 0; }
.card { background:#1e293b; border-radius:8px; padding:12px; text-align:center; }
.card.weekend { border:1px solid #475569; }
.card.actual { border:2px solid #22d3ee; background:#1a2535; }
.card .day { font-size:12px; color:#94a3b8; font-weight:600; }
.card .date { font-size:10px; color:#64748b; }
.card .price { font-size:22px; font-weight:700; margin:6px 0 2px; }
.card .label { font-size:9px; color:#94a3b8; }
.price-low { color:#4ade80; }
.price-mid { color:#facc15; }
.price-high { color:#f97316; }
.price-vhigh { color:#ef4444; }

/* Legend row */
.legend { display:flex; gap:16px; flex-wrap:wrap; margin:8px 0; font-size:11px; color:#94a3b8; }
.legend span { display:flex; align-items:center; gap:4px; }
.legend .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }

.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media(max-width:700px){ .grid-2{grid-template-columns:1fr;} }

footer { margin-top:24px; padding:10px 0; text-align:center; color:#475569; font-size:10px;
  border-top:1px solid #1e293b; }
footer a { color:#60a5fa; text-decoration:none; }
</style></head>
<body>

<h1>Electricity Price Forecast</h1>
<p class="sub" id="info"></p>

<h2>Weekly Overview — Daily Cost by Usage Duration</h2>
<p class="sub">D(k) = average consumer price for the k cheapest hours.
D(4) = cheapest 4h, D(8) = cheapest 8h, D(24) = daily average.</p>
<div class="cards" id="day-cards"></div>

<h2>D(k) Duration Costs (Consumer c/kWh)</h2>
<p class="sub">Actual (filled) + forecast (open) duration curve costs per day.
Lower = cheaper electricity for that usage pattern.</p>
<div class="box"><canvas id="dkChart" height="280"></canvas></div>

<h2>Duration Curve — Day Detail</h2>
<p class="sub">Click a day above to see its full 24-level D(k) curve.
X = cheapest hours used, Y = average cost.</p>
<div class="box"><canvas id="dayCurve" height="200"></canvas></div>

<h2>Hourly Consumer Price (c/kWh)</h2>
<p class="sub">Total price including spot, transfer, tax and VAT. Hover for spot EUR/MWh and weather.</p>
<div class="box"><canvas id="hourlyChart" height="250"></canvas></div>

<h2>Weather Forecast</h2>
<div class="grid-2">
  <div class="box"><canvas id="windChart" height="160"></canvas></div>
  <div class="box"><canvas id="tempChart" height="160"></canvas></div>
</div>

<footer>
  Forecast generated <span id="gen-time"></span> |
  <a href="https://github.com/watti-matti/HA-spot-price-predictor">HA Spot Price Predictor</a>
</footer>

<script>
const F = ''' + forecast_json + ''';
const H = F.hourly, DK = F.daily_dk;

function toC(s,night){ var t=night?''' + str(_night_transfer) + ''':''' + str(_day_transfer) + '''; return (Math.max(0,s)/1000+t+''' + str(_tax) + '''+''' + str(_seller) + ''')*''' + str(_vat) + '''*100; }

function priceClass(c){
  if(c<8) return 'price-low';
  if(c<10) return 'price-mid';
  if(c<14) return 'price-high';
  return 'price-vhigh';
}

// Info line
document.getElementById('info').textContent =
  'Generated: '+F.generated+' | Operator: '+F.consumer.operator+
  ' | '+H.length+' hours | Duration model: '+(DK.length?DK.length+' days':'unavailable');
document.getElementById('gen-time').textContent = F.generated;

// ── Daily cards ──
(function(){
  const el=document.getElementById('day-cards');
  DK.forEach((d,i)=>{
    const isWe=d.dow>=5;
    const isActual=d.source==='actual';
    const dayName=F.day_names[d.dow];
    const dateShort=d.date.substring(5);
    const cls=(isWe?'card weekend':'card')+(isActual?' actual':'');
    el.innerHTML+=
      '<div class="'+cls+'" onclick="showDay('+i+')" style="cursor:pointer">'+
      '<div class="day">'+dayName+(isActual?' &#9679;':'')+'</div>'+
      '<div class="date">'+dateShort+'</div>'+
      '<div class="price '+priceClass(d.d4)+'">'+d.d4.toFixed(1)+'</div>'+
      '<div class="label">D(4) 4h c/kWh</div>'+
      '<div style="margin-top:6px;font-size:10px;color:#94a3b8">'+
        '<span style="color:#22d3ee">1h:</span> '+d.d1.toFixed(1)+' &nbsp; '+
        '<span style="color:#f97316">8h:</span> '+d.d8.toFixed(1)+' &nbsp; '+
        '<span style="color:#ef4444">24h:</span> '+d.d24.toFixed(1)+
      '</div></div>';
  });
})();

// ── D(k) line chart (actual + forecast) ──
(function(){
  if(!DK.length) return;
  const labels=DK.map(d=>(d.source==='actual'?'\\u2588 ':'')+F.day_names[d.dow]+'\\n'+d.date.substring(5));
  // Point styles: filled circle for actual, open circle for forecast
  const ptStyle=DK.map(d=>d.source==='actual'?'circle':'circle');
  const ptBg=(color)=>DK.map(d=>d.source==='actual'?color:'transparent');
  const ptBorder=(color)=>DK.map(d=>color);
  const ptBw=DK.map(d=>d.source==='actual'?0:2);
  new Chart(document.getElementById('dkChart').getContext('2d'),{type:'line',
    data:{labels,datasets:[
      {label:'D(1) Cheapest 1h',data:DK.map(d=>d.d1),borderColor:'#22d3ee',borderWidth:2,pointRadius:5,
       pointBackgroundColor:ptBg('#22d3ee'),pointBorderColor:ptBorder('#22d3ee'),pointBorderWidth:ptBw,fill:false,tension:0.3},
      {label:'D(4) Cheapest 4h',data:DK.map(d=>d.d4),borderColor:'#facc15',borderWidth:2.5,pointRadius:6,
       pointBackgroundColor:ptBg('#facc15'),pointBorderColor:ptBorder('#facc15'),pointBorderWidth:ptBw,fill:false,tension:0.3},
      {label:'D(8) Cheapest 8h',data:DK.map(d=>d.d8),borderColor:'#f97316',borderWidth:2,pointRadius:5,
       pointBackgroundColor:ptBg('#f97316'),pointBorderColor:ptBorder('#f97316'),pointBorderWidth:ptBw,fill:false,tension:0.3},
      {label:'D(24) Daily Average',data:DK.map(d=>d.d24),borderColor:'#ef4444',borderWidth:2,pointRadius:5,
       pointBackgroundColor:ptBg('#ef4444'),pointBorderColor:ptBorder('#ef4444'),pointBorderWidth:ptBw,fill:false,tension:0.3},
    ]},
    options:{responsive:true,animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:11},boxWidth:14,padding:12}},
        tooltip:{backgroundColor:'#1e2433',borderColor:'#374151',borderWidth:1,
          titleColor:'#e2e8f0',bodyColor:'#e2e8f0',
          callbacks:{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(1)+' c/kWh'}}},
      scales:{
        x:{grid:{color:'#1e293b'},ticks:{color:'#e2e8f0',font:{size:10},maxRotation:0}},
        y:{title:{display:true,text:'Consumer price (c/kWh)',color:'#e2e8f0'},
          grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}});
})();

// ── Day detail curve ──
let dayCurveChart=null;
function showDay(idx){
  const d=DK[idx];
  if(!d||!d.dk_cons) return;
  const labels=d.dk_cons.map((_,i)=>(i+1)+'h');
  const data={labels,datasets:[
    {label:'Duration D(k)',data:d.dk_cons,borderColor:'#facc15',borderWidth:2.5,
     pointRadius:3,pointBackgroundColor:'#facc15',fill:false,tension:0.2},
  ]};
  // Add hourly model sorted prices if available
  if(d.hourly_sorted_cons&&d.hourly_sorted_cons.length){
    data.datasets.push({label:'Hourly model (sorted)',data:d.hourly_sorted_cons.slice(0,d.dk_cons.length).map(
      (v,i,a)=>{let s=0;for(let j=0;j<=i;j++)s+=a[j];return s/(i+1);}),
      borderColor:'#60a5fa',borderWidth:1.5,borderDash:[5,3],
      pointRadius:2,pointBackgroundColor:'#60a5fa',fill:false,tension:0.2});
  }
  const cfg={type:'line',data,
    options:{responsive:true,animation:{duration:200},
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10}}},
        title:{display:true,text:F.day_names[d.dow]+' '+d.date+' — Duration Curve',
          color:'#f8fafc',font:{size:13}}},
      scales:{x:{title:{display:true,text:'Cheapest hours used',color:'#e2e8f0'},
        grid:{color:'#1e293b'},ticks:{color:'#e2e8f0'}},
        y:{title:{display:true,text:'Avg consumer price (c/kWh)',color:'#e2e8f0'},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}};
  if(dayCurveChart){dayCurveChart.data=data;dayCurveChart.options.plugins.title.text=cfg.options.plugins.title.text;dayCurveChart.update();}
  else dayCurveChart=new Chart(document.getElementById('dayCurve').getContext('2d'),cfg);
}
if(DK.length) showDay(0);

// ── Shared x-axis config (linear scale with explicit ticks) ──
const xTickMap={};
const xGridLines=new Set();
const days=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
H.forEach((h,i)=>{
  if(h.local_hour===0){
    const d=new Date(h.local_date);
    xTickMap[i]=days[d.getDay()]+' '+h.local_date.substring(5);
    xGridLines.add(i);
  }
});
function makeXScale(){
  return {type:'linear',min:0,max:H.length-1,
    grid:{color:function(ctx){return xGridLines.has(ctx.tick.value)?'#475569':'#1e293b';}},
    afterBuildTicks:function(axis){
      axis.ticks=Object.keys(xTickMap).map(k=>({value:Number(k)}));
    },
    ticks:{color:'#e2e8f0',font:{size:9},maxRotation:0,autoSkip:false,
      callback:function(val){return xTickMap[val]||'';}
    }};
}

// ── Hourly price chart ──
(function(){
  if(!H.length) return;
  const data=H.map((h,i)=>({x:i,y:h.price_cons}));

  new Chart(document.getElementById('hourlyChart').getContext('2d'),{type:'line',
    data:{datasets:[
      {label:'Consumer price (c/kWh)',data:data,
       borderColor:'#60a5fa',borderWidth:1.5,pointRadius:0,fill:true,
       backgroundColor:'rgba(96,165,250,0.1)',tension:0.2},
    ]},
    options:{responsive:true,animation:false,
      plugins:{
        legend:{labels:{color:'#e2e8f0',font:{size:10}}},
        tooltip:{backgroundColor:'#1e2433',borderColor:'#374151',borderWidth:1,
          titleColor:'#e2e8f0',bodyColor:'#e2e8f0',
          callbacks:{
            title:function(ctx){return H[ctx[0].dataIndex].local;},
            afterLabel:function(ctx){
              const h=H[ctx.dataIndex];
              return 'Spot: '+h.price_eur_mwh+' EUR/MWh\\nWind: '+h.wind+' m/s  Temp: '+h.temp+'C';
            }
          }},
        annotation:undefined},
      scales:{
        x:makeXScale(),
        y:{title:{display:true,text:'Consumer price (c/kWh)',color:'#e2e8f0'},
          grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}});
})();

// ── Wind + Temperature charts ──
(function(){
  new Chart(document.getElementById('windChart').getContext('2d'),{type:'line',
    data:{datasets:[
      {label:'Wind (m/s)',data:H.map((h,i)=>({x:i,y:h.wind})),borderColor:'#22d3ee',borderWidth:1.5,
       pointRadius:0,fill:true,backgroundColor:'rgba(34,211,238,0.08)',tension:0.3},
    ]},
    options:{responsive:true,animation:false,
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10}}}},
      scales:{x:makeXScale(),
        y:{title:{display:true,text:'Wind speed (m/s)',color:'#e2e8f0'},
          grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}});

  new Chart(document.getElementById('tempChart').getContext('2d'),{type:'line',
    data:{datasets:[
      {label:'Temperature (\u00b0C)',data:H.map((h,i)=>({x:i,y:h.temp})),borderColor:'#f97316',borderWidth:1.5,
       pointRadius:0,fill:true,backgroundColor:'rgba(249,115,22,0.08)',tension:0.3},
    ]},
    options:{responsive:true,animation:false,
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10}}}},
      scales:{x:makeXScale(),
        y:{title:{display:true,text:'Temperature (\u00b0C)',color:'#e2e8f0'},
          grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}});
})();
</script></body></html>'''

out_path = Path("output/forecast.html")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nForecast dashboard saved to: {out_path}")
print(f"View: file:///{out_path.resolve()}")
