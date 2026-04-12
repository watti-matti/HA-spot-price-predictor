"""
API clients for all data sources, driven by region config.

Data source groups:
  Required: Spot prices (Sahkotin) + Weather (Open-Meteo)
  Cross-border (optional): Neighboring zone prices (elprisetjustnu.se, Elering)
  Nuclear (optional): Grid data (Fingrid, requires free API key)

Each fetcher reads its configuration from the region YAML and handles
retries, rate limiting, and unit conversion.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spot prices (required)
# ---------------------------------------------------------------------------

def fetch_prices(
    config: dict[str, Any],
    start: datetime,
    end: datetime,
) -> pd.Series:
    """Fetch spot prices from the configured price source.

    Returns:
        pd.Series with UTC DatetimeIndex, values in EUR/MWh.
    """
    src = config["price_source"]
    url = src["url"]
    divisor = src.get("divisor", 1)

    logger.info("Fetching prices from %s (%s -> %s)", src["name"], start.date(), end.date())
    dates, values = [], []
    chunk = start

    while chunk < end:
        chunk_end = min(chunk + timedelta(days=180), end)
        params = {
            "start": chunk.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 200:
                    break
                logger.warning("  HTTP %d from %s, retry %d/3", r.status_code, src["name"], attempt + 1)
            except requests.RequestException as e:
                logger.warning("  Request error: %s, retry %d/3", e, attempt + 1)
            time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"Failed to fetch prices from {src['name']} after 3 retries")

        pts = r.json().get("prices", [])
        for pt in pts:
            dates.append(pt["date"])
            values.append(float(pt["value"]) / divisor)

        logger.info("  %s -> %s: %d rows", chunk.date(), chunk_end.date(), len(pts))
        chunk = chunk_end
        time.sleep(0.5)

    s = pd.Series(values, index=pd.to_datetime(dates, utc=True), name="price_eur_mwh")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    logger.info("  Total: %d price rows", len(s))
    return s


# ---------------------------------------------------------------------------
# Weather data (required)
# ---------------------------------------------------------------------------

def fetch_weather(
    config: dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo and compute capacity-weighted averages.

    Args:
        config: Region config dict.
        start_date: "YYYY-MM-DD" start.
        end_date: "YYYY-MM-DD" end.

    Returns:
        DataFrame with columns: wind_speed_weighted, solar_irradiance_weighted,
        temperature_weighted. UTC DatetimeIndex.
    """
    ws = config["weather_source"]
    locations = ws["locations"]
    archive_url = ws["archive_url"]
    variables = ws.get("variables", {})
    wind_var = variables.get("wind", "wind_speed_120m")
    solar_var = variables.get("solar", "global_tilted_irradiance_instant")
    temp_var = variables.get("temperature", "temperature_2m")
    tilt = ws.get("solar_tilt_deg", 45)

    logger.info("Fetching weather from Open-Meteo (%s -> %s, %d locations)",
                start_date, end_date, len(locations))

    index = None
    wind_w = solar_w = temp_w = None
    wind_weight_sum = 0.0
    solar_weight_sum = 0.0
    temp_weight_sum = 0.0

    for loc in locations:
        name = loc["name"]
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "tilt": tilt,
            "hourly": f"{wind_var},{solar_var},{temp_var}",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "start_date": start_date,
            "end_date": end_date,
        }
        try:
            r = requests.get(archive_url, params=params, timeout=30)
            if r.status_code != 200:
                logger.warning("  %s -> HTTP %d, skipping", name, r.status_code)
                continue
        except requests.RequestException as e:
            logger.warning("  %s -> %s, skipping", name, e)
            continue

        h = r.json().get("hourly", {})
        idx = pd.to_datetime(h["time"], utc=True)
        w = np.array(h.get(wind_var, [0] * len(idx)), dtype=float)
        s = np.array(h.get(solar_var, [0] * len(idx)), dtype=float)
        t = np.array(h.get(temp_var, [0] * len(idx)), dtype=float)

        # Replace NaN with 0
        w = np.nan_to_num(w, 0.0)
        s = np.nan_to_num(s, 0.0)
        t = np.nan_to_num(t, 0.0)

        ww = loc.get("wind_weight", 0)
        sw = loc.get("solar_weight", 0)
        tw = loc.get("temp_weight", 0)

        wind_weight_sum += ww
        solar_weight_sum += sw
        temp_weight_sum += tw

        if index is None:
            index = idx
            wind_w = w * ww
            solar_w = s * sw
            temp_w = t * tw
        else:
            n = min(len(index), len(idx))
            wind_w = wind_w[:n] + w[:n] * ww
            solar_w = solar_w[:n] + s[:n] * sw
            temp_w = temp_w[:n] + t[:n] * tw
            index = index[:n]

        logger.info("  OK %s (w=%.2f, s=%.2f, t=%.2f)", name, ww, sw, tw)
        time.sleep(0.4)

    if index is None:
        raise RuntimeError("No weather data fetched from any location")

    # Normalize: divide by sum of available weights so result is a
    # proper weighted average regardless of how many locations succeeded
    df = pd.DataFrame({
        "wind_speed_weighted": wind_w / wind_weight_sum if wind_weight_sum > 0 else wind_w,
        "solar_irradiance_weighted": solar_w / solar_weight_sum if solar_weight_sum > 0 else solar_w,
        "temperature_weighted": temp_w / temp_weight_sum if temp_weight_sum > 0 else temp_w,
    }, index=index)
    df.index.name = "time_utc"
    logger.info("  Weather: %d rows (weights: wind=%.2f, solar=%.2f, temp=%.2f)",
                len(df), wind_weight_sum, solar_weight_sum, temp_weight_sum)
    return df


# ---------------------------------------------------------------------------
# Cross-border: Neighboring zone prices (optional)
# ---------------------------------------------------------------------------

def fetch_neighbor_prices(
    config: dict[str, Any],
    start: datetime,
    end: datetime,
) -> dict[str, pd.Series]:
    """Fetch neighboring zone spot prices for cross-border spread calculation.

    Returns:
        Dict mapping zone prefix (e.g., "se1") to pd.Series of EUR/MWh prices.
        Zones that fail to fetch are omitted silently.
    """
    sources = config.get("neighbor_price_sources", [])
    if not sources:
        logger.info("No neighbor price sources configured, skipping cross-border features")
        return {}

    results: dict[str, pd.Series] = {}

    for src in sources:
        name = src["name"]
        zone = src["zone"]
        prefix = src["feature_prefix"]
        factor = src.get("to_eur_mwh_factor", 1)
        url_template = src["url"]
        bulk_history = src.get("bulk_history", True)
        use_for = src.get("use_for", "both")

        # Skip sources that don't support bulk history (e.g., mgrey.se)
        if not bulk_history or use_for == "inference":
            logger.info("Skipping %s (%s) - not suitable for bulk training history", zone, name)
            continue

        logger.info("Fetching %s (%s) prices...", zone, name)

        source_type = src.get("source_type", "").lower()

        try:
            if "elering" in name.lower() or source_type == "elering":
                series = _fetch_elering(url_template, start, end, factor, zone=zone)
            elif source_type == "elpriset":
                series = _fetch_elpriset(url_template, zone, start, end, factor)
            elif "mgrey" in name.lower():
                series = _fetch_mgrey(url_template, zone, start, end, factor)
            else:
                logger.warning("  Unknown source type: %s, skipping", name)
                continue

            if series is not None and len(series) > 0:
                results[prefix] = series
                logger.info("  %s: %d hours fetched", zone, len(series))
            else:
                logger.warning("  %s: no data returned", zone)

        except Exception as e:
            logger.warning("  %s fetch failed: %s, skipping", zone, e)

    return results


def _fetch_mgrey(
    url_template: str,
    zone: str,
    start: datetime,
    end: datetime,
    factor: float,
) -> pd.Series | None:
    """Fetch Swedish prices from mgrey.se, one day at a time."""
    rows: list[dict] = []
    current = start.date() if hasattr(start, 'date') else start

    if isinstance(current, datetime):
        current = current.date()
    end_date = end.date() if isinstance(end, datetime) else end

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = url_template.replace("{date}", date_str)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if zone in data:
                    for h in data[zone]:
                        ts = pd.Timestamp(
                            f"{date_str} {h['hour']:02d}:00:00",
                            tz="Europe/Stockholm"
                        ).tz_convert("UTC")
                        rows.append({
                            "date": ts,
                            "price": h["price_eur"] * factor,
                        })
        except Exception:
            pass  # Skip failed days

        current += timedelta(days=1)
        time.sleep(0.05)

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("date")
    return df["price"].sort_index()


def _fetch_elpriset(
    url_template: str,
    zone: str,
    start: datetime,
    end: datetime,
    factor: float,
) -> pd.Series | None:
    """Fetch Swedish prices from elprisetjustnu.se, one day at a time.

    URL format: /api/v1/prices/{year}/{MM-DD}_{zone}.json
    Returns EUR_per_kWh which we multiply by factor (1000) to get EUR/MWh.
    History available from 2022.
    """
    rows: list[dict] = []
    current = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    total_days = (end_date - current).days
    fetched = 0

    while current <= end_date:
        year = current.strftime("%Y")
        mmdd = current.strftime("%m-%d")
        url = url_template.replace("{year}", year).replace("{mmdd}", mmdd).replace("{zone}", zone)

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for entry in data:
                    ts = pd.to_datetime(entry["time_start"], utc=True)
                    rows.append({
                        "date": ts,
                        "price": entry["EUR_per_kWh"] * factor,
                    })
            # 404 = date not available (before history start), skip silently
        except Exception:
            pass

        current += timedelta(days=1)
        fetched += 1
        if fetched % 100 == 0:
            logger.info("    %s: %d/%d days fetched (%d rows so far)",
                        zone, fetched, total_days, len(rows))
        time.sleep(0.03)  # Light rate limiting

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    logger.info("  elpriset %s: %d hourly rows", zone, len(df))
    return df["price"]


def _fetch_elering(
    url: str,
    start: datetime,
    end: datetime,
    factor: float,
    zone: str = "ee",
) -> pd.Series | None:
    """Fetch prices from Elering dashboard API in 90-day chunks.

    Elering provides prices for ee, fi, lv, lt zones.
    """
    all_dfs: list[pd.DataFrame] = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=90), end)
        params = {
            "start": chunk_start.strftime("%Y-%m-%dT00:00:00.000Z"),
            "end": chunk_end.strftime("%Y-%m-%dT23:59:59.999Z"),
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                logger.warning("  Elering HTTP %d for %s-%s",
                               r.status_code, chunk_start.date(), chunk_end.date())
                chunk_start = chunk_end
                continue

            data = r.json()
            zone_key = zone.lower()
            if "data" not in data or zone_key not in data["data"]:
                chunk_start = chunk_end
                continue

            rows = data["data"][zone_key]
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df["price"] = df["price"] * factor
                all_dfs.append(df[["date", "price"]].set_index("date"))

        except Exception as e:
            logger.warning("  Elering chunk error: %s", e)

        chunk_start = chunk_end
        time.sleep(0.3)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Resample to hourly if sub-hourly
    hourly = combined.resample("h").mean().dropna()
    logger.info("  Elering %s: %d hourly rows fetched", zone.upper(), len(hourly))
    return hourly["price"]


# ---------------------------------------------------------------------------
# Nuclear: Grid data (Fingrid, optional)
# ---------------------------------------------------------------------------

def fetch_grid_data(
    config: dict[str, Any],
    start: datetime,
    end: datetime,
) -> dict[str, pd.Series]:
    """Fetch grid data from sources requiring API keys.

    Returns:
        Dict mapping feature_name to pd.Series. Sources whose env_key is
        missing are silently skipped.
    """
    sources = config.get("grid_sources", [])
    if not sources:
        logger.info("No grid sources configured, skipping nuclear features")
        return {}

    results: dict[str, pd.Series] = {}

    for src in sources:
        name = src["name"]
        env_key = src["env_key"]
        api_key = os.environ.get(env_key, "").strip()
        feature_name = src["feature_name"]
        max_value = src.get("max_value", 1.0)

        if not api_key:
            logger.info("  Skipping %s (no %s env var)", name, env_key)
            continue

        logger.info("Fetching %s from Fingrid...", name)

        try:
            series = _fetch_fingrid_dataset(
                url=src["url"],
                api_key=api_key,
                start=start,
                end=end,
            )
            if series is not None and len(series) > 0:
                # Normalize: divide by max_value to get relative scale
                # Flow data can be negative (import) or positive (export)
                # Nuclear data is always positive
                series = series / max_value
                results[feature_name] = series
                logger.info("  %s: %d rows, range [%.3f, %.3f] (normalized by %d MW)",
                            name, len(series), series.min(), series.max(), max_value)
            else:
                logger.warning("  %s: no data returned", name)

        except Exception as e:
            logger.warning("  %s fetch failed: %s, skipping", name, e)

    return results


def _fetch_fingrid_dataset(
    url: str,
    api_key: str,
    start: datetime,
    end: datetime,
) -> pd.Series | None:
    """Fetch a single Fingrid dataset and return hourly-resampled series."""
    headers = {"x-api-key": api_key}

    # Fingrid API uses startTime/endTime query params
    params = {
        "startTime": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "endTime": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "format": "json",
        "pageSize": 20000,
    }

    all_rows: list[dict] = []
    page = 1
    total_expected = None
    max_conn_retries = 3

    while True:
        params["page"] = page
        conn_retries = 0
        r = None
        while True:
            try:
                r = requests.get(url, headers=headers, params=params, timeout=60)
                if r.status_code == 429:
                    # Rate limited — always wait and retry (don't count toward max)
                    retry_after = max(int(r.headers.get("Retry-After", 10)), 6)
                    logger.info("  Rate limited on page %d, waiting %ds...",
                                page, retry_after)
                    time.sleep(retry_after)
                    continue
                break  # Got a non-429 response
            except requests.RequestException as e:
                conn_retries += 1
                if conn_retries < max_conn_retries:
                    logger.info("  Fingrid request error (retry %d/%d): %s",
                                conn_retries, max_conn_retries, e)
                    time.sleep(10)
                else:
                    logger.warning("  Fingrid request failed after %d retries: %s",
                                   max_conn_retries, e)
                    break

        if r is None or r.status_code != 200:
            if r is not None:
                logger.warning("  Fingrid HTTP %d (page %d)", r.status_code, page)
            break

        data = r.json()
        rows = data.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        # Check pagination (total/lastPage only present on page 1)
        pagination = data.get("pagination", {})
        if total_expected is None:
            total_expected = pagination.get("total", len(all_rows))
        last_page = pagination.get("lastPage", page)
        if len(all_rows) >= total_expected:
            break
        logger.info("    Page %d/%d: %d/%d rows fetched",
                    page, last_page, len(all_rows), total_expected)
        page += 1
        # Fingrid rate limit: 10 requests/min → 6s between requests
        time.sleep(6)

    if not all_rows:
        return None

    if total_expected and len(all_rows) < total_expected:
        logger.warning("  Incomplete fetch: got %d/%d rows", len(all_rows), total_expected)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["startTime"], utc=True)
    df = df[["date", "value"]].set_index("date").sort_index()

    # Resample to hourly (mean of sub-hourly intervals)
    hourly = df.resample("h").mean().dropna()
    logger.info("  Resampled to %d hourly rows", len(hourly))
    return hourly["value"]
