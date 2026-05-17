"""Deterministic clear-sky GHI baseline for Finnish locations.

Used by v2.5.3+ as the structural baseline of the solar production
sub-model. Output is modulated by Open-Meteo `cloud_cover` to produce
the final PV-production estimate that feeds the FI price model.

This module is intentionally tiny and dependency-free (only numpy +
stdlib) so that it can be called both from the runtime coordinator and
from offline study scripts without dragging in pvlib or similar.

Two clear-sky models are exposed:

* :func:`haurwitz_ghi` — single-formula model with NO atmospheric
  inputs, depends only on the cosine of the solar zenith angle. Fast,
  deterministic, suitable as the production default.

* :func:`ineichen_perez_ghi` — slightly more accurate model that uses a
  monthly Linke turbidity climatology. The default climatology bundled
  here is a Finland-wide average; per-site values can be passed in.

Solar geometry is computed by :func:`solar_zenith_cos` using the
NOAA / Spencer 1971 formulas (adequate for hourly modelling; accurate
to ~0.01°).

Reference:
    Haurwitz (1945, 1948), Ineichen & Perez (2002), Spencer (1971).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

# Solar constant (W/m²) used for the top-of-atmosphere reference.
SOLAR_CONSTANT_W_M2 = 1367.0

# Finland-wide monthly Linke turbidity climatology (rough average across
# the seven FI sites used by the integration). Higher in summer due to
# elevated aerosol and water-vapour load. Replace with per-site values
# if/when published in `data/finland.yaml`.
LINKE_TURBIDITY_FI_MONTHLY = (
    2.4, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.4, 3.0, 2.7, 2.5, 2.4,
)


# ── Solar geometry ──────────────────────────────────────────────────


def _day_of_year_fraction(ts: datetime) -> float:
    """Fractional day-of-year in [0, 1) including UTC time-of-day."""
    start = datetime(ts.year, 1, 1, tzinfo=timezone.utc)
    seconds_in_year = (datetime(ts.year + 1, 1, 1, tzinfo=timezone.utc)
                       - start).total_seconds()
    return (ts - start).total_seconds() / seconds_in_year


def solar_zenith_cos(ts: datetime, lat_deg: float, lon_deg: float) -> float:
    """Cosine of the solar zenith angle at (lat, lon) and UTC time `ts`.

    Uses Spencer's (1971) Fourier expansion for declination and the
    equation of time. Accurate to ~0.01° — adequate for hourly clear-sky
    modelling.

    Returns 0.0 when the sun is below the horizon (so the multiplication
    by GHI naturally zeroes out night-time radiation).
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # Day angle (radians) — Spencer 1971
    doy_frac = _day_of_year_fraction(ts)
    gamma = 2.0 * math.pi * doy_frac

    # Declination (radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.001480 * math.sin(3 * gamma)
    )

    # Equation of time (minutes)
    eqt_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Local solar time (hours, on lon meridian)
    utc_hours = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    solar_time = utc_hours + lon_deg / 15.0 + eqt_min / 60.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))

    lat_rad = math.radians(lat_deg)
    cos_z = (math.sin(lat_rad) * math.sin(decl)
             + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    return max(0.0, cos_z)


def extraterrestrial_irradiance(ts: datetime) -> float:
    """Top-of-atmosphere normal irradiance with eccentricity correction."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    gamma = 2.0 * math.pi * _day_of_year_fraction(ts)
    # Eccentricity correction (Spencer 1971)
    eccentricity = (
        1.000110
        + 0.034221 * math.cos(gamma)
        + 0.001280 * math.sin(gamma)
        + 0.000719 * math.cos(2 * gamma)
        + 0.000077 * math.sin(2 * gamma)
    )
    return SOLAR_CONSTANT_W_M2 * eccentricity


# ── Clear-sky GHI formulas ──────────────────────────────────────────


def haurwitz_ghi(cos_zenith: float) -> float:
    """Haurwitz (1945) clear-sky GHI, W/m².

    Single-formula model. Accurate to ~5 % vs measured GHI under
    cloud-free conditions in mid-latitudes; sufficient as a baseline.

    Returns 0.0 when the sun is below the horizon.
    """
    if cos_zenith <= 0.0:
        return 0.0
    return 1098.0 * cos_zenith * math.exp(-0.057 / cos_zenith)


def ineichen_perez_ghi(
    cos_zenith: float,
    extraterrestrial: float,
    linke_turbidity: float,
    altitude_m: float = 0.0,
) -> float:
    """Ineichen-Perez (2002) clear-sky GHI, W/m².

    Slightly better than Haurwitz under hazy / high-aerosol conditions
    because it uses the Linke turbidity factor. Needs a turbidity
    climatology (a monthly average per site is fine; we ship a
    Finland-wide one in :data:`LINKE_TURBIDITY_FI_MONTHLY`).
    """
    if cos_zenith <= 0.0:
        return 0.0

    # Airmass (Kasten-Young approximation)
    zenith_deg = math.degrees(math.acos(cos_zenith))
    airmass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg) ** -1.6364)

    # Altitude correction factors (Ineichen 2002)
    fh1 = math.exp(-altitude_m / 8000.0)
    fh2 = math.exp(-altitude_m / 1250.0)
    cg1 = 5.09e-5 * altitude_m + 0.868
    cg2 = 3.92e-5 * altitude_m + 0.0387

    ghi = (cg1 * extraterrestrial * cos_zenith
           * math.exp(-cg2 * airmass * (fh1 + fh2 * (linke_turbidity - 1.0)))
           * math.exp(0.01 * airmass ** 1.8))
    return max(0.0, ghi)


def ineichen_perez_ghi_at(ts: datetime, lat_deg: float, lon_deg: float,
                          altitude_m: float = 0.0) -> float:
    """Convenience wrapper: Ineichen-Perez GHI at (lat, lon, ts) using
    the bundled Finland monthly Linke climatology."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    cos_z = solar_zenith_cos(ts, lat_deg, lon_deg)
    e0 = extraterrestrial_irradiance(ts)
    linke = LINKE_TURBIDITY_FI_MONTHLY[ts.month - 1]
    return ineichen_perez_ghi(cos_z, e0, linke, altitude_m)


# ── Vectorised helpers for offline studies ─────────────────────────


def clear_sky_series(
    timestamps: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    altitude_m: float = 0.0,
    model: str = "haurwitz",
) -> np.ndarray:
    """Compute a vector of clear-sky GHI (W/m²) for an array of UTC
    datetime64[ns] timestamps at a single site.

    Args:
        timestamps: 1-D array of numpy datetime64 (UTC).
        lat_deg, lon_deg: site coordinates.
        altitude_m: site elevation (Ineichen only).
        model: ``"haurwitz"`` or ``"ineichen"``.

    Returns:
        1-D numpy array of GHI in W/m².
    """
    if model not in ("haurwitz", "ineichen"):
        raise ValueError(f"unknown clear-sky model: {model!r}")

    out = np.empty(len(timestamps), dtype=float)
    for i, ts64 in enumerate(timestamps):
        # numpy datetime64 → python datetime, UTC
        ts = datetime.fromtimestamp(
            ts64.astype("datetime64[s]").astype("int64"),
            tz=timezone.utc,
        )
        if model == "haurwitz":
            cos_z = solar_zenith_cos(ts, lat_deg, lon_deg)
            out[i] = haurwitz_ghi(cos_z)
        else:
            cos_z = solar_zenith_cos(ts, lat_deg, lon_deg)
            e0 = extraterrestrial_irradiance(ts)
            linke = LINKE_TURBIDITY_FI_MONTHLY[ts.month - 1]
            out[i] = ineichen_perez_ghi(cos_z, e0, linke, altitude_m)
    return out


def predict_solar_mw(
    timestamps: np.ndarray,
    cloud_cover_pct: np.ndarray,
    artifact: dict,
    sites: list[dict] | None = None,
) -> np.ndarray:
    """Inference for the v2.5.3 solar production sub-model — Fingrid-free.

    Given a frozen artifact (from :func:`build_artifact`, persisted as
    JSON), an hourly UTC timestamp array, and the matching capacity-
    weighted Open-Meteo `cloud_cover` series, return predicted Finnish
    solar production in MW.

    Runtime characteristics:

    - No Fingrid call. The training-time `capacity_MW` is baked into
      the artifact's `K = gain · capacity_ref` scalar.
    - No re-fitting. The artifact is a frozen object; refresh it
      offline every few months as installed capacity drifts.
    - Pure deterministic function of the inputs.

    Args:
        timestamps: 1-D numpy datetime64 array (UTC).
        cloud_cover_pct: matching 1-D array of capacity-weighted
            cloud cover (%, 0–100) from Open-Meteo.
        artifact: dict loaded from the persisted JSON. Must contain
            keys ``clear_sky_model``, ``modulator_form``, ``alpha``,
            ``K``. Optional ``modulator_params`` for non-default forms.
        sites: optional override of the weighted FI site list. If
            ``None``, uses ``artifact["sites"]``.

    Returns:
        1-D numpy array of predicted production in MW. Values < 0
        are clipped to 0 (capacity floor; alpha can be slightly
        negative due to OLS, but production never is).
    """
    if sites is None:
        sites = artifact["sites"]

    # Capacity-weighted clear-sky GHI across the configured sites.
    ghi = np.zeros(len(timestamps), dtype=float)
    w_total = 0.0
    for site in sites:
        sw = float(site.get("solar_weight", 0.0))
        if sw <= 0:
            continue
        ghi += sw * clear_sky_series(
            timestamps,
            lat_deg=float(site["lat"]),
            lon_deg=float(site["lon"]),
            model=artifact["clear_sky_model"],
            altitude_m=float(site.get("altitude_m", 0.0)),
        )
        w_total += sw
    if w_total > 0:
        ghi /= w_total

    mod = cloudiness_modulator(
        cloud_cover_pct,
        form=artifact["modulator_form"],
        params=tuple(artifact.get("modulator_params") or ()) or None,
    )
    pred = float(artifact["alpha"]) + float(artifact["K"]) * ghi * mod
    return np.clip(pred, 0.0, None)


def build_artifact(
    *,
    clear_sky_model: str,
    modulator_form: str,
    alpha: float,
    gain: float,
    capacity_ref_mw: float,
    sites: list[dict],
    train_window: tuple[str, str] | None = None,
    test_metrics: dict | None = None,
    modulator_params: tuple[float, ...] | None = None,
    notes: str = "",
) -> dict:
    """Build the frozen-artifact dict for persistence.

    The runtime inference only ever sees `K = gain · capacity_ref_mw`,
    so capacity does NOT need to be fetched at inference time. The
    artifact records ``capacity_ref_mw`` separately for traceability —
    if the operator wants to re-scale the model to a different capacity
    without re-fitting they can adjust K proportionally.
    """
    return {
        "version": "2.5.3",
        "clear_sky_model": clear_sky_model,
        "modulator_form": modulator_form,
        "modulator_params": list(modulator_params) if modulator_params else None,
        "alpha": float(alpha),
        "gain": float(gain),
        "capacity_ref_mw": float(capacity_ref_mw),
        "K": float(gain) * float(capacity_ref_mw),
        "sites": [
            {"name": s.get("name"),
             "lat": float(s["lat"]),
             "lon": float(s["lon"]),
             "solar_weight": float(s.get("solar_weight", 0.0)),
             "altitude_m": float(s.get("altitude_m", 0.0))}
            for s in sites if float(s.get("solar_weight", 0.0)) > 0
        ],
        "train_window": list(train_window) if train_window else None,
        "test_metrics": test_metrics or {},
        "notes": notes,
    }


def cloudiness_modulator(
    cloud_cover_pct: np.ndarray,
    form: str = "kasten_czeplak",
    params: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Map total cloud cover (%) to a GHI-attenuation factor in [0, 1].

    Three forms supported (selected via :func:`fit_modulator`):

    * ``"linear"`` — ``1 − a · c/100``
    * ``"affine_floor"`` — ``max(0, 1 − a · c/100) + b``
    * ``"kasten_czeplak"`` — ``1 − 0.75 · (c/100)^3.4`` (empirical default)
    """
    c = np.clip(np.asarray(cloud_cover_pct, dtype=float), 0.0, 100.0) / 100.0
    if form == "linear":
        a = (params or (0.75,))[0]
        return np.clip(1.0 - a * c, 0.0, 1.0)
    if form == "affine_floor":
        a, b = params or (0.75, 0.05)
        return np.clip(np.maximum(0.0, 1.0 - a * c) + b, 0.0, 1.0)
    if form == "kasten_czeplak":
        return np.clip(1.0 - 0.75 * c ** 3.4, 0.0, 1.0)
    raise ValueError(f"unknown modulator form: {form!r}")
