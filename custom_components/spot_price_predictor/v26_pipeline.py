"""v2.6.0 — Production wiring of the L1+L2+L3+L4 prediction pipeline.

This module encapsulates the four-layer architecture plus the v2.5.14
softplus floor and v2.5.15 hourly DtACI calibrators (bias corrector,
fan-chart, refit monitor) into a single class that the coordinator
calls per update cycle.

Design principle: ADDITIVE integration. The existing v2.2 9-feature
Ridge model continues to populate the established `forecast` and
`duration_forecast` attributes unchanged. v26 outputs appear as
additional fields on each row so existing dashboards / automations
keep working, while consumers that want the richer signal can opt in
by reading the new keys.

Pipeline (per hour h of the 168-hour forecast horizon):

  L1 seasonal_fi(h) = P_hour[h.hour] + P_day[h.weekday] + P_week[h.week]
                       — from data/seasonal_components_default.json

  L2 Ridge contribution
       = β · [Y_fi_lag168, is_workday(h),
              Y_sigmoid_wind_rho(h), Y_solar_effective(h), Y_temp(h)]
       — coefficients from data/spike_model_default.json

  L3 AR(1) contribution at horizon h
       = φ^h · ε(t0-1)   where ε(t0-1) is the most-recent observed
                          deseasonalized FI price residual

  L1+L2+L3 mean      = seasonal_fi(h) + ridge(h) + ar_corr(h)
  + softplus floor   = floored at −5 EUR/MWh (v2.5.14)
  + hourly bias EMA  = − bias_estimate (v2.5.15)
  → final point forecast (the `spot_eur_mwh` value on each forecast row)

  L4 GPD POT fan-chart
       sample 500 paths from Normal-body + GPD-tail mixture
       compute P5 / P25 / P50 / P75 / P95 per hour
       → fan-chart attributes per forecast row

  D(k) for each day in the forecast horizon
       cumulative sort of the 24 hourly predictions
       cheap[i] = mean of (i+1) cheapest hours for i in 0..23
       peak [i] = mean of (i+1) priciest hours for i in 0..23

Persistent state under `<config>/.storage/spot_price_predictor_v26/`:
  hourly_bias.json        EMA bias of L1+L2+L3+floor mean predictions
  hourly_fan_chart.json   DtACI bundle per coverage target
  refit_monitor.json      drift trigger state
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import seasonal_decomposition as _sd
from . import solar_clear_sky as _scs
from . import price_floor as _pf
from . import hourly_calibration as _hc

_LOGGER = logging.getLogger(__name__)

# Default Ridge feature ordering matches v2.5.13 V_sigmoid_full.
V26_FEATURES = (
    "intercept",          # constant 1
    "Y_fi_lag168",
    "is_workday",
    "Y_sigmoid_wind_rho",
    "Y_solar_effective",
    "Y_temp",
)


# ── Physics features (vectorised) ──────────────────────────────────


def _air_density(temp_c: np.ndarray, pressure_pa: float = 101_325.0,
                 R_specific: float = 287.05) -> np.ndarray:
    T_K = np.asarray(temp_c, dtype=float) + 273.15
    return pressure_pa / (R_specific * T_K)


def _sigmoid_turbine_rho(wind: np.ndarray, temp_c: np.ndarray,
                          v_mid: float = 7.5, k_steep: float = 1.5,
                          rho_ref: float = 1.225) -> np.ndarray:
    wind = np.asarray(wind, dtype=float)
    sigmoid = 1.0 / (1.0 + np.exp(-(wind - v_mid) / k_steep))
    rho = _air_density(temp_c)
    return sigmoid * (rho / rho_ref)


def _solar_effective(ghi: np.ndarray, temp_c: np.ndarray,
                      coeff_per_C: float = 0.004,
                      noct_coeff: float = 0.03) -> np.ndarray:
    ghi = np.asarray(ghi, dtype=float)
    cell_temp = np.asarray(temp_c, dtype=float) + noct_coeff * ghi
    derating = 1.0 - coeff_per_C * np.maximum(0.0, cell_temp - 25.0)
    return ghi * derating


# ── Pipeline ───────────────────────────────────────────────────────


class V26Pipeline:
    """Runs the four-layer prediction + floor + calibrators on every
    coordinator update cycle.

    Loads frozen artifacts at __init__:
      data/seasonal_components_default.json
      data/spike_model_default.json

    Maintains three persistent calibrator state files under
    `<config>/.storage/spot_price_predictor_v26/`. State is loaded on
    construction if present, otherwise initialised cold and saved
    after the first update.
    """

    def __init__(self, data_dir: Path, storage_dir: Path) -> None:
        self._data_dir    = Path(data_dir)
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._seasonal_artifact = self._load_json(
            self._data_dir / "seasonal_components_default.json")
        self._spike_artifact = self._load_json(
            self._data_dir / "spike_model_default.json")

        # Ridge β vector ordering must match V26_FEATURES
        self._ridge_coef = np.asarray(
            self._spike_artifact.get("ridge_coef", [0.0] * len(V26_FEATURES)),
            dtype=float,
        )
        # AR(1) coefficient on the deseasonalized FI residual
        self._ar1_phi = float(self._spike_artifact.get("ar1_phi", 0.0))
        # L4 GPD POT parameters for fan-chart sampling
        self._gpd_right    = self._spike_artifact.get("gpd_right")
        self._gpd_left     = self._spike_artifact.get("gpd_left")
        # Body Normal: μ, σ from η training stats
        stats = self._spike_artifact.get("stats", {})
        self._eta_mu    = float(stats.get("eta_train_mean", 0.0))
        self._eta_sigma = float(stats.get("eta_train_sigma", 25.0))

        # Calibrators with persistent state
        self._bias = self._load_calibrator(
            self._storage_dir / "hourly_bias.json",
            _hc.HourlyBiasCorrector,
            default_kwargs=dict(halflife_days=14.0, warmup_hours=168),
        )
        self._fan = self._load_calibrator(
            self._storage_dir / "hourly_fan_chart.json",
            _hc.HourlyFanChartCalibrator,
            default_kwargs=dict(target_coverages=(0.5, 0.9),
                                 window=720, min_warmup=24),
        )
        self._refit = self._load_calibrator(
            self._storage_dir / "refit_monitor.json",
            _hc.RefitMonitor,
            default_kwargs=dict(target_coverage=0.9, drift_pp=0.05,
                                 persistence_steps=14 * 24),
        )
        # Cache of the most-recent observed η so AR(1) has a starting
        # point for forecasting. Updated when we see new actuals.
        self._last_eta: float | None = None

    # ── Artifact / state I/O ───────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            _LOGGER.warning("v26: artifact missing: %s", path)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            _LOGGER.warning("v26: failed to read %s: %s", path, e)
            return {}

    @staticmethod
    def _load_calibrator(path: Path, cls, default_kwargs: dict):
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                return cls.from_dict(d)
            except Exception as e:
                _LOGGER.warning("v26: failed to restore %s; resetting (%s)",
                                 path, e)
        return cls(**default_kwargs)

    def save_state(self) -> None:
        try:
            (self._storage_dir / "hourly_bias.json").write_text(
                json.dumps(self._bias.to_dict()), encoding="utf-8")
            (self._storage_dir / "hourly_fan_chart.json").write_text(
                json.dumps(self._fan.to_dict()), encoding="utf-8")
            (self._storage_dir / "refit_monitor.json").write_text(
                json.dumps(self._refit.to_dict()), encoding="utf-8")
        except Exception as e:
            _LOGGER.warning("v26: state save failed: %s", e)

    # ── L1 seasonal lookup ────────────────────────────────────────

    def _seasonal_fi(self, timestamps: np.ndarray) -> np.ndarray:
        comp = (self._seasonal_artifact.get("components") or {}).get("fi")
        if not comp:
            return np.zeros(len(timestamps), dtype=float)
        return _sd.compute_seasonal_part(timestamps, comp)

    def _deseasonalize_input(self, name: str,
                              values: np.ndarray,
                              timestamps: np.ndarray) -> np.ndarray:
        """Y_X = X − seasonal_X using shipped per-input components."""
        comp = (self._seasonal_artifact.get("components") or {}).get(name)
        if not comp:
            # No seasonal for this input → use raw centered
            v = np.asarray(values, dtype=float)
            return v - float(np.nanmean(v))
        seasonal = _sd.compute_seasonal_part(timestamps, comp)
        return np.asarray(values, dtype=float) - seasonal

    # ── L2 Ridge features + prediction ─────────────────────────────

    def _build_features(self, timestamps: np.ndarray,
                        wind: np.ndarray, solar: np.ndarray,
                        temp: np.ndarray,
                        Y_fi_lag168: np.ndarray) -> np.ndarray:
        """Returns design matrix (n, 6) with columns matching V26_FEATURES."""
        n = len(timestamps)
        is_workday = self._is_workday(timestamps).astype(float)
        # Physics-derived features
        wind_rho = _sigmoid_turbine_rho(wind, temp)
        solar_eff = _solar_effective(solar, temp)
        # Deseasonalize the physics features (matches v2.5.13 fit)
        Y_wind_rho  = wind_rho  - np.mean(wind_rho)   # local centering
        Y_solar_eff = solar_eff - np.mean(solar_eff)
        Y_temp      = self._deseasonalize_input("temp", temp, timestamps)
        X = np.column_stack([
            np.ones(n),
            Y_fi_lag168,
            is_workday,
            Y_wind_rho,
            Y_solar_eff,
            Y_temp,
        ])
        return X

    @staticmethod
    def _is_workday(timestamps: np.ndarray) -> np.ndarray:
        secs = timestamps.astype("datetime64[s]").astype("int64")
        days = secs // 86400
        # 1970-01-01 = Thursday (weekday 3)
        weekday = (days + 3) % 7
        return weekday < 5

    # ── L3 AR(1) per-horizon decay ─────────────────────────────────

    def _ar_contribution(self, n_steps: int) -> np.ndarray:
        """φ^h · ε(t0-1) for h=1..n_steps. If we have no observed η
        yet, returns zeros (cold-start)."""
        if self._last_eta is None or self._ar1_phi == 0:
            return np.zeros(n_steps, dtype=float)
        h = np.arange(1, n_steps + 1, dtype=float)
        return (self._ar1_phi ** h) * float(self._last_eta)

    # ── L4 GPD POT fan-chart sampling ──────────────────────────────

    def _sample_fan_chart(self, mean_pred: np.ndarray,
                          n_samples: int = 500, seed: int = 0
                          ) -> dict[str, np.ndarray]:
        """Per-hour fan-chart bands by sampling η from Normal-body +
        GPD-tail mixture and adding to mean_pred."""
        rng = np.random.default_rng(seed)
        right = self._gpd_right or {}
        left  = self._gpd_left  or {}
        threshold = float(right.get("threshold", self._eta_sigma * 1.5))
        xi_r = float(right.get("shape", 0.0))
        sg_r = float(right.get("scale", self._eta_sigma))
        p_r  = float(right.get("p_exceed", 0.05))
        xi_l = float(left.get("shape", 0.0))
        sg_l = float(left.get("scale", self._eta_sigma))
        p_l  = float(left.get("p_exceed", 0.05))
        mu, sigma = self._eta_mu, self._eta_sigma
        n_hours = len(mean_pred)
        samples = np.empty((n_samples, n_hours), dtype=float)
        for s in range(n_samples):
            u = rng.uniform(size=n_hours)
            shock = np.empty(n_hours, dtype=float)
            body_mask = (u >= p_l) & (u < 1 - p_r)
            n_body = int(body_mask.sum())
            if n_body > 0:
                body = rng.normal(mu, sigma, size=n_body)
                body = np.clip(body, -threshold, threshold)
                shock[body_mask] = body
            right_mask = u >= 1 - p_r
            n_right = int(right_mask.sum())
            if n_right > 0:
                if abs(xi_r) < 1e-9:
                    exc = rng.exponential(scale=sg_r, size=n_right)
                else:
                    exc = sg_r / xi_r * (rng.uniform(size=n_right) ** (-xi_r) - 1.0)
                shock[right_mask] = threshold + np.maximum(0.0, exc)
            left_mask = u < p_l
            n_left = int(left_mask.sum())
            if n_left > 0:
                if abs(xi_l) < 1e-9:
                    exc = rng.exponential(scale=sg_l, size=n_left)
                else:
                    exc = sg_l / xi_l * (rng.uniform(size=n_left) ** (-xi_l) - 1.0)
                shock[left_mask] = -threshold - np.maximum(0.0, exc)
            samples[s, :] = mean_pred + shock
        return {
            "P5":  np.percentile(samples, 5,  axis=0),
            "P25": np.percentile(samples, 25, axis=0),
            "P50": np.percentile(samples, 50, axis=0),
            "P75": np.percentile(samples, 75, axis=0),
            "P95": np.percentile(samples, 95, axis=0),
        }

    # ── Public API ────────────────────────────────────────────────

    def compute_forecast(
        self, timestamps: np.ndarray,
        wind: np.ndarray, solar: np.ndarray, temp: np.ndarray,
        recent_fi_residuals: dict[str, float] | None = None,
        enable_fan_chart: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute the v2.6.0 hourly forecast.

        Args:
            timestamps: 1-D numpy datetime64 array for the forecast horizon.
            wind / solar / temp: matching arrays (Open-Meteo forecast values).
            recent_fi_residuals: optional dict {"lag168": np.ndarray(n)}
                providing Y_fi at t-168 for each forecast hour, AND
                {"last_eta": float} providing the most-recent observed
                post-AR residual (for L3 AR(1) propagation).
            enable_fan_chart: if True, also compute P5/P25/P50/P75/P95.

        Returns:
            Dict with keys:
                mean_eur_mwh:  shape (n,) point forecast in EUR/MWh
                bias_eur_mwh:  scalar — the bias estimate applied
                P5..P95_eur_mwh: shape (n,) fan-chart bands (if enabled)
        """
        n = len(timestamps)
        if n == 0:
            return {"mean_eur_mwh": np.array([]),
                    "bias_eur_mwh": 0.0}

        # L1
        seasonal = self._seasonal_fi(timestamps)

        # Y_fi_lag168 = caller provides (typically from observed
        # FI residuals 7 days ago). If not, fall back to zeros.
        lag168 = np.zeros(n, dtype=float)
        if recent_fi_residuals and "lag168" in recent_fi_residuals:
            lag168_arr = np.asarray(recent_fi_residuals["lag168"], dtype=float)
            if lag168_arr.shape == (n,):
                lag168 = lag168_arr
        if recent_fi_residuals and "last_eta" in recent_fi_residuals:
            self._last_eta = float(recent_fi_residuals["last_eta"])

        # L2 Ridge
        X = self._build_features(timestamps, wind, solar, temp, lag168)
        ridge = X @ self._ridge_coef

        # L3 AR(1) propagation
        ar_corr = self._ar_contribution(n)

        # Mean before floor / bias
        mean = seasonal + ridge + ar_corr

        # softplus floor
        mean = _pf.apply_floor(mean, floor=_pf.DEFAULT_FLOOR_EUR_MWH)

        # Hourly bias correction (small constant offset, slow-moving)
        bias = self._bias.bias_estimate if self._bias.warm else 0.0
        mean_corrected = np.array([self._bias.correct(float(v)) for v in mean])

        out: dict[str, Any] = {
            "mean_eur_mwh": mean_corrected,
            "bias_eur_mwh": float(bias),
        }

        if enable_fan_chart:
            fan = self._sample_fan_chart(mean_corrected)
            for k, v in fan.items():
                out[f"{k}_eur_mwh"] = v

        return out

    def compute_duration_curves(
        self, hourly_prediction: np.ndarray, timestamps: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Aggregate hourly forecast into per-day D(k) for k=1..24.

        Returns:
            list of dicts (one per day) with keys:
                date: ISO date string
                dk_cheap_eur_mwh: list[24] mean of i+1 cheapest hours
                dk_peak_eur_mwh:  list[24] mean of i+1 priciest hours
                hours_in_day:    int (24 for full days, less for edges)
        """
        import pandas as pd
        ts = pd.DatetimeIndex(timestamps, tz="UTC")
        ser = pd.Series(hourly_prediction, index=ts)
        out = []
        for date, day in ser.groupby(ts.date):
            vals = np.sort(day.values)
            if len(vals) < 1:
                continue
            n = len(vals)
            counts = np.arange(1, n + 1, dtype=float)
            cum_low  = np.cumsum(vals) / counts
            cum_high = np.cumsum(vals[::-1]) / counts
            # Pad to 24 if a partial day
            cheap = list(cum_low)  + [float("nan")] * (24 - n)
            peak  = list(cum_high) + [float("nan")] * (24 - n)
            out.append({
                "date": str(date),
                "hours_in_day": n,
                "dk_cheap_eur_mwh": cheap[:24],
                "dk_peak_eur_mwh":  peak[:24],
            })
        return out

    def update_with_actuals(
        self, predicted: np.ndarray, actual: np.ndarray,
    ) -> dict[str, Any]:
        """Feed the calibrators (bias + fan-chart) with realised (pred,
        actual) pairs. Update last_eta for next AR forecast. Returns a
        diagnostics dict including the refit_recommended flag."""
        predicted = np.asarray(predicted, dtype=float)
        actual    = np.asarray(actual,    dtype=float)
        n = min(len(predicted), len(actual))
        for i in range(n):
            self._bias.update(float(predicted[i]), float(actual[i]))
            self._fan.update(float(predicted[i]),  float(actual[i]))
        # Track most-recent η = actual − ridge_pred_at_that_time
        # Caller passes those if it can; otherwise we approximate with
        # the last (actual − predicted) which conflates AR contribution
        # but is fine as a cold-start placeholder.
        if n > 0:
            self._last_eta = float(actual[n - 1] - predicted[n - 1])

        # Watch realised coverage of the 0.9 band for drift
        diag = self._fan.diagnostics()
        for tc, d in diag.items():
            if abs(tc - 0.9) < 1e-9:
                self._refit.observe(
                    realised_coverage=d["effective_coverage"],
                    timestamp_iso=datetime.now(timezone.utc).isoformat(),
                )
                break

        return {
            "bias_warm":            self._bias.warm,
            "bias_estimate":        self._bias.bias_estimate,
            "refit_recommended":    self._refit.refit_recommended,
            "consecutive_drift_hours": self._refit.consecutive_drift_hours,
            "fan_diagnostics":      diag,
        }
