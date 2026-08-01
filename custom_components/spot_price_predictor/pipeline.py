"""Production wiring of the L1+L2+L3+L4 prediction pipeline.

This module encapsulates the four-layer architecture plus the softplus
floor and the hourly DtACI calibrators (bias corrector, fan-chart,
refit monitor) into a single class that the coordinator calls per
update cycle.

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
  + softplus floor   = floored at −5 EUR/MWh
  + hourly bias EMA  = − bias_estimate
  → final point forecast (the `spot_eur_mwh` value on each forecast row)

  L4 GPD POT fan-chart
       sample 500 paths from Normal-body + GPD-tail mixture
       compute P5 / P25 / P50 / P75 / P95 per hour
       → fan-chart attributes per forecast row

  D(k) for each day in the forecast horizon
       cumulative sort of the 24 hourly predictions
       cheap[i] = mean of (i+1) cheapest hours for i in 0..23
       peak [i] = mean of (i+1) priciest hours for i in 0..23

Persistent calibrator state lives under `<config>/.storage/`
(see the storage directory configured by the coordinator):
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

# Default Ridge feature ordering. The shipped artifact's
# `ridge_features` field is authoritative at runtime — the pipeline
# reads it and builds the design matrix in that exact order. This
# constant is only the fallback when the artifact lacks the field.
RIDGE_FEATURES = (
    "intercept",          # constant 1
    "Y_fi_lag168",
    "is_workday",
    "Y_sigmoid_wind_rho",
    "Y_solar_effective",
    "Y_temp",
    # v2.17.0 — cross-border features, LAGGED 168 h. Through v2.16 these
    # were same-hour prices, which cannot be known when the forecast is
    # made: SE1/SE3/EE clear in the SAME day-ahead auction as FI, so
    # observing them implies the FI price is published too. See
    # NEIGHBOUR_LAG_HOURS below.
    "Y_se1_lag168",
    "Y_se3_lag168",
    "Y_ee_lag168",
    # v2.17.1 — demand. The pipeline previously had NO demand signal at
    # all: wind/solar are supply, and temperature is only a heating proxy
    # that goes silent in summer (corr(temp, net load) = -0.70 winter but
    # +0.12 summer). The holiday flag is the part that pays: public
    # holidays have weekend-like demand, and `is_workday` alone priced
    # Midsummer and Christmas as ordinary working days.
    #
    # `Y_netload_lag168` was shipped in v2.17.0 and REMOVED in v2.17.1:
    # on the correct training window it was worth -0.04 % MAE, and its
    # fitted coefficient was negative (-1.25) — not a demand relationship
    # but a collinearity artifact, since last week's demand is already
    # embedded in last week's price (corr = +0.587). The pipeline still
    # accepts `netload_lag168` so a future artifact can use it, but no
    # shipped model does. The demand signal that genuinely carries
    # information is SAME-HOUR net load (coefficient +18.9, corr +0.587),
    # which Fingrid publishes day-ahead only — see docs/BACKLOG.md.
    "is_holiday",
)

# ── Cross-border look-back (v2.17.0) ───────────────────────────────
#
# FI, SE1, SE3 and EE clear simultaneously in the day-ahead auction, so a
# same-hour neighbour price is never observable before the target it is
# meant to predict (corr(FI(t), SE3(t)) = +0.82 contemporaneous vs +0.66
# at 24 h lag). Using it made the model a follower of a market that
# publishes at the same instant as the answer: it suppressed the physical
# wind coefficient by more than half (-44.6 vs -93.0) and left the
# coefficients mis-specified for the hours production must actually
# forecast, where the feature is necessarily absent.
#
# The features are therefore built from prices `NEIGHBOUR_LAG_HOURS` in
# the past, which are genuinely known at forecast time for the whole
# horizon. Leak-free evaluation (studies/honest_horizon_study.py):
# MAE 35.46 -> 27.30 (-23 %), winter 46.57 -> 31.54 (-32 %), bias
# -7.29 -> -4.15. It also removes the +2 d/+3 d discontinuity, which was
# a symptom of the same defect.
NEIGHBOUR_LAG_HOURS: int = 168

# Names of neighbour-price zones consumed by Pipeline.compute_forecast
# when the caller supplies the optional `neighbour_prices_lag168` arg.
_NEIGHBOUR_ZONES: tuple[str, ...] = ("se1", "se3", "ee")

# ── Economic sign invariant (v2.16.0) ──────────────────────────────
#
# Wind and PV are zero-marginal-cost generation. More of either shifts
# the merit order to the right, so their price effect can only be
# negative or nil — NEVER positive. Any fit that assigns a positive
# coefficient to these features is describing a confound (in Finland:
# clear winter skies are cold and expensive; sunny days here are sunny
# in Sweden too, so the neighbour-price channel already carries the PV
# signal), not a causal price response.
#
# A positive coefficient here is not a tuning question — it makes the
# model raise its forecast when irradiance rises, which is indefensible
# and gets structurally worse as PV capacity grows. The trainer fits
# these features under a <= 0 constraint, and the runtime clamps them
# defensively so that no artifact, however produced, can reintroduce
# the inversion in production.
NON_POSITIVE_FEATURES: tuple[str, ...] = (
    "Y_solar_effective",
    "Y_sigmoid_wind_rho",
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


class Pipeline:
    """Runs the four-layer prediction + floor + calibrators on every
    coordinator update cycle.

    Loads frozen artifacts at __init__:
      data/seasonal_components_default.json
      data/spike_model_default.json

    Maintains three persistent calibrator state files under
    `<config>/.storage/spot_price_predictor_pipeline/`. State is loaded on
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

        # Seasonal components for the derived physics features
        # (solar_effective / sigmoid_wind_rho) so the runtime deseasonalises
        # them the SAME way the trainer did. Previously the pipeline used a
        # local mean while the trainer fit seasonal components — a
        # train/inference mismatch. Absent (older artifacts) -> local mean.
        self._physics_seasonal = self._spike_artifact.get("physics_seasonal") or {}

        # Ridge β vector. The artifact carries the feature names it was
        # trained on; the pipeline builds the design matrix in that
        # order. Falls back to RIDGE_FEATURES if the artifact omits the
        # field (legacy artifacts).
        self._ridge_coef = np.asarray(
            self._spike_artifact.get("ridge_coef", [0.0] * len(RIDGE_FEATURES)),
            dtype=float,
        )
        feats = self._spike_artifact.get("ridge_features")
        if isinstance(feats, list) and len(feats) == len(self._ridge_coef) - 1:
            # Artifact stores features WITHOUT the intercept; prepend.
            self._features: tuple[str, ...] = ("intercept", *feats)
        elif isinstance(feats, list) and len(feats) == len(self._ridge_coef):
            self._features = tuple(feats)
        else:
            self._features = RIDGE_FEATURES[: len(self._ridge_coef)]
        # Economic sign invariant — clamp any positive zero-marginal-cost
        # coefficient to zero before it can reach a forecast.
        self._enforce_physics_signs()
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
            _hc.PerHourBiasCorrector,
            default_kwargs=dict(halflife_days=14.0, warmup_updates=14),
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

    def _enforce_physics_signs(self) -> None:
        """Clamp zero-marginal-cost coefficients to <= 0 (see
        `NON_POSITIVE_FEATURES`).

        This is a defensive runtime guard, not the primary fix — the
        trainer already fits these features under a sign constraint. It
        exists so that a hand-edited, third-party, or historical artifact
        can never make the forecast rise when wind or irradiance rises.
        Clamping to exactly 0.0 removes the feature's influence rather
        than inventing a magnitude we have not fitted.
        """
        for name in NON_POSITIVE_FEATURES:
            if name not in self._features:
                continue
            i = self._features.index(name)
            if i < len(self._ridge_coef) and self._ridge_coef[i] > 0.0:
                _LOGGER.error(
                    "pipeline: artifact violates the zero-marginal-cost sign "
                    "invariant — %s coefficient %+.6f > 0 would make the "
                    "forecast RISE with more generation. Clamping to 0. "
                    "Retrain with the sign constraint (see "
                    "studies/build_fresh_spike_model.py).",
                    name, float(self._ridge_coef[i]),
                )
                self._ridge_coef[i] = 0.0

    # ── Artifact / state I/O ───────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            _LOGGER.warning("pipeline:artifact missing: %s", path)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            _LOGGER.warning("pipeline:failed to read %s: %s", path, e)
            return {}

    @staticmethod
    def _load_calibrator(path: Path, cls, default_kwargs: dict):
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                return cls.from_dict(d)
            except Exception as e:
                _LOGGER.warning("pipeline:failed to restore %s; resetting (%s)",
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
            _LOGGER.warning("pipeline:state save failed: %s", e)

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

    def _deseasonalize_physics(self, name: str,
                                values: np.ndarray,
                                timestamps: np.ndarray) -> np.ndarray:
        """Deseasonalise a derived physics feature (solar_effective /
        sigmoid_wind_rho) using the components the spike trainer stored, so
        train and inference match exactly. Falls back to local-mean centering
        when the artifact predates this (legacy behaviour)."""
        comp = self._physics_seasonal.get(name)
        v = np.asarray(values, dtype=float)
        if not comp:
            return v - float(np.mean(v))
        return v - _sd.compute_seasonal_part(timestamps, comp)

    # ── L2 Ridge features + prediction ─────────────────────────────

    def _build_features(
        self, timestamps: np.ndarray,
        wind: np.ndarray, solar: np.ndarray, temp: np.ndarray,
        Y_fi_lag168: np.ndarray,
        neighbour_prices_lag168: Mapping[str, np.ndarray] | None = None,
        netload_lag168: np.ndarray | None = None,
        is_holiday: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the L2 Ridge design matrix in the order declared by
        the artifact's `ridge_features` field (see `__init__`).

        `neighbour_prices_lag168` must hold, for each forecast hour t, the
        neighbour zone price at **t − NEIGHBOUR_LAG_HOURS** — a value that
        is genuinely known when the forecast is made. Passing same-hour
        prices here would reintroduce the v2.16 leak (see
        NEIGHBOUR_LAG_HOURS); the parameter is named for the contract so
        that cannot happen silently. Missing zones contribute zero.

        `netload_lag168` likewise holds net load (consumption − wind −
        solar, MW) at t − NEIGHBOUR_LAG_HOURS. `is_holiday` is a 0/1 flag
        per forecast hour. Both are optional; absent, they contribute zero.
        """
        n = len(timestamps)
        # Workday excludes public holidays when the caller supplies them —
        # a weekday holiday has weekend-like demand, and treating it as a
        # normal working day is a systematic error on ~900 hours/window.
        wd = self._is_workday(timestamps).astype(float)
        hol = (np.zeros(n, dtype=float) if is_holiday is None
               else np.clip(np.asarray(is_holiday, dtype=float), 0.0, 1.0))
        if hol.shape != (n,):
            hol = np.zeros(n, dtype=float)
        is_workday = wd * (1.0 - hol)
        # Lagged timestamps — the neighbour/net-load features are
        # deseasonalised against the seasonal profile of the hour they
        # were actually observed in, not the forecast hour.
        lag_ts = (timestamps.astype("datetime64[s]")
                  - np.timedelta64(NEIGHBOUR_LAG_HOURS * 3600, "s"))
        # Physics-derived features (intermediate; deseasonalised below).
        wind_rho = _sigmoid_turbine_rho(wind, temp)
        solar_eff = _solar_effective(solar, temp)
        # Deseasonalise with the trainer's stored components (consistent);
        # legacy artifacts without them fall back to local-mean centering.
        Y_wind_rho  = self._deseasonalize_physics("sigmoid_wind_rho", wind_rho, timestamps)
        Y_solar_eff = self._deseasonalize_physics("solar_effective", solar_eff, timestamps)
        Y_temp      = self._deseasonalize_input("temp", temp, timestamps)

        # Cross-border zones — deseasonalised raw prices using the
        # shipped L1 components; zeros if not supplied. NaN entries
        # (alignment gaps) are filled post-deseasonalisation so the
        # Ridge term contributes zero for that hour.
        Y_zone: dict[str, np.ndarray] = {}
        np_arr = neighbour_prices_lag168 or {}
        for zone in _NEIGHBOUR_ZONES:
            raw = np_arr.get(zone) if isinstance(np_arr, Mapping) else None
            if raw is None:
                Y_zone[zone] = np.zeros(n, dtype=float)
                continue
            raw_arr = np.asarray(raw, dtype=float)
            if raw_arr.shape != (n,):
                Y_zone[zone] = np.zeros(n, dtype=float)
                continue
            # If the entire zone is missing (all NaN), short-circuit.
            if not np.any(np.isfinite(raw_arr)):
                Y_zone[zone] = np.zeros(n, dtype=float)
                continue
            # Fill alignment gaps with the zone's local mean so the
            # subsequent deseasonalisation step doesn't propagate NaN.
            finite = raw_arr[np.isfinite(raw_arr)]
            filled = np.where(np.isfinite(raw_arr), raw_arr,
                              float(np.mean(finite)))
            y = self._deseasonalize_input(zone, filled, lag_ts)
            # Final guard against any residual NaN.
            y = np.where(np.isfinite(y), y, 0.0)
            Y_zone[zone] = y

        # Net load at t-168 h, deseasonalised against its own stored
        # components and expressed in GW so the coefficient is readable.
        Y_netload = np.zeros(n, dtype=float)
        if netload_lag168 is not None:
            nl = np.asarray(netload_lag168, dtype=float)
            if nl.shape == (n,) and np.any(np.isfinite(nl)):
                finite = nl[np.isfinite(nl)]
                nl = np.where(np.isfinite(nl), nl, float(np.mean(finite)))
                yv = self._deseasonalize_physics("netload", nl, lag_ts) / 1000.0
                Y_netload = np.where(np.isfinite(yv), yv, 0.0)

        # Map feature name → column array.
        named: dict[str, np.ndarray] = {
            "intercept":          np.ones(n, dtype=float),
            "Y_fi_lag168":        Y_fi_lag168,
            "is_workday":         is_workday,
            "Y_sigmoid_wind_rho": Y_wind_rho,
            "Y_solar_effective":  Y_solar_eff,
            "Y_temp":             Y_temp,
            "Y_se1_lag168":       Y_zone["se1"],
            "Y_se3_lag168":       Y_zone["se3"],
            "Y_ee_lag168":        Y_zone["ee"],
            "Y_netload_lag168":   Y_netload,
            "is_holiday":         hol,
        }

        cols = []
        for f in self._features:
            if f in named:
                cols.append(named[f])
            else:
                # Unknown feature in artifact — zero-pad so the ridge
                # coefficient applied to it contributes nothing rather
                # than crashing the inference cycle.
                _LOGGER.warning("pipeline:unknown feature in artifact: %s", f)
                cols.append(np.zeros(n, dtype=float))
        return np.column_stack(cols)

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
        neighbour_prices_lag168: Mapping[str, np.ndarray] | None = None,
        netload_lag168: np.ndarray | None = None,
        is_holiday: np.ndarray | None = None,
        enable_fan_chart: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute the hourly forecast.

        Args:
            timestamps: 1-D numpy datetime64 array for the forecast horizon.
            wind / solar / temp: matching arrays (Open-Meteo forecast values).
            recent_fi_residuals: optional dict {"lag168": np.ndarray(n)}
                providing Y_fi at t-168 for each forecast hour, AND
                {"last_eta": float} providing the most-recent observed
                post-AR residual (for L3 AR(1) propagation).
            neighbour_prices_lag168: optional dict mapping zone name
                (``"se1"``, ``"se3"``, ``"ee"``) to raw EUR/MWh prices at
                **t − NEIGHBOUR_LAG_HOURS** for each forecast hour t —
                i.e. prices already published when the forecast is made.
                Do NOT pass same-hour prices: those clear in the same
                auction as the FI target and are unknowable at forecast
                time (this was the v2.16 leak). Missing zones contribute
                zero.
            netload_lag168: optional array of net load (consumption −
                wind − solar, MW) at t − NEIGHBOUR_LAG_HOURS.
            is_holiday: optional 0/1 array marking public holidays for
                each forecast hour; also suppresses ``is_workday``.
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
        X = self._build_features(
            timestamps, wind, solar, temp, lag168,
            neighbour_prices_lag168=neighbour_prices_lag168,
            netload_lag168=netload_lag168,
            is_holiday=is_holiday)
        ridge = X @ self._ridge_coef

        # L3 AR(1) propagation
        ar_corr = self._ar_contribution(n)

        # Mean before floor / bias
        mean = seasonal + ridge + ar_corr

        # softplus floor
        mean = _pf.apply_floor(mean, floor=_pf.DEFAULT_FLOOR_EUR_MWH)

        # Per-hour-of-day bias correction (slow-moving EMAs, one per UTC
        # hour — see PerHourBiasCorrector). Bins still warming pass the
        # forecast through unchanged.
        bias = self._bias.bias_estimate if self._bias.warm else 0.0
        hours = (timestamps.astype("datetime64[s]").astype("int64")
                 // 3600) % 24
        mean_corrected = np.array([
            self._bias.correct(float(v), int(h))
            for v, h in zip(mean, hours)
        ])

        out: dict[str, Any] = {
            "mean_eur_mwh": mean_corrected,
            # Pre-correction mean — the coordinator records THIS for later
            # (forecast, actual) reconciliation so the bias EMAs learn on
            # the raw pipeline output, not on their own corrections.
            "mean_uncorrected_eur_mwh": mean,
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
        timestamps: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Feed the calibrators (bias + fan-chart) with realised (pred,
        actual) pairs. Update last_eta for next AR forecast. Returns a
        diagnostics dict including the refit_recommended flag.

        `predicted` should be the PRE-correction pipeline mean
        (``mean_uncorrected_eur_mwh``) so the bias EMAs learn the raw
        model error rather than chasing their own corrections.
        `timestamps` (datetime64, UTC) route each pair to its
        hour-of-day bias bin; when omitted, the bias corrector is
        skipped (the fan-chart calibrator still updates).
        """
        predicted = np.asarray(predicted, dtype=float)
        actual    = np.asarray(actual,    dtype=float)
        n = min(len(predicted), len(actual))
        hours = None
        if timestamps is not None:
            ts = np.asarray(timestamps)
            if ts.shape[0] >= n:
                hours = (ts.astype("datetime64[s]").astype("int64")
                         // 3600) % 24
        for i in range(n):
            if hours is not None:
                self._bias.update(float(predicted[i]), float(actual[i]),
                                  int(hours[i]))
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
