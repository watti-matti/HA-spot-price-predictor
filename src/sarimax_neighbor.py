"""
Hourly SARIMAX trainer for neighbor-zone spot prices (SE1, SE3, EE).

Adapts the reference daily SARIMAX from
`SW resources for sport price estimation/sarimax_nordpool.py` to hourly
granularity. Because SARIMAX with a 168-hour seasonal state is
computationally prohibitive (state dim ~168), we use the equivalent
"regression with ARMA errors" formulation:

  Model:  SARIMAX(2, 0, 1)(0, 0, 0, 0)
  Exog (captures all periodic structure):
    - Day-of-week dummies (Sun=ref): mon..sat
    - Holiday flag (country-specific via `holidays` package)
    - Diurnal Fourier (period 24, K=2): sin_d1..cos_d2
    - Diurnal x weekend interaction: weekend-shape vs workday-shape
    - Annual Fourier (period 365.25 d, K=2): sin_y1..cos_y2

This captures the weekly + daily + annual structure that the original
SARIMA seasonal state would have, with O(n) instead of O(n*s^2)
fit cost. Empirically equivalent for forecasting accuracy in
electricity price literature.

Training is PC-only (statsmodels). The fitted model exports a
serializable dict that pure-Python inference on HA can replay.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Country mapping for neighbor zones
ZONE_COUNTRIES: dict[str, str] = {
    "se1": "SE",
    "se3": "SE",
    "ee":  "EE",
}


def build_calendar_features_hourly(
    date_index: pd.DatetimeIndex,
    country: str = "SE",
    include_diurnal_fourier: bool = True,
    diurnal_K: int = 2,
    include_weekend_interaction: bool = True,
    include_annual_fourier: bool = True,
    annual_K: int = 2,
) -> pd.DataFrame:
    """Build hourly calendar exog regressors aligned with `date_index`.

    Columns:
      mon, tue, wed, thu, fri, sat                       (Sun=ref, day-of-week dummies)
      is_holiday                                          (country-specific)
      sin_d1, cos_d1, ..., sin_dK, cos_dK                 (diurnal cycle, period 24)
      sin_d1_we, cos_d1_we, ...                           (diurnal x weekend interaction)
      sin_y1, cos_y1, ..., sin_yK, cos_yK                 (annual cycle)
    """
    import holidays as _holidays

    country_map = {
        "SE": _holidays.Sweden,
        "EE": _holidays.Estonia,
        "FI": _holidays.Finland,
        "NO": _holidays.Norway,
        "DK": _holidays.Denmark,
        "DE": _holidays.Germany,
    }
    HolidayClass = country_map.get(country.upper(), _holidays.Sweden)
    years = date_index.year.unique().tolist()
    all_years = sorted(set(years + [max(years) + 1]))
    country_holidays = HolidayClass(years=all_years)

    df = pd.DataFrame(index=date_index)

    # Day-of-week dummies (Sunday = reference)
    dow = date_index.dayofweek
    for i, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat"]):
        df[name] = (dow == i).astype(int)

    # Holiday flag (compares date part only)
    holiday_dates = set(country_holidays)
    df["is_holiday"] = pd.Series(
        [d.date() in holiday_dates for d in date_index],
        index=date_index,
    ).astype(int)

    is_weekend = (dow.values >= 5).astype(float) if hasattr(dow, "values") else (np.asarray(dow) >= 5).astype(float)

    # Diurnal Fourier (period 24 hours)
    if include_diurnal_fourier:
        hour_fraction = date_index.hour.astype(float).values / 24.0
        for k in range(1, diurnal_K + 1):
            sin_d = np.sin(2 * np.pi * k * hour_fraction)
            cos_d = np.cos(2 * np.pi * k * hour_fraction)
            df[f"sin_d{k}"] = sin_d
            df[f"cos_d{k}"] = cos_d
            if include_weekend_interaction:
                df[f"sin_d{k}_we"] = sin_d * is_weekend
                df[f"cos_d{k}_we"] = cos_d * is_weekend

    # Annual Fourier
    if include_annual_fourier:
        doy = date_index.dayofyear.astype(float).values
        hod = date_index.hour.astype(float).values / 24.0
        t = (doy + hod) / 365.25
        for k in range(1, annual_K + 1):
            df[f"sin_y{k}"] = np.sin(2 * np.pi * k * t)
            df[f"cos_y{k}"] = np.cos(2 * np.pi * k * t)

    return df


def build_calendar_features_hour_workday(
    date_index: pd.DatetimeIndex,
    country: str = "SE",
) -> pd.DataFrame:
    """Option A — Hour-of-day x workday/weekend dummies.

    Strict structural superset of AR(2)'s profile_wd[24] + profile_we[24].
    Calendar columns (~46):
      h1..h23                  (23 hour-of-day dummies, hour 0 = ref)
      h1_we..h23_we            (23 hour-of-day x weekend interactions)
      is_weekend               (overall weekend level shift)
      is_holiday               (country-specific)
      sin_y1..cos_y2           (annual Fourier, K=2)
    """
    import holidays as _holidays

    country_map = {
        "SE": _holidays.Sweden, "EE": _holidays.Estonia,
        "FI": _holidays.Finland, "NO": _holidays.Norway,
    }
    HolidayClass = country_map.get(country.upper(), _holidays.Sweden)
    years = date_index.year.unique().tolist()
    all_years = sorted(set(years + [max(years) + 1]))
    country_holidays = HolidayClass(years=all_years)

    df = pd.DataFrame(index=date_index)

    # Hour-of-day dummies (hour 0 = reference)
    hod = np.asarray(date_index.hour)
    is_weekend = (np.asarray(date_index.dayofweek) >= 5).astype(int)
    for h in range(1, 24):
        df[f"h{h}"] = (hod == h).astype(int)
    df["is_weekend"] = is_weekend
    # Hour x weekend interactions
    for h in range(1, 24):
        df[f"h{h}_we"] = (hod == h).astype(int) * is_weekend

    holiday_dates = set(country_holidays)
    df["is_holiday"] = pd.Series(
        [d.date() in holiday_dates for d in date_index],
        index=date_index,
    ).astype(int)

    # Annual Fourier (small contribution but captures slow drift)
    doy = date_index.dayofyear.astype(float).values
    hod_f = date_index.hour.astype(float).values / 24.0
    t = (doy + hod_f) / 365.25
    for k in range(1, 3):
        df[f"sin_y{k}"] = np.sin(2 * np.pi * k * t)
        df[f"cos_y{k}"] = np.cos(2 * np.pi * k * t)

    return df


def build_calendar_features_hour_of_week(
    date_index: pd.DatetimeIndex,
    country: str = "SE",
) -> pd.DataFrame:
    """Option B — Full hour-of-week dummies (167 features).

    Strictly more expressive than AR(2)'s profile structure: each
    (hour, day-of-week) cell gets its own free intercept. Total 167
    dummies (Sunday h0 = reference).

    Calendar columns (~172):
      how_1..how_167           (Sunday h0 = ref)
      is_holiday               (country-specific)
      sin_y1..cos_y2           (annual Fourier, K=2)
    """
    import holidays as _holidays

    country_map = {
        "SE": _holidays.Sweden, "EE": _holidays.Estonia,
        "FI": _holidays.Finland, "NO": _holidays.Norway,
    }
    HolidayClass = country_map.get(country.upper(), _holidays.Sweden)
    years = date_index.year.unique().tolist()
    all_years = sorted(set(years + [max(years) + 1]))
    country_holidays = HolidayClass(years=all_years)

    df = pd.DataFrame(index=date_index)

    # hour-of-week index: 0..167 (Mon h0=0, ..., Sun h23=167)
    # Use Sunday h0=0..23 as the *first* slots so we can drop how_0 as ref
    # Actually simplest: how = dayofweek*24 + hour (Mon=0 -> Sun=6 mapping in pandas)
    # Drop how_0 (Monday hour 0) as reference
    how = date_index.dayofweek.values * 24 + date_index.hour.values  # 0..167
    for k in range(1, 168):
        df[f"how_{k}"] = (how == k).astype(int)

    holiday_dates = set(country_holidays)
    df["is_holiday"] = pd.Series(
        [d.date() in holiday_dates for d in date_index],
        index=date_index,
    ).astype(int)

    doy = date_index.dayofyear.astype(float).values
    hod_f = date_index.hour.astype(float).values / 24.0
    t = (doy + hod_f) / 365.25
    for k in range(1, 3):
        df[f"sin_y{k}"] = np.sin(2 * np.pi * k * t)
        df[f"cos_y{k}"] = np.cos(2 * np.pi * k * t)

    return df


class HourlyNordPoolSARIMAX:
    """Hourly SARIMAX wrapper for Nord Pool neighbor zones.

    Default architecture:  SARIMAX(2, 0, 1)(0, 0, 0, 0) with rich calendar exog.
    The "regression with ARMA errors" formulation captures weekly/daily/annual
    patterns through exog regressors instead of a 168-period seasonal state
    (which has prohibitive O(s^2) Kalman filter cost for hourly data).

    Calendar exog regressors:
      - day-of-week dummies (Sun=ref): mon..sat
      - is_holiday (country-specific)
      - diurnal Fourier (period 24, K=2): sin/cos at 1x and 2x daily frequency
      - diurnal x weekend interaction: separate diurnal shape on weekends
      - annual Fourier (period 365.25 d, K=2)

    Total exog: ~16 regressors. ARMA(2,1) handles short-term residual dynamics.
    """

    def __init__(
        self,
        country: str = "SE",
        order: tuple[int, int, int] = (2, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        diurnal_K: int = 2,
        annual_K: int = 2,
        include_weekend_interaction: bool = True,
        exog_mode: str = "fourier",
    ):
        """
        Parameters
        ----------
        exog_mode : str
            "fourier"        — original: ~14 features (dow dummies + diurnal Fourier
                               + weekend interaction + annual Fourier).
            "hour-workday"   — option A: ~46 features (hour-of-day x workday/weekend
                               dummies + holiday + annual Fourier). Strictly matches
                               AR(2)'s profile_wd/profile_we structure.
            "hour-of-week"   — option B: ~172 features (full hour-of-week dummies
                               + holiday + annual Fourier). Strictly more expressive
                               than AR(2).
        """
        self.country = country
        self.order = order
        self.seasonal_order = seasonal_order
        self.diurnal_K = diurnal_K
        self.annual_K = annual_K
        self.include_weekend_interaction = include_weekend_interaction
        self.exog_mode = exog_mode
        self.result_: Any = None
        self.exog_cols_: list[str] | None = None
        self._train_index: pd.DatetimeIndex | None = None

    def _build_exog(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        if self.exog_mode == "hour-workday":
            return build_calendar_features_hour_workday(idx, country=self.country)
        if self.exog_mode == "hour-of-week":
            return build_calendar_features_hour_of_week(idx, country=self.country)
        # Default: "fourier"
        cal = build_calendar_features_hourly(
            idx,
            country=self.country,
            include_diurnal_fourier=self.diurnal_K > 0,
            diurnal_K=self.diurnal_K,
            include_weekend_interaction=self.include_weekend_interaction,
            include_annual_fourier=self.annual_K > 0,
            annual_K=self.annual_K,
        )
        return cal

    def fit(self, prices: pd.Series, *, verbose: bool = False) -> "HourlyNordPoolSARIMAX":
        """Fit SARIMAX to an hourly price series.

        Parameters
        ----------
        prices : pd.Series with DatetimeIndex (hourly frequency)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must have a DatetimeIndex")

        cal = self._build_exog(prices.index)
        self.exog_cols_ = list(cal.columns)
        exog_train = cal

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                prices,
                exog=exog_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.result_ = model.fit(disp=False, maxiter=200)
        self._train_index = prices.index

        if verbose:
            logger.info(self.result_.summary())

        return self

    def forecast(self, horizon: int = 168) -> pd.Series:
        """Forecast `horizon` hours after training data.

        Returns hourly EUR/MWh forecasts as a pd.Series.
        """
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .forecast()")

        last = self._train_index[-1]
        future = pd.date_range(
            start=last + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )
        cal_future = self._build_exog(future)
        exog_future = cal_future[self.exog_cols_]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.result_.get_forecast(steps=horizon, exog=exog_future)
            mean = pred.predicted_mean
        return pd.Series(mean.values, index=future, name="forecast_eur_mwh")

    def export_coefs(self) -> dict[str, Any]:
        """Serialize fitted model for pure-Python HA inference.

        The exported dict carries enough state to reproduce forecasts via the
        SARIMAX recursion without statsmodels.
        """
        if self.result_ is None:
            raise RuntimeError("Call .fit() before export_coefs()")

        # Extract parameter slices by name pattern. statsmodels names them like:
        #   ar.L1, ar.L2, ma.L1, ar.S.L168, ma.S.L168, sigma2, <exog name>, ...
        params = self.result_.params
        names = list(params.index) if hasattr(params, "index") else list(self.result_.param_names)
        vals = list(params.values) if hasattr(params, "values") else list(params)
        idx = {n: i for i, n in enumerate(names)}

        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        phi = [vals[idx[f"ar.L{i}"]] for i in range(1, p + 1)] if p > 0 else []
        theta = [vals[idx[f"ma.L{i}"]] for i in range(1, q + 1)] if q > 0 else []
        Phi = [vals[idx[f"ar.S.L{i*s}"]] for i in range(1, P + 1)] if P > 0 else []
        Theta = [vals[idx[f"ma.S.L{i*s}"]] for i in range(1, Q + 1)] if Q > 0 else []
        sigma2 = vals[idx["sigma2"]] if "sigma2" in idx else float("nan")

        exog_coefs = [float(vals[idx[c]]) for c in (self.exog_cols_ or [])]

        # Last `s + p + max(d, 1)` observations needed to seed recursion.
        # For SARIMAX(p,d,q)(P,D,Q)[s] with d=0, D=1, we need >= s + max(p, P*s, q, Q*s) values.
        n_state = s + max(p, P * s, q, Q * s)
        train_y = self.result_.data.endog
        last_obs = [float(x) for x in train_y[-n_state:]]

        # Innovations (residuals) — last (q + s*Q) values are needed for MA terms.
        n_innov = max(q + s * Q, 1)
        last_innov = [float(x) for x in self.result_.resid[-n_innov:]]

        return {
            "model_type": "sarimax",
            "country": self.country,
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "exog_cols": list(self.exog_cols_ or []),
            "exog_coefs": [float(c) for c in exog_coefs],
            "phi":   [float(c) for c in phi],
            "theta": [float(c) for c in theta],
            "Phi":   [float(c) for c in Phi],
            "Theta": [float(c) for c in Theta],
            "sigma2": float(sigma2),
            "last_obs":   last_obs,
            "last_innov": last_innov,
            "diurnal_K": self.diurnal_K,
            "annual_K": self.annual_K,
            "include_weekend_interaction": self.include_weekend_interaction,
        }


def train_sarimax_neighbors(
    neighbor_prices: dict[str, pd.Series],
    *,
    order: tuple[int, int, int] = (2, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    diurnal_K: int = 2,
    annual_K: int = 2,
    include_weekend_interaction: bool = True,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train one HourlyNordPoolSARIMAX per zone and export coefficients.

    Parameters
    ----------
    neighbor_prices : dict mapping zone key (se1/se3/ee) -> hourly pd.Series

    Returns
    -------
    dict mapping zone key -> exported coefficient dict (JSON-serializable)
    """
    out: dict[str, dict[str, Any]] = {}
    for zone, prices in neighbor_prices.items():
        country = ZONE_COUNTRIES.get(zone.lower(), "SE")
        logger.info("Training SARIMAX for %s (country=%s, n_obs=%d)",
                    zone, country, len(prices))
        model = HourlyNordPoolSARIMAX(
            country=country,
            order=order,
            seasonal_order=seasonal_order,
            diurnal_K=diurnal_K,
            annual_K=annual_K,
            include_weekend_interaction=include_weekend_interaction,
        )
        model.fit(prices, verbose=verbose)
        out[zone] = model.export_coefs()
        logger.info("  %s done — phi=%s, sigma2=%.2f",
                    zone, out[zone]["phi"], out[zone]["sigma2"])
    return out
