"""NPK-CVaR hedge analysis tool — Python port of npk_cvar_hedge_demo.m.

Implements the validation methodology specified in the v2.4.x → v2.5.0 plan:
every new model variant must reduce out-of-sample CVaR on a hedge-ratio
optimization to be accepted. The hedge mechanic is the user's MATLAB-validated
"if test CVaR drops, the feature captures real signal; if unchanged, it's noise."

Functions:
    fit_seasonal_hdw(x, ts) -> P_hour, P_day, P_week, seasonal_t, Y_t
    fit_ou_ar1(Y) -> {lambda, mu, sigma, half_life, b}
    npk_cvar_objective(h, v, rS, rF, alpha) -> J
    optimize_hedge(rS, rF, alpha=0.05, train_frac=0.55) -> dict
    historical_cvar(L, alpha) -> float
    acf(y, lags) -> array

References:
    Moazeni, Powell, Hajimiragha (IEEE TPwrS, Sept 2013).
    Cartea & Figueroa (Applied Mathematical Finance 12(4), 2005).
    User's MATLAB study in studies/Matlab_study_on_CVAR/.

All pure NumPy + SciPy; no homeassistant dependency; runs standalone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import erfc


# ----------------------------------------------------------------------------
# Seasonal decomposition (sequential subtraction per Moazeni-Powell)
# ----------------------------------------------------------------------------


def fit_seasonal_hdw(
    x: np.ndarray,
    ts: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential-subtraction seasonal decomposition: x = P_hour + P_day + P_week + Y.

    Step 1: P_hour[h] = mean(x | hour=h)
    Step 2: P_day[d]  = mean(x - P_hour[hour] | day-of-week=d)
    Step 3: P_week[w] = mean(x - P_hour[hour] - P_day[dow] | week-of-year=w)

    Result: E[seasonal] = E[x], E[Y] = 0 exactly.

    Parameters
    ----------
    x : array (N,)
        Input variable observations.
    ts : DatetimeIndex (N,)
        Timestamps (any timezone; local time is what matters for hour/dow).

    Returns
    -------
    P_hour : array (24,)    hour-of-day seasonal factors
    P_day  : array (7,)     day-of-week seasonal factors (0=Mon)
    P_week : array (53,)    week-of-year seasonal factors
    seasonal : array (N,)   P_hour[h] + P_day[d] + P_week[w] per observation
    Y : array (N,)          residual x - seasonal, zero-mean by construction
    """
    x = np.asarray(x, dtype=float)
    if len(x) != len(ts):
        raise ValueError(f"length mismatch: x={len(x)} vs ts={len(ts)}")
    if x.size == 0:
        raise ValueError("empty input")

    h = ts.hour.to_numpy()
    d = ts.weekday.to_numpy()
    w = (ts.isocalendar().week.to_numpy()) - 1
    w = np.clip(w, 0, 52)

    # Step 1: hour-of-day mean on raw x
    P_hour = np.zeros(24)
    for i in range(24):
        mask = (h == i)
        if mask.any():
            P_hour[i] = np.nanmean(x[mask])

    # Step 2: day-of-week mean on (x - P_hour[h])
    r1 = x - P_hour[h]
    P_day = np.zeros(7)
    for i in range(7):
        mask = (d == i)
        if mask.any():
            P_day[i] = np.nanmean(r1[mask])

    # Step 3: week-of-year mean on (x - P_hour[h] - P_day[dow])
    r2 = r1 - P_day[d]
    P_week = np.zeros(53)
    valid_weeks = []
    for i in range(53):
        mask = (w == i)
        if mask.any():
            P_week[i] = np.nanmean(r2[mask])
            valid_weeks.append(i)

    # Nearest-neighbour fill for unobserved weeks
    if len(valid_weeks) < 53 and len(valid_weeks) > 0:
        valid_arr = np.array(valid_weeks)
        for i in range(53):
            if i not in valid_arr:
                nearest = valid_arr[np.argmin(np.abs(valid_arr - i))]
                P_week[i] = P_week[nearest]

    seasonal = P_hour[h] + P_day[d] + P_week[w]
    Y = x - seasonal
    return P_hour, P_day, P_week, seasonal, Y


# ----------------------------------------------------------------------------
# OU / AR(1) fit on deseasonalized residual
# ----------------------------------------------------------------------------


def fit_ou_ar1(Y: np.ndarray, dt_hours: float = 1.0) -> dict:
    """Fit AR(1) discrete-time approximation of Ornstein-Uhlenbeck.

    Model: Y_{k+1} = a + b*Y_k + eps
    OU mapping: b = exp(-lambda*dt), a = mu*(1-b)
    Volatility: sigma = std(eps) / sqrt((1 - exp(-2*lambda*dt)) / (2*lambda))

    Parameters
    ----------
    Y : array (N,)
        Deseasonalized residual.
    dt_hours : float
        Sampling interval in hours (default 1.0).

    Returns
    -------
    dict with keys: a, b, lambda_per_hour, mu, sigma, half_life_hours,
                    residuals, n_samples
    """
    Y = np.asarray(Y, dtype=float)
    Y = Y[np.isfinite(Y)]
    if len(Y) < 10:
        raise ValueError(f"not enough samples to fit OU: {len(Y)}")

    Yk = Y[:-1]
    Yk1 = Y[1:]
    X = np.column_stack([np.ones_like(Yk), Yk])
    beta, *_ = np.linalg.lstsq(X, Yk1, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    eps = Yk1 - X @ beta

    b_clamped = float(np.clip(b, 1e-6, 0.999999))
    lam = -np.log(b_clamped) / dt_hours
    mu = a / (1.0 - b_clamped)
    half_life = np.log(2.0) / lam
    scale = np.sqrt((1.0 - np.exp(-2.0 * lam * dt_hours)) / (2.0 * lam))
    sigma = float(np.std(eps) / max(scale, 1e-12))

    return {
        "a": a,
        "b": b,
        "lambda_per_hour": float(lam),
        "mu": float(mu),
        "sigma": sigma,
        "half_life_hours": float(half_life),
        "residuals": eps,
        "n_samples": int(len(Y)),
    }


# ----------------------------------------------------------------------------
# NPK-CVaR objective (Rockafellar + Gaussian kernel smoothing)
# ----------------------------------------------------------------------------


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erfc (no scipy.stats import)."""
    return 0.5 * erfc(-z / np.sqrt(2.0))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def npk_cvar_objective(
    h: float, v: float, rS: np.ndarray, rF: np.ndarray, alpha: float
) -> float:
    """Rockafellar form of kernel-smoothed CVaR.

    CVaR(L) = min_v  v + (1/alpha) * E[(L - v)+]
    where L = -(rS - h*rF) is the loss of the hedged portfolio.
    The expectation is taken under a Gaussian KDE with Silverman bandwidth.

    Parameters
    ----------
    h : float       hedge ratio
    v : float       VaR-style threshold (loss domain)
    rS : array      spot returns/changes
    rF : array      futures/forecast returns/changes
    alpha : float   tail probability (0 < alpha < 1; e.g. 0.05 = 95 % conf)

    Returns
    -------
    float : objective value (lower is better)
    """
    L = -(rS - h * rF)
    T = len(L)
    sigma = float(np.nanstd(L))
    bw = max(1.06 * sigma * T ** (-1.0 / 5.0), 1e-8)
    z = (v - L) / bw
    Phi = _norm_cdf(z)
    phi = _norm_pdf(z)
    tail_mean_plus = float(np.nanmean((L - v) * (1.0 - Phi) + bw * phi))
    return v + (1.0 / alpha) * tail_mean_plus


def historical_cvar(L: np.ndarray, alpha: float) -> float:
    """Empirical CVaR: mean of losses above the (1-alpha) quantile.

    Used as a sanity check vs the kernel-smoothed objective.
    """
    L = np.asarray(L, dtype=float)
    L = L[np.isfinite(L)]
    if L.size == 0:
        return float("nan")
    q = float(np.quantile(L, 1.0 - alpha))
    tail = L[L >= q]
    return float(np.mean(tail)) if tail.size else float("nan")


def optimize_hedge(
    rS: np.ndarray,
    rF: np.ndarray,
    alpha: float = 0.05,
    train_frac: float = 0.55,
    h_bounds: tuple[float, float] = (-5.0, 5.0),
) -> dict:
    """Optimize the hedge ratio h to minimize kernel CVaR; report train + test.

    Train/test split is chronological (first train_frac for fitting, remainder
    for out-of-sample evaluation) — matches the MATLAB demo's protocol.

    Returns
    -------
    dict with: h_hat, v_hat, cvar_train_kernel, cvar_train_hist_unhedged,
               cvar_train_hist_hedged, cvar_test_hist_unhedged,
               cvar_test_hist_hedged, n_train, n_test, alpha, train_frac
    """
    rS = np.asarray(rS, dtype=float)
    rF = np.asarray(rF, dtype=float)
    mask = np.isfinite(rS) & np.isfinite(rF)
    rS, rF = rS[mask], rF[mask]
    n = len(rS)
    if n < 100:
        raise ValueError(f"not enough samples for hedge optimization: {n}")
    n_train = max(50, int(train_frac * n))
    rS_tr, rF_tr = rS[:n_train], rF[:n_train]
    rS_te, rF_te = rS[n_train:], rF[n_train:]

    # Initial guess: minimum-variance hedge ratio h_mv = cov(S, F) / var(F)
    var_F = float(np.var(rF_tr))
    if var_F < 1e-12:
        h0 = 0.0
    else:
        h0 = float(np.cov(rS_tr, rF_tr)[0, 1] / var_F)
    h0 = float(np.clip(h0, h_bounds[0], h_bounds[1]))

    L0 = -(rS_tr - h0 * rF_tr)
    v0 = float(np.quantile(L0, 1.0 - alpha))

    def obj_pen(x):
        h_pen = float(np.clip(x[0], h_bounds[0], h_bounds[1]))
        v = float(x[1])
        penalty = 1e3 * (
            max(0.0, h_bounds[0] - x[0]) ** 2
            + max(0.0, x[0] - h_bounds[1]) ** 2
        )
        return npk_cvar_objective(h_pen, v, rS_tr, rF_tr, alpha) + penalty

    result = optimize.minimize(
        obj_pen,
        x0=np.array([h0, v0]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000, "maxfev": 20000},
    )
    h_hat = float(np.clip(result.x[0], h_bounds[0], h_bounds[1]))
    v_hat = float(result.x[1])

    return {
        "h_hat": h_hat,
        "v_hat": v_hat,
        "alpha": alpha,
        "train_frac": train_frac,
        "n_train": n_train,
        "n_test": n - n_train,
        "cvar_train_kernel": float(
            npk_cvar_objective(h_hat, v_hat, rS_tr, rF_tr, alpha)
        ),
        "cvar_train_hist_unhedged": historical_cvar(-rS_tr, alpha),
        "cvar_train_hist_hedged": historical_cvar(-(rS_tr - h_hat * rF_tr), alpha),
        "cvar_test_hist_unhedged": historical_cvar(-rS_te, alpha),
        "cvar_test_hist_hedged": historical_cvar(-(rS_te - h_hat * rF_te), alpha),
    }


def acf(y: np.ndarray, lags: list[int] | tuple[int, ...]) -> dict[int, float]:
    """Sample autocorrelation at specified lags. Used for OU diagnostics."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < max(lags) + 1:
        raise ValueError(f"need ≥ {max(lags) + 1} samples for lag {max(lags)}")
    yc = y - y.mean()
    denom = float(np.sum(yc * yc))
    out = {}
    for k in lags:
        out[k] = float(np.sum(yc[:-k] * yc[k:]) / denom) if denom > 0 else 0.0
    return out


# ----------------------------------------------------------------------------
# Convenience: run full pipeline on a (timestamps, prices) series
# ----------------------------------------------------------------------------


def run_baseline_hedge_analysis(
    ts: pd.DatetimeIndex,
    P: np.ndarray,
    *,
    alpha: float = 0.05,
    futures_lag_hours: int = 48,
    train_frac: float = 0.55,
    mode: str = "raw",
) -> dict:
    """End-to-end NPK-CVaR hedge analysis on a single price series.

    Replicates the MATLAB demo's two modes:

    mode='raw':
        rS = diff(P), rF = diff(seasonal forecast shifted by futures_lag_hours).
        Expected result on MATLAB benchmarks: h ≈ 0.99, ~7.9 % CVaR reduction.

    mode='deseasonalized':
        rS = diff(Y, lag=futures_lag_hours+1), rF = diff(Y, lag=1) shifted.
        Expected: h ≈ -0.12 at 2h lag; ~0 hedgeable risk at ≥ 48h lag.

    Parameters
    ----------
    ts : DatetimeIndex
    P : array (N,) prices
    alpha : tail probability
    futures_lag_hours : lag of the hedge instrument
    train_frac : training fraction (chronological split)
    mode : 'raw' or 'deseasonalized'

    Returns
    -------
    dict with seasonal decomposition stats, OU fit, hedge results.
    """
    P_hour, P_day, P_week, seasonal, Y = fit_seasonal_hdw(P, ts)
    ou = fit_ou_ar1(Y)

    if mode == "raw":
        # Use future-shifted seasonal as the "futures" hedge instrument
        Fwd = np.concatenate(
            [seasonal[futures_lag_hours:], np.repeat(seasonal[-1], futures_lag_hours)]
        )
        rS = np.diff(P)
        rF = np.diff(Fwd)
        return_label = "ΔP (EUR/MWh)"
    elif mode == "deseasonalized":
        dY = np.diff(Y)
        rS = dY[futures_lag_hours:]
        rF = dY[: -futures_lag_hours if futures_lag_hours > 0 else len(dY)]
        return_label = "ΔY (EUR/MWh)"
    else:
        raise ValueError(f"mode must be 'raw' or 'deseasonalized', got {mode!r}")

    hedge = optimize_hedge(rS, rF, alpha=alpha, train_frac=train_frac)
    variance_total = float(np.var(P))
    variance_seasonal = float(np.var(seasonal))
    variance_resid = float(np.var(Y))

    return {
        "mode": mode,
        "futures_lag_hours": futures_lag_hours,
        "alpha": alpha,
        "n_observations": int(len(P)),
        "return_label": return_label,
        "seasonal": {
            "P_hour": P_hour,
            "P_day": P_day,
            "P_week": P_week,
            "var_pct_of_total": 100.0 * variance_seasonal / variance_total if variance_total else 0.0,
        },
        "residual_Y": {
            "var_pct_of_total": 100.0 * variance_resid / variance_total if variance_total else 0.0,
            "mean": float(np.mean(Y)),
            "std": float(np.std(Y)),
        },
        "ou": ou,
        "hedge": hedge,
    }
