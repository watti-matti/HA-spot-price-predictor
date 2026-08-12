"""The backtest harness's PRODUCTION config must equal the real pipeline.

Backlog D6: the harness refit L1 every month while production runs the
frozen artifact, so offline work concluded weekday *under*-prediction
while the field saw *over*-prediction. Every model experiment is judged
against this harness, so if its reference config drifts from
`Pipeline.compute_forecast` the conclusions drift with it.

This test pins them together on synthetic inputs. It also catches the
failure that let the drift go unnoticed for three releases: between
v2.17.0 and v2.18.0 the harness asserted an eight-feature list against a
nine-feature artifact and could not run at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SPP = REPO / "custom_components" / "spot_price_predictor"

# Import the pipeline as a package member (it uses `from . import ...`).
_pkg = types.ModuleType("spp_parity")
_pkg.__path__ = [str(SPP)]
sys.modules["spp_parity"] = _pkg
for _m in ("seasonal_decomposition", "solar_clear_sky", "price_floor",
           "bias_corrector", "dtaci", "hourly_calibration", "pipeline"):
    _s = importlib.util.spec_from_file_location(f"spp_parity.{_m}", SPP / f"{_m}.py")
    _mod = importlib.util.module_from_spec(_s)
    sys.modules[f"spp_parity.{_m}"] = _mod
    _s.loader.exec_module(_mod)
pipeline_mod = sys.modules["spp_parity.pipeline"]


def _harness_production_fn():
    """Load `build_production_prediction` without importing the whole
    harness module, which pulls in the study data pipeline."""
    src = (REPO / "studies" / "backtest_harness.py").read_text(encoding="utf-8")
    start = src.index("def build_production_prediction")
    end = src.index("def build_predictions")
    ns = {
        "np": np, "pd": pd, "json": json,
        "sd": sys.modules["spp_parity.seasonal_decomposition"],
        "_pf": sys.modules["spp_parity.price_floor"],
        "DATA_DIR": SPP / "data",
        "SEASONAL_ARTIFACT": json.loads(
            (SPP / "data" / "seasonal_components_default.json").read_text(
                encoding="utf-8")),
        "NEIGHBOUR_LAG_HOURS": 168,
        "PRICE_FLOOR": -5.0,
    }
    sys.path.insert(0, str(SPP))
    from holidays import build_holiday_set          # noqa: E402
    ns["build_holiday_set"] = build_holiday_set
    exec(compile(src[start:end], "backtest_harness.py", "exec"), ns)
    return ns["build_production_prediction"]


@pytest.fixture(scope="module")
def synthetic():
    """420 hours of deterministic weather and neighbour prices."""
    n = 420
    idx = pd.date_range("2026-03-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    wind = 3.0 + 6.0 * (1 + np.sin(np.arange(n) / 17.0)) + rng.normal(0, 0.4, n)
    ghi = np.clip(400 * np.sin(np.arange(n) * np.pi / 12.0), 0, None)
    temp = 2.0 + 8.0 * np.sin(np.arange(n) / 24.0) + rng.normal(0, 0.5, n)
    nb = {z: 30.0 + 20.0 * np.sin(np.arange(n) / (11.0 + i))
          for i, z in enumerate(("se1", "se3", "ee"))}
    return idx, wind, ghi, temp, nb


def test_harness_production_matches_pipeline(synthetic, tmp_path):
    idx, wind, ghi, temp, nb = synthetic
    n = len(idx)
    lag = pipeline_mod.NEIGHBOUR_LAG_HOURS

    # --- harness side: a df shaped like build_dataframe's output ---
    df = pd.DataFrame({
        "fi": np.zeros(n),                     # unused by the production path
        "sigmoid_wind_rho": pipeline_mod._sigmoid_turbine_rho(wind, temp),
        "solar_effective": pipeline_mod._solar_effective(ghi, temp),
        "temp": temp,
        **nb,
    }, index=idx)
    harness_pred = _harness_production_fn()(df)

    # --- pipeline side: same inputs, neighbours pre-lagged by the caller ---
    p = pipeline_mod.Pipeline(data_dir=SPP / "data", storage_dir=tmp_path)
    hol = {z: pd.Series(v, index=idx).shift(lag).fillna(0.0).values
           for z, v in nb.items()}
    is_holiday = np.zeros(n)          # March 2026 window has none
    out = p.compute_forecast(
        timestamps=idx.tz_localize(None).values.astype("datetime64[ns]"),
        wind=wind, solar=ghi, temp=temp,
        neighbour_prices_lag168=hol,
        is_holiday=is_holiday,
        enable_fan_chart=False,
    )
    pipe_pred = out["mean_eur_mwh"]

    # Compare only where the 168 h lag is defined for both sides. The
    # harness fills the leading window with 0.0 after deseasonalising,
    # the pipeline receives 0.0 from the caller — same value, but the
    # deseasonalisation reference differs there, so skip it.
    m = np.arange(n) >= lag
    assert m.sum() > 200
    diff = np.abs(np.asarray(harness_pred)[m] - np.asarray(pipe_pred)[m])
    assert diff.max() < 1e-6, (
        f"harness PRODUCTION and Pipeline.compute_forecast disagree by up to "
        f"{diff.max():.6f} EUR/MWh. The harness is the reference every model "
        f"experiment is judged against; if it drifts from the deployed "
        f"pipeline the conclusions drift with it (docs/BACKLOG.md D6)."
    )


def test_harness_feature_list_matches_the_shipped_artifact():
    """The mismatch that silently broke the harness for three releases."""
    src = (REPO / "studies" / "backtest_harness.py").read_text(encoding="utf-8")
    ns: dict = {}
    start = src.index("FEATS = [")
    exec(src[start:src.index("\n", src.index("]", start))], ns)
    art = json.loads(
        (SPP / "data" / "spike_model_default.json").read_text(encoding="utf-8"))
    assert ns["FEATS"] == list(art["ridge_features"]), (
        "studies/backtest_harness.py FEATS is out of step with the shipped "
        "artifact. Update FEATS and build_production_prediction together."
    )


def test_pipeline_coefficient_vector_has_the_intercept_first():
    """The alignment trap: `ridge_coef` carries the intercept, but
    `ridge_features` omits it. Read them zipped and the wind coefficient
    is applied to solar."""
    art = json.loads(
        (SPP / "data" / "spike_model_default.json").read_text(encoding="utf-8"))
    assert len(art["ridge_coef"]) == len(art["ridge_features"]) + 1
    feats = ["intercept"] + list(art["ridge_features"])
    wind = art["ridge_coef"][feats.index("Y_sigmoid_wind_rho")]
    solar = art["ridge_coef"][feats.index("Y_solar_effective")]
    assert wind < -1.0, f"wind coefficient {wind} is not a wind-scale value"
    assert -1.0 < solar <= 0.0, f"solar coefficient {solar} out of range"
