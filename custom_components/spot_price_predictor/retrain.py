"""v2.8.0 — Consolidated model-retraining orchestrator.

Single entry point that refits every artifact the v26 pipeline depends on:

  L1 seasonal      → data/seasonal_components_default.json
  L2+L3+L4 spike   → data/spike_model_default.json
  L4 solar sub-model → data/solar_submodel_default.json   (optional)

Each layer's refit reads cached parquets under `output/`. The solar
sub-model additionally needs a Fingrid API key to refresh the ENTSO-E-
equivalent dataset 248 (Finnish solar generation forecast). If no key
is available the solar refit is skipped and the existing artifact is
left in place.

Designed to be invoked from a Home Assistant service handler so the
operator can refresh the model on demand (e.g. when the RefitMonitor
flags drift, or quarterly as a hygiene task). Each layer's refit
returns metadata; the orchestrator atomically writes the new artifact
and the integration's V26Pipeline reloads on the next coordinator cycle.

This module deliberately does NOT depend on Home Assistant — it is
pure-python so it can be exercised standalone via
`python -m custom_components.spot_price_predictor.retrain` for
debugging.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

_LOGGER = logging.getLogger(__name__)

# Layers retrainable
ALL_LAYERS = ("seasonal", "spike", "solar")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to a temp file in the same directory, then rename.

    Atomic on POSIX; on Windows it's two ops but still safer than
    truncating the live file. Avoids leaving a partially-written
    artifact if anything crashes mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # On Windows os.replace is atomic if the destination exists
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def _ensure_repo_imports(repo_root: Path) -> None:
    """Add repo + studies + custom_components to sys.path so the
    layer-specific helpers can be imported without going through the
    package __init__ (which depends on Home Assistant)."""
    for p in (repo_root,
              repo_root / "studies",
              repo_root / "custom_components" / "spot_price_predictor"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# ── Layer refitters ────────────────────────────────────────────────


def retrain_seasonal(repo_root: Path, data_dir: Path) -> dict:
    """Refit L1 seasonal components from the latest cached parquets.

    Returns metadata: train window, per-input variance reduction,
    output artifact path.
    """
    _ensure_repo_imports(repo_root)
    import build_seasonal_components as bsc

    _LOGGER.info("retrain L1 seasonal: refitting from cached parquets")
    # The build script's main() writes the artifact to the data dir
    # using its own constants — we run it directly here.
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        bsc.main()
    finally:
        os.chdir(old_cwd)

    artifact = data_dir / "seasonal_components_default.json"
    if not artifact.exists():
        raise FileNotFoundError(f"seasonal artifact missing after refit: {artifact}")
    meta = json.loads(artifact.read_text(encoding="utf-8"))
    return {
        "layer": "seasonal",
        "artifact": str(artifact),
        "version": meta.get("version", "?"),
        "train_window": meta.get("train_window"),
        "stats": meta.get("stats", {}),
        "inputs_fit": list((meta.get("components") or {}).keys()),
    }


def retrain_spike_model(repo_root: Path, data_dir: Path) -> dict:
    """Refit L2 Ridge + L3 AR(1) + L4 GPD POT in one pass.

    Writes data/spike_model_default.json. Returns the L4 GPD POT
    diagnostics + CVaR back-test summary.
    """
    _ensure_repo_imports(repo_root)
    import v2513_layer4_spike_model as l4

    _LOGGER.info("retrain L2+L3+L4: refitting Ridge + AR(1) + GPD POT")
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        l4.main()
    finally:
        os.chdir(old_cwd)

    artifact = data_dir / "spike_model_default.json"
    if not artifact.exists():
        raise FileNotFoundError(f"spike artifact missing after refit: {artifact}")
    meta = json.loads(artifact.read_text(encoding="utf-8"))
    return {
        "layer": "spike",
        "artifact": str(artifact),
        "version": meta.get("version", "?"),
        "ar1_phi": meta.get("ar1_phi"),
        "ridge_feature_count": len(meta.get("ridge_features", [])),
        "gpd_right_shape": (meta.get("gpd_right") or {}).get("shape"),
        "gpd_left_shape":  (meta.get("gpd_left")  or {}).get("shape"),
        "train_window": meta.get("train_window"),
        "cvar_backtest": meta.get("cvar_backtest", []),
    }


def retrain_solar_submodel(
    repo_root: Path, data_dir: Path,
    fingrid_api_key: str | None = None,
) -> dict:
    """Refit the v2.5.3 clear-sky × cloudiness solar sub-model.

    Requires Fingrid API key (dataset 248 = FI solar generation).
    If `fingrid_api_key` is None, looks for `FINGRID_API_KEY` env var
    or skips the refit and returns a skip marker.
    """
    if not fingrid_api_key:
        fingrid_api_key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not fingrid_api_key:
        _LOGGER.warning("retrain solar: no Fingrid API key — skipping")
        return {
            "layer": "solar",
            "skipped": True,
            "reason": "no FINGRID_API_KEY provided",
        }

    _ensure_repo_imports(repo_root)
    import solar_clear_sky_submodel as scsm

    _LOGGER.info("retrain solar: refitting clear-sky × cloudiness sub-model")
    os.environ["FINGRID_API_KEY"] = fingrid_api_key
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        scsm.main()
    finally:
        os.chdir(old_cwd)

    artifact = data_dir / "solar_submodel_default.json"
    if not artifact.exists():
        raise FileNotFoundError(f"solar artifact missing after refit: {artifact}")
    meta = json.loads(artifact.read_text(encoding="utf-8"))
    return {
        "layer": "solar",
        "artifact": str(artifact),
        "version": meta.get("version", "?"),
        "clear_sky_model": meta.get("clear_sky_model"),
        "modulator_form": meta.get("modulator_form"),
        "capacity_ref_mw": meta.get("capacity_ref_mw"),
        "test_metrics": meta.get("test_metrics", {}),
    }


# ── Orchestrator ───────────────────────────────────────────────────


def retrain_all(
    repo_root: Path | None = None,
    data_dir: Path | None = None,
    layers: Iterable[str] | None = None,
    fingrid_api_key: str | None = None,
) -> dict:
    """Refit one or more model artifacts.

    Args:
        repo_root: project root containing `studies/` and `output/`.
            Defaults to two levels up from this module.
        data_dir: directory where artifact JSON files live. Defaults
            to `custom_components/spot_price_predictor/data/`.
        layers: subset of `("seasonal", "spike", "solar")` to refit.
            Defaults to all three.
        fingrid_api_key: needed for the solar layer only. Falls back
            to `FINGRID_API_KEY` env var.

    Returns:
        Aggregate result dict with `started_at`, `completed_at`, and
        per-layer metadata.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    layers = list(layers) if layers else list(ALL_LAYERS)

    started = datetime.now(timezone.utc).isoformat()
    results: dict[str, Any] = {}

    for layer in layers:
        try:
            if layer == "seasonal":
                results[layer] = retrain_seasonal(repo_root, data_dir)
            elif layer == "spike":
                results[layer] = retrain_spike_model(repo_root, data_dir)
            elif layer == "solar":
                results[layer] = retrain_solar_submodel(
                    repo_root, data_dir, fingrid_api_key)
            else:
                results[layer] = {"layer": layer, "error":
                                   f"unknown layer name: {layer!r}"}
        except Exception as e:
            _LOGGER.exception("retrain %s failed", layer)
            results[layer] = {"layer": layer, "error": str(e)}

    completed = datetime.now(timezone.utc).isoformat()
    return {
        "started_at":    started,
        "completed_at":  completed,
        "layers":        layers,
        "results":       results,
        "ok":            all(
            "error" not in r and not r.get("skipped", False)
            for r in results.values()
        ),
    }


# ── CLI entrypoint ────────────────────────────────────────────────


def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Retrain the v26 model artifacts.")
    parser.add_argument("--layers", nargs="+", choices=list(ALL_LAYERS),
                        default=list(ALL_LAYERS),
                        help="Subset of layers to retrain.")
    parser.add_argument("--fingrid-key", default=None,
                        help="Fingrid API key (for solar layer).")
    args = parser.parse_args()

    result = retrain_all(layers=args.layers,
                          fingrid_api_key=args.fingrid_key)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
