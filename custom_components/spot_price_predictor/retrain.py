"""Consolidated model-retraining orchestrator.

Single entry point that refits every artifact the prediction pipeline depends on:

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
and the integration's Pipeline reloads on the next coordinator cycle.

This module deliberately does NOT depend on Home Assistant — it is
pure-python so it can be exercised standalone. Run it as a FILE, not
with ``-m``: ``python custom_components/spot_price_predictor/retrain.py``
(the ``-m`` form imports the package ``__init__``, which pulls in Home
Assistant and fails on a training PC). The ``scripts/retrain_model.*``
helpers wrap this.
"""

from __future__ import annotations

import contextlib
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


def pipeline_ridge_features(repo_root: Path) -> tuple[str, ...]:
    """Read `pipeline.RIDGE_FEATURES` without importing the module.

    `pipeline.py` uses package-relative imports (`from . import ...`),
    so a bare `import pipeline` raises, and importing it as a package
    member drags in Home Assistant — which is absent on a training PC.
    The feature list is a module-level tuple of string literals, so the
    source can simply be parsed. Static reading is also the safer choice
    for a guard: it cannot be perturbed by import side effects.
    """
    src = (repo_root / "custom_components" / "spot_price_predictor"
           / "pipeline.py").read_text(encoding="utf-8")
    import ast

    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "RIDGE_FEATURES"
               for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError(
        "could not find RIDGE_FEATURES in pipeline.py — the guard cannot "
        "verify the artifact, so refusing rather than installing blind.")


def _validate_spike_payload(meta: dict, source: Path, repo_root: Path) -> None:
    """Refuse a spike artifact the running pipeline cannot consume.

    This guard exists because of a real, silent regression. Until
    v2.18.2 this module refitted via ``v2513_layer4_spike_model``, a
    v2.5.13-era trainer that writes a FIVE-feature model — no neighbour
    prices, no holiday — over its own short train window. It replaced
    the shipped nine-feature ``v2.17.1`` artifact with no error at all;
    the only symptom was a worse forecast. Validating the candidate
    against the pipeline's own feature tuple turns any future
    trainer/pipeline divergence into a loud failure here, before the
    shipped artifact is touched.
    """
    # pipeline.RIDGE_FEATURES carries "intercept" in slot 0; the
    # artifact's `ridge_features` names only the real columns, with the
    # intercept implied as the first coefficient. Compare accordingly.
    full = list(pipeline_ridge_features(repo_root))
    if not full or full[0] != "intercept":
        raise RuntimeError(
            "pipeline.RIDGE_FEATURES no longer starts with 'intercept'; the "
            "guard's alignment assumption is stale. Refusing to install.")
    expected = full[1:]
    got = list(meta.get("ridge_features") or [])
    if got != expected:
        raise ValueError(
            f"spike feature mismatch in {source.name}: pipeline expects "
            f"{len(expected)} features {expected}, trainer wrote {len(got)} "
            f"{got}. Refusing to install."
        )
    coef = meta.get("ridge_coef") or []
    if len(coef) != len(full):
        raise ValueError(
            f"spike coefficient mismatch in {source.name}: {len(coef)} "
            f"coefficients for {len(expected)} features — the intercept is "
            f"first, so {len(full)} are required."
        )
    if not meta.get("physics_seasonal"):
        raise ValueError(
            f"spike artifact {source.name} has no `physics_seasonal` block, "
            "so the pipeline would deseasonalise the wind/solar physics "
            "terms inconsistently with training."
        )


def retrain_spike_model(repo_root: Path, data_dir: Path) -> dict:
    """Refit L2 Ridge + L3 AR(1) + L4 GPD POT in one pass.

    Writes data/spike_model_default.json. Returns the L4 GPD POT
    diagnostics + CVaR back-test summary.

    Refits with ``studies/build_fresh_spike_model.py`` — the trainer
    that produces the shipped nine-feature ``v2.17.1/leakfree-holiday``
    form. That script deliberately writes a CANDIDATE under output/
    instead of overwriting the shipped artifact, so this function
    validates the candidate and only then installs it atomically.
    """
    _ensure_repo_imports(repo_root)
    import build_fresh_spike_model as bfsm

    _LOGGER.info("retrain L2+L3+L4: refitting Ridge + AR(1) + GPD POT")
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        bfsm.main()
    finally:
        os.chdir(old_cwd)

    candidate = repo_root / "output" / "spike_model_fresh.json"
    if not candidate.exists():
        raise FileNotFoundError(
            f"spike candidate missing after refit: {candidate}")
    meta = json.loads(candidate.read_text(encoding="utf-8"))
    _validate_spike_payload(meta, candidate, repo_root)

    artifact = data_dir / "spike_model_default.json"
    _atomic_write_json(artifact, meta)
    _LOGGER.info("retrain L2+L3+L4: installed %d-feature %s, window %s",
                 len(meta.get("ridge_features") or []),
                 meta.get("version"), meta.get("train_window"))
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

    No Fingrid API key is needed. The sub-model used to fetch datasets
    248 and 267 live on every run; it now reads the national PV series
    (247) and installed capacity (267) from the cached data store, so
    the only network call left is Open-Meteo cloud cover, which is
    unauthenticated. `fingrid_api_key` is accepted and exported for
    backward compatibility but is no longer required.

    The data store itself still needs a key to populate those columns —
    `python -m src.data_store update` — and `load_pv_from_store` raises
    with that instruction if they are absent.
    """
    if not fingrid_api_key:
        fingrid_api_key = os.environ.get("FINGRID_API_KEY", "").strip()

    _ensure_repo_imports(repo_root)
    import solar_clear_sky_submodel as scsm

    _LOGGER.info("retrain solar: refitting clear-sky × cloudiness sub-model")
    if fingrid_api_key:
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


ARTIFACT_NAMES = {
    "seasonal": "seasonal_components_default.json",
    "spike":    "spike_model_default.json",
    "solar":    "solar_submodel_default.json",
}


class _Transaction:
    """Handle yielded by `_artifacts_transaction`; commit to keep writes."""

    def __init__(self) -> None:
        self.committed = False
        # Set by the context manager after a rollback; None when the
        # transaction committed and so nothing was restored.
        self.restore_verified: bool | None = None

    def commit(self) -> None:
        self.committed = True


@contextlib.contextmanager
def _artifacts_transaction(data_dir: Path, names: Iterable[str]):
    """All-or-nothing artifact writes.

    The shipped artifacts are snapshotted on entry and restored on exit
    unless the block calls `commit()`. This makes a batch retrain
    transactional: `retrain_all` catches each layer's exception and
    carries on, so without this a run that fails half-way leaves the
    earlier layers' writes in place. That is how the 2026-10-12 run put
    a regressed five-feature spike artifact on disk while reporting
    `ok: false` — the caller's `|| goto :fail` blocked staging, but the
    shipped file had already been overwritten.
    """
    tx = _Transaction()
    names = list(names)
    saved = {n: (data_dir / n).read_bytes()
             for n in names if (data_dir / n).exists()}
    absent = [n for n in names if n not in saved]
    try:
        yield tx
    finally:
        if not tx.committed:
            for n, blob in saved.items():
                (data_dir / n).write_bytes(blob)
            for n in absent:
                p = data_dir / n
                if p.exists():
                    p.unlink()
            # Verify rather than assume. Callers report
            # `shipped_artifacts_untouched` to the operator, and a
            # rollback that silently half-worked would be worse than no
            # rollback at all — it would be an untrue reassurance.
            bad = [n for n, blob in saved.items()
                   if (data_dir / n).read_bytes() != blob]
            bad += [n for n in absent if (data_dir / n).exists()]
            tx.restore_verified = not bad
            if bad:
                raise RuntimeError(
                    "artifact rollback FAILED to restore: "
                    f"{', '.join(sorted(bad))} in {data_dir}. The shipped "
                    "artifacts may be modified — check `git status` and "
                    "restore with `git checkout -- <path>` before using "
                    "the model.")


@contextlib.contextmanager
def _artifacts_preserved(data_dir: Path, names: Iterable[str]):
    """Restore the shipped artifacts on exit, whatever happened inside.

    The layer trainers are study scripts that write to the shipped
    artifact path using their own module constants, and the solar one
    builds that path inline at two separate sites. Rather than refactor
    three scripts to accept an output directory, snapshot the bytes and
    put them back. That makes candidate mode non-destructive regardless
    of where a trainer decides to write — including when it crashes
    half-way, which is exactly how the 2026-10-12 run left a regressed
    spike artifact in place.
    """
    names = list(names)
    saved = {n: (data_dir / n).read_bytes()
             for n in names if (data_dir / n).exists()}
    absent = [n for n in names if n not in saved]
    try:
        yield
    finally:
        for n, blob in saved.items():
            (data_dir / n).write_bytes(blob)
        # A trainer may have created an artifact that did not exist
        # before; leaving it behind would silently become "shipped".
        for n in absent:
            p = data_dir / n
            if p.exists():
                p.unlink()


def retrain_all(
    repo_root: Path | None = None,
    data_dir: Path | None = None,
    layers: Iterable[str] | None = None,
    fingrid_api_key: str | None = None,
    candidate_dir: Path | None = None,
    allow_partial: bool = False,
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
        allow_partial: keep the artifacts written by layers that did
            succeed even when another layer failed. Off by default: a
            batch retrain is all-or-nothing, because a half-updated set
            of artifacts is not a model anyone validated.
        candidate_dir: when set, the refitted artifacts are collected
            here and the shipped ones under `data_dir` are restored
            byte-for-byte. Nothing the running integration reads is
            changed, so a candidate can be trained and evaluated while
            production keeps serving the current model.

    Returns:
        Aggregate result dict with `started_at`, `completed_at`, and
        per-layer metadata.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    layers = list(layers) if layers else list(ALL_LAYERS)

    names = [ARTIFACT_NAMES[l] for l in layers if l in ARTIFACT_NAMES]

    if candidate_dir is not None:
        candidate_dir = Path(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)

    with _artifacts_transaction(data_dir, names) as tx:
        res = _retrain_layers(repo_root, data_dir, layers, fingrid_api_key)

        if candidate_dir is not None:
            for name in names:
                src = data_dir / name
                if src.exists():
                    shutil.copy2(src, candidate_dir / name)
            # Never commit in candidate mode: the shipped artifacts get
            # restored and the candidate lives only under candidate_dir.
        else:
            # All-or-nothing. A SKIPPED layer (e.g. solar without a
            # Fingrid key) is not a failure and must not discard the
            # layers that did refit, so key the decision on errors only
            # — unlike `ok`, which also reports False for a skip.
            failed = sorted(l for l, r in res["results"].items()
                            if "error" in r)
            if failed and not allow_partial:
                res["rolled_back"] = True
                res["rolled_back_because"] = failed
                res["artifacts_installed"] = []
                _LOGGER.error(
                    "retrain rolled back: %s failed, so the shipped "
                    "artifacts were restored. Nothing was installed.",
                    ", ".join(failed))
            else:
                tx.commit()
                res["rolled_back"] = False
                res["artifacts_installed"] = names
                if failed:
                    _LOGGER.warning(
                        "retrain kept partial results (--allow-partial): %s "
                        "failed", ", ".join(failed))

    # Outside the `with`, so the transaction has finished restoring and
    # verifying. Report what was checked, not what was intended.
    if candidate_dir is not None:
        res["candidate_dir"] = str(candidate_dir)
        res["shipped_artifacts_untouched"] = bool(tx.restore_verified)
    return res


def _retrain_layers(
    repo_root: Path,
    data_dir: Path,
    layers: list[str],
    fingrid_api_key: str | None,
) -> dict:

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
        description="Retrain the prediction-pipeline model artifacts.")
    parser.add_argument("--layers", nargs="+", choices=list(ALL_LAYERS),
                        default=list(ALL_LAYERS),
                        help="Subset of layers to retrain.")
    parser.add_argument("--fingrid-key", default=None,
                        help="Fingrid API key (for solar layer).")
    parser.add_argument(
        "--candidate-dir", default=None, metavar="DIR",
        help="Train into DIR and restore the shipped artifacts, so the "
             "running model is left untouched. Evaluate the candidate "
             "before promoting it.")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Keep artifacts from layers that succeeded even if another "
             "layer failed. Default is all-or-nothing rollback.")
    args = parser.parse_args()

    result = retrain_all(layers=args.layers,
                          fingrid_api_key=args.fingrid_key,
                          candidate_dir=args.candidate_dir,
                          allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
