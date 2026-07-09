"""Spot Price Predictor integration for Home Assistant."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
import homeassistant.helpers.config_validation as cv

from .const import DOMAIN, PLATFORMS
from .coordinator import SpotPriceCoordinator
from .model import SpotPriceModel, DEFAULT_COEFS_PATH

_LOGGER = logging.getLogger(__name__)

# Path for user-uploaded coefficients (persists across restarts)
USER_COEFS_DIR = Path(__file__).parent / "data"
USER_COEFS_PATH = USER_COEFS_DIR / "model_coefs_user.json"

SERVICE_UPLOAD_COEFFICIENTS = "upload_coefficients"
SERVICE_RESET_COEFFICIENTS = "reset_coefficients"
SERVICE_MODEL_INFO = "model_info"
SERVICE_FORCE_REFRESH = "force_refresh"
# v2.8.0 — consolidated retraining of the L1+L2+L3+L4 pipeline
SERVICE_RETRAIN_MODELS = "retrain_models"

UPLOAD_SCHEMA = vol.Schema({
    vol.Optional("file_path"): cv.string,
    vol.Optional("json_data"): cv.string,
})

RETRAIN_SCHEMA = vol.Schema({
    # Subset of {"seasonal", "spike", "solar"}. Omit for all three.
    vol.Optional("layers"): vol.All(cv.ensure_list, [cv.string]),
    # Fingrid API key for solar layer (otherwise read from FINGRID_API_KEY env).
    vol.Optional("fingrid_api_key"): cv.string,
})


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Spot Price Predictor from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    try:
        model = await SpotPriceModel.async_load()
    except Exception as err:
        _LOGGER.error("Unexpected error loading model: %s. Using null model.", err)
        model = SpotPriceModel._null_model()
    coordinator = SpotPriceCoordinator(hass, entry, model=model)
    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register listener for options flow changes -> reload integration
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    # v2.12.0 — fast PV-nowcast poll. Registered only when a measured-PV
    # sensor is configured; the tick self-guards otherwise. Cleaned up on
    # unload. The 6-hour coordinator interval is untouched — this timer
    # only requests a (gated) refresh when the live sky drifts from the
    # published forecast.
    if getattr(coordinator, "pv_measured_entity", ""):
        from datetime import timedelta

        from homeassistant.helpers.event import async_track_time_interval

        from .const import PV_NOWCAST_POLL_SECONDS

        entry.async_on_unload(
            async_track_time_interval(
                hass,
                coordinator._async_pv_nowcast_tick,
                timedelta(seconds=PV_NOWCAST_POLL_SECONDS),
            )
        )

    # Register services (once, not per entry)
    if not hass.services.has_service(DOMAIN, SERVICE_UPLOAD_COEFFICIENTS):
        _register_services(hass)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    # Unregister services if no entries left
    if not hass.data.get(DOMAIN):
        hass.services.async_remove(DOMAIN, SERVICE_UPLOAD_COEFFICIENTS)
        hass.services.async_remove(DOMAIN, SERVICE_RESET_COEFFICIENTS)
        hass.services.async_remove(DOMAIN, SERVICE_MODEL_INFO)
        hass.services.async_remove(DOMAIN, SERVICE_FORCE_REFRESH)
        hass.services.async_remove(DOMAIN, SERVICE_RETRAIN_MODELS)

    return unload_ok


async def _async_update_listener(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Reload integration when options change."""
    _LOGGER.info("Configuration updated, reloading integration")
    await hass.config_entries.async_reload(entry.entry_id)


def _register_services(hass: HomeAssistant) -> None:
    """Register integration services."""

    async def handle_upload_coefficients(call: ServiceCall) -> None:
        """Upload new model coefficients from file or JSON string.

        After training on PC, use this service to update the model:
          python -m src.train_model --region finland --fingrid-key YOUR_KEY
          # Then call this service with file_path pointing to output/model_coefs.json
        """
        file_path = call.data.get("file_path")
        json_data = call.data.get("json_data")

        if not file_path and not json_data:
            _LOGGER.error("Either file_path or json_data must be provided")
            return

        try:
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    _LOGGER.error("Coefficients file not found: %s", file_path)
                    return
                with open(path, "r", encoding="utf-8") as f:
                    coefs = json.load(f)
            else:
                coefs = json.loads(json_data)

            # Validate structure (v2.0 log-linear model format)
            required_keys = ["intercept", "features", "feature_names"]
            missing = [k for k in required_keys if k not in coefs]
            if missing:
                _LOGGER.error("Invalid coefficients: missing keys %s", missing)
                return

            # Save to user coefficients path
            USER_COEFS_DIR.mkdir(parents=True, exist_ok=True)
            with open(USER_COEFS_PATH, "w", encoding="utf-8") as f:
                json.dump(coefs, f, indent=2)

            _LOGGER.info(
                "Model coefficients uploaded: %s, %d features, version %s",
                USER_COEFS_PATH,
                coefs.get("feature_count", "?"),
                coefs.get("model_version", "?"),
            )

            # Reload all coordinators to use new coefficients
            for entry_id, coordinator in hass.data.get(DOMAIN, {}).items():
                if isinstance(coordinator, SpotPriceCoordinator):
                    coordinator.model = SpotPriceModel.load(USER_COEFS_PATH)
                    await coordinator.async_request_refresh()

            hass.components.persistent_notification.async_create(
                f"Model coefficients updated successfully.\n"
                f"Features: {coefs.get('feature_count', '?')}\n"
                f"Version: {coefs.get('model_version', '?')}\n"
                f"Data sources: {coefs.get('data_sources', {})}",
                title="Spot Price Predictor",
                notification_id=f"{DOMAIN}_coefs_updated",
            )

        except json.JSONDecodeError as err:
            _LOGGER.error("Invalid JSON in coefficients: %s", err)
        except Exception as err:
            _LOGGER.exception("Failed to upload coefficients: %s", err)

    async def handle_reset_coefficients(call: ServiceCall) -> None:
        """Reset to bundled default coefficients."""
        if USER_COEFS_PATH.exists():
            USER_COEFS_PATH.unlink()
            _LOGGER.info("User coefficients removed, reverting to defaults")

        for entry_id, coordinator in hass.data.get(DOMAIN, {}).items():
            if isinstance(coordinator, SpotPriceCoordinator):
                coordinator.model = SpotPriceModel.load()
                await coordinator.async_request_refresh()

        hass.components.persistent_notification.async_create(
            "Model reset to bundled default coefficients.",
            title="Spot Price Predictor",
            notification_id=f"{DOMAIN}_coefs_reset",
        )

    async def handle_model_info(call: ServiceCall) -> None:
        """Show current model information as a persistent notification."""
        using_user = USER_COEFS_PATH.exists()
        path = USER_COEFS_PATH if using_user else DEFAULT_COEFS_PATH

        try:
            with open(path, "r", encoding="utf-8") as f:
                coefs = json.load(f)

            metrics = coefs.get("metrics", {})
            info = (
                f"**Model source:** {'User-uploaded' if using_user else 'Bundled default'}\n"
                f"**Version:** {coefs.get('model_version', 'unknown')}\n"
                f"**Features:** {coefs.get('feature_count', '?')}\n"
                f"**Data sources:** {coefs.get('data_sources', {})}\n"
                f"**MAE:** {metrics.get('mae', '?')} EUR/MWh\n"
                f"**R²:** {metrics.get('r2', '?')}\n"
                f"**Train samples:** {metrics.get('train_samples', '?')}\n"
                f"**Test samples:** {metrics.get('test_samples', '?')}\n"
                f"**Duration model:** {'Yes' if 'duration_model' in coefs else 'No'}\n\n"
                f"To retrain, run on your PC:\n"
                f"```\n"
                f"python -m src.train_model --region finland --fingrid-key YOUR_KEY\n"
                f"```\n"
                f"Then upload with the `{DOMAIN}.{SERVICE_UPLOAD_COEFFICIENTS}` service."
            )

            hass.components.persistent_notification.async_create(
                info,
                title="Spot Price Predictor — Model Info",
                notification_id=f"{DOMAIN}_model_info",
            )
        except Exception as err:
            _LOGGER.error("Failed to read model info: %s", err)

    async def handle_force_refresh(call: ServiceCall) -> None:
        """Force an immediate data refresh on all coordinators."""
        _LOGGER.info("Manual refresh triggered via service call")
        for entry_id, coordinator in hass.data.get(DOMAIN, {}).items():
            if isinstance(coordinator, SpotPriceCoordinator):
                await coordinator.async_request_refresh()
        _LOGGER.info("Manual refresh completed")

    async def handle_retrain_models(call: ServiceCall) -> None:
        """Refit all (or a subset of) the prediction-pipeline artifacts.

        Runs the orchestrator from `retrain.py` in an executor so the
        long-running CPU+IO work doesn't block the HA event loop. Once
        complete, every active coordinator gets a fresh Pipeline
        instance pointing at the refreshed artifacts.
        """
        from . import retrain as _retrain
        from .pipeline import Pipeline

        layers = call.data.get("layers")
        fingrid_key = call.data.get("fingrid_api_key")

        _LOGGER.info("retrain_models invoked: layers=%s", layers or "all")

        def _run() -> dict:
            return _retrain.retrain_all(
                layers=layers,
                fingrid_api_key=fingrid_key,
            )

        try:
            result = await hass.async_add_executor_job(_run)
        except Exception as err:
            _LOGGER.exception("retrain_models failed: %s", err)
            hass.components.persistent_notification.async_create(
                f"Model retraining failed: {err}",
                title="Spot Price Predictor",
                notification_id=f"{DOMAIN}_retrain_failed",
            )
            return

        # Reload the Pipeline on every active coordinator so the new
        # artifacts take effect immediately (without waiting for a
        # full HA restart).
        data_dir = Path(__file__).parent / "data"
        reloaded = 0
        for entry_id, coordinator in hass.data.get(DOMAIN, {}).items():
            if not isinstance(coordinator, SpotPriceCoordinator):
                continue
            try:
                storage_dir = Path(
                    hass.config.path(".storage", "spot_price_predictor_pipeline"))
                coordinator._pipeline = Pipeline(
                    data_dir=data_dir, storage_dir=storage_dir)
                await coordinator.async_request_refresh()
                reloaded += 1
            except Exception as e:
                _LOGGER.warning(
                    "Failed to reload Pipeline on %s: %s", entry_id, e)

        # Summary notification
        ok = result.get("ok", False)
        layers_done = list(result.get("results", {}).keys())
        details_lines = []
        for layer, r in result.get("results", {}).items():
            if r.get("error"):
                details_lines.append(f"- **{layer}**: error — {r['error']}")
            elif r.get("skipped"):
                details_lines.append(
                    f"- **{layer}**: skipped — {r.get('reason', '?')}")
            else:
                details_lines.append(f"- **{layer}**: refit OK")
        msg = (
            f"Retraining {'succeeded ✓' if ok else 'finished with issues'} "
            f"on {len(layers_done)} layer(s).\n"
            f"Coordinator reloads: {reloaded}\n\n"
            + "\n".join(details_lines)
        )
        hass.components.persistent_notification.async_create(
            msg, title="Spot Price Predictor — Retraining",
            notification_id=f"{DOMAIN}_retrain_complete",
        )
        # Emit a HA event so automations can chain on completion
        hass.bus.async_fire(
            f"{DOMAIN}_models_retrained",
            {"result": result, "reloaded_coordinators": reloaded},
        )
        _LOGGER.info("retrain_models complete: ok=%s, layers=%s, reloaded=%d",
                     ok, layers_done, reloaded)

    hass.services.async_register(
        DOMAIN, SERVICE_UPLOAD_COEFFICIENTS, handle_upload_coefficients,
        schema=UPLOAD_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_RESET_COEFFICIENTS, handle_reset_coefficients,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_MODEL_INFO, handle_model_info,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_FORCE_REFRESH, handle_force_refresh,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_RETRAIN_MODELS, handle_retrain_models,
        schema=RETRAIN_SCHEMA,
    )
