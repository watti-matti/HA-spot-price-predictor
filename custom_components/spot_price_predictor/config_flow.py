"""Config flow for Spot Price Predictor."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api_client import SpotPriceApiClient
from .const import (
    DOMAIN,
    CONF_REGION,
    CONF_OPERATOR,
    CONF_FINGRID_API_KEY,
    CONF_ENABLE_NEIGHBOR_PRICES,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
    CONF_SELLER_MARGIN,
    DEFAULT_SELLER_MARGIN,
    CONF_NORDPOOL_ENTITY,
    CONF_ENABLE_PV_SELLING,
    CONF_PV_SELL_COMMISSION,
    DEFAULT_PV_SELL_COMMISSION,
    CONF_ENABLE_DTACI_DK,
    DEFAULT_ENABLE_DTACI_DK,
    CONF_PV_CAPACITY_KWP,
    CONF_PV_TILT_DEG,
    CONF_PV_AZIMUTH_DEG,
    CONF_PV_SYSTEM_EFFICIENCY,
    CONF_PV_EXTERNAL_ENTITY,
    CONF_PV_EXPORT_GRID_FEE,
    CONF_BASELOAD_KWH_PER_HOUR,
    CONF_BASELOAD_DAY_FACTOR,
    CONF_BASELOAD_NIGHT_FACTOR,
    CONF_ANNUAL_CONSUMPTION_KWH,
    CONF_CONSUMPTION_ENTITY,
    DEFAULT_PV_CAPACITY_KWP,
    DEFAULT_PV_TILT_DEG,
    DEFAULT_PV_AZIMUTH_DEG,
    DEFAULT_PV_SYSTEM_EFFICIENCY,
    DEFAULT_PV_EXPORT_GRID_FEE,
    DEFAULT_BASELOAD_KWH_PER_HOUR,
    DEFAULT_BASELOAD_DAY_FACTOR,
    DEFAULT_BASELOAD_NIGHT_FACTOR,
    DEFAULT_ANNUAL_CONSUMPTION_KWH,
    DEFAULT_CONSUMPTION_ENTITY,
    REGIONS,
    OPERATORS,
    DEFAULT_VAT_MULTIPLIER,
    DEFAULT_ENERGY_TAX,
)

_LOGGER = logging.getLogger(__name__)


def _infer_legacy_annual_kwh(current: dict) -> float:
    """Compute annual_consumption_kwh equivalent of legacy v2.3 baseload.

    Used to populate the Options form's default when migrating an entry
    that was created in v2.3.x. Mirrors the inference in coordinator
    `__init__` so the user sees a consistent number.
    """
    if CONF_ANNUAL_CONSUMPTION_KWH in current:
        return float(current[CONF_ANNUAL_CONSUMPTION_KWH])
    base = float(current.get(
        CONF_BASELOAD_KWH_PER_HOUR, DEFAULT_BASELOAD_KWH_PER_HOUR))
    day = float(current.get(
        CONF_BASELOAD_DAY_FACTOR, DEFAULT_BASELOAD_DAY_FACTOR))
    night = float(current.get(
        CONF_BASELOAD_NIGHT_FACTOR, DEFAULT_BASELOAD_NIGHT_FACTOR))
    avg = base * ((day * 15 + night * 9) / 24.0)
    return round(avg * 8760.0, 0) or DEFAULT_ANNUAL_CONSUMPTION_KWH


class SpotPricePredictorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Spot Price Predictor."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 1: Select region."""
        if user_input is not None:
            self._data[CONF_REGION] = user_input[CONF_REGION]
            return await self.async_step_operator()

        schema = vol.Schema({
            vol.Required(CONF_REGION, default="finland"): vol.In(
                {k: v for k, v in REGIONS.items()}
            ),
        })
        return self.async_show_form(step_id="user", data_schema=schema)

    async def async_step_operator(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 2: Select operator and tariff."""
        if user_input is not None:
            self._data[CONF_OPERATOR] = user_input[CONF_OPERATOR]
            if user_input[CONF_OPERATOR] == "custom":
                self._data[CONF_CUSTOM_DAY_RATE] = user_input.get(
                    CONF_CUSTOM_DAY_RATE, 0.05
                )
                self._data[CONF_CUSTOM_NIGHT_RATE] = user_input.get(
                    CONF_CUSTOM_NIGHT_RATE, 0.04
                )
                self._data[CONF_CUSTOM_VAT] = user_input.get(
                    CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER
                )
                self._data[CONF_CUSTOM_ENERGY_TAX] = user_input.get(
                    CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX
                )
            else:
                # Store operator defaults so they're visible in options flow
                op = OPERATORS.get(user_input[CONF_OPERATOR], OPERATORS["elenia"])
                self._data[CONF_CUSTOM_DAY_RATE] = op["day_rate"]
                self._data[CONF_CUSTOM_NIGHT_RATE] = op["night_rate"]
                self._data[CONF_CUSTOM_VAT] = DEFAULT_VAT_MULTIPLIER
                self._data[CONF_CUSTOM_ENERGY_TAX] = DEFAULT_ENERGY_TAX
            self._data[CONF_SELLER_MARGIN] = user_input.get(
                CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN
            )
            self._data[CONF_NORDPOOL_ENTITY] = user_input.get(
                CONF_NORDPOOL_ENTITY, ""
            )
            self._data[CONF_ENABLE_PV_SELLING] = user_input.get(
                CONF_ENABLE_PV_SELLING, False
            )
            self._data[CONF_PV_SELL_COMMISSION] = user_input.get(
                CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION
            )
            return await self.async_step_optional_apis()

        operator_options = {k: v["name"] for k, v in OPERATORS.items()}
        schema = vol.Schema({
            vol.Required(CONF_OPERATOR, default="elenia"): vol.In(operator_options),
            vol.Optional(
                CONF_SELLER_MARGIN,
                default=DEFAULT_SELLER_MARGIN,
                description={"suggested_value": DEFAULT_SELLER_MARGIN},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.10)),
            vol.Optional(
                CONF_CUSTOM_DAY_RATE,
                default=0.0361,
                description={"suggested_value": 0.0361},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.50)),
            vol.Optional(
                CONF_CUSTOM_NIGHT_RATE,
                default=0.0220,
                description={"suggested_value": 0.0220},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.50)),
            vol.Optional(
                CONF_CUSTOM_VAT,
                default=DEFAULT_VAT_MULTIPLIER,
                description={"suggested_value": DEFAULT_VAT_MULTIPLIER},
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=2.0)),
            vol.Optional(
                CONF_CUSTOM_ENERGY_TAX,
                default=DEFAULT_ENERGY_TAX,
                description={"suggested_value": DEFAULT_ENERGY_TAX},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.20)),
            vol.Optional(CONF_NORDPOOL_ENTITY, default=""): str,
            vol.Optional(CONF_ENABLE_PV_SELLING, default=False): bool,
            vol.Optional(
                CONF_PV_SELL_COMMISSION,
                default=DEFAULT_PV_SELL_COMMISSION,
                description={"suggested_value": DEFAULT_PV_SELL_COMMISSION},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.10)),
        })
        return self.async_show_form(step_id="operator", data_schema=schema)

    async def async_step_optional_apis(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 3: Optional API keys and data source selection."""
        errors: dict[str, str] = {}

        if user_input is not None:
            fingrid_key = user_input.get(CONF_FINGRID_API_KEY, "").strip()
            if fingrid_key:
                # Validate key
                session = async_get_clientsession(self.hass)
                client = SpotPriceApiClient(session, fingrid_key)
                valid = await client.validate_fingrid_key()
                if not valid:
                    errors["base"] = "invalid_fingrid_key"

            if not errors:
                if fingrid_key:
                    self._data[CONF_FINGRID_API_KEY] = fingrid_key
                self._data[CONF_ENABLE_NEIGHBOR_PRICES] = user_input.get(
                    CONF_ENABLE_NEIGHBOR_PRICES, True
                )
                self._data[CONF_ENABLE_DTACI_DK] = user_input.get(
                    CONF_ENABLE_DTACI_DK, DEFAULT_ENABLE_DTACI_DK
                )

                return await self.async_step_pv_system()

        schema = vol.Schema({
            vol.Optional(CONF_FINGRID_API_KEY, default=""): str,
            vol.Optional(CONF_ENABLE_NEIGHBOR_PRICES, default=True): bool,
            vol.Optional(
                CONF_ENABLE_DTACI_DK,
                default=DEFAULT_ENABLE_DTACI_DK,
            ): bool,
        })
        return self.async_show_form(
            step_id="optional_apis", data_schema=schema, errors=errors
        )

    async def async_step_pv_system(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Step 4: Optional household PV system + baseload (skip if no PV)."""
        if user_input is not None:
            self._data[CONF_PV_CAPACITY_KWP] = float(
                user_input.get(CONF_PV_CAPACITY_KWP, DEFAULT_PV_CAPACITY_KWP)
            )
            self._data[CONF_PV_TILT_DEG] = float(
                user_input.get(CONF_PV_TILT_DEG, DEFAULT_PV_TILT_DEG)
            )
            self._data[CONF_PV_AZIMUTH_DEG] = float(
                user_input.get(CONF_PV_AZIMUTH_DEG, DEFAULT_PV_AZIMUTH_DEG)
            )
            self._data[CONF_PV_SYSTEM_EFFICIENCY] = float(
                user_input.get(CONF_PV_SYSTEM_EFFICIENCY, DEFAULT_PV_SYSTEM_EFFICIENCY)
            )
            self._data[CONF_PV_EXTERNAL_ENTITY] = user_input.get(
                CONF_PV_EXTERNAL_ENTITY, ""
            )
            self._data[CONF_PV_EXPORT_GRID_FEE] = float(
                user_input.get(CONF_PV_EXPORT_GRID_FEE, DEFAULT_PV_EXPORT_GRID_FEE)
            )
            self._data[CONF_ANNUAL_CONSUMPTION_KWH] = float(
                user_input.get(CONF_ANNUAL_CONSUMPTION_KWH,
                               DEFAULT_ANNUAL_CONSUMPTION_KWH)
            )
            self._data[CONF_CONSUMPTION_ENTITY] = (user_input.get(
                CONF_CONSUMPTION_ENTITY, DEFAULT_CONSUMPTION_ENTITY) or "")

            title = f"Spot Price ({REGIONS.get(self._data.get(CONF_REGION, 'finland'), 'Finland')})"
            return self.async_create_entry(title=title, data=self._data)

        # Stability invariant: the price forecaster's baseload must be a
        # deterministic function of (config + long-window EMA, time). The
        # `annual_consumption_kwh` value represents typical TOTAL annual
        # household demand including PV self-consumption AND optimizer-
        # controlled loads (heat pump, EV, sauna, water heater). Static
        # config cannot create optimizer feedback. The optional
        # `consumption_entity` reads any HA consumption sensor and
        # internally smooths it over a 14-day window with 5 % hysteresis,
        # long enough that EMHASS's daily decisions don't propagate back.
        # See TECHNICAL_GUIDE "PV-aware pricing" for the worked Case A vs
        # Case B example.
        schema = vol.Schema({
            vol.Optional(
                CONF_PV_CAPACITY_KWP, default=DEFAULT_PV_CAPACITY_KWP,
                description={"suggested_value": DEFAULT_PV_CAPACITY_KWP},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=50.0)),
            vol.Optional(
                CONF_PV_TILT_DEG, default=DEFAULT_PV_TILT_DEG,
                description={"suggested_value": DEFAULT_PV_TILT_DEG},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=90.0)),
            vol.Optional(
                CONF_PV_AZIMUTH_DEG, default=DEFAULT_PV_AZIMUTH_DEG,
                description={"suggested_value": DEFAULT_PV_AZIMUTH_DEG},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=360.0)),
            vol.Optional(
                CONF_PV_SYSTEM_EFFICIENCY, default=DEFAULT_PV_SYSTEM_EFFICIENCY,
                description={"suggested_value": DEFAULT_PV_SYSTEM_EFFICIENCY},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=1.0)),
            vol.Optional(CONF_PV_EXTERNAL_ENTITY, default=""): str,
            vol.Optional(
                CONF_PV_EXPORT_GRID_FEE, default=DEFAULT_PV_EXPORT_GRID_FEE,
                description={"suggested_value": DEFAULT_PV_EXPORT_GRID_FEE},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.10)),
            # v2.4 baseload schema (replaces 3 v2.3 fields).
            vol.Optional(
                CONF_ANNUAL_CONSUMPTION_KWH,
                default=DEFAULT_ANNUAL_CONSUMPTION_KWH,
                description={"suggested_value": DEFAULT_ANNUAL_CONSUMPTION_KWH},
            ): vol.All(vol.Coerce(float), vol.Range(min=500.0, max=50000.0)),
            vol.Optional(
                CONF_CONSUMPTION_ENTITY,
                default=DEFAULT_CONSUMPTION_ENTITY,
            ): str,
        })
        return self.async_show_form(step_id="pv_system", data_schema=schema)

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "SpotPriceOptionsFlow":
        """Get the options flow handler."""
        return SpotPriceOptionsFlow(config_entry)


class SpotPriceOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow — change operator, tariffs, API keys after setup."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Options step: modify operator, tariffs, and API settings."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate Fingrid key if provided
            fingrid_key = user_input.get(CONF_FINGRID_API_KEY, "").strip()
            if fingrid_key:
                session = async_get_clientsession(self.hass)
                client = SpotPriceApiClient(session, fingrid_key)
                valid = await client.validate_fingrid_key()
                if not valid:
                    errors["base"] = "invalid_fingrid_key"

            if not errors:
                # Update the config entry data
                new_data = dict(self._config_entry.data)
                new_data[CONF_OPERATOR] = user_input.get(
                    CONF_OPERATOR, new_data.get(CONF_OPERATOR, "elenia")
                )
                new_data[CONF_CUSTOM_DAY_RATE] = user_input.get(
                    CONF_CUSTOM_DAY_RATE, 0.0361
                )
                new_data[CONF_CUSTOM_NIGHT_RATE] = user_input.get(
                    CONF_CUSTOM_NIGHT_RATE, 0.0220
                )
                new_data[CONF_CUSTOM_VAT] = user_input.get(
                    CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER
                )
                new_data[CONF_CUSTOM_ENERGY_TAX] = user_input.get(
                    CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX
                )
                new_data[CONF_SELLER_MARGIN] = user_input.get(
                    CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN
                )
                new_data[CONF_ENABLE_NEIGHBOR_PRICES] = user_input.get(
                    CONF_ENABLE_NEIGHBOR_PRICES, True
                )
                new_data[CONF_ENABLE_DTACI_DK] = user_input.get(
                    CONF_ENABLE_DTACI_DK, DEFAULT_ENABLE_DTACI_DK
                )
                if fingrid_key:
                    new_data[CONF_FINGRID_API_KEY] = fingrid_key
                elif not user_input.get(CONF_FINGRID_API_KEY):
                    new_data.pop(CONF_FINGRID_API_KEY, None)

                new_data[CONF_NORDPOOL_ENTITY] = user_input.get(
                    CONF_NORDPOOL_ENTITY, ""
                )
                new_data[CONF_ENABLE_PV_SELLING] = user_input.get(
                    CONF_ENABLE_PV_SELLING, False
                )
                new_data[CONF_PV_SELL_COMMISSION] = user_input.get(
                    CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION
                )
                # Household PV system + baseload (Phase 1)
                new_data[CONF_PV_CAPACITY_KWP] = float(user_input.get(
                    CONF_PV_CAPACITY_KWP, DEFAULT_PV_CAPACITY_KWP
                ))
                new_data[CONF_PV_TILT_DEG] = float(user_input.get(
                    CONF_PV_TILT_DEG, DEFAULT_PV_TILT_DEG
                ))
                new_data[CONF_PV_AZIMUTH_DEG] = float(user_input.get(
                    CONF_PV_AZIMUTH_DEG, DEFAULT_PV_AZIMUTH_DEG
                ))
                new_data[CONF_PV_SYSTEM_EFFICIENCY] = float(user_input.get(
                    CONF_PV_SYSTEM_EFFICIENCY, DEFAULT_PV_SYSTEM_EFFICIENCY
                ))
                new_data[CONF_PV_EXTERNAL_ENTITY] = user_input.get(
                    CONF_PV_EXTERNAL_ENTITY, ""
                )
                new_data[CONF_PV_EXPORT_GRID_FEE] = float(user_input.get(
                    CONF_PV_EXPORT_GRID_FEE, DEFAULT_PV_EXPORT_GRID_FEE
                ))
                new_data[CONF_ANNUAL_CONSUMPTION_KWH] = float(user_input.get(
                    CONF_ANNUAL_CONSUMPTION_KWH,
                    DEFAULT_ANNUAL_CONSUMPTION_KWH
                ))
                new_data[CONF_CONSUMPTION_ENTITY] = (user_input.get(
                    CONF_CONSUMPTION_ENTITY,
                    DEFAULT_CONSUMPTION_ENTITY
                ) or "")
                # Drop legacy v2.3 baseload fields when the user re-saves
                # via Options. The coordinator's migration logic still
                # honours them on first load if present, but once the user
                # touches Options we move cleanly to the v2.4 schema.
                for k in (CONF_BASELOAD_KWH_PER_HOUR,
                          CONF_BASELOAD_DAY_FACTOR,
                          CONF_BASELOAD_NIGHT_FACTOR):
                    new_data.pop(k, None)

                self.hass.config_entries.async_update_entry(
                    self._config_entry, data=new_data
                )
                return self.async_create_entry(title="", data={})

        # Pre-fill with current values
        current = self._config_entry.data
        operator_options = {k: v["name"] for k, v in OPERATORS.items()}

        schema = vol.Schema({
            vol.Required(
                CONF_OPERATOR,
                default=current.get(CONF_OPERATOR, "elenia"),
            ): vol.In(operator_options),
            vol.Optional(
                CONF_SELLER_MARGIN,
                default=current.get(CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN),
                description={"suggested_value": current.get(CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.10)),
            vol.Optional(
                CONF_CUSTOM_DAY_RATE,
                default=current.get(CONF_CUSTOM_DAY_RATE, 0.0361),
                description={"suggested_value": current.get(CONF_CUSTOM_DAY_RATE, 0.0361)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.50)),
            vol.Optional(
                CONF_CUSTOM_NIGHT_RATE,
                default=current.get(CONF_CUSTOM_NIGHT_RATE, 0.0220),
                description={"suggested_value": current.get(CONF_CUSTOM_NIGHT_RATE, 0.0220)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.50)),
            vol.Optional(
                CONF_CUSTOM_VAT,
                default=current.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER),
                description={"suggested_value": current.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER)},
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=2.0)),
            vol.Optional(
                CONF_CUSTOM_ENERGY_TAX,
                default=current.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX),
                description={"suggested_value": current.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.20)),
            vol.Optional(
                CONF_ENABLE_NEIGHBOR_PRICES,
                default=current.get(CONF_ENABLE_NEIGHBOR_PRICES, True),
            ): bool,
            vol.Optional(
                CONF_ENABLE_DTACI_DK,
                default=current.get(CONF_ENABLE_DTACI_DK, DEFAULT_ENABLE_DTACI_DK),
            ): bool,
            vol.Optional(
                CONF_FINGRID_API_KEY,
                default=current.get(CONF_FINGRID_API_KEY, ""),
            ): str,
            vol.Optional(
                CONF_NORDPOOL_ENTITY,
                default=current.get(CONF_NORDPOOL_ENTITY, ""),
            ): str,
            vol.Optional(
                CONF_ENABLE_PV_SELLING,
                default=current.get(CONF_ENABLE_PV_SELLING, False),
            ): bool,
            vol.Optional(
                CONF_PV_SELL_COMMISSION,
                default=current.get(CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION),
                description={"suggested_value": current.get(CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.10)),
            vol.Optional(
                CONF_PV_CAPACITY_KWP,
                default=current.get(CONF_PV_CAPACITY_KWP, DEFAULT_PV_CAPACITY_KWP),
                description={"suggested_value": current.get(CONF_PV_CAPACITY_KWP, DEFAULT_PV_CAPACITY_KWP)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=50.0)),
            vol.Optional(
                CONF_PV_TILT_DEG,
                default=current.get(CONF_PV_TILT_DEG, DEFAULT_PV_TILT_DEG),
                description={"suggested_value": current.get(CONF_PV_TILT_DEG, DEFAULT_PV_TILT_DEG)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=90.0)),
            vol.Optional(
                CONF_PV_AZIMUTH_DEG,
                default=current.get(CONF_PV_AZIMUTH_DEG, DEFAULT_PV_AZIMUTH_DEG),
                description={"suggested_value": current.get(CONF_PV_AZIMUTH_DEG, DEFAULT_PV_AZIMUTH_DEG)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=360.0)),
            vol.Optional(
                CONF_PV_SYSTEM_EFFICIENCY,
                default=current.get(CONF_PV_SYSTEM_EFFICIENCY, DEFAULT_PV_SYSTEM_EFFICIENCY),
                description={"suggested_value": current.get(CONF_PV_SYSTEM_EFFICIENCY, DEFAULT_PV_SYSTEM_EFFICIENCY)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=1.0)),
            vol.Optional(
                CONF_PV_EXTERNAL_ENTITY,
                default=current.get(CONF_PV_EXTERNAL_ENTITY, ""),
            ): str,
            vol.Optional(
                CONF_PV_EXPORT_GRID_FEE,
                default=current.get(CONF_PV_EXPORT_GRID_FEE, DEFAULT_PV_EXPORT_GRID_FEE),
                description={"suggested_value": current.get(CONF_PV_EXPORT_GRID_FEE, DEFAULT_PV_EXPORT_GRID_FEE)},
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.10)),
            # v2.4 baseload schema. If the entry only carries legacy v2.3
            # `baseload_kwh_per_hour`, infer the equivalent annual value
            # for the form default; the user can keep it or re-tune.
            vol.Optional(
                CONF_ANNUAL_CONSUMPTION_KWH,
                default=current.get(
                    CONF_ANNUAL_CONSUMPTION_KWH,
                    _infer_legacy_annual_kwh(current),
                ),
                description={"suggested_value": current.get(
                    CONF_ANNUAL_CONSUMPTION_KWH,
                    _infer_legacy_annual_kwh(current),
                )},
            ): vol.All(vol.Coerce(float), vol.Range(min=500.0, max=50000.0)),
            vol.Optional(
                CONF_CONSUMPTION_ENTITY,
                default=current.get(
                    CONF_CONSUMPTION_ENTITY, DEFAULT_CONSUMPTION_ENTITY,
                ),
            ): str,
        })
        return self.async_show_form(
            step_id="init", data_schema=schema, errors=errors
        )
