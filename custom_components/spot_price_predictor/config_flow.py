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
    REGIONS,
    OPERATORS,
    DEFAULT_VAT_MULTIPLIER,
    DEFAULT_ENERGY_TAX,
)

_LOGGER = logging.getLogger(__name__)


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

                title = f"Spot Price ({REGIONS.get(self._data.get(CONF_REGION, 'finland'), 'Finland')})"
                return self.async_create_entry(title=title, data=self._data)

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
        })
        return self.async_show_form(
            step_id="init", data_schema=schema, errors=errors
        )
