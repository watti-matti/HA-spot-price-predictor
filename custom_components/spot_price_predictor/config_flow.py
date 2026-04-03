"""Config flow for Spot Price Predictor."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api_client import SpotPriceApiClient
from .const import (
    DOMAIN,
    CONF_REGION,
    CONF_OPERATOR,
    CONF_FINGRID_API_KEY,
    CONF_ENABLE_TIER2,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
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
    ) -> config_entries.ConfigFlowResult:
        """Step 1: Select region."""
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")

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
    ) -> config_entries.ConfigFlowResult:
        """Step 2: Select operator and tariff."""
        if user_input is not None:
            self._data[CONF_OPERATOR] = user_input[CONF_OPERATOR]
            if user_input[CONF_OPERATOR] == "custom":
                self._data[CONF_CUSTOM_DAY_RATE] = user_input.get(CONF_CUSTOM_DAY_RATE, 0.05)
                self._data[CONF_CUSTOM_NIGHT_RATE] = user_input.get(CONF_CUSTOM_NIGHT_RATE, 0.04)
                self._data[CONF_CUSTOM_VAT] = user_input.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER)
                self._data[CONF_CUSTOM_ENERGY_TAX] = user_input.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX)
            return await self.async_step_optional_apis()

        operator_options = {k: v["name"] for k, v in OPERATORS.items()}
        schema = vol.Schema({
            vol.Required(CONF_OPERATOR, default="elenia"): vol.In(operator_options),
            vol.Optional(CONF_CUSTOM_DAY_RATE, default=0.05): vol.Coerce(float),
            vol.Optional(CONF_CUSTOM_NIGHT_RATE, default=0.04): vol.Coerce(float),
            vol.Optional(CONF_CUSTOM_VAT, default=DEFAULT_VAT_MULTIPLIER): vol.Coerce(float),
            vol.Optional(CONF_CUSTOM_ENERGY_TAX, default=DEFAULT_ENERGY_TAX): vol.Coerce(float),
        })
        return self.async_show_form(step_id="operator", data_schema=schema)

    async def async_step_optional_apis(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 3: Optional API keys and tier selection."""
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
                self._data[CONF_ENABLE_TIER2] = user_input.get(CONF_ENABLE_TIER2, False)

                title = f"Spot Price Predictor ({REGIONS.get(self._data.get(CONF_REGION, 'finland'), 'Finland')})"
                return self.async_create_entry(title=title, data=self._data)

        schema = vol.Schema({
            vol.Optional(CONF_FINGRID_API_KEY, default=""): str,
            vol.Optional(CONF_ENABLE_TIER2, default=False): bool,
        })
        return self.async_show_form(
            step_id="optional_apis", data_schema=schema, errors=errors
        )
