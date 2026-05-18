# Installation Guide / Asennusohje

Step-by-step setup of Spot Price Predictor in Home Assistant.

## Prerequisites

- Home Assistant **2024.1.0** or newer.
- [HACS](https://hacs.xyz/) installed.
- Optional: a [Nordpool integration](https://github.com/custom-components/nordpool) (only needed if you want the actual-price sensors).
- Optional: [ApexCharts Card](https://github.com/RomRider/apexcharts-card) for the bundled dashboard, installable via HACS → Frontend.

## Step 1 — Add the custom repository to HACS

1. Open **HACS → Integrations**.
2. Click the **⋮** menu → **Custom repositories**.
3. Enter the repository URL: `https://github.com/watti-matti/HA-spot-price-predictor`.
4. Type: **Integration**. Click **ADD**.

![HACS Custom Repository](docs/screenshots/install-01-hacs-repo.png)

## Step 2 — Download

1. Find **Spot Price Predictor** in HACS Integrations.
2. Click **Download**.
3. **Restart Home Assistant** — required before the integration can be configured.

![HACS Download](docs/screenshots/install-02-hacs-download.png)

## Step 3 — Add the integration

1. **Settings → Devices & Services → + Add Integration**.
2. Search for **Spot Price Predictor** and click it.

![Add Integration](docs/screenshots/install-03-add-integration.png)

## Step 4 — Select region

Currently supported: **Finland**. Submit.

![Select Region](docs/screenshots/install-04-region.png)

## Step 5 — Operator, tariffs, and Nordpool

This step configures consumer pricing and (optionally) the actual-price sensors.

![Operator Configuration](docs/screenshots/install-05-operator.png)

**Operator selection** (`operator`):

| ID | Day rate (07–22) | Night rate (22–07) |
|---|:---:|:---:|
| `elenia` | 3.61 c/kWh | 2.20 c/kWh |
| `caruna_espoo` | 2.21 c/kWh | 2.21 c/kWh |
| `caruna_north` | 4.07 c/kWh | 2.49 c/kWh |
| `helen` | 3.54 c/kWh | 3.54 c/kWh |
| `custom` | user-defined | user-defined |

For yleissiirto (general transfer / flat tariff), set day and night rates equal — or pick an operator that already has them equal (Caruna Espoo, Helen).

**Price parameters** (all excl. VAT):

- **`seller_margin`** — from your electricity contract (e.g. 0.00383 EUR/kWh). Default 0.0.
- **`custom_day_rate`**, **`custom_night_rate`** — only relevant when `operator = custom`.
- **`custom_vat`** — VAT multiplier. Default 1.255 (25.5 %).
- **`custom_energy_tax`** — Default 0.02325 EUR/kWh (class I, 2026).

**Nordpool (optional)**:

- **`nordpool_entity`** — leave empty to skip the actual-price sensors. If you have a Nordpool integration, paste its sensor entity ID (e.g. `sensor.nordpool_kwh_fi_eur_3_10_0`).
- **`enable_pv_selling`** — when on (and a Nordpool entity is configured), the selling-price sensor is also created.
- **`pv_sell_commission`** — your retailer's selling commission (e.g. 0.002 EUR/kWh = 0.2 c/kWh).

Submit.

## Step 6 — Optional data sources

![Optional APIs](docs/screenshots/install-06-apis.png)

- **`fingrid_api_key`** — optional. Free email registration at [data.fingrid.fi](https://data.fingrid.fi). Enables Fingrid data fetches (nuclear, consumption / wind / solar forecasts). The non-seasonal spot model does not consume these signals today, but they feed the duration model and the solar sub-model retraining.
- **`enable_neighbor_prices`** — default on. Fetches SE1/SE3/EE day-ahead spot prices for the duration model and dashboard context.
- **`enable_dtaci_dk`** — default off. Turns on the per-(direction, k) DtACI calibration that wraps the D(k) curves with adaptive 90 % bands. Warmup is ~5 days of reconciled daily updates.

Submit.

## Step 7 — PV system (optional)

![PV Parameters](docs/screenshots/install-07-device-PV-parameters.png)

Leave `pv_capacity_kwp` at 0 to disable PV-aware outputs entirely (all PV-related sensor attributes are absent).

With PV enabled, the integration produces:

- a marginal `effective_eur_kwh` per forecast hour, bounded analytically in `[sell, buy]`;
- a `dk_cheap_pv_eur_kwh[24]` / `dk_peak_pv_eur_kwh[24]` pair per day on the duration sensor;
- convenience scalars `today_cheap_pv_*h_eur_kwh` / `today_peak_pv_*h_eur_kwh`.

| Field | Default | Notes |
|---|---|---|
| `pv_capacity_kwp` | 0.0 | 0 disables PV. |
| `pv_tilt_deg` | 45 | Matches Open-Meteo's fetch tilt. |
| `pv_azimuth_deg` | 180 | 0 = N, 90 = E, 180 = S, 270 = W. |
| `pv_system_efficiency` | 0.85 | Lumped DC/AC + soiling + losses. |
| `pv_external_entity` | "" | Override the internal estimate with any HA sensor whose attributes match one of: `forecast` list-of-dict (kWh), `wh_hours` dict (Wh), `watts` dict (W), or `irradiance` list (auto-detected W/kWh). |
| `pv_export_grid_fee` | 0 EUR/kWh | Extra fee on exported energy (above seller commission). |
| `annual_consumption_kwh` | 12 000 | Typical TOTAL annual household demand from the bill, including PV self-consumption AND optimizer-controlled loads (heat pump, EV, sauna, water heater). Drives baseload via a Finnish residential monthly seasonal profile. |
| `consumption_entity` | "" | Optional. Any HA consumption sensor: cumulative-kWh counter, daily/monthly `utility_meter`, or instantaneous-power sensor (W/kW). Auto-detected; smoothed on a 14-day rolling window with a 5 % hysteresis dead-band so optimizer rescheduling cannot feed back into next cycle's forecast. |

## Step 8 — Verify

The integration creates a device named **Spot Price Predictor** with sensor entities. Open **Settings → Devices & Services → Spot Price Predictor → Sensors** to verify:

| Sensor | Always created |
|---|:-:|
| `sensor.spot_price_predictor_price_forecast` | yes |
| `sensor.spot_price_predictor_duration_forecast` | yes |
| `sensor.spot_price_predictor_spot_electricity_price` | only with `nordpool_entity` |
| `sensor.spot_price_predictor_spot_electricity_selling_price` | only with `nordpool_entity` + `enable_pv_selling` |

![Sensors](docs/screenshots/install-09-sensors.png)

## Step 9 — Add the dashboard (optional)

A ready-made Lovelace dashboard is available at [`ha_dashboard.yaml`](ha_dashboard.yaml). Install [ApexCharts Card](https://github.com/RomRider/apexcharts-card) via HACS → Frontend, then add the dashboard:

1. Go to your HA dashboard → **Edit** → **+ Add Card** → **Manual**.
2. Paste the YAML from `ha_dashboard.yaml`.
3. Adjust entity IDs if your installation differs.

![Dashboard Example](docs/screenshots/example_UI.png)

The dashboard shows the 7-day consumer-price trend, today's D(k) cheap/peak curve, wind speed, and (when configured) Nordpool actual price for comparison.

## Changing settings later

**Settings → Devices & Services → Spot Price Predictor → Configure** opens the options flow. All wizard fields can be re-edited there. Changes apply on the next coordinator cycle (no restart needed).

## Retraining (optional)

The integration ships with pre-trained artifacts under `custom_components/spot_price_predictor/data/` — no retraining is required for normal use. If you want to refit against more recent data (e.g. after a regime shift, a tariff change, or when `RefitMonitor` flags coverage drift), call the service from **Developer Tools → Services**:

```yaml
service: spot_price_predictor.retrain_models
data:
  layers: ["seasonal", "spike", "solar"]   # omit to refit all three
  # fingrid_api_key: "..."                  # only needed for the solar layer
```

The service rewrites the three artifacts atomically and the coordinators auto-reload them on the next update cycle. On completion it fires the `spot_price_predictor_models_retrained` Home Assistant event so automations can react (e.g. send a notification).

## Troubleshooting

**No sensors visible after installation.**
Make sure you completed the full configuration wizard (Steps 3–7). Check **Settings → System → Logs** for errors containing `spot_price_predictor`. Try removing the integration and adding it again.

**`spot_electricity_price` shows wrong values.**
Verify that the `nordpool_entity` is correct. The sensor applies the same overhead as the forecast (`spot + seller_margin + transfer + energy_tax) × VAT`).

**Forecast looks inaccurate.**
The bundled artifacts were trained on recent Finnish data. For best accuracy, refit quarterly via `spot_price_predictor.retrain_models`. If `pipeline_diagnostics.refit_recommended` becomes `true`, the calibrator has detected sustained coverage drift and is suggesting a refit.

**DtACI bands are missing.**
DtACI is opt-in (`enable_dtaci_dk`). After enabling, the calibrator needs ≈ 5 days of reconciled daily updates before the bands open up — the `dtaci_warmup_status` attribute on the duration sensor reports the current state.

---

*[(Suomenkieliset tekniset ohjeet)](TEKNINEN_TOTEUTUS.md)*
