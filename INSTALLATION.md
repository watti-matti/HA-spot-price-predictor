# Installation

Text-only by design — Home Assistant's UI changes faster than screenshots can be maintained, and a stale picture is worse than none.

**Requirements:** Home Assistant with [HACS](https://hacs.xyz). No API keys are needed; every data source the forecast depends on is free and open.

---

## 1. Add the repository to HACS

**HACS → ⋮ (top right) → Custom repositories**

- Repository: `https://github.com/watti-matti/HA-spot-price-predictor`
- Type: **Integration**

Add it, then search HACS for **Spot Price Predictor** and download it. Restart Home Assistant.

## 2. Add the integration

**Settings → Devices & Services → Add Integration → Spot Price Predictor**

The setup wizard has **four steps**. Only the first two need any thought; the last two can be accepted as-is.

### Step 1 — Region

One field. `finland` is the only region currently shipped, and it is the default. Continue.

### Step 2 — Operator and pricing

Pick your distribution network operator so transfer tariffs are right. Built-in rates (EUR/kWh, excl. VAT):

| Operator | Day | Night |
|---|--:|--:|
| `elenia` | 0.0361 | 0.0220 |
| `caruna_espoo` | 0.0221 | 0.0221 |
| `caruna_north` | 0.0407 | 0.0249 |
| `helen` | 0.0354 | 0.0354 |
| `custom` | you set them | you set them |

Day rate applies 07:00–22:00 local, night rate 22:00–07:00. Choose `custom` if your operator is not listed or your contract differs — you then also set VAT (default `1.255`) and energy tax (default `0.02325` EUR/kWh).

Other fields in this step:

| Field | Default | What it does |
|---|---|---|
| `seller_margin` | `0.0` | Your electricity retailer's margin, EUR/kWh. Add it or your consumer prices read low. |
| `nordpool_entity` | empty | Entity ID of an existing Nordpool integration sensor. Set it to also get realised spot-price sensors alongside the forecast. |
| `enable_pv_selling` | off | Adds a selling-price sensor. Needs `nordpool_entity`. |
| `pv_sell_commission` | `0.002` | Your retailer's commission on exported energy, EUR/kWh. |

> **Check `seller_margin` against a real bill.** It is the single most common reason consumer prices look wrong, and the default of 0 is right for almost nobody.

### Step 3 — Optional APIs

All three can be left alone.

| Field | Default | What it does |
|---|---|---|
| `fingrid_api_key` | empty | Free key from [data.fingrid.fi](https://data.fingrid.fi). Feeds diagnostics only — **the price forecast does not use it.** |
| `enable_neighbor_prices` | on | Swedish and Estonian prices as model inputs. Free, no key. Leave on. |
| `enable_dtaci_dk` | off | Adaptive confidence bands on the duration curves. Warms up over ~5 days. |

### Step 4 — PV and consumption

Leave `pv_capacity_kwp` at `0` if you have no solar — everything PV-related is then simply absent, and you are done.

With PV:

| Field | Default | Notes |
|---|---|---|
| `pv_capacity_kwp` | `0` | Your array size. `0` disables all PV outputs. |
| `pv_tilt_deg` | `45` | Panel tilt. |
| `pv_azimuth_deg` | `180` | 0 = N, 90 = E, 180 = S, 270 = W. |
| `pv_system_efficiency` | `0.85` | Lumped DC/AC, soiling and wiring losses. |
| `pv_external_entity` | empty | Use an existing PV-forecast sensor instead of the internal estimate. |
| `pv_measured_power_entity` | empty | A live PV power sensor. Enables the intraday nowcast correction. |
| `pv_export_grid_fee` | `0` | Grid fee on exported energy, on top of the seller commission. |

Consumption, used for the PV-aware cost figures:

| Field | Default | Notes |
|---|---|---|
| `annual_consumption_kwh` | `12000` | **Total** yearly consumption from your bill — including heat pump, EV and water heater. |
| `consumption_entity` | empty | Any consumption sensor (cumulative kWh, utility meter, or instantaneous power). Auto-detected and smoothed. |
| `consumption_profile_entity` | empty | An external hourly profile sensor, if you run one. |

Submit. The integration creates a device named **Spot Price Predictor**.

## 3. Verify

**Settings → Devices & Services → Spot Price Predictor**

| Sensor | Created |
|---|---|
| Price Forecast | always |
| Duration Forecast | always |
| Spot Price Forecast FI | always — Nordpool-compatible, drop-in for EMHASS and ApexCharts |
| Effective Wind Speed | always — diagnostic |
| Spot Electricity Price | only with `nordpool_entity` |
| Spot Electricity Selling Price | only with `nordpool_entity` + `enable_pv_selling` |

Open **Price Forecast** and check its attributes: `forecast` should hold ~170 hourly rows, each with `spot_eur_mwh` and `consumer_eur_kwh`. If it does, you are running.

The first forecast appears within a few minutes. Bias correction needs a few days of realised prices before it starts adjusting — see the note below.

## 4. Dashboard (optional)

Ready-made ApexCharts cards are in [`docs/yaml_examples/`](docs/yaml_examples/). Copy one into a manual dashboard card. `forecast_v2_11_dashboard.yaml` is the recommended starting point.

---

## Changing settings later

**Settings → Devices & Services → Spot Price Predictor → Configure.** Every field above can be changed without reinstalling.

## What to expect from accuracy

MAE is about **24 EUR/MWh against a mean price near 50** — this is a *ranking* tool, not a price oracle. The cheap-hour/expensive-hour ordering that drives EV charging and deferrable loads is reliable well before the absolute level is. See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for the measured numbers.

After installing, or after any model update, the bias corrector starts cold and needs a few days of realised prices before it corrects. Forecasts are usable immediately but slightly less accurate during that window.

## Updating

HACS notifies you of new releases. Download and restart Home Assistant. Your settings and accumulated calibration history are preserved.

## Troubleshooting

**No sensors after adding the integration** — restart Home Assistant. HACS downloads files but the integration only loads on restart.

**Consumer prices look wrong** — check `seller_margin` first, then the operator selection. Together they account for most of the difference between spot and what you actually pay.

**Forecast stuck or missing** — check **Settings → System → Logs** for `spot_price_predictor`. The integration logs a warning rather than failing hard when an upstream API is unavailable.

**Refit prompt** — if the `pipeline_diagnostics` attribute shows `refit_recommended: true`, the calibrator has seen sustained drift. Call the `spot_price_predictor.retrain_models` service, or wait for the next release; shipped artifacts are refreshed periodically.
