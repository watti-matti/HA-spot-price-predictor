# Installation Guide / Asennusohje

Step-by-step installation guide for Spot Price Predictor.

## Prerequisites

- Home Assistant 2024.1.0 or newer
- [HACS](https://hacs.xyz/) installed
- Optional: [Nordpool integration](https://github.com/custom-components/nordpool) for actual price comparison
- Optional: [ApexCharts Card](https://github.com/RomRider/apexcharts-card) for dashboard (install via HACS Frontend)

## Step 1: Add Repository to HACS

1. Open **HACS** → **Integrations**
2. Click the **⋮** menu (top right) → **Custom repositories**
3. Enter URL: `https://github.com/watti-matti/HA-spot-price-predictor`
4. Category: **Integration**
5. Click **Add**

![HACS Custom Repository](docs/screenshots/install-01-hacs-repo.png)

## Step 2: Download Integration

1. Search for **"Spot Price Predictor"** in HACS Integrations
2. Click **Download**
3. **Restart Home Assistant**

![HACS Download](docs/screenshots/install-02-hacs-download.png)

## Step 3: Add Integration

1. Go to **Settings** → **Devices & Services**
2. Click **+ Add Integration** (bottom right)
3. Search for **"Spot Price Predictor"**

![Add Integration](docs/screenshots/install-03-add-integration.png)

## Step 4: Select Region

Select your electricity market region. Currently supported: **Finland**.

![Select Region](docs/screenshots/install-04-region.png)

## Step 5: Configure Operator & Tariffs

Configure your electricity pricing:

1. **Operator** — Select your network operator (Elenia, Caruna Espoo, Caruna North, Helen) or **Custom** for manual rates
2. **Seller's margin** — Your energy retailer's margin from your contract (EUR/kWh, excl. VAT). Default 0.00.
3. **Day/Night transfer rates** — Pre-filled based on operator selection. For **yleissiirto** (general transfer), set both to the same value.
4. **VAT multiplier** — 1.255 (25.5%) for Finland
5. **Energy tax** — 0.02325 EUR/kWh (class I, 2026)
6. **Nordpool entity** — If you have a Nordpool integration, enter the entity ID (e.g., `sensor.nordpool_kwh_fi_eur_3_10_0`) to get actual price comparison sensors. Leave empty to skip.
7. **PV selling price** — Check to enable a selling price sensor for solar panel owners. Set the commission (e.g., 0.002 = 0.2 c/kWh).

![Operator Config](docs/screenshots/install-05-operator.png)

## Step 6: Optional Data Sources

1. **Cross-border price data** — Enabled by default. Uses free Swedish and Estonian price data to improve predictions. No API key needed.
2. **Fingrid API key** — Optional. Register for free at [data.fingrid.fi](https://data.fingrid.fi) to add nuclear production and cross-border power flow data (Tier 3 features).
3. **Cheapest hours search window** — Configure where to look for the cheapest hours:
   - **Start offset**: Hours from now (default 24 = tomorrow)
   - **Duration**: Window length in hours (default 48 = 2 days)

![Optional APIs](docs/screenshots/install-06-apis.png)

## Step 7: Done!

After completing the setup, the following sensors are created:

### Forecast Sensors (always created)

| Sensor | Description |
|--------|-------------|
| Spot Price Forecast | Predicted price (EUR/MWh) with 170h forecast |
| Consumer Price | Total price (EUR/kWh) with tariff, VAT, tax |
| Cheapest Hours | Best time windows for flexible loads |
| Week Price Stats | Weekly min/avg/max consumer price |

### Spot Price Sensors (if Nordpool entity configured)

| Sensor | Description |
|--------|-------------|
| Spot Electricity Price | Actual buying price with continuous timeline |
| Spot Electricity Selling Price | Selling price for solar PV (if enabled) |

![Sensors Created](docs/screenshots/install-07-sensors.png)

## Step 8: Add Dashboard (Optional)

1. Install [ApexCharts Card](https://github.com/RomRider/apexcharts-card) via HACS → Frontend
2. Copy the dashboard YAML from [apexcharts_dashboard.yaml](docs/yaml_examples/apexcharts_dashboard.yaml)
3. In HA, go to your dashboard → **Edit** → **+ Add Card** → **Manual** → paste the YAML

The dashboard shows actual prices vs forecast for ground truth comparison:

![Dashboard](docs/screenshots/install-08-dashboard.png)

## Changing Settings Later

Click **Configure** on the integration card in **Settings → Devices & Services** to change:
- Operator and tariff rates
- Seller's margin
- Nordpool entity and PV settings
- Fingrid API key
- Cheapest hours search window

![Options](docs/screenshots/install-09-options.png)

## Troubleshooting

**No sensors visible after installation:**
- Make sure you completed the config flow (Step 3-6 above)
- Check **Settings → System → Logs** for errors containing `spot_price_predictor`

**Cheapest Hours shows "unavailable":**
- The search window may extend beyond the forecast range (170h). Reduce the start offset + duration to stay within 170 hours total.

**Forecast values seem wrong:**
- The bundled model was trained on 2024-2026 Finnish data. Retrain with your own data for better accuracy — see [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).

---

*[(Suomenkieliset ohjeet)](TEKNINEN_TOTEUTUS.md)*
