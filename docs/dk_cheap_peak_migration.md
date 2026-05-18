# D(k) Cheap/Peak Sensor Schema

The Duration Forecast sensor exposes the daily price duration curve as **two complementary 24-level arrays per direction** (cheap and peak), in both spot EUR/MWh and consumer EUR/kWh. All four arrays are 0-indexed.

## Canonical attributes

Each entry in `daily_forecast[]` carries:

| Attribute | Shape | Unit | Definition |
|-----------|-------|------|------------|
| `dk_cheap_eur_mwh` | float[24] | EUR/MWh | `dk_cheap_eur_mwh[i]` = mean spot price of the (i+1) cheapest hours of the day. Monotone non-decreasing in i. |
| `dk_peak_eur_mwh`  | float[24] | EUR/MWh | `dk_peak_eur_mwh[i]`  = mean spot price of the (i+1) priciest hours of the day. Monotone non-increasing in i. |
| `dk_cheap_eur_kwh` | float[24] | EUR/kWh | Same cheapest-end curve in consumer price (per-hour day/night tariff applied before sorting). |
| `dk_peak_eur_kwh`  | float[24] | EUR/kWh | Same priciest-end curve in consumer price. |

PV-aware variants `dk_cheap_pv_eur_kwh[24]` / `dk_peak_pv_eur_kwh[24]` are added when the integration is configured with a non-zero `pv_capacity_kwp`.

## Why cheap and peak

The cheap-end curve answers "what's the best achievable cost per kWh if I run a deferrable load for k hours?" — CVaR at α = (i+1)/24 in the lower tail. The peak-end curve answers "what's the worst-case cost if I'm forced to run during the priciest k hours?" — CVaR at α = (i+1)/24 in the upper tail.

## Identities

The full-day mean is recovered at the last index in either direction:

```
dk_cheap_eur_mwh[23] == dk_peak_eur_mwh[23] == daily_average_spot
dk_cheap_eur_kwh[23] == dk_peak_eur_kwh[23] == daily_average_consumer
```

## Access patterns

```python
# Cheapest 4 hours of day d (consumer EUR/kWh)
daily_forecast[d]["dk_cheap_eur_kwh"][3]

# Priciest 8 hours of day d (spot EUR/MWh)
daily_forecast[d]["dk_peak_eur_mwh"][7]

# Today's cheapest hour (consumer EUR/kWh)
daily_forecast[0]["dk_cheap_eur_kwh"][0]
```

## Convenience scalars on the sensor

The Duration Forecast sensor surfaces the most-used `dk_cheap_eur_kwh` / `dk_peak_eur_kwh` lookups as flat attributes:

| Attribute | Source |
|-----------|--------|
| `today_cheap_1h_eur_kwh` | `daily_forecast[0].dk_cheap_eur_kwh[0]` |
| `today_cheap_4h_eur_kwh` | `daily_forecast[0].dk_cheap_eur_kwh[3]` |
| `today_cheap_8h_eur_kwh` | `daily_forecast[0].dk_cheap_eur_kwh[7]` |
| `today_cheap_12h_eur_kwh` | `daily_forecast[0].dk_cheap_eur_kwh[11]` |
| `today_peak_1h_eur_kwh`  | `daily_forecast[0].dk_peak_eur_kwh[0]` |
| `today_peak_4h_eur_kwh`  | `daily_forecast[0].dk_peak_eur_kwh[3]` |
| `today_peak_8h_eur_kwh`  | `daily_forecast[0].dk_peak_eur_kwh[7]` |
| `today_peak_12h_eur_kwh` | `daily_forecast[0].dk_peak_eur_kwh[11]` |

The PV-aware variants (`today_cheap_pv_*h_eur_kwh`, `today_peak_pv_*h_eur_kwh`) follow the same shape and are populated only when PV is configured.
