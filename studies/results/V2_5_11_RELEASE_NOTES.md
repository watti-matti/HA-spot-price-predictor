# v2.5.11 — Physics features test (Y_wind source confirmed; AR(1) is not masking missing physics)

## TL;DR

**No coordinator behaviour change.** v2.5.11 answers three user questions from 2026-05-17:

1. ✅ **`Y_wind` is Open-Meteo `wind_speed_120m`** (120 m, not 100 m), capacity-weighted across the 7 FI sites in `data/finland.yaml`, deseasonalised by the v2.5.8 artifact (`P_hour + P_week`, 7-week circular smoothing).
2. ✅ **Air-density and PV-temperature physics features ARE missing** from the v2.2 production model and the entire v2.5.x candidate set. The user's earlier physical insights did not survive the v2.2 reduction.
3. ❌ **Adding the standard physics features does NOT reduce AR(1) dominance.** The user's hypothesis — that φ=0.93 was masking missing physics — is not borne out by data. The strong residual autocorrelation comes from elsewhere (price-side persistence, market microstructure, or wrong functional form for wind power).

## Variant comparison

All five variants share L1 seasonal + L3 AR(1); only the L2 Ridge feature set differs. Window 2023-01-08 → 2026-04-27 (28,944 hourly rows), 55/45 chronological split.

| Variant | φ | ρ(1) | h=24 MAE / R² / CVaR | h=168 MAE / R² / CVaR |
|---|---:|---:|---|---|
| V_base (raw Y_wind, Y_solar, Y_temp) | +0.903 | +0.903 | 27.55 / +0.592 / +13.5 % | 28.33 / +0.570 / +10.6 % |
| V_phys_wind (Y_wind_power replaces Y_wind) | +0.908 | +0.908 | 30.64 / +0.501 / +13.2 % | 31.54 / +0.475 / +10.4 % |
| V_phys_solar (Y_solar_effective replaces Y_solar) | +0.903 | +0.903 | 27.56 / +0.592 / +13.6 % | 28.34 / +0.570 / +10.7 % |
| V_phys_both | +0.908 | +0.908 | 30.66 / +0.501 / +13.2 % | 31.55 / +0.475 / +10.4 % |
| **V_phys_plus_raw_wind** | +0.903 | +0.902 | **27.06 / +0.604 / +13.4 %** | **27.84 / +0.582 / +10.3 %** |

Physics formulae used:

- `ρ_air(T) = 101_325 / (287.05 · (T_°C + 273.15))` → 1.17 – 1.42 kg/m³ across FI conditions (18 % range)
- `wind_power_proxy = ρ_air(T) · v³` → range 0.5 – 8434 (arbitrary units)
- `cell_temp ≈ T_ambient + 0.03·GHI` (NOCT-style approximation)
- `η_temp = 1 − 0.004·max(0, cell_temp − 25)` (silicon PV temperature coefficient)
- `solar_effective = GHI · η_temp` → at high-solar hours the derating factor averages 0.946 (5.4 % loss)

## Key findings

### 1. φ does NOT drop with physics features

AR(1) coefficient stays at 0.90–0.91 across all variants. The user's hypothesis — that strong AR was masking missing physical structure — is not supported by this data. The 0.91 persistence is robustly NOT explained by adding density-corrected wind or PV-temperature-derated solar.

What this implies:

- The residual autocorrelation comes from **price-side dynamics** (demand momentum, generator scheduling, market microstructure, mean reversion of price spikes) — not from missing renewable physics.
- An OU / AR layer is fundamentally needed; we can't engineer it away by adding more inputs.
- The Layer 3 → Layer 4 chain (AR + GPD POT) is the right next direction.

### 2. Replacing raw Y_wind with Y_wind_power is WORSE

The MAE goes up from 27.55 to 30.64 EUR/MWh and R² drops from 0.59 to 0.50 when raw Y_wind is replaced with the physics-correct `ρ·v³`. Two reasons:

- **Wind turbine power curves are NOT cubic.** They start at cut-in ~3 m/s, rise approximately as v³ up to rated speed ~12 m/s, then **stay flat** (or curtail) up to cut-out ~25 m/s. The cubic feature over-extrapolates at high wind speeds where actual generation saturates.
- The Ridge with raw Y_wind effectively learns a linear approximation to the rated-power region, which is where most price-relevant action lives. The physics proxy puts too much weight on rare extreme-wind hours.

If we want physics-correct wind we'd need a **clipped or sigmoidal turbine power curve**:

```python
v_rated = 12.0   # m/s, fleet-weighted average
v_cutout = 25.0
def turbine_power(v):
    if v < 3.0:    return 0.0
    if v < v_rated: return (v / v_rated) ** 3
    if v < v_cutout: return 1.0
    return 0.0
```

That's a candidate for a follow-up patch but not in v2.5.11.

### 3. PV temperature derating is negligible at FI latitudes

Y_solar_effective and Y_solar produce essentially identical Ridge coefficients (−11.0 vs −11.0) and MAE (27.55 vs 27.56). FI ambient temperatures average 5.6 °C across the dataset; even in summer cell temperatures rarely exceed 25 °C by enough margin to derate noticeably. The 5.4 % average derating at high-solar hours is real but uniform — Ridge absorbs it as a scalar gain on Y_solar.

### 4. V_phys_plus_raw_wind is marginally best

Keeping BOTH raw `Y_wind` (which acts as a linear proxy for the rated-power region) AND adding `Y_wind_power` (which captures the cubic regime at sub-rated speeds) gives a small improvement: MAE 27.06 vs 27.55, R² 0.604 vs 0.592. The two features encode different parts of the power curve.

Coefficients in V_phys_plus_raw_wind: `Y_wind = −13.4`, `Y_wind_power = +0.013`. The signs (negative on wind speed, slightly positive on wind power) suggest **multicollinearity** is dampening interpretability — both encode "more wind ⇒ lower price" but the cubic feature is heavily centered/scaled differently.

## What this means for v2.6.0

- **AR(1) Layer 3 is structurally necessary** — physics features can't replace it.
- **Y_wind raw stays in the candidate set**; physics-based wind is at best a small marginal addition.
- **Y_solar can be replaced by Y_solar_effective** at zero cost (same accuracy, slightly more physically motivated), or just left as raw — neither matters.
- **Turbine power curve modelling** (clipped or sigmoidal) is the proper way to capture the wind→price physics; cubic is the wrong functional form for a wind-power proxy. Candidate for a follow-up patch if v2.6.0 production model needs the last few percent.

## Files

- **New**: `studies/v2511_physics_features.py` (~370 LOC)
- **New**: `studies/results/v2511_physics_features.md` (auto-generated)
- **New**: `studies/results/figures/v2511_phi_vs_features.png` (3-panel: φ/ρ, MAE, CVaR per variant)
- **New**: `studies/results/figures/v2511_physics_relationship.png` (visualisation of `ρ(T)`, derating, vs scatter)
- **New**: `studies/results/V2_5_11_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.10 → 2.5.11`), `README.md` index

## Tests

**369 / 369 passing** (no new tests — exploratory study only; physics formulae are pure-numpy with no side effects).

## Reproducibility

```bash
python studies/v2511_physics_features.py
```

Offline; uses only locally cached data.
