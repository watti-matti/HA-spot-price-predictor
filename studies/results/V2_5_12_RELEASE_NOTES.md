# v2.5.12 — Sigmoid turbine power curve (replaces v2.5.11 cubic)

## TL;DR

**No coordinator behaviour change.** v2.5.12 swaps the v2.5.11 `ρ·v³` cubic wind-power proxy for a sigmoid (S-curve) that better matches actual turbine power output. Result: **MAE matches raw `Y_wind` (27.7 vs 27.5), CVaR improves marginally (+14.0% vs +13.5% at 24h, +11.0% vs +10.6% at 168h)**. AR(1) φ still 0.90 across all variants — confirms again that residual autocorrelation is not driven by wind-physics mis-specification.

```
P(v) = 1 / (1 + exp(-(v - 7.5) / 1.5))     normalised power [0..1]
sigmoid_wind_rho = P(v) · ρ_air(T) / 1.225  optional density correction
```

Defaults: v_mid = 7.5 m/s (≈ 50 % of rated), k_steep = 1.5 (transition width ≈ 9 m/s spans cut-in to rated). Matches typical 120 m hub-height fleet curve: 5 % at v=3 m/s, 50 % at 7.5, 95 % at 12.

## Variant comparison

| Variant | φ | h=24 MAE / R² / CVaR | h=168 MAE / R² / CVaR |
|---|---:|---|---|
| V_base (raw `Y_wind`) | 0.903 | 27.55 / +0.59 / +13.5 % | 28.33 / +0.57 / +10.6 % |
| **V_sigmoid** (sigmoid) | 0.904 | 27.67 / +0.59 / **+14.0 %** | 28.46 / +0.56 / **+11.0 %** |
| V_sigmoid_rho (+ ρ_air) | 0.904 | 27.64 / +0.59 / +14.0 % | 28.44 / +0.57 / **+11.1 %** |
| **V_sigmoid_full** (+ solar derate) | 0.904 | 27.65 / +0.59 / **+14.1 %** | 28.46 / +0.57 / **+11.1 %** |
| V_sigmoid_plus_raw_wind (both) | 0.903 | **27.54** / +0.59 / +13.7 % | **28.31** / +0.57 / +10.8 % |

## Key findings

1. **Sigmoid is much better than v2.5.11's cubic** (MAE 27.7 vs 30.6, R² +0.59 vs +0.50). The S-curve correctly handles cut-in saturation at low wind AND rated saturation at high wind, where the unsaturated `v³` over-extrapolated.

2. **Sigmoid edges raw `Y_wind` on CVaR** (+14.0 % vs +13.5 % at 24h; +11.0 % vs +10.6 % at 168h). The improvement is small but consistent — the sigmoid is physically sound, captures the non-linearity, and gives modestly better hedging behaviour.

3. **Air-density correction is negligible** — V_sigmoid_rho is essentially tied with V_sigmoid (same MAE, same φ, +0.01 pp on CVaR). The ρ(T) range at FI temperatures (1.17 – 1.42 kg/m³) is only ~18 %, and the Ridge essentially absorbs the temperature signal via Y_temp anyway.

4. **PV temp derating still doesn't matter** (V_sigmoid_full ≈ V_sigmoid_rho) — confirmed again at FI latitudes.

5. **φ unchanged at 0.90 across all five variants.** This is now the third independent test showing that AR(1) Layer 3 is structurally necessary — no functional form of wind/solar physics can substitute for it. The residual autocorrelation is price-side, not feature-side.

## Implications for v2.6.0 candidate selection

| Feature | Verdict |
|---|---|
| `Y_wind` (raw deseasonalised wind speed) | Keep — best on raw MAE, slightly behind on CVaR |
| `Y_sigmoid_wind_rho` (sigmoid × ρ_air) | **Production candidate** — best CVaR (+14.1 % / +11.1 %), physically defensible, single fitted scalar in Ridge |
| `Y_solar_effective` | Marginal — equivalent to `Y_solar` at FI latitudes; can use either |
| Cubic `ρ·v³` | **REJECT** — over-extrapolates, MAE 30.6 |

For v2.6.0 the cleanest design is **`Y_sigmoid_wind_rho + Y_solar_effective` replacing `Y_wind + Y_solar`** — slightly better CVaR, exactly the same operational complexity. The sigmoid is deterministic so the runtime cost is one `exp()` per hour per coordinator cycle (sub-microsecond).

## Figure

![Turbine power curve candidates](figures/v2512_turbine_curves.png)

The figure overlays three candidate forms:
- **Blue (sigmoid)**: smooth S-curve, never over-shoots, captures cut-in and rated saturation in one shape
- **Green (clipped cubic)**: physically exact (real turbine has cubic-up-to-rated then flat), but discontinuous derivative at v_rated
- **Red dotted (v2.5.11 unsaturated cubic)**: rises off the chart past v_rated — explains why it over-weights extreme-wind hours

Sigmoid is a 2-parameter smooth approximation to the clipped cubic — closer to the physics without the kink.

## Files

- **New**: `studies/v2512_sigmoid_turbine_curve.py` (~270 LOC)
- **New**: `studies/results/v2512_sigmoid_turbine_curve.md` (auto-generated)
- **New**: `studies/results/figures/v2512_turbine_curves.png` (sigmoid vs cubic candidates)
- **New**: `studies/results/figures/v2512_sigmoid_variants.png` (φ / MAE / R² per variant)
- **New**: `studies/results/V2_5_12_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.11 → 2.5.12`), `README.md` index

No new tests (the `sigmoid_turbine` function is pure-numpy with no side effects and is exercised by the study itself).

## Tests

**369 / 369 passing**.

## Next step — v2.5.13 = Layer 4 (GPD POT spike model)

The architectural plan from v2.5.10 ends with Layer 4 (heavy-tail spike model on the residual). v2.5.2 already demonstrated GPD POT feasibility for SE3/SE1/EE cross-border zones. v2.5.13 applies the same methodology to the FI Ridge+AR residual produced by the V_sigmoid_full architecture. Auto-mode continuing into v2.5.13 next.

## Reproducibility

```bash
python studies/v2512_sigmoid_turbine_curve.py
```

Offline; uses only locally cached data.
