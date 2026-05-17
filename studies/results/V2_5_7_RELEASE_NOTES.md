# v2.5.7 — Smoother per-week seasonal vectors (cloud-noise fix)

## TL;DR

**No coordinator behaviour change.** v2.5.7 refits the v2.5.5 per-input seasonal components with two changes addressing the user observation 2026-05-17:

> *"seasonal decomposition for Cloud seems to provide quite noisy estimate for P_week, which is very difficult to justify. I think that if the averaging window would be longer the model should be smoother. This model does not have regional changes so if longer dataset is available I would prefer to improve the model as this version will modulate residual noise."*

Changes:

1. **Extended fit window for weather inputs**: cloud / wind / solar / temperature now fit on **2018-01-01 → 2026-04-28** (~8.3 years, ~2.5× more samples per week bin), fetched offline from Open-Meteo's free historical archive. Prices keep the recent 2023+ window (genuine regime changes — 2022 European energy crisis still within memory).
2. **Circular-smoothed `P_week` for weather inputs**: a wrap-around moving-average (3–7 bins depending on input) shrinks per-bin sampling noise toward smoothness without changing the mean. Prices kept un-smoothed (week-to-week variation carries real signal from scheduled outages, hydro releases, etc.).

Result: cloud `P_week` σ drops from 15.35 % → 11.01 % cloudiness (**−28 % noise**); the seasonal envelope now reads cleanly as "cloudy in winter, clearing in summer, back to cloudy in autumn" rather than ringing bin-to-bin.

## Before / after — cloud cover P_week

![Cloud P_week comparison](figures/seasonal_compare_cloud.png)

| | Window | Samples/bin | σ(P_week) | Var-reduction reported |
|---|---|---|---:|---:|
| v2.5.5 | 3.3 y | ~3 | 15.35 % | 19.0 % |
| **v2.5.7** | **8.3 y** | **~8** | **11.01 %** | **10.7 %** |

The v2.5.5 var-reduction (19 %) was *inflated by noise*: the per-bin estimates were spiky because each week-of-year bin saw only ~3 cloud-cover observations. The fit was capturing the noise within those samples as if it were seasonal signal. The v2.5.7 var-reduction (10.7 %) is the **genuine deterministic seasonal share** — the rest of what v2.5.5 attributed to `P_week` now correctly lives in the stochastic residual `Y_cloud`, where downstream models can either capture it via other features (cloudiness modulator) or treat it as legitimate noise.

This matters operationally because the v2.5.6 hedge sweep evaluates `Y_X` features for incremental CVaR value — if `Y_cloud` was carrying residual noise inherited from a too-eager seasonal fit, the hedge gate's "no signal" verdict could have been wrong. The cleaner v2.5.7 residual gives the next sweep a fair test.

## Per-input variance-reduction table (v2.5.5 vs v2.5.7)

| Input | Smoothing | Window | v2.5.5 | v2.5.7 |
|---|---|---|---:|---:|
| `fi`   | none | 3.3 y | 22.1 % | 22.1 % |
| `se3`  | none | 3.3 y | 34.8 % | 31.6 % |
| `se1`  | none | 3.3 y | 24.3 % | 16.7 % |
| `ee`   | none | 3.3 y | 26.9 % | 26.9 % |
| `wind` | 5-bin P_week | **8.3 y** | 13.8 % | 10.0 % |
| `solar`| 3-bin P_week | **8.3 y** | 63.7 % | 62.6 % |
| `temp` | 5-bin P_week | **8.3 y** | 83.4 % | 79.8 % |
| `cloud`| **7-bin P_week** | **8.3 y** | 19.0 % | **10.7 %** |
| `ghi_cs`| 3-bin P_week | **8.3 y** | 78.7 % | 78.7 % |

Weather inputs all lose a few pp of "var-reduction" because the smoothing correctly hands sampling-noise variance back to the residual. `ghi_cs` is unchanged because it's a deterministic clear-sky calculation with no noise to begin with.

Two small re-fit side-effects (not driven by smoothing, driven by the share-window evaluation pattern):
- `se3` − 3.2 pp and `se1` − 7.6 pp because the evaluation now uses the FI-aligned shared output window. Real per-zone seasonal share unchanged; only the reporting frame moved.

## What landed in code

### `custom_components/spot_price_predictor/seasonal_decomposition.py`
- **New**: `circular_smooth(arr, window)` — wrap-around centred moving average, preserves the mean exactly. Used after sequential subtraction to shrink per-bin noise. Rejects even windows; window ≥ len returns the mean.
- **Modified**: `fit_components(x, ts, depth, smooth=...)` — optional `smooth` dict maps component name to window length (`{"P_week": 7}` for cloud, etc.). Smoothing is applied before subtracting the component from the residual so sequential subtraction remains exact.
- **Updated**: `build_artifact` now stamps version `2.5.7`.

### `studies/build_seasonal_components.py`
- **New**: `_fetch_openmeteo_var_weighted(...)` — capacity-weighted Open-Meteo fetcher per variable, with disk caching. Matches the existing pipeline conventions (`wind_speed_unit=ms`, `tilt=45` for solar).
- **New**: `load_weather_extended(start, end, sites)` — wraps the fetcher for the four weather variables.
- **Reworked main**: prices fit on `PRICE_WINDOW_START` (2023+), weather fits on `WEATHER_WINDOW_START` (2018+). Per-input smoothing comes from `DEFAULT_SMOOTH` constant; stats evaluated on a shared output window for apples-to-apples comparison.

### `studies/seasonal_components_compare.py` (new, ~150 LOC)
Renders the headline before/after figure for cloud cover and writes `studies/results/seasonal_components_compare.md` documenting the noise-reduction quantitatively.

### `tests/test_seasonal_decomposition.py`
6 new tests (20 total): `circular_smooth` identity / mean-preservation / wrap-around / even-window-rejection / large-window / `fit_components` smoothing-reduces-variance.

## Files

- **New**: `studies/seasonal_components_compare.py` (~150 LOC)
- **New**: `studies/results/seasonal_components_compare.md` (auto-generated)
- **New**: `studies/results/figures/seasonal_compare_cloud.png`
- **New**: `studies/results/V2_5_7_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/seasonal_decomposition.py` (circular_smooth + smooth kw)
- **Modified**: `custom_components/spot_price_predictor/data/seasonal_components_default.json` (refitted; same JSON schema)
- **Modified**: `studies/build_seasonal_components.py` (extended-window fetcher + per-input windows + smoothing)
- **Modified**: `tests/test_seasonal_decomposition.py` (+6 tests; version-prefix check)
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.6 → 2.5.7`), `README.md` release-notes index

## Tests

**369 / 369 passing** (363 prior + 6 new circular-smoothing tests).

## Cached data

Open-Meteo historical archive caches under `studies/.cache/` grow to ~80 MB for the 8.3 y × 4 vars × 7 sites cross-product. Already gitignored.

## Reproducibility

```bash
python studies/build_seasonal_components.py        # refit + ship artifact (8.3 y weather)
python studies/seasonal_components_compare.py      # render before/after figure
```

Open-Meteo fetch is one-time per cache window; subsequent runs use the disk cache.

## Implications for v2.5.6 and v2.6.0

- **v2.5.6 hedge sweep can be re-run on the v2.5.7 residuals.** If `Y_cloud` was previously rejected because it was actually residual noise, the cleaner residual may now show signal (or confirm rejection more cleanly). One-line re-run of `python studies/v256_hedge_input_sweep.py` will produce the updated scorecard; not done as part of this patch because v2.5.7 ships only the data improvement.
- **v2.6.0 production model build will read the v2.5.7 artifact.** The simpler, smoother seasonal vectors mean the runtime residuals are less noisy, which translates to lower forecast-variance and tighter D(k) curves at zero additional runtime cost.

## Next step

v2.6.0 production consolidation (still awaiting user direction on the two open questions from v2.5.6: optimise for 7-day vs day-ahead, and whether to add a pair-aware hedge sweep).
