# Training baseline

The reference point every future training is measured against. Recorded
in `studies/results/BASELINE.json`, verified by
`python studies/verify_baseline.py`, guarded by
`tests/test_baseline_integrity.py`.

No number in this document was typed by hand. All of them come from
`BASELINE.json`, which can only be written from a measured harness run.

## Why this exists

On 2026-10-13 a branch-vs-production comparison appeared to show the
branch changing the forecast by up to **0.0154 EUR/MWh**. It did not.
The two checkouts held different vintages of the Open-Meteo weather
cache, so the two runs were scored on different input data. Syncing the
cache made the results byte-identical.

Nothing in the output revealed this. `backtest_harness.json` recorded
`snapshot_id` for the data store and nothing about the weather, the
artifacts, or the code. Two numbers from different moments were only
loosely comparable, and "the model changed" was indistinguishable from
"the inputs changed".

## The five inputs

A forecast number is reproducible only if all five are fixed:

| input | what it is | in git? |
|---|---|---|
| `inference_code` | the forecast-reachable modules | yes |
| `shipped_artifacts` | the 4 JSON files the pipeline loads | yes |
| `harness_code` | the scoring path itself | yes |
| `data_store_parquets` | prices, neighbours, grid, weather archive | **no** |
| `weather_cache` | `studies/.cache/wx_*.parquet` | **no** |

The last two are gitignored and **per-checkout**, which is the trap. The
weather cache is incremental and append-only, so two clones silently
hold different vintages of recent hours. `tests/` can only assert on the
three tracked components; `verify_baseline.py` checks all five.

### `inference_code` is derived, not hand-listed

The reachable set is computed from the import graph each run, starting
from `coordinator` and `pipeline`. A hand-written list was tried first
and was wrong in both directions: it missed ten modules the forecast
really reaches (`features`, `dtaci_integration`, `pv_estimate`,
`pv_nowcast`, `pv_aware_cvar`, `pv_cost_kernel`,
`consumption_profile_loader`, `dk_dtaci`, `dk_utils`,
`price_forecast_verifier`, `holidays`) and included `sensor.py`, which
reads coordinator output and cannot affect it.

An incomplete list is the worst possible failure here — it means a
change to real forecast code does not invalidate the baseline. So
adding an import to the forecast path now changes the fingerprint on
its own, with no list to remember to update. `retrain.py` is excluded
because it is imported lazily inside a service handler and never runs
while forecasting.

## How to compare fairly

**The recorded numbers are a reproducibility reference, not a yardstick
to hold a future model against.**

The harness eval window ends wherever the data store ends, so
refreshing data changes which hours are scored. A model measured on
2026-11 data and compared against a baseline recorded on 2026-10 data
differs for two reasons at once, and you cannot separate them.

| question | method |
|---|---|
| Is this code change a no-op? | `verify_baseline.py` must report an **IDENTICAL** results hash. Exact and unambiguous. |
| Is this new model better? | One harness run containing **both** configs (`PRODUCTION` vs `FRESH`/candidate), so both are scored on the same rows. |
| Can I quote a baseline MAE? | Only alongside its fingerprint, and only if `verify` passes. Otherwise it is void. |

Never compare numbers across two runs. If you must, sync
`studies/.cache/wx_*.parquet` and `data_store/` from the reference
checkout first, then confirm with `verify_baseline.py`.

## Recorded baseline

Fingerprint, abbreviated — see `BASELINE.json` for full hashes:

```
inference_code       44664da11eef27ca…      (22 modules, derived)
shipped_artifacts    25b3bf142ae9c371…
harness_code         79464edb8fc3f362…
data_store_snapshot  22a27347590b74d6
data_store_parquets  a414929bf0273411…
weather_cache        6676888069645026…      (28 files)
```

Artifacts: `spike_model_default.json` at `v2.17.1/leakfree-holiday`,
**9 features**, train window ending 2026-06-26;
`seasonal_components_default.json` at `2.5.8`, window ending
2026-07-15.

Measured, `PRODUCTION` config (the frozen artifact, i.e. what the field
actually runs), `n_eval = 11,319`, eval window 2025-07-01 … 2026-10-16:

| segment | MAE | bias |
|---|--:|--:|
| ALL | 24.5307 | −0.6455 |
| WINTER Dec−Feb | 31.9710 | −1.6376 |
| SUMMER May−Jul | 19.9836 | −4.6170 |
| midday 8−12 UTC | 24.4061 | +2.5866 |
| evening 15−19 UTC | 32.3112 | −5.1564 |
| tail p95 price | 86.9614 | −86.6710 |

The `tail p95 price` row is not a typo: bias ≈ −MAE means the model
under-predicts essentially every hour in the top 5% of prices. That is
the conditional-spread defect (model amplitude slope 0.094 against an
actual 1.330), and it is the open defect with the largest headroom.

## Reproducing it

```bash
python -m src.data_store status          # confirm the snapshot matches
python studies/backtest_harness.py       # ~2 min
python studies/verify_baseline.py        # must say IDENTICAL
```

If `verify` reports a mismatch it names the component and what to do.
Common cases:

- **`weather_cache` DIFFERS** — different clone or a fetch extended it.
  Copy `studies/.cache/wx_*.parquet` from the reference checkout.
- **`data_store_*` DIFFERS** — data was refreshed, so the eval window
  moved. Do not compare to the recorded numbers; run both configs in one
  harness run.
- **`shipped_artifacts` DIFFERS** — the model changed. Re-record
  deliberately, and note that installing a new artifact **wipes every
  per-hour bias calibrator**, which costs 2–3 days of warm-up before
  field accuracy is representative.
- **`harness_code` DIFFERS** — the scoring path changed, which moves the
  numbers without moving the model. Re-record, and treat older numbers
  as incomparable.

## Re-recording

```bash
python studies/backtest_harness.py
python studies/verify_baseline.py --record
```

`--record` refuses to write unless a harness result exists and its
`snapshot_id` matches the current tree, so a baseline can never be
pinned to data that did not produce it. Do not edit `BASELINE.json` by
hand; `tests/test_baseline_integrity.py` cross-checks the recorded
artifact versions and feature counts against what is on disk, and
requires a 64-character results hash, so a hand-written entry fails.
