# Demand Analysis Findings

## Experiment: Finland Electricity Demand vs Price Prediction

Branch: `feature/demand-analysis`
Data: Fingrid dataset #124 (electricity consumption), 1 year (Apr 2025 - Apr 2026)

## Key Findings

### 1. Demand Peak ≠ Price Peak

The demand and price peaks occur at different hours:

| Metric | AM peak | PM peak |
|--------|:---:|:---:|
| **Actual demand** | hour 11 (10,509 MW) | hour 19 (10,404 MW) |
| **Actual price** | hour 9 | hour 19 |
| **Model Gaussian** | hour 9 | hour 19 |

The **price peaks at 9h** not because demand is highest then, but because the **demand ramp is steepest** at 6-7h (+495 MW/h). The morning ramp-up from night minimum to daytime plateau drives the price spike. By hour 11 when demand is actually highest, prices have already moderated.

The **PM peak aligns** (both demand and price peak at 19h) because the evening ramp-up from afternoon to evening coincides with the actual demand peak.

**Conclusion: Our model's Gaussian centers (9h/19h) are correctly tuned to price peaks, not demand peaks.**

### 2. Demand Shape: Plateau, Not Peak

Workday demand forms a **flat plateau from 8h-19h** (>90% of max), not sharp Gaussian peaks:

```
Night minimum:  9,228 MW (hour 03)
Daytime plateau: 10,346-10,509 MW (hours 08-19, amplitude only 163 MW)
Max:            10,509 MW (hour 11)
Daily range:    1,281 MW (12.2% of max)
```

The Gaussian model captures the **price-relevant** transitions (ramp-up at 6-9h, ramp-down at 20-23h) rather than the flat demand plateau.

### 3. Seasonal Demand Variation

| Season | Avg demand | Peak shift |
|--------|:---:|:---:|
| Winter (Jan-Feb) | 12,956 MW | AM peak at 9-11h, PM at 18-19h |
| Summer (Jun-Jul) | 8,146 MW | AM peak at 12h, PM at 14h |
| Ratio | 1.59x | PM peak shifts 5 hours |

Winter/summer ratio of 1.59x is already captured by the model's HDD (heating degree days) and month_sin/cos features.

### 4. Demand-Price Correlation

- Overall: r = 0.436
- Winter: r = 0.550 (strong — heating drives both demand and price)
- Summer: r = 0.113 (weak — price driven by wind/solar, not demand)

### 5. Adding Demand as a Model Feature: No Improvement

| Configuration | MAE (EUR/MWh) | R² | Delta |
|---------------|:---:|:---:|:---:|
| **Baseline (38 features)** | **3.465** | **0.537** | — |
| + demand_norm | 3.781 | 0.483 | -0.316 (worse) |
| + demand_norm + ramp | 3.762 | 0.487 | -0.298 (worse) |
| + demand_norm + ramp + deviation | 3.624 | 0.511 | -0.159 (worse) |

**Adding actual demand data makes the model worse.** This is because:
1. The existing weather features (temperature, HDD) already capture the demand-relevant signal
2. Adding correlated demand data introduces multicollinearity that degrades Ridge regression
3. At inference time, we don't have future demand data anyway — we'd need to forecast demand first, adding another source of error

### 6. Demand Ramp Rate Analysis

| Hour | Ramp (MW/h) | Significance |
|------|:---:|:---|
| 06 | **+495** | Steepest morning ramp — drives price spike |
| 07 | +317 | Continued ramp |
| 21 | **-245** | Steepest evening decline |
| 00 | -260 | Night decline |

The morning ramp at 06-07h is the strongest hourly change, confirming why prices spike at 9h (2-3 hours after the steepest ramp, when supply struggles to match the rapid demand increase).

### 7. The Real Driver: European Market Coupling

The most significant finding is that **Finnish price peaks are not driven by Finnish demand at all**, but are **imported through the interconnected Nordic/European electricity market**.

#### Correlation with Finnish spot price

| Factor | Correlation (r) | Role |
|--------|:---:|:---|
| **SE1 price** | **0.727** | Strongest — Nordic price coupling |
| **EE price** | **0.723** | Baltic-Continental coupling |
| **SE3 price** | **0.713** | Central Sweden coupling |
| Net import | 0.477 | Flow direction matters |
| FI demand | 0.407 | Weakest — flat plateau doesn't drive peaks |

Finnish spot price correlates **almost twice as strongly** with neighboring prices (r ≈ 0.72) as with Finnish demand (r = 0.41).

#### Daily price amplitude comparison

| Market | Daily amplitude | Min | Max |
|--------|:---:|:---:|:---:|
| Finland | 6.4 EUR/MWh | 2.4 | 8.8 |
| Sweden SE3 | 50.1 EUR/MWh | 29.4 | 79.5 |
| Estonia | 103.8 EUR/MWh | 53.1 | 156.9 |

Estonia's daily price swing is **16x larger** than Finland's. The double-peak signal flows from high-amplitude neighbors into low-amplitude Finland through the interconnectors.

#### Price propagation mechanism

```
Continental Europe (Germany, Poland)
  │  Strong industrial peaks at 8-9h
  │  Strong residential peaks at 18-20h
  │
  ├── Denmark → Sweden SE3 → Sweden SE1 → FINLAND
  │
  └── Poland → Baltics (EE) → FINLAND
```

1. **Continental European demand** has strong double peaks (industrial morning + residential evening)
2. These propagate through **Nord Pool day-ahead price coupling** across interconnectors
3. **Finland's own demand is flat** (plateau 8-19h), but imported prices from the coupled market carry the European demand pattern into Finnish spot prices
4. Finland is a **net importer** (avg +500 to +1,000 MW during daytime), so interconnector price dynamics dominate

#### Hourly price profile: Finland vs neighbors (workday average)

```
Hour |   FI  |  SE1  |  SE3  |   EE   | FI demand | Net import
-----+-------+-------+-------+--------+-----------+-----------
  03 |   2.9 |  22.1 |  40.1 |   49.2 |     9,228 |      +386
  06 |   4.8 |  23.9 |  43.8 |   86.2 |     9,911 |      +760
  08 |   8.8 |  38.7 |  78.1 |  142.9 |    10,393 |      +842
  09 |   9.1 |  46.7 |  89.2 |  125.7 |    10,470 |      +898  ← AM price peak
  11 |   7.0 |  42.1 |  68.7 |   90.2 |    10,509 |      +996  ← demand peak
  14 |   5.7 |  32.3 |  47.2 |   83.0 |    10,378 |    +1,020
  19 |   8.3 |  43.6 |  87.2 |  138.8 |    10,404 |      +745  ← PM price peak
  20 |   8.1 |  41.0 |  92.0 |  146.9 |    10,319 |      +707
```

The FI price profile mirrors the SE1/SE3/EE profiles, not the FI demand profile. The demand is essentially flat from 8-19h while prices track the European pattern.

### 8. Why This Validates the Model Architecture

The Gaussian demand features at 9h/19h in our model are **proxies for the European demand cycle**, not the Finnish demand cycle. This explains why:

1. **Finnish demand features don't improve the model** — they measure the wrong signal (flat plateau vs sharp peaks)
2. **Cross-border price spreads (Tier 2) DO improve it** — they directly capture the price coupling effect
3. **The model's time-of-day features work** — hour_sin/cos + Gaussians capture the imported daily pattern
4. **The model's seasonal features work** — month_sin/cos + HDD capture winter demand amplification which does affect both Finnish and European demand

## Conclusion

**The current model's demand representation is adequate.** The Gaussian peaks at 9h/19h correctly capture price-relevant demand transitions that are imported from the European market through Nord Pool price coupling, rather than the flat Finnish demand plateau.

Adding Fingrid demand data as a direct feature does not improve predictions because:
1. The price-relevant signal originates from European demand, not Finnish demand
2. Weather features already proxy for demand (temperature → heating → demand)
3. Demand data would not be available at forecast time without a separate demand prediction model
4. Cross-border price spreads (Tier 2) already capture the market coupling effect

**Recommendation:** Keep the current Gaussian demand model. For further improvement, focus on:
- Cross-border price spread dynamics (Tier 2) — this is where the demand signal enters Finnish prices
- Nuclear outage prediction (Tier 3) — these are the largest supply-side events
- Weather forecast accuracy — drives both Finnish demand and Nordic wind/hydro supply
