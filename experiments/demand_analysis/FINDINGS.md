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

## Conclusion

**The current model's demand representation is adequate.** The Gaussian peaks at 9h/19h correctly capture price-relevant demand transitions rather than the flat demand plateau. Adding Fingrid demand data as a direct feature does not improve predictions because:

1. Weather features already proxy for demand (temperature → heating → demand)
2. Demand data would not be available at forecast time without a separate demand prediction model
3. The price-relevant signal is the demand *ramp rate*, which the Gaussian shape approximates well

**Recommendation:** Keep the current Gaussian demand model. If further improvement is needed, focus on:
- Better nuclear outage prediction (Tier 3)
- Cross-border price spread dynamics (Tier 2)
- Weather forecast accuracy (more locations or higher-resolution wind data)
