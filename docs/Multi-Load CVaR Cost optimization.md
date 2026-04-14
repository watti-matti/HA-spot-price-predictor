# Multi-Load CVaR Cost Optimization

Plan for integrating duration-sorted price distribution D(k) with
thermal energy optimization for prosumer load scheduling.

**Status**: Plan phase — spot price CVaR forecasting to be finalized first.

---

## 1. Theoretical Foundation: D(k) as CVaR

The duration-sorted price distribution D(k) is mathematically equivalent
to Conditional Value-at-Risk (CVaR), also called Expected Shortfall,
applied to the intra-day price distribution.

For a day with N = 24 hours, at level alpha = k/N:

- **VaR_alpha** = p_(k) = the k-th order statistic (alpha-quantile)
- **CVaR_alpha** = E[price | price <= VaR_alpha] = (1/k) * sum_{i=1}^{k} p_(i) = **D(k)**

The consumer optimization "schedule my load during the cheapest k hours"
is exactly **CVaR minimization** of the electricity cost distribution.

### Key identity

```
D(k) = (1/k) * sum_{i=1}^{k} p_(i)     where p_(i) are ascending-sorted prices
```

For each deferrable load:

```
k_hours   = ceil(energy_need_kWh / nominal_power_kW)
cost_rate = D(k_hours)                                  [c/kWh]
daily_cost = energy_need_kWh * D(k_hours) / 100          [EUR]
```

The D(k) value at the load's operating point IS the minimum achievable
average price per kWh, given optimal scheduling into the cheapest k hours.

### Inverse: marginal price from D(k)

```
p_(k) = k * D(k) - (k-1) * D(k-1)      (standard order statistics result)
```

This recovers the sorted price vector from the duration curve, enabling
segment-to-full-day reconstruction.

---

## 2. System Architecture

```
+---------------------+     +----------------------+
| Thermal Model       |     | Price Forecast       |
| (energy need/day)   |     | (170h hourly prices) |
|                     |     |                      |
| load -> Q_kWh[7d]   |     | day -> prices[24h]   |
+----------+----------+     +----------+-----------+
           |                            |
           v                            v
+--------------------------------------------------+
| Per-Day D(k) Computation                         |
|                                                  |
| For each day d=0..6:                             |
|   Sort 24 hourly consumer prices                 |
|   D(k) = (1/k) * sum p_(i) for k=1..24          |
|                                                  |
| For each load:                                   |
|   k = ceil(Q_kWh[d] / P_nominal)                 |
|   cost_rate = D(k)                               |
|   total_cost = Q_kWh[d] * D(k) / 100             |
+----------+---------------------------------------+
           |
           v
+--------------------------------------------------+
| PV-Aware Net Price Adjustment                    |
|                                                  |
| net_price_h = grid_price_h                       |
|   - min(PV_h, P_load) / P_load                   |
|     * (grid_price_h - feed_in_rate)               |
|                                                  |
| D_net(k, load) computed on net prices             |
| Note: D_net is load-specific (different P_load)   |
+----------+---------------------------------------+
           |
           v
+--------------------------------------------------+
| Power Budget Coordination                        |
|                                                  |
| For each hour: sum nominal power of loads that   |
| would schedule that hour -> detect > 25 kW       |
|                                                  |
| Conflict resolution: push lowest-priority load   |
| to next cheapest available hour -> D(k')         |
+----------+---------------------------------------+
           |
           v
    Dashboard / HA Sensors
```

---

## 3. Data Sources

### Spot price forecast (watti-matti)

- 170-hour hourly price forecast from Ridge regression model
- Consumer price: `(max(0, spot) / 1000 + 0.0361 + 0.02325) * 1.255 * 100` [c/kWh]
- D(k) computed per day from 24 hourly consumer prices
- Duration model: Ridge + PAVA with lambda = 0.990 (69-day half-life)

### Thermal energy need (thermal-energy-optimization)

7 loads across 7 thermal zones:

| Load                  | Power (kW) | Typical Q (kWh/d) | k hours | Zone              |
|-----------------------|------------|--------------------|---------| ------------------|
| Heat pump downstairs  | 3.5        | 14-28              | 4-8     | living_area       |
| Heat pump upstairs    | 2.5        | 8-15               | 3-6     | bedrooms          |
| Floor heating bath    | 1.1        | 4-10               | 4-9     | bathroom          |
| Floor heating hallway | 0.6        | 3-6                | 5-10    | living_area       |
| Floor heating garage  | 1.5        | 3-5                | 2-3     | garage            |
| Mass storage heater   | 2.0        | 4-11               | 2-6     | workshop          |
| Water boiler          | 3.0        | 3-9                | 1-3     | water_storage     |
| EV charger            | 11.0       | 11-44              | 1-4     | standalone        |

Energy need estimation per zone:

```
Q_thermal = sum_h { max(UA_eff * (T_set - T_out) - A_eff * solar_h - Q_int/24, 0) }
Q_electric = Q_thermal / COP
k_hours = ceil(Q_electric / nominal_power_kW)
```

Parameters learned via RWLS with lambda = 0.97 (23-day half-life).

### PV forecast

- Source: `sensor.meteo_7day_forecast_total` (168 hourly watts)
- Arrays: south (29.92 m^2, 20.7% eff, tilt 45 deg) + west (13.1 m^2, 21.3% eff, tilt 30 deg)
- Capacity: 5.0 kW peak
- Feed-in value: 0.05 EUR/kWh (`solar_self_use_value` in loads.yaml)

### System constraints

- Maximum simultaneous electrical load: 25.0 kW
- Price ceiling: 0.25 EUR/kWh (hard block)
- Thermal buffer: 10% reserve on estimates
- Mass heater: 0-6 daily hours max
- Water boiler: 1-8 daily hours max
- EV charger: 0-12 daily hours max

---

## 4. PV Integration into D(k)

### Why PV forecast is required for optimality

Without PV, each hour has a single grid price. With PV generation G_h
at hour h, the effective cost of running load with power P_load becomes:

```
               grid_price_h                              if G_h = 0
net_price_h =  grid_price_h * (1 - G_h/P_load)
               + feed_in * G_h/P_load                    if G_h < P_load
               feed_in                                   if G_h >= P_load
```

This means D(k) must be computed on **net prices**, not grid prices.
Without PV forecast, the hour ordering is wrong: a grid-expensive midday
hour may be the cheapest hour net of solar.

### Seasonal impact at 60 deg N, 5 kW PV

| Season   | Daily PV (kWh) | Impact on D(k)                              |
|----------|----------------|----------------------------------------------|
| Dec-Jan  | 0-0.5          | Negligible -- grid-only D(k) is near-optimal |
| Mar, Oct | 5-10           | Moderate -- midday ranking shifts 1-3 places  |
| May-Jul  | 20-30          | Large -- midday hours become cheapest         |
| Jun peak | 30-35          | Critical -- 6-8 hours effectively free        |

### Load-specific D(k) curves

Net price depends on load power, so D(k) becomes device-specific:

```
D_net(k, load_A) != D_net(k, load_B)    when P_A != P_B
```

A 3.0 kW water boiler absorbs more PV fraction per hour than a 0.6 kW
floor heater, resulting in different effective price rankings.

---

## 5. Dashboard Visualization Design

### Panel 1: 7-Day Price Landscape (top, full width)

Horizontal strip: 168 hours as colored cells, day boundaries marked.
Color scale: consumer c/kWh (blue -> yellow -> red).
PV generation overlay: semi-transparent green bars showing hourly production.

Purpose: context -- see which days/hours are cheap vs. expensive,
where PV shifts the effective cost.

### Panel 2: Duration Curves with Load Markers (left, main insight)

7 small-multiple D(k) curves, one per day.
- X-axis: k hours (1-24)
- Y-axis: D(k) in c/kWh
- Solid line: grid-only D(k)
- Dashed line: PV-adjusted D_net(k) (per-load or representative load)
- Each load plotted as a labeled dot at position (k_hours, D_net(k))
- Dot size proportional to energy need
- Color = load identity (consistent across all panels)

Key insight: **the dot's y-position IS the load's average cost per kWh**.

```
  D(k) |            o--- daily avg (24h reference)
  c/kWh|         o----- heat pump 8h
       |      o-------- water heater 6h
       |    o---------- EV 4h
       |  o------------ cheapest 1h
       +-------------------------------- k hours
```

### Panel 3: Load Cost Trajectories (right, decision support)

Line chart: 7 days on x-axis, daily cost in EUR on y-axis.
One line per load, color-coded with markers.

- Star marker on cheapest day per load
- Dashed line showing cost without PV (grid-only D(k))
- Shaded area between grid and net cost = PV savings per load

Answers: "Tuesday is cheapest for water heater, Thursday for EV".

### Panel 4: Power Budget Timeline (middle, coordination)

For each day: 24-hour stacked bar showing optimal load allocation.
Each load's cheapest k hours shaded in its color.

```
  kW  | ############### EV 11kW (conflict at hours 2-4)
  25  |------- budget limit --------------------------
      | ####  heat pump 3.5kW
      | ###   water boiler 3kW
      | ##    floor heating 1.5kW
      +----------------------------------------------- hour
       0   4   8   12  16  20  24
```

Red line at 25 kW budget limit. Conflict hours highlighted where
independent scheduling exceeds budget.

PV generation shown as green area (negative direction): loads scheduled
during PV hours reduce grid demand.

### Panel 5: Deferral Savings Matrix (bottom, actionable)

Table: rows = loads, columns = days 0-6.
Cell value: daily cost in EUR if load runs on that day.
Cell color: green (cheapest) -> red (most expensive).

```
                Mon   Tue   Wed   Thu   Fri   Sat   Sun   Savings
Heat pump      1.24  1.18  *0.98 1.31  1.15  1.22  1.28  0.33 EUR
Water boiler   0.45  *0.31 0.38  0.52  0.44  0.41  0.47  0.21 EUR
EV charger     2.80  2.65  *2.10 2.95  2.73  2.81  2.88  0.85 EUR
Floor bath     0.22  0.19  *0.16 0.24  0.21  0.20  0.23  0.08 EUR
```

Last column: savings vs. flat-rate (running any day at uniform cost).
Answers: "How much do I save by deferring water heater to Wednesday?"

---

## 6. Implementation Phases

### Phase 1: Spot Price CVaR Forecasting (current)

Finalize D(k) duration model for publication:

1. **Duration model in `src/train_model.py`**: Weighted Ridge per
   (segment, duration_level), lambda = 0.990, PAVA post-processing.
2. **Model coefficients**: D(k) prediction coefficients in `model_coefs.json`
   alongside existing hourly price model.
3. **HA inference**: PAVA-constrained duration prediction in `model.py`
   (pure Python, no numpy).
4. **Sensor output**: D(k) for k = 1, 4, 6, 8, 12, 24 as HA sensor
   attributes, enabling EMHASS integration.
5. **Full-day curve reconstruction**: 4-segment merge for 24-hour D(k).
6. **Dashboard**: Duration curve visualization in HA Lovelace card.

**Deliverable**: Published v1.7.0 with D(k) sensor attributes.

### Phase 2: Thermal Load Integration

Connect thermal energy optimization with D(k) forecast:

1. **Energy need -> k mapping**: For each load, compute
   `k = ceil(Q_electric / P_nominal)` from thermal model output.
2. **D(k) lookup**: Map k to D(k) from spot price predictor sensor.
3. **Daily cost estimation**: `cost = Q * D(k) / 100` for each load.
4. **7-day cost trajectory**: Extend D(k) computation to all 7 forecast
   days using 170-hour price forecast.
5. **HA sensor**: Per-load cost estimate sensor
   `sensor.thermal_opt_{load_id}_estimated_cost`.

**Deliverable**: Cost trajectory sensors in HA.

### Phase 3: PV-Aware Net Price D(k)

Integrate PV forecast into duration curve computation:

1. **Net price computation**: For each load, compute load-specific
   net prices using PV forecast from `sensor.meteo_7day_forecast_total`.
2. **Load-specific D_net(k)**: Sort net prices per load, compute
   PV-adjusted duration curves.
3. **PV savings quantification**: Difference between grid D(k) and
   net D_net(k) per load per day.
4. **Seasonal adaptation**: Net price impact varies from negligible
   (winter) to dominant (summer).

**Deliverable**: PV-aware D(k) sensors with savings attribution.

### Phase 4: Power Budget Coordination

Multi-load scheduling with finite power constraint:

1. **Conflict detection**: For each hour, sum nominal power of all loads
   that would independently schedule that hour.
2. **Priority-based resolution**: When sum exceeds 25 kW budget, push
   lowest-priority load (`allocation_weight`) to next cheapest hour.
3. **Coordinated D(k')**: Compute degraded duration cost D(k') for
   displaced loads.
4. **LP formulation**: Joint optimization as multi-asset CVaR portfolio
   with power budget constraint.

**Deliverable**: Coordinated scheduling with conflict resolution.

### Phase 5: Dashboard Prototype

Build 5-panel HTML dashboard using Chart.js:

1. Prototype with historical data from duration_study.py output.
2. Representative load parameters from loads.yaml.
3. Simulated PV from Open-Meteo solar irradiance archive.
4. Static HTML with interactive day selector and load toggles.
5. Integration path: HA Lovelace custom card or iframe.

**Deliverable**: `output/cvar_cost_dashboard.html` prototype.

---

## 7. Scientific References

### CVaR / Expected Shortfall (the D(k) identity)

1. Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of
   Conditional Value-at-Risk." *Journal of Risk, 2*(3), 21-41.
   [docs/Literature/rtr179-CVaR1.pdf]

2. Rockafellar, R.T. & Uryasev, S. (2002). "Conditional Value-at-Risk
   for General Loss Distributions." *Journal of Banking & Finance,
   26*(7), 1443-1471. [docs/Literature/rtr187-CVaR2.pdf]

3. Pflug, G.Ch. (2000). "Some Remarks on the Value-at-Risk and the
   Conditional Value-at-Risk." In *Probabilistic Constrained
   Optimization*, Springer.

### Price Duration Curves

4. Stoft, S. (2002). *Power System Economics: Designing Markets for
   Electricity.* IEEE/Wiley.

5. Weron, R. (2014). "Electricity price forecasting: A review of the
   state-of-the-art with a look into the future." *International
   Journal of Forecasting, 30*(4), 1030-1044.
   [docs/Literature/1-s2.0-S0169207014001083-main.pdf]

6. Joskow, P.L. (2007). "Competitive Electricity Markets and Investment
   in New Generating Capacity." In *The New Energy Paradigm*, Oxford
   University Press.

### Exponentially Weighted Ridge Regression

7. Hoerl, A.E. & Kennard, R.W. (1970). "Ridge Regression: Biased
   Estimation for Nonorthogonal Problems." *Technometrics, 12*(1),
   55-67.

8. Ljung, L. (1999). *System Identification: Theory for the User*
   (2nd ed.). Prentice Hall. -- Sec. 11.3: recursive least squares with
   exponential forgetting.

9. Haykin, S. (2002). *Adaptive Filter Theory* (4th ed.). Prentice
   Hall. -- Ch. 13: RLS with forgetting factor.

10. Gaillard, P., Goude, Y. & Nedellec, R. (2016). "Additive models
    and robust aggregation for GEFCom2014 probabilistic electric load
    and electricity price forecasting." *Int. J. Forecasting, 32*(3),
    1038-1050.

### Log Transformation and Back-Transform Bias

11. Box, G.E.P. & Cox, D.R. (1964). "An Analysis of Transformations."
    *J. Royal Statistical Society, Series B, 26*(2), 211-252.

12. Weron, R. (2006). *Modeling and Forecasting Electricity Loads and
    Prices: A Statistical Approach.* Wiley.

13. Duan, N. (1983). "Smearing Estimate: A Nonparametric
    Retransformation Method." *JASA, 78*(383), 605-610.

### Isotonic Regression (PAVA)

14. Barlow, R.E., Bartholomew, D.J., Bremner, J.M. & Brunk, H.D.
    (1972). *Statistical Inference Under Order Restrictions.* Wiley.

15. de Leeuw, J., Hornik, K. & Mair, P. (2009). "Isotone Optimization
    in R: Pool-Adjacent-Violators Algorithm (PAVA) and Active Set
    Methods." *J. Statistical Software, 32*(5), 1-24.

### Hierarchical Forecasting

16. Hong, T., Pinson, P. & Fan, S. (2014). "Global Energy Forecasting
    Competition 2012." *Int. J. Forecasting, 30*(2), 357-363.

17. Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. & Shang, H.L.
    (2011). "Optimal Combination Forecasts for Hierarchical Time
    Series." *Computational Statistics & Data Analysis, 55*(9),
    2579-2589.

### Spearman Rank Correlation

18. Spearman, C. (1904). "The proof and measurement of association
    between two things." *American J. Psychology, 15*(1), 72-101.

19. Conover, W.J. (1999). *Practical Nonparametric Statistics* (3rd
    ed.). Wiley.

### Order Statistics

20. David, H.A. & Nagaraja, H.N. (2003). *Order Statistics* (3rd ed.).
    Wiley. -- Sec. 2.3: linear functions of order statistics; the
    p_(k) = k*D(k) - (k-1)*D(k-1) reconstruction.

### Prosumer Scheduling with PV + Thermal Storage

21. Salpakari, J. & Lund, P.D. (2016). "Optimal and rule-based control
    strategies for energy flexibility in buildings with PV." *Energy
    and Buildings, 120*, 98-109. -- Finnish context, heat pump + water
    heater + PV spot price optimization.
    [docs/Literature/salpakari_lund_optimal_and_rule_based_control_APEN_2016.pdf]

22. Brahman, F., Honarmand, M. & Jadid, S. (2015). "Optimal electrical
    and thermal energy management of a residential energy hub,
    integrating demand response and energy storage system." *Int. J.
    Electrical Power & Energy Systems, 64*, 1067-1079.

23. Luthander, R., Widen, J., Nilsson, D. & Palm, J. (2015).
    "Photovoltaic self-consumption in buildings: A review." *Applied
    Energy, 142*, 80-94.

### Stochastic Optimization with Renewable Uncertainty

24. Conejo, A.J., Carrion, M. & Morales, J.M. (2010). *Decision Making
    Under Uncertainty in Electricity Markets.* Springer. -- Ch. 7:
    prosumer CVaR with distributed generation.
    [docs/Literature/Conejo,  Decision Making Under Uncertainty in Electricity Markets.pdf]

25. Morales, J.M., Conejo, A.J., Madsen, H., Pinson, P. & Zugno, M.
    (2014). *Integrating Renewables in Electricity Markets.* Springer.
    -- Sec. 5.3: net demand approach.

26. Parisio, A., Rikos, E. & Glielmo, L. (2014). "A Model Predictive
    Control Approach to Microgrid Operation Optimization." *IEEE Trans.
    Control Systems Technology, 22*(5), 1813-1827.

27. Oldewurtel, F. et al. (2012). "Use of model predictive control
    and weather forecasts for energy efficient building climate
    control." *Energy and Buildings, 45*, 15-27.

### Demand Response and Flexible Load Scheduling

28. Ottesen, S.O., Tomasgard, A. & Fleten, S.E. (2018). "Multi
    building portfolio energy management and stochastic programming."
    *Applied Energy, 228*, 2181-2198.
    [docs/Literature/Optimal_demand_response_aggregation_in_wholesale_e.pdf]

29. Erdinc, O., Paterakis, N.G., Pappi, I.N., Bakirtzis, A.G. &
    Catalao, J.P.S. (2015). "A new perspective for sizing of
    distributed generation and energy storage for smart households
    under demand response." *Applied Energy, 143*, 26-37.

### Value of Forecast Information

30. Birge, J.R. & Louveaux, F. (2011). *Introduction to Stochastic
    Programming* (2nd ed.). Springer. -- Sec. 4.2: Expected Value of
    Perfect Information (EVPI).

31. Nowotarski, J. & Weron, R. (2018). "Recent advances in electricity
    price forecasting: A review of the state-of-the-art with a look
    into the future." *Renewable and Sustainable Energy Reviews, 81*,
    1548-1568.

---

## 8. Known Limitations

### Duration Model

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Segment reconstruction amplifies bias | D(1) > D(24) Spearman artifact | Direct full-day D(k) model |
| Single lambda forgets seasonality | Poor winter prediction after long summer | Explicit seasonal features (month_sin/cos) |
| No regime detection | ~69 days to adapt after market shocks | Regime-switching or adaptive lambda |
| Log back-transform bias | Systematic underestimation | Duan's smearing estimator [13] |
| Cross-segment independence | Ignores night-midday correlation | Joint multivariate model |
| Spearman-MSE mismatch | MSE minimization != rank maximization | Learning-to-rank methods |

### PV Integration

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Load-specific D(k) curves | N loads * 7 days * 24 levels = combinatorial | Pre-compute for active loads only |
| PV forecast uncertainty | Cloud transients change hour ordering | Robust CVaR with PV scenarios |
| Feed-in tariff assumption | Fixed f = 0.05 EUR/kWh may not hold | Dynamic tariff from retailer API |
| Net metering interaction | Self-consumption priority rules vary | Configuration per retailer |

### Power Budget Coordination

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Greedy priority resolution | Not globally optimal | Full LP with power constraint |
| EV dominates budget (11 kW) | Crowds out thermal loads | Time-of-use constraints for EV |
| Phase balance not modeled | 3-phase limit != sum of phases | Phase-aware constraint |

---

## 9. Metrics and Validation

### Duration model performance (lambda = 0.990)

Full-history (1271 days):

| Level | Spearman rho | MAE (c/kWh) | Bias (c/kWh) |
|-------|-------------|-------------|--------------|
| D(1) cheapest 1h | 0.384 | 0.369 | +0.021 |
| D(4) EV charge 4h | 0.407 | 0.398 | +0.029 |
| D(8) heat pump 8h | 0.412 | 0.454 | +0.059 |
| D(24) daily avg | 0.381 | 0.900 | +0.460 |

Latest 90-day window:

| Level | Spearman rho |
|-------|-------------|
| D(4) | 0.597 |
| D(8) | 0.580 |
| D(24) | 0.469 |

### Target metrics for multi-load system

| Metric | Target | Method |
|--------|--------|--------|
| Rank accuracy (Spearman) | rho > 0.5 for D(4), D(8) | Rolling 90-day evaluation |
| Cost estimation error | < 0.5 EUR/day per load | Backtest vs. actual consumption |
| PV savings capture | > 80% of optimal self-consumption | Compare grid vs. net D(k) scheduling |
| Power budget violations | 0 per day | Constraint enforcement |
| Scheduling latency | < 5 seconds for 7-day plan | Benchmark on HA hardware |
