# Diagrams — SUPERSEDED, do not use

⚠️ **Every file in this directory describes an architecture the project no
longer has.** They were drawn 2026-04-03…04-13, before the integration was
rewritten as a Python custom component with the four-layer L1+L2+L3+L4
pipeline. None of them is referenced from any document.

Concretely, `architecture-overview.drawio` shows:

| the diagram says | the code actually does |
|---|---|
| "Home Assistant (always-on, **Jinja2 inference**)" | inference is Python — `pipeline.py`, class `Pipeline` |
| "Spot Price Forecast (**Jinja2 model eval**)" | `Pipeline.compute_forecast()` |
| "Feature Engineering (**28–38 features**)" | the L2 ridge has **9** features |
| "**Two-Stage** Ridge Regression + Piecewise Calibration" | L1 seasonal + L2 ridge + L3 AR(1) + L4 GPD POT |
| "**REST Sensors** (7–11×)" | `api_client.py` fetches directly |
| "**mgrey.se** + Elering" | elprisetjustnu.se + Elering |
| "`model_coefs.json` (coefficients + feature names)" | that is the *legacy* base model, whose output the pipeline overwrites |

Additionally `component-responsibility`, `interface-contract` and
`system-boundary` have no rendered PNG, so they cannot be viewed on
GitHub at all.

**Authoritative architecture documentation is [TECHNICAL_GUIDE.md](../../TECHNICAL_GUIDE.md).**
Redrawing these against the current pipeline is tracked in
[BACKLOG.md](../BACKLOG.md).
