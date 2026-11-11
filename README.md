# Hydrocline: Drift Engine v0.1

Hydrocline is a proposed digital twin of the ocean for Nigeria, built for the NIMASA Blue Economy Accelerator Programme (NIMASA-BEAP) 2026. This repository holds the first working module: a Monte Carlo drift forecaster for oil spills and people in the water.

## What it does

Given a release point, ocean current, wind and a time horizon, it releases thousands of virtual particles, moves each one with its own sampled current, wind drift (windage) and random diffusion, and reports where they end up. It gives a 95% search area and the chance, timing and location of reaching the coast.

## Results from the two demo scenarios

| Scenario | Result |
|---|---|
| A. Oil release off the Niger Delta, 72 h | 60% of particles reach shore, first arrival at 31 h, all landings on the mangrove coast, about 40 km of shoreline exposed |
| B. Person in the water off Lagos, 24 h | 95% search area of 2,024 km2 versus 11,230 km2 for a naive circle (82% smaller); 176 km2 vs 702 km2 at 6 h; 604 km2 vs 2,808 km2 at 12 h |

Figures are the .png files in this repository.

## Run it

```bash
pip install -r requirements.txt
python hydrocline_drift_prototype.py
```

The script prints the statistics and writes the forecast figure.

## Honest limits

- Currents and winds are **synthetic inputs** chosen to be plausible, not real model or satellite data.
- The coastline is a **simplified polyline**, not a survey-grade shoreline.
- The engine has **not been validated** against observed drift or past incidents. That is the purpose of the pilot we are asking NIMASA-BEAP to support.
- Readiness: engine at TRL 3, full system at TRL 2.

## Next steps

1. Replace synthetic forcing with real current and wind data (for example Copernicus Marine and ERA5).
2. Replace the polyline with a proper coastline.
3. Back-test against anonymised past incidents with RMRCC and C4i.
4. Add the ecosystem layer (mangroves, fish nurseries).

## Team

Elijah Chiemelie Dominic (Marine Biology, UNILAG) and Igbinosa Nosa Delight (Chemical Engineering, UNILAG).

Licence: MIT.
