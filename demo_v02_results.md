# Hydrocline v0.2: end-to-end demo results

Pipeline: Copernicus/ERA5-format forcing files -> ingestion -> drift engine (2,000 particles) -> real Natural Earth coastline -> ecosystem layer.

Forcing values are illustrative, not observed. Results show the system works end to end, not real-world accuracy.

## A. Oil release off the western Niger Delta, 72 h

- Reach shore within 72 h: 40%
- First landfall: 41 h, median 64 h

| Coastal unit | Sensitivity | Share of landings | Shore stretch hit (west to east) |
|---|---|---|---|
| Barrier-lagoon coast (Badagry to Lekki) | Moderate: sandy beach, lagoons behind | 0.0% | 0 km |
| Mahin mud coast (Ondo) | High: muddy shore, fish nurseries | 0.0% | 0 km |
| Arcuate Niger Delta | Very high: mangrove forest | 100.0% | 67 km |
| Strand coast (Imo River to Cross River) | Very high: mangrove estuaries | 0.0% | 0 km |

## B. Person in water 25 km off Lagos, 24 h

| Time | Hydrocline 95% search area | Plain circle | Smaller by |
|---|---|---|---|
| t + 6 h | 213 km² | 702 km² | 70% |
| t + 12 h | 573 km² | 2,808 km² | 80% |
| t + 24 h | 1,937 km² | 11,230 km² | 83% |
