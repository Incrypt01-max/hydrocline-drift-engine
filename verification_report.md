# Hydrocline v0.2: lab verification report

**8 of 8 tests passed.** Each test compares the engine with an exact answer or real geometry.

| Test | Component | Expected | Engine | Error | Tolerance | Result |
|---|---|---|---|---|---|---|
| T1 Steady current, 24 h | Drift engine | 25.92 km | 25.89 km | -0.112 % | 0.5 % | PASS |
| T2 Wind drag (3% windage), 24 h | Drift engine | 25.92 km | 25.89 km | -0.112 % | 0.5 % | PASS |
| T3 Turbulent spreading (20,000 particles) | Drift engine | sd 1.859 km | sd 1.857 km | -0.195 % | 3.0 % | PASS |
| T4 Time-varying (rotating) current, 0 to 24 h | Drift engine | exact circle, radius 2.75 km | max miss 0.6 m | 0.576 m | 50 m | PASS |
| T5 Read CMEMS + ERA5 NetCDF, run 12 h | Forcing ingestion | position 5.50 km north | miss 31 m | 31.5 m | 100 m | PASS |
| T6 Landfall on real Lagos shoreline | Real coastline | 14.41 h at 6.433 N | 14.50 h at 6.434 N | 5.41 min | 10 min | PASS |
| T7 Time-step convergence (48 h, sheared field) | Drift engine | error shrinks about 4x when step halves | 3028 m at 1 h step, 91.6 m at 10 min | 0.231 (order shortfall) | 0.3 (order shortfall) | PASS |
| T8 Same inputs give same forecast | Drift engine | identical | identical | 0 deg | 0 deg | PASS |

Time-step convergence order: 1.77 (2.0 expected for midpoint stepping).

Coastline: Natural Earth 10 m land and coastline (public domain), clipped to 2 to 10 E, 3 to 8 N.

Scope: this verifies that each critical function works as designed in the lab (TRL 3). It is not validation against observed incidents, which is the next stage (TRL 4 to 5).
