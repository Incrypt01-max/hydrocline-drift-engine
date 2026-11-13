"""
Hydrocline v0.2: lab verification of every critical component (TRL 3 evidence).
Each test compares the engine with an exact (analytical) answer or a known geometry.
Run: python verify_physics.py   -> verification_report.md, verification_tests.png
"""
import os, tempfile, numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from hydrocline_engine import (run_ensemble, constant_field, read_forcing, Coast, M_PER_DEG)

HERE = os.path.dirname(os.path.abspath(__file__))
coast = Coast(os.path.join(HERE, "nigeria_land_ne10m.geojson"), os.path.join(HERE, "nigeria_coastline_ne10m.geojson"))
ZERO = constant_field(0.0, 0.0)
rows = []

def km(lon0, lat0, lon1, lat1):
    p0, p1 = np.deg2rad(lat0), np.deg2rad(lat1)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.deg2rad(lon1 - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def add(test, component, expected, got, err, tol, unit):
    rows.append(dict(test=test, component=component, expected=expected, engine=got,
                     error=f"{err:.3g} {unit}", tolerance=f"{tol} {unit}", result="PASS" if abs(err) <= tol else "FAIL"))

# T1 uniform current: 0.30 m/s east for 24 h = 25.92 km
r = run_ensemble(3.4, 4.5, 24, 1, constant_field(0.30, 0.0), ZERO)
d = km(3.4, 4.5, r["lon"][0], r["lat"][0]); exp = 0.30 * 86400 / 1000
add("T1 Steady current, 24 h", "Drift engine", f"{exp:.2f} km", f"{d:.2f} km", 100 * (d - exp) / exp, 0.5, "%")

# T2 windage: 10 m/s wind x 3% = 0.30 m/s
r = run_ensemble(3.4, 4.5, 24, 1, ZERO, constant_field(10.0, 0.0), windage=(0.03, 0.0))
d = km(3.4, 4.5, r["lon"][0], r["lat"][0])
add("T2 Wind drag (3% windage), 24 h", "Drift engine", f"{exp:.2f} km", f"{d:.2f} km", 100 * (d - exp) / exp, 0.5, "%")

# T3 diffusion: variance per axis = 2 K t
K, T = 20.0, 24.0
r = run_ensemble(3.4, 4.5, T, 20000, ZERO, ZERO, diff_k=K, seed=7)
x = (r["lon"] - 3.4) * M_PER_DEG * np.cos(np.deg2rad(4.5)); y = (r["lat"] - 4.5) * M_PER_DEG
var_exp = 2 * K * T * 3600; var_got = 0.5 * (x.var() + y.var())
add("T3 Turbulent spreading (20,000 particles)", "Drift engine", f"sd {np.sqrt(var_exp)/1000:.3f} km",
    f"sd {np.sqrt(var_got)/1000:.3f} km", 100 * (var_got - var_exp) / var_exp, 3.0, "%")

# T4 time-varying rotating current: exact circle of radius U/omega
U, w = 0.20, 2 * np.pi / (24 * 3600)
rot = lambda t, lon, lat: (np.full(np.shape(lon), U * np.cos(w * t * 3600)), np.full(np.shape(lon), U * np.sin(w * t * 3600)))
errs, traj = [], []
for h in (3, 6, 9, 12, 18, 24):
    r = run_ensemble(3.4, 4.5, h, 1, rot, ZERO)
    xe, ye = U / w * np.sin(w * h * 3600), U / w * (1 - np.cos(w * h * 3600))
    xg = (r["lon"][0] - 3.4) * M_PER_DEG * np.cos(np.deg2rad(4.5)); yg = (r["lat"][0] - 4.5) * M_PER_DEG
    errs.append(np.hypot(xg - xe, yg - ye)); traj.append((h, xe, ye, xg, yg))
add("T4 Time-varying (rotating) current, 0 to 24 h", "Drift engine", "exact circle, radius 2.75 km",
    f"max miss {max(errs):.1f} m", max(errs), 50, "m")

# T5 ingestion round trip: same rotating current written as CMEMS NetCDF, null wind as ERA5 NetCDF
tmp = tempfile.mkdtemp(); t = pd.date_range("2026-01-01", periods=49, freq="h")
lat = np.linspace(3.5, 6.5, 13); lon = np.linspace(2.5, 6.5, 17); hh = np.arange(49)
uo = (U * np.cos(w * hh * 3600))[:, None, None, None] * np.ones((49, 1, 13, 17))
vo = (U * np.sin(w * hh * 3600))[:, None, None, None] * np.ones((49, 1, 13, 17))
xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), uo), "vo": (("time", "depth", "latitude", "longitude"), vo)},
           coords=dict(time=t, depth=[0.494], latitude=lat, longitude=lon)).to_netcdf(f"{tmp}/cmems.nc")
xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), np.zeros((49, 13, 17))),
            "v10": (("valid_time", "latitude", "longitude"), np.zeros((49, 13, 17)))},
           coords=dict(valid_time=t, latitude=lat[::-1], longitude=lon)).to_netcdf(f"{tmp}/era5.nc")
cur = read_forcing(f"{tmp}/cmems.nc", "current", t[0]); wnd = read_forcing(f"{tmp}/era5.nc", "wind", t[0])
r = run_ensemble(3.4, 4.5, 12, 1, cur, wnd)
xe, ye = 0.0, 2 * U / w
xg = (r["lon"][0] - 3.4) * M_PER_DEG * np.cos(np.deg2rad(4.5)); yg = (r["lat"][0] - 4.5) * M_PER_DEG
e5 = np.hypot(xg - xe, yg - ye)
add("T5 Read CMEMS + ERA5 NetCDF, run 12 h", "Forcing ingestion", "position 5.50 km north", f"miss {e5:.0f} m", e5, 100, "m")

# T6 beaching on the real coastline: 0.5 m/s north from 26 km off Lagos
clat = coast.coast_lat(3.40)
r = run_ensemble(3.40, 6.20, 30, 1, constant_field(0.0, 0.5), ZERO, coast=coast)
t_exp = (clat - 6.20) * M_PER_DEG / 0.5 / 3600
add("T6 Landfall on real Lagos shoreline", "Real coastline", f"{t_exp:.2f} h at {clat:.3f} N",
    f"{r['beach_t'][0]:.2f} h at {r['lat'][0]:.3f} N", (r["beach_t"][0] - t_exp) * 60, 10, "min")

# T7 convergence in a sheared, spatially varying field
shear = lambda t, lon, lat: (0.3 + 0.5 * np.sin(2 * np.pi * (lat - 4.0) / 0.15), 0.4 * np.cos(2 * np.pi * (lon - 3.0) / 0.15))
ref = run_ensemble(3.4, 4.5, 48, 1, shear, ZERO, dt=30)
conv = []
for dtt in (3600, 1800, 900, 600):
    r = run_ensemble(3.4, 4.5, 48, 1, shear, ZERO, dt=dtt)
    conv.append((dtt, km(ref["lon"][0], ref["lat"][0], r["lon"][0], r["lat"][0]) * 1000))
order = np.log(conv[0][1] / conv[1][1]) / np.log(2)
add("T7 Time-step convergence (48 h, sheared field)", "Drift engine", "error shrinks about 4x when step halves",
    f"{conv[0][1]:.0f} m at 1 h step, {conv[-1][1]:.1f} m at 10 min", -(order - 2.0) if order < 2 else 0.0, 0.3, "(order shortfall)")

# T8 reproducibility
a = run_ensemble(4.62, 4.55, 24, 500, shear, ZERO, diff_k=20, cur_scale_sd=0.3, seed=11)
b = run_ensemble(4.62, 4.55, 24, 500, shear, ZERO, diff_k=20, cur_scale_sd=0.3, seed=11)
diff = float(np.max(np.abs(a["lon"] - b["lon"])) + np.max(np.abs(a["lat"] - b["lat"])))
add("T8 Same inputs give same forecast", "Drift engine", "identical", "identical" if diff == 0 else f"{diff:g}", diff, 0, "deg")

df = pd.DataFrame(rows)
print(df[["test", "expected", "engine", "result"]].to_string(index=False))
print(f"\nConvergence order: {order:.2f} (2 = second order)")

# figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)
th = np.linspace(0, 24, 200)
ax[0].plot(U / w * np.sin(w * th * 3600) / 1000, U / w * (1 - np.cos(w * th * 3600)) / 1000, color="#90CAF9", lw=6, label="Exact answer")
ax[0].plot([p[3] / 1000 for p in traj], [p[4] / 1000 for p in traj], "o", color="#0A2540", label="Engine")
ax[0].set_title("T4: rotating current, exact vs engine", color="#0A2540", fontweight="bold"); ax[0].set_xlabel("km east"); ax[0].set_ylabel("km north"); ax[0].axis("equal"); ax[0].legend()
xx = np.linspace(-8, 8, 200); sd = np.sqrt(var_exp) / 1000
ax[1].hist(x / 1000, bins=60, density=True, color="#1E88E5", alpha=0.7, label="Engine (20,000 particles)")
ax[1].plot(xx, np.exp(-xx ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), color="#0A2540", lw=2.5, label="Exact (2Kt)")
ax[1].set_title("T3: turbulent spreading after 24 h", color="#0A2540", fontweight="bold"); ax[1].set_xlabel("km east of release"); ax[1].legend()
ax[2].loglog([c[0] / 60 for c in conv], [c[1] for c in conv], "o-", color="#1565C0", label="Engine error")
ax[2].loglog([60, 10], [conv[0][1], conv[0][1] / 36], "--", color="#90CAF9", label="Second-order slope")
ax[2].set_title("T7: error falls as time step shrinks", color="#0A2540", fontweight="bold"); ax[2].set_xlabel("time step (minutes)"); ax[2].set_ylabel("position error (m)"); ax[2].legend()
fig.suptitle("Hydrocline v0.2 lab verification: engine matches exact answers", color="#0A2540", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "verification_tests.png"), facecolor="white")

with open(os.path.join(HERE, "verification_report.md"), "w") as f:
    f.write("# Hydrocline v0.2: lab verification report\n\n")
    f.write(f"**{(df.result == 'PASS').sum()} of {len(df)} tests passed.** Each test compares the engine with an exact answer or real geometry.\n\n")
    f.write("| Test | Component | Expected | Engine | Error | Tolerance | Result |\n|---|---|---|---|---|---|---|\n")
    for _, r_ in df.iterrows():
        f.write(f"| {r_.test} | {r_.component} | {r_.expected} | {r_.engine} | {r_.error} | {r_.tolerance} | {r_.result} |\n")
    f.write(f"\nTime-step convergence order: {order:.2f} (2.0 expected for midpoint stepping).\n\n")
    f.write("Coastline: Natural Earth 10 m land and coastline (public domain), clipped to 2 to 10 E, 3 to 8 N.\n\n")
    f.write("Scope: this verifies that each critical function works as designed in the lab (TRL 3). "
            "It is not validation against observed incidents, which is the next stage (TRL 4 to 5).\n")
