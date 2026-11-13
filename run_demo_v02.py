"""
Hydrocline v0.2 end-to-end demo: all four critical components working together.
  forcing files (CMEMS + ERA5 format) -> ingestion -> drift engine -> real coastline -> ecosystem layer
Forcing values are ILLUSTRATIVE (typical Guinea Current with tidal pulsing, south-westerly monsoon wind
with a daily cycle). Swap in downloaded Copernicus Marine and ERA5 files to run on real conditions.
Run: python run_demo_v02.py  -> demo_v02_results.md, demo_v02_map.png
"""
import os, numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box
from hydrocline_engine import read_forcing, run_ensemble, Coast, area95_km2, habitat_exposure, COASTAL_UNITS

HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, "demo_forcing"); os.makedirs(FD, exist_ok=True)
coast = Coast(os.path.join(HERE, "nigeria_land_ne10m.geojson"), os.path.join(HERE, "nigeria_coastline_ne10m.geojson"))
T0 = pd.Timestamp("2026-07-01T00:00"); t = pd.date_range(T0, periods=97, freq="h"); h = np.arange(97)
lat = np.round(np.arange(3.0, 7.01, 0.25), 2); lon = np.round(np.arange(2.0, 10.01, 0.25), 2)
LA, LO = np.meshgrid(lat, lon, indexing="ij")

def write_cmems(path, speed, toward_deg):
    tide = 1.0 + 0.25 * np.sin(2 * np.pi * h / 12.42)              # M2 tidal pulsing
    near = 1.0 - 0.3 * np.exp(-((LA - 4.6) / 0.6) ** 2)              # slower over the shelf
    s = speed * tide[:, None, None] * near[None]
    d = np.deg2rad(toward_deg)
    uo, vo = (s * np.sin(d))[:, None], (s * np.cos(d))[:, None]
    xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), uo), "vo": (("time", "depth", "latitude", "longitude"), vo)},
               coords=dict(time=t, depth=[0.494], latitude=lat, longitude=lon)).to_netcdf(path)

def write_era5(path, speed, from_deg):
    sp = speed + 1.0 * np.sin(2 * np.pi * (h - 14) / 24)          # afternoon sea breeze peak
    d = np.deg2rad(from_deg + 180)
    u10 = (sp * np.sin(d))[:, None, None] * np.ones((1, lat.size, lon.size)); v10 = (sp * np.cos(d))[:, None, None] * np.ones((1, lat.size, lon.size))
    xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), u10[:, ::-1]), "v10": (("valid_time", "latitude", "longitude"), v10[:, ::-1])},
               coords=dict(valid_time=t, latitude=lat[::-1], longitude=lon)).to_netcdf(path)

write_cmems(f"{FD}/cmems_currents_A.nc", 0.35, 80); write_cmems(f"{FD}/cmems_currents_B.nc", 0.30, 85)
write_era5(f"{FD}/era5_wind.nc", 5.0, 225)
wind = read_forcing(f"{FD}/era5_wind.nc", "wind", T0)

# Scenario A: oil release off the western Niger Delta, 72 h
curA = read_forcing(f"{FD}/cmems_currents_A.nc", "current", T0)
A = run_ensemble(4.62, 4.55, 72, 2000, curA, wind, coast=coast, windage=(0.03, 0.005), cur_scale_sd=0.34,
                 cur_rot_sd_deg=20, diff_k=30, seed=2026, release_sd_deg=0.01, snapshots=(24, 48, 72))
bA = ~A["alive"]; expo = habitat_exposure(A, coast)
# Scenario B: person in water 25 km off Lagos, 24 h
curB = read_forcing(f"{FD}/cmems_currents_B.nc", "current", T0)
B = run_ensemble(3.40, 6.20, 24, 2000, curB, wind, coast=coast, windage=(0.011, 0.004), cur_scale_sd=0.40,
                 cur_rot_sd_deg=25, diff_k=20, seed=2027, release_sd_deg=0.01, snapshots=(6, 12, 24))
vmax = (0.30 + 2 * 0.12) + (0.011 + 2 * 0.004) * (5.0 + 2 * 1.5)  # same plain-circle rule as v0.1
areas = {k: area95_km2(B["snaps"][k][0][B["snaps"][k][2]], B["snaps"][k][1][B["snaps"][k][2]]) for k in (6.0, 12.0, 24.0)}
naive = {k: np.pi * (vmax * k * 3600 / 1000) ** 2 for k in (6.0, 12.0, 24.0)}

summary = {
    "A_reach_shore_pct": round(100 * bA.mean()), "A_first_h": round(float(np.nanmin(A["beach_t"]))),
    "A_median_h": round(float(np.nanmedian(A["beach_t"]))),
    "B_area24": round(areas[24.0]), "B_naive24": round(naive[24.0]), "B_smaller24": round(100 * (1 - areas[24.0] / naive[24.0])),
}
print(summary); print(expo.to_string(index=False))
for k in (6.0, 12.0, 24.0):
    print(f"B t+{k:.0f}h: {areas[k]:.0f} km2 vs {naive[k]:.0f} km2 ({100*(1-areas[k]/naive[k]):.0f}% smaller)")

# figure
UCOL = ["#90CAF9", "#42A5F5", "#0D47A1", "#1565C0"]; SH = ["#90CAF9", "#1E88E5", "#0D47A1"]
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), dpi=150)
for a_, (xl, yl) in zip(ax, [((3.9, 7.6), (3.7, 6.3)), ((2.9, 4.2), (5.85, 6.65))]):
    a_.set_facecolor("#F0F7FE")
    for g in coast.land.geoms:
        a_.fill(*g.exterior.xy, color="#D9E4EF", ec="#0F3B66", lw=0.8, zorder=2)
    for (name, lo, hi, sens), c in zip(COASTAL_UNITS, UCOL):
        part = coast.lines.intersection(box(lo, 4.0, hi, 8.0))
        for g in getattr(part, "geoms", [part]):
            a_.plot(*g.xy, color=c, lw=4, zorder=3, solid_capstyle="round")
    a_.set_xlim(*xl); a_.set_ylim(*yl); a_.set_xlabel("Longitude (°E)"); a_.set_ylabel("Latitude (°N)"); a_.grid(color="white")
for (k, c) in zip((24.0, 48.0, 72.0), SH):
    lo_, la_, al_ = A["snaps"][k]; ax[0].scatter(lo_, la_, s=3, color=c, alpha=0.5, zorder=4)
ax[0].scatter([4.62], [4.55], marker="*", s=300, color="#F57C00", ec="#0A2540", zorder=6)
ax[0].set_title(f"A. Oil release, 72 h: {summary['A_reach_shore_pct']}% reach shore, first landfall {summary['A_first_h']} h",
                loc="left", color="#0A2540", fontweight="bold")
ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f"t + {int(k)} h") for k, c in zip((24, 48, 72), SH)] +
             [Line2D([], [], color=c, lw=4, label=n.split(" (")[0]) for (n, *_), c in zip(COASTAL_UNITS, UCOL)],
             loc="upper left", fontsize=8.5, framealpha=0.95)
for (k, c) in zip((6.0, 12.0, 24.0), SH):
    lo_, la_, al_ = B["snaps"][k]; ax[1].scatter(lo_[al_], la_[al_], s=3, color=c, alpha=0.5, zorder=4)
ax[1].scatter([3.40], [6.20], marker="*", s=300, color="#F57C00", ec="#0A2540", zorder=6)
ax[1].set_title(f"B. Person in water off Lagos, 24 h: search area {summary['B_smaller24']}% smaller", loc="left", color="#0A2540", fontweight="bold")
ax[1].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f"t + {int(k)} h") for k, c in zip((6, 12, 24), SH)], loc="lower right")
fig.suptitle("Hydrocline v0.2: real coastline, forcing read from Copernicus/ERA5-format files, ecosystem layer",
             color="#0A2540", fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.01, "Coastline: Natural Earth 10 m (public domain). Forcing values illustrative. Coastal units after Ibe (1988), approximate. Not yet validated on real incidents.",
         ha="center", fontsize=8.5, color="#374151")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(os.path.join(HERE, "demo_v02_map.png"), facecolor="white")

with open(os.path.join(HERE, "demo_v02_results.md"), "w") as f:
    f.write("# Hydrocline v0.2: end-to-end demo results\n\n")
    f.write("Pipeline: Copernicus/ERA5-format forcing files -> ingestion -> drift engine (2,000 particles) -> real Natural Earth coastline -> ecosystem layer.\n\n")
    f.write("Forcing values are illustrative, not observed. Results show the system works end to end, not real-world accuracy.\n\n")
    f.write(f"## A. Oil release off the western Niger Delta, 72 h\n\n- Reach shore within 72 h: {summary['A_reach_shore_pct']}%\n- First landfall: {summary['A_first_h']} h, median {summary['A_median_h']} h\n\n")
    f.write("| Coastal unit | Sensitivity | Share of landings | Shore stretch hit (west to east) |\n|---|---|---|---|\n")
    for _, r in expo.iterrows():
        f.write(f"| {r.unit} | {r.sensitivity} | {r.share_of_landings_pct}% | {r.shore_stretch_km} km |\n")
    f.write("\n## B. Person in water 25 km off Lagos, 24 h\n\n| Time | Hydrocline 95% search area | Plain circle | Smaller by |\n|---|---|---|---|\n")
    for k in (6.0, 12.0, 24.0):
        f.write(f"| t + {k:.0f} h | {areas[k]:,.0f} km² | {naive[k]:,.0f} km² | {100*(1-areas[k]/naive[k]):.0f}% |\n")
