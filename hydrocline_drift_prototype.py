"""
Hydrocline Drift Engine - proof-of-concept (v0.1)
-------------------------------------------
Monte Carlo Lagrangian drift forecast for the Nigerian coast.

What it does
  * Releases an ensemble of virtual particles at an incident position.
  * Moves each particle with: surface current + wind drag (windage) + turbulent diffusion.
  * Each particle samples its own current, wind and windage from an uncertainty range,
    so the spread of the cloud IS the forecast uncertainty.
  * Stops particles that reach land and reports where and when they arrive.

Two scenarios
  A. Oil release 120 km off the western Niger Delta, 72-hour forecast.
  B. Person in water from a capsized canoe 25 km south of Lagos, 24-hour forecast.

IMPORTANT (honest scope)
  Forcing here is CLIMATOLOGICAL and ILLUSTRATIVE (typical Guinea Current and
  south-westerly monsoon wind), and the coastline is a simplified polyline.
  It proves the core algorithm runs end to end. It is NOT yet validated against
  observed drifter tracks or satellite-detected slicks. Next step: swap in
  Copernicus Marine current/wind forecasts and validate against real cases.

Run:  python hydrocline_drift_prototype.py
Needs: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon

RNG = np.random.default_rng(2026)
M_PER_DEG = 111_320.0

# Simplified Nigerian coastline, west to east (lon, lat). Schematic only.
COAST = np.array([
    (2.70, 6.37), (3.40, 6.42), (4.00, 6.43), (4.50, 6.30), (4.90, 5.95),
    (5.05, 5.75), (5.20, 5.55), (5.35, 5.35), (5.40, 5.10), (5.50, 4.85),
    (5.65, 4.65), (5.90, 4.42), (6.07, 4.30), (6.25, 4.30), (6.70, 4.35),
    (7.00, 4.40), (7.55, 4.48), (8.00, 4.52), (8.35, 4.60), (8.60, 4.50),
])
LAND = np.vstack([COAST, (8.60, 7.2), (2.70, 7.2)])
LAND_PATH = Path(LAND)
# Mangrove-fringed delta coast (schematic longitude band)
MANGROVE_LON = (5.05, 7.60)


def run(lon0, lat0, hours, n, cur_speed, cur_dir, wind_speed, wind_from,
        windage, diff_k, dt=600.0, snapshots=()):
    """Advance an ensemble. Directions in degrees: current 'toward', wind 'from'."""
    lon = np.full(n, lon0) + RNG.normal(0, 0.01, n)
    lat = np.full(n, lat0) + RNG.normal(0, 0.01, n)
    # per-particle forcing (ensemble uncertainty)
    cs = np.clip(RNG.normal(cur_speed[0], cur_speed[1], n), 0.02, None)
    cd = np.deg2rad(RNG.normal(cur_dir[0], cur_dir[1], n))
    ws = np.clip(RNG.normal(wind_speed[0], wind_speed[1], n), 0.5, None)
    wd = np.deg2rad(RNG.normal(wind_from[0], wind_from[1], n) + 180.0)  # toward
    wf = np.clip(RNG.normal(windage[0], windage[1], n), 0.002, None)
    u = cs * np.sin(cd) + wf * ws * np.sin(wd)
    v = cs * np.cos(cd) + wf * ws * np.cos(wd)
    sigma = np.sqrt(2.0 * diff_k * dt)  # random-walk step, metres
    alive = np.ones(n, bool)
    beach_t = np.full(n, np.nan)
    steps = int(hours * 3600 / dt)
    snaps = {}
    for k in range(1, steps + 1):
        dx = u * dt + RNG.normal(0, sigma, n)
        dy = v * dt + RNG.normal(0, sigma, n)
        lat = np.where(alive, lat + dy / M_PER_DEG, lat)
        lon = np.where(alive, lon + dx / (M_PER_DEG * np.cos(np.deg2rad(lat))), lon)
        hit = alive & LAND_PATH.contains_points(np.column_stack([lon, lat]))
        beach_t[hit] = k * dt / 3600.0
        alive &= ~hit
        t = k * dt / 3600.0
        for s in snapshots:
            if abs(t - s) < dt / 7200.0:
                snaps[s] = (lon.copy(), lat.copy(), alive.copy())
    return lon, lat, alive, beach_t, snaps


def area95_km2(lon, lat):
    """Area of the 95% probability ellipse of a particle cloud, km^2."""
    x = (lon - lon.mean()) * M_PER_DEG * np.cos(np.deg2rad(lat.mean())) / 1000.0
    y = (lat - lat.mean()) * M_PER_DEG / 1000.0
    cov = np.cov(np.vstack([x, y]))
    return float(np.pi * 5.991 * np.sqrt(max(np.linalg.det(cov), 0.0)))


# ---------------- Scenario A: oil release ----------------
A = dict(lon0=4.62, lat0=4.55, hours=72, n=2000,
         cur_speed=(0.35, 0.12), cur_dir=(80, 20),
         wind_speed=(5.0, 1.5), wind_from=(225, 20),
         windage=(0.03, 0.005), diff_k=30.0, snapshots=(24, 48, 72))
lonA, latA, aliveA, btA, snapA = run(**A)
beached = ~aliveA
pct_beached = 100.0 * beached.mean()
first_arrival = np.nanmin(btA) if beached.any() else float("nan")
median_arrival = np.nanmedian(btA) if beached.any() else float("nan")
on_mangrove = beached & (lonA >= MANGROVE_LON[0]) & (lonA <= MANGROVE_LON[1])
pct_mangrove = 100.0 * on_mangrove.sum() / max(beached.sum(), 1)
hit_lo, hit_hi = (np.percentile(lonA[beached], [5, 95]) if beached.any() else (np.nan, np.nan))
coast_km = (hit_hi - hit_lo) * M_PER_DEG * np.cos(np.deg2rad(4.8)) / 1000.0

# ---------------- Scenario B: person in water ----------------
B = dict(lon0=3.40, lat0=6.20, hours=24, n=2000,
         cur_speed=(0.30, 0.12), cur_dir=(85, 25),
         wind_speed=(5.0, 1.5), wind_from=(225, 20),
         windage=(0.011, 0.004), diff_k=20.0, snapshots=(6, 12, 24))
lonB, latB, aliveB, btB, snapB = run(**B)
areas = {h: area95_km2(snapB[h][0][snapB[h][2]], snapB[h][1][snapB[h][2]]) for h in (6, 12, 24)}
# naive search: circle whose radius is the fastest plausible drift (mean + 2 sd)
vmax = (0.30 + 2 * 0.12) + (0.011 + 2 * 0.004) * (5.0 + 2 * 1.5)
naive = {h: np.pi * (vmax * h * 3600 / 1000.0) ** 2 for h in (6, 12, 24)}

print("=== Scenario A: oil release, 72 h ===")
print(f"particles: {A['n']}")
print(f"reached shore within 72 h: {pct_beached:.0f}%")
print(f"first shoreline arrival: {first_arrival:.0f} h   median arrival: {median_arrival:.0f} h")
print(f"share of landings on mangrove-fringed coast: {pct_mangrove:.0f}%")
print(f"shoreline exposed (5th-95th pct): lon {hit_lo:.2f}E to {hit_hi:.2f}E  (~{coast_km:.0f} km)")
print("=== Scenario B: person in water, 24 h ===")
for h in (6, 12, 24):
    print(f"t+{h:>2} h: 95% search area {areas[h]:.0f} km2  vs naive circle {naive[h]:.0f} km2 "
          f"({100 * (1 - areas[h] / naive[h]):.0f}% smaller)")
print(f"reached shore within 24 h: {100 * (~aliveB).mean():.0f}%")

# ---------------- Figures ----------------
from matplotlib.lines import Line2D
G_DEEP, G_FOREST, G_EMER, G_PALE = "#0A2540", "#0F3B66", "#1E88E5", "#F0F7FE"
SHADES = ["#90CAF9", "#1E88E5", "#0D47A1"]


def draw(ax, which, fs=10.0, stats=True):
    ax.set_facecolor(G_PALE)
    ax.add_patch(Polygon(LAND, closed=True, facecolor="#D9E4EF", edgecolor=G_FOREST, lw=1.2, zorder=2))
    m = (COAST[:, 0] >= MANGROVE_LON[0]) & (COAST[:, 0] <= MANGROVE_LON[1])
    ax.plot(COAST[m, 0], COAST[m, 1], color="#2E7D32", lw=5, alpha=0.9, zorder=3, solid_capstyle="round")
    ax.grid(color="white", lw=0.8)
    ax.set_xlabel("Longitude (°E)", fontsize=fs); ax.set_ylabel("Latitude (°N)", fontsize=fs)
    ax.tick_params(labelsize=fs)
    for sp in ax.spines.values():
        sp.set_color(G_FOREST)
    if which == "A":
        cfg, snaps, hrs, start = A, snapA, (24, 48, 72), "Release point"
        ax.set_xlim(3.9, 7.2); ax.set_ylim(3.6, 6.2)
        ax.text(6.0, 5.55, "NIGERIA", color=G_FOREST, fontsize=fs + 2, fontweight="bold", zorder=5)
        title = "A. Oil release 120 km offshore: 72-hour drift forecast"
        note = (f"{pct_beached:.0f}% of the ensemble reaches shore within 72 h\n"
                f"First arrival {first_arrival:.0f} h, median {median_arrival:.0f} h\n"
                f"{pct_mangrove:.0f}% of landings hit mangrove coast (~{coast_km:.0f} km exposed)")
        nx, ny = 3.97, 3.68
    else:
        cfg, snaps, hrs, start = B, snapB, (6, 12, 24), "Last known position"
        ax.set_xlim(3.1, 4.3); ax.set_ylim(5.85, 6.6)
        ax.text(3.2, 6.5, "LAGOS", color=G_FOREST, fontsize=fs + 2, fontweight="bold", zorder=5)
        title = "B. Person in water off Lagos: 24-hour search area"
        note = "95% search area vs naive circle\n" + "\n".join(
            f"t + {h} h: {areas[h]:,.0f} km² vs {naive[h]:,.0f} km²  ({100 * (1 - areas[h] / naive[h]):.0f}% smaller)" for h in hrs)
        nx, ny = 3.13, 5.87
    for h, c in zip(hrs, SHADES):
        lo, la, al = snaps[h]
        keep = al if which == "B" else np.ones_like(al)
        ax.scatter(lo[keep], la[keep], s=3, color=c, alpha=0.55, zorder=4)
    ax.scatter([cfg["lon0"]], [cfg["lat0"]], marker="*", s=300, color="#F57C00", edgecolor=G_DEEP, zorder=6)
    ax.set_title(title, color=G_DEEP, fontsize=fs + 2, fontweight="bold", loc="left")
    if stats:
        ax.text(nx, ny, note, fontsize=fs - 0.5, color=G_DEEP, va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=G_EMER, lw=1.2), zorder=7)
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=f"t + {h} h") for h, c in zip(hrs, SHADES)]
    handles += [Line2D([], [], marker="*", ls="", color="#F57C00", mec=G_DEEP, ms=13, label=start),
                Line2D([], [], color="#2E7D32", lw=5, label="Mangrove-fringed coast (schematic)")]
    ax.legend(handles=handles, loc="upper right" if which == "B" else "upper left", fontsize=fs - 1.5, framealpha=0.95)


fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=160)
draw(axes[0], "A"); draw(axes[1], "B")
fig.suptitle("Hydrocline Drift Engine v0.1: proof-of-concept ensemble forecast (2,000 particles per scenario)",
             color=G_DEEP, fontsize=13.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.012,
         "Prototype run with climatological, illustrative forcing (Guinea Current, south-westerly monsoon wind) and a simplified coastline. "
         "Not yet validated against observed drift data.", ha="center", fontsize=8.5, color="#374151")
fig.tight_layout(rect=(0, 0.03, 1, 0.955))
fig.savefig("hydrocline_drift_forecast.png", facecolor="white")
for w in ("A", "B"):
    f2, a2 = plt.subplots(figsize=(8, 6.2), dpi=150)
    draw(a2, w, fs=13.0, stats=False)
    f2.tight_layout()
    f2.savefig(f"hydrocline_panel_{w.lower()}.png", facecolor="white")
print("saved figures")
