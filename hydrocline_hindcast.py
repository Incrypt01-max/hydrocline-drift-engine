"""
Hydrocline Hindcast Validator (v0.1)
------------------------------------
Replays a real drift event with REAL ocean currents and winds and scores the forecast
against the OBSERVED track (a drifter buoy, a GPS-tagged object, or a slick centroid).

Inputs you download
  1. Currents  : Copernicus Marine (CMEMS) NetCDF with eastward/northward surface velocity
                 (variables uo, vo; coordinates time, latitude, longitude, optional depth).
  2. Winds     : ERA5 NetCDF with 10 m wind (variables u10, v10; time or valid_time).
  3. Track     : CSV with columns  time,lat,lon   (UTC, ISO format). First row = start point.

Run
  python hydrocline_hindcast.py --currents cur.nc --winds wind.nc --track track.csv
  python hydrocline_hindcast.py --selftest        # checks the pipeline on synthetic data

Outputs: hindcast_scores.csv and hindcast_plot.png

Scores (per observation time)
  * separation  : distance from the ensemble centre to the observed position (km)
  * inside95    : is the observed position inside the 95% search ellipse?
  * area95      : 95% ellipse area (km2)
  * skill       : Liu-Weisberg skill score, 1 - sum(separation) / sum(observed path length).
                  1 is perfect, 0 is no better than assuming the object never moved.

Honest notes
  * Tune only windage and diffusion, within physical ranges, and report the values used.
  * Never tune per event until it fits. Fix the settings, then run many events.
  * Euler time stepping, 1 h step. Near-coast CMEMS currents are coarse.
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd, xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH = 6371.0
M_PER_DEG = 111_320.0

def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1; dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def pick(ds, names):
    for n in names:
        if n in ds.variables or n in ds.coords or n in ds.dims:
            return n
    raise KeyError(f"None of {names} found in dataset: {list(ds.variables)}")

def load_field(path, uname, vname, t0):
    """Return a function f(t_hours, lon, lat) -> (u, v) in m/s, plus the time span in hours."""
    ds = xr.open_dataset(path)
    tn = pick(ds, ["time", "valid_time"]); la = pick(ds, ["latitude", "lat"]); lo = pick(ds, ["longitude", "lon"])
    u, v = ds[pick(ds, uname)], ds[pick(ds, vname)]
    for dim in list(u.dims):
        if dim not in (tn, la, lo):  # depth or ensemble dims: take the first (surface) level
            u, v = u.isel({dim: 0}), v.isel({dim: 0})
    u, v = u.transpose(tn, la, lo), v.transpose(tn, la, lo)
    lat, lon = u[la].values, u[lo].values
    if lat[0] > lat[-1]:
        u, v, lat = u.isel({la: slice(None, None, -1)}), v.isel({la: slice(None, None, -1)}), lat[::-1]
    if lon[0] > lon[-1]:
        u, v, lon = u.isel({lo: slice(None, None, -1)}), v.isel({lo: slice(None, None, -1)}), lon[::-1]
    th = ((pd.to_datetime(u[tn].values) - t0) / pd.Timedelta(hours=1)).values.astype(float)
    fu = RegularGridInterpolator((th, lat, lon), u.values, bounds_error=False, fill_value=np.nan)
    fv = RegularGridInterpolator((th, lat, lon), v.values, bounds_error=False, fill_value=np.nan)
    def f(t, lon_p, lat_p):
        pts = np.column_stack([np.full_like(lon_p, t), lat_p, lon_p])
        return fu(pts), fv(pts)
    return f, (th.min(), th.max())

def ellipse95(lon, lat):
    x = (lon - lon.mean()) * M_PER_DEG * np.cos(np.deg2rad(lat.mean())) / 1000.0
    y = (lat - lat.mean()) * M_PER_DEG / 1000.0
    cov = np.cov(np.vstack([x, y]))
    return cov, float(np.pi * 5.991 * np.sqrt(max(np.linalg.det(cov), 0.0)))

def inside95(lon, lat, olon, olat):
    cov, _ = ellipse95(lon, lat)
    d = np.array([(olon - lon.mean()) * M_PER_DEG * np.cos(np.deg2rad(lat.mean())) / 1000.0,
                  (olat - lat.mean()) * M_PER_DEG / 1000.0])
    try:
        return bool(d @ np.linalg.inv(cov) @ d <= 5.991)
    except np.linalg.LinAlgError:
        return False

def simulate(cur, wind, lon0, lat0, hours, n, windage, windage_sd, diff_k, seed=1):
    rng = np.random.default_rng(seed)
    lon = np.full(n, lon0) + rng.normal(0, 0.005, n); lat = np.full(n, lat0) + rng.normal(0, 0.005, n)
    wf = np.clip(rng.normal(windage, windage_sd, n), 0.0, None)
    dt = 3600.0; sigma = np.sqrt(2.0 * diff_k * dt)
    traj = [(lon.copy(), lat.copy())]
    for h in range(int(hours)):
        uc, vc = cur(h, lon, lat); uw, vw = wind(h, lon, lat)
        ok = np.isfinite(uc) & np.isfinite(vc)  # NaN = land or outside the data: freeze particle
        uw = np.where(np.isfinite(uw), uw, 0.0); vw = np.where(np.isfinite(vw), vw, 0.0)
        dx = (np.where(ok, uc, 0) + wf * uw) * dt + rng.normal(0, sigma, n) * ok
        dy = (np.where(ok, vc, 0) + wf * vw) * dt + rng.normal(0, sigma, n) * ok
        lat = lat + np.where(ok, dy, 0) / M_PER_DEG
        lon = lon + np.where(ok, dx, 0) / (M_PER_DEG * np.cos(np.deg2rad(lat)))
        traj.append((lon.copy(), lat.copy()))
    return traj

def score(traj, track, t0):
    hrs = ((track["time"] - t0) / pd.Timedelta(hours=1)).values
    rows, path_len = [], 0.0
    for i in range(1, len(track)):
        path_len_i = haversine_km(track.lon[0], track.lat[0], track.lon[i], track.lat[i])
        h = int(round(hrs[i])); h = min(h, len(traj) - 1)
        lo, la = traj[h]
        sep = float(haversine_km(lo.mean(), la.mean(), track.lon[i], track.lat[i]))
        _, area = ellipse95(lo, la)
        rows.append(dict(hours=hrs[i], observed_lon=track.lon[i], observed_lat=track.lat[i],
                         ensemble_lon=lo.mean(), ensemble_lat=la.mean(), separation_km=round(sep, 2),
                         inside95=inside95(lo, la, track.lon[i], track.lat[i]), area95_km2=round(area, 1),
                         observed_path_km=round(float(path_len_i), 2)))
    df = pd.DataFrame(rows)
    skill = 1.0 - df.separation_km.sum() / max(df.observed_path_km.sum(), 1e-9)
    return df, max(0.0, skill)

def plot(traj, track, df, out):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=140)
    lo, la = traj[-1]
    ax.scatter(lo, la, s=4, color="#90CAF9", alpha=0.5, label="Final ensemble")
    ax.plot(df.ensemble_lon, df.ensemble_lat, "-o", color="#0F3B66", ms=4, label="Forecast centre")
    ax.plot(track.lon, track.lat, "-s", color="#F57C00", ms=5, label="Observed track")
    ax.scatter([track.lon[0]], [track.lat[0]], marker="*", s=250, color="#F57C00", edgecolor="#0A2540", zorder=5, label="Start")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Hydrocline hindcast: forecast versus observed", color="#0A2540", fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def run(cur_path, wind_path, track_path, args, outdir="."):
    track = pd.read_csv(track_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    t0 = track["time"][0]; hours = (track["time"].iloc[-1] - t0) / pd.Timedelta(hours=1)
    cur, span = load_field(cur_path, ["uo", "eastward_sea_water_velocity"], ["vo", "northward_sea_water_velocity"], t0)
    wind, _ = load_field(wind_path, ["u10"], ["v10"], t0)
    traj = simulate(cur, wind, track.lon[0], track.lat[0], hours, args.n, args.windage, args.windage_sd, args.diff_k)
    df, skill = score(traj, track, t0)
    df.to_csv(os.path.join(outdir, "hindcast_scores.csv"), index=False)
    plot(traj, track, df, os.path.join(outdir, "hindcast_plot.png"))
    print(df.to_string(index=False))
    print(f"\nObserved inside 95% ellipse: {int(df.inside95.sum())} of {len(df)}")
    print(f"Mean separation: {df.separation_km.mean():.1f} km | Liu-Weisberg skill: {skill:.2f} (1 perfect, 0 no better than no drift)")
    print(f"Settings used: windage {args.windage} (sd {args.windage_sd}), diffusion {args.diff_k} m2/s, {args.n} particles")
    return df, skill

def selftest(args):
    """Synthetic fields + a 'truth' drifter with different settings. Proves the pipeline end to end."""
    d = tempfile.mkdtemp(); t = pd.date_range("2026-01-01", periods=73, freq="h")
    lat = np.linspace(5.0, 7.0, 21); lon = np.linspace(2.5, 4.5, 21)
    T, LA, LO = np.meshgrid(np.arange(73), lat, lon, indexing="ij")
    uo = 0.30 + 0.05 * np.sin(LO) + 0.0 * T; vo = 0.05 + 0.03 * np.cos(LA) + 0.0 * T
    u10 = 4.0 + 0 * T; v10 = 2.0 + 0 * T
    xr.Dataset({"uo": (("time", "latitude", "longitude"), uo), "vo": (("time", "latitude", "longitude"), vo)},
               coords=dict(time=t, latitude=lat, longitude=lon)).to_netcdf(f"{d}/cur.nc")
    xr.Dataset({"u10": (("time", "latitude", "longitude"), u10), "v10": (("time", "latitude", "longitude"), v10)},
               coords=dict(time=t, latitude=lat, longitude=lon)).to_netcdf(f"{d}/wind.nc")
    cur, _ = load_field(f"{d}/cur.nc", ["uo"], ["vo"], t[0]); wind, _ = load_field(f"{d}/wind.nc", ["u10"], ["v10"], t[0])
    truth = simulate(cur, wind, 3.4, 6.0, 48, 1, 0.015, 0.0, 5.0, seed=99)  # one 'real' drifter
    rows = [dict(time=t[h], lat=truth[h][1][0], lon=truth[h][0][0]) for h in range(0, 49, 6)]
    pd.DataFrame(rows).to_csv(f"{d}/track.csv", index=False)
    args.windage, args.windage_sd = 0.012, 0.004  # model does not know the true windage
    print("SELF-TEST on synthetic data (not a real validation)\n")
    return run(f"{d}/cur.nc", f"{d}/wind.nc", f"{d}/track.csv", args)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hydrocline hindcast validator")
    ap.add_argument("--currents"); ap.add_argument("--winds"); ap.add_argument("--track")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--windage", type=float, default=0.011)
    ap.add_argument("--windage_sd", type=float, default=0.004); ap.add_argument("--diff_k", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(a)
    elif a.currents and a.winds and a.track: run(a.currents, a.winds, a.track, a)
    else: ap.error("give --currents, --winds and --track, or use --selftest")
