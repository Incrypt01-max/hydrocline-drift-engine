"""
Hydrocline Engine v0.2
======================
Core library for Hydrocline, a digital twin of Nigeria's ocean.

Critical components (each one proven in the lab, see verify_physics.py and run_demo_v02.py)
  1. FORCING INGESTION  read_forcing(): Copernicus Marine (CMEMS) and ERA5 NetCDF conventions,
                        time-varying gridded fields, interpolated in space and time.
  2. DRIFT ENGINE       run_ensemble(): Monte Carlo Lagrangian ensemble, current + windage +
                        turbulent diffusion, 2nd-order (midpoint) time stepping.
  3. REAL COASTLINE     Coast: Natural Earth 10 m land and coastline (public domain),
                        beaching detected on real geometry.
  4. ECOSYSTEM LAYER    habitat_exposure(): classifies where particles land by coastal unit and
                        sensitivity, and the length of coastline exposed.

Honest scope: lab proof of concept (TRL 3). Not yet validated against observed incidents.
"""
import json
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape, box
from scipy.interpolate import RegularGridInterpolator

M_PER_DEG = 111_320.0
R_EARTH_KM = 6371.0


# ---------------------------------------------------------------- 1. forcing ingestion
def _pick(ds, names):
    for n in names:
        if n in ds.variables or n in ds.coords or n in ds.dims:
            return n
    raise KeyError(f"none of {names} in {list(ds.variables)}")


def read_forcing(path, kind, t0):
    """Read CMEMS currents (kind='current') or ERA5 10 m wind (kind='wind') from NetCDF.

    Handles: time or valid_time; latitude/lat; longitude/lon; extra dims such as depth
    (surface level taken); descending latitude (ERA5 default). Returns f(t_hours, lon, lat) -> (u, v).
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    un = ["uo", "eastward_sea_water_velocity"] if kind == "current" else ["u10"]
    vn = ["vo", "northward_sea_water_velocity"] if kind == "current" else ["v10"]
    tn, la, lo = _pick(ds, ["time", "valid_time"]), _pick(ds, ["latitude", "lat"]), _pick(ds, ["longitude", "lon"])
    u, v = ds[_pick(ds, un)], ds[_pick(ds, vn)]
    for d in list(u.dims):
        if d not in (tn, la, lo):
            u, v = u.isel({d: 0}), v.isel({d: 0})
    u, v = u.transpose(tn, la, lo), v.transpose(tn, la, lo)
    if u[la].values[0] > u[la].values[-1]:
        u, v = u.isel({la: slice(None, None, -1)}), v.isel({la: slice(None, None, -1)})
    if u[lo].values[0] > u[lo].values[-1]:
        u, v = u.isel({lo: slice(None, None, -1)}), v.isel({lo: slice(None, None, -1)})
    th = ((pd.to_datetime(u[tn].values) - pd.Timestamp(t0)) / pd.Timedelta(hours=1)).values.astype(float)
    grid = (th, u[la].values, u[lo].values)
    fu = RegularGridInterpolator(grid, np.nan_to_num(u.values), bounds_error=False, fill_value=None)
    fv = RegularGridInterpolator(grid, np.nan_to_num(v.values), bounds_error=False, fill_value=None)

    def f(t, lon, lat):
        p = np.column_stack([np.full(np.shape(lon), t), lat, lon])
        return fu(p), fv(p)
    return f


def constant_field(u, v):
    return lambda t, lon, lat: (np.full(np.shape(lon), u, float), np.full(np.shape(lon), v, float))


# ---------------------------------------------------------------- 3. real coastline
class Coast:
    def __init__(self, land_geojson, coastline_geojson=None):
        g = json.load(open(land_geojson))
        self.land = shapely.union_all([shape(f["geometry"]) for f in g["features"]])
        shapely.prepare(self.land)
        self.lines = None
        if coastline_geojson:
            c = json.load(open(coastline_geojson))
            self.lines = shapely.union_all([shape(f["geometry"]) for f in c["features"]])

    def is_land(self, lon, lat):
        return shapely.contains_xy(self.land, lon, lat)

    def coast_lat(self, lon):
        """Southernmost shoreline latitude crossing a meridian (for tests)."""
        seg = shapely.LineString([(lon, 3.0), (lon, 8.0)]).intersection(self.land)
        ys = [c[1] for g in getattr(seg, "geoms", [seg]) for c in g.coords]
        return min(ys)

    def length_km(self, lon_min, lon_max):
        """Real coastline length (km) between two meridians."""
        if self.lines is None:
            return np.nan
        part = self.lines.intersection(box(lon_min, 4.0, lon_max, 8.0))  # lat >= 4 N excludes Bioko island
        total = 0.0
        for g in getattr(part, "geoms", [part]):
            xy = np.asarray(g.coords)
            if len(xy) < 2:
                continue
            dx = np.diff(xy[:, 0]) * np.cos(np.deg2rad(xy[1:, 1]))
            dy = np.diff(xy[:, 1])
            total += np.sum(np.hypot(dx, dy)) * M_PER_DEG / 1000.0
        return total


# ---------------------------------------------------------------- 2. drift engine
def run_ensemble(lon0, lat0, hours, n, current, wind, coast=None, windage=(0.03, 0.0),
                 cur_scale_sd=0.0, cur_rot_sd_deg=0.0, diff_k=0.0, dt=600.0, seed=0,
                 release_sd_deg=0.0, snapshots=()):
    """Advance an ensemble of n particles.

    current, wind: f(t_hours, lon, lat) -> (u, v) in m/s.
    Ensemble uncertainty: per-particle current scale ~N(1, cur_scale_sd), rotation ~N(0, cur_rot_sd_deg),
    windage ~N(windage[0], windage[1]); random walk with diffusivity diff_k (m2/s).
    Midpoint (RK2) advection. Particles that enter land stop (beached).
    """
    rng = np.random.default_rng(seed)
    lon = lon0 + rng.normal(0, release_sd_deg, n) if release_sd_deg else np.full(n, float(lon0))
    lat = lat0 + rng.normal(0, release_sd_deg, n) if release_sd_deg else np.full(n, float(lat0))
    scale = np.clip(rng.normal(1.0, cur_scale_sd, n), 0.05, None) if cur_scale_sd else np.ones(n)
    rot = np.deg2rad(rng.normal(0.0, cur_rot_sd_deg, n)) if cur_rot_sd_deg else np.zeros(n)
    wf = np.clip(rng.normal(windage[0], windage[1], n), 0.0, None) if windage[1] else np.full(n, windage[0])
    cr, sr = np.cos(rot), np.sin(rot)

    def vel(t, x, y):
        uc, vc = current(t, x, y)
        uw, vw = wind(t, x, y)
        uc, vc = scale * (uc * cr - vc * sr), scale * (uc * sr + vc * cr)
        return uc + wf * uw, vc + wf * vw

    def move(x, y, u, v, h):
        y2 = y + v * h / M_PER_DEG
        x2 = x + u * h / (M_PER_DEG * np.cos(np.deg2rad(0.5 * (y + y2))))
        return x2, y2

    sigma = np.sqrt(2.0 * diff_k * dt)
    alive = np.ones(n, bool)
    beach_t = np.full(n, np.nan)
    steps = int(round(hours * 3600.0 / dt))
    snaps = {0.0: (lon.copy(), lat.copy(), alive.copy())}
    for k in range(steps):
        t = k * dt / 3600.0
        u1, v1 = vel(t, lon, lat)
        xm, ym = move(lon, lat, u1, v1, 0.5 * dt)
        u2, v2 = vel(t + 0.5 * dt / 3600.0, xm, ym)
        nx, ny = move(lon, lat, u2, v2, dt)
        if diff_k:
            ny = ny + rng.normal(0, sigma, n) / M_PER_DEG
            nx = nx + rng.normal(0, sigma, n) / (M_PER_DEG * np.cos(np.deg2rad(ny)))
        lon = np.where(alive, nx, lon)
        lat = np.where(alive, ny, lat)
        if coast is not None:
            hit = alive & coast.is_land(lon, lat)
            beach_t[hit] = (k + 1) * dt / 3600.0
            alive &= ~hit
        tn = (k + 1) * dt / 3600.0
        for s in snapshots:
            if abs(tn - s) < 0.5 * dt / 3600.0:
                snaps[float(s)] = (lon.copy(), lat.copy(), alive.copy())
    return dict(lon=lon, lat=lat, alive=alive, beach_t=beach_t, snaps=snaps)


def area95_km2(lon, lat):
    x = (lon - lon.mean()) * M_PER_DEG * np.cos(np.deg2rad(lat.mean())) / 1000.0
    y = (lat - lat.mean()) * M_PER_DEG / 1000.0
    return float(np.pi * 5.991 * np.sqrt(max(np.linalg.det(np.cov(np.vstack([x, y]))), 0.0)))


# ---------------------------------------------------------------- 4. ecosystem layer
# Nigeria's four coastal geomorphic units (after Ibe, 1988). Boundaries are approximate meridians;
# to be replaced with Global Mangrove Watch habitat polygons in the next version.
COASTAL_UNITS = [
    ("Barrier-lagoon coast (Badagry to Lekki)", 2.70, 4.40, "Moderate: sandy beach, lagoons behind"),
    ("Mahin mud coast (Ondo)", 4.40, 5.05, "High: muddy shore, fish nurseries"),
    ("Arcuate Niger Delta", 5.05, 7.30, "Very high: mangrove forest"),
    ("Strand coast (Imo River to Cross River)", 7.30, 8.70, "Very high: mangrove estuaries"),
]


def habitat_exposure(res, coast, bin_deg=0.05):
    """Where beached particles land: share per coastal unit and km of real coastline touched."""
    b = ~res["alive"]
    lon_b = res["lon"][b]
    rows = []
    for name, lo, hi, sens in COASTAL_UNITS:
        m = (lon_b >= lo) & (lon_b < hi)
        bins = np.unique(np.floor(lon_b[m] / bin_deg))
        # shore stretch hit, measured west to east (conservative: ignores creek and estuary shorelines)
        km = len(bins) * bin_deg * M_PER_DEG / 1000.0 * np.cos(np.deg2rad(np.mean(res["lat"][b][m]))) if len(bins) else 0.0
        rows.append(dict(unit=name, sensitivity=sens, share_of_landings_pct=round(100 * m.sum() / max(b.sum(), 1), 1),
                         shore_stretch_km=round(km)))
    return pd.DataFrame(rows)
