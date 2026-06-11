"""Spatial interpolation of per-city warming trends (Phase 3).

Two interpolators over :data:`src.trends.DEFAULT_TRENDS_PATH` (one row per
city-location, in ``slope_c_per_decade``):

- :func:`idw_interpolate` -- inverse-distance weighting, implemented from
  scratch on great-circle distances.
- :func:`ordinary_kriging_interpolate` -- thin wrapper around
  ``pykrige.ok.OrdinaryKriging`` with ``coordinates_type="geographic"``.

Both exactly reproduce the input value at a query point that coincides with
a known location (see the "exact recovery" tests).

**Why k-nearest-neighbor (local) kriging.** A *global* ordinary-kriging fit
over all ~3,500 city-locations is numerically ill-conditioned (pykrige warns
of an (n+1)x(n+1) matrix with rcond ~1e-18) and an O(n^3) refit per
leave-one-out fold is computationally infeasible. Instead, the variogram is
fit once globally (:func:`fit_variogram_parameters`) and reused for small,
well-conditioned local fits over each point's ``k`` nearest neighbors -- a
standard "moving neighborhood" kriging setup. :func:`idw_interpolate` accepts
the same ``k`` so :func:`leave_one_out_cv` compares both methods on identical
neighborhoods.

Rendering masks the interpolated grid to land using Natural Earth's 110m
land polygons (downloaded once into ``data/raw/natural_earth/``, as these are
not part of the project's two Kaggle datasets).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from pykrige.ok import OrdinaryKriging
from scipy.linalg import LinAlgWarning
from scipy.spatial import cKDTree
from shapely import Geometry, contains_xy

from src.data_io import PROJECT_ROOT, RAW_DIR
from src.trends import DEFAULT_TRENDS_PATH

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EARTH_RADIUS_KM = 6371.0088
DEFAULT_VALUE_COL = "slope_c_per_decade"
DEFAULT_K_NEIGHBORS = 30
DEFAULT_VARIOGRAM_MODEL = "spherical"
DEFAULT_GRID_RESOLUTION = 2.0  # degrees

# Natural Earth 110m land polygons (~70 KB) -- not one of the project's two
# Kaggle datasets, so cached separately under data/raw/.
LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
LAND_ZIP_PATH = RAW_DIR / "natural_earth" / "ne_110m_land.zip"


def haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km between paired (lon, lat) points.

    Inputs follow numpy broadcasting, so passing column and row vectors
    produces a full pairwise distance matrix.

    Args:
        lon1, lat1: Longitude/latitude of the first point(s), in degrees.
        lon2, lat2: Longitude/latitude of the second point(s), in degrees.

    Returns:
        Distance(s) in km, broadcast to the common shape of the inputs.
    """
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(a, dtype=float)) for a in (lon1, lat1, lon2, lat2)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_unit_sphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(lon, lat) in degrees -> (n, 3) Cartesian points on the unit sphere.

    Nearest neighbors by Euclidean (chordal) distance in this space match
    nearest neighbors by great-circle distance, so a plain
    :class:`scipy.spatial.cKDTree` can be used for spherical KNN queries.
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def _knn_indices(
    known_lon: np.ndarray,
    known_lat: np.ndarray,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
    k: int,
) -> np.ndarray:
    """Indices into `known_*` of the k nearest points to each query point.

    Returns:
        Array of shape (len(query_lon), k).
    """
    k = min(k, len(known_lon))
    tree = cKDTree(_to_unit_sphere(known_lon, known_lat))
    _, idx = tree.query(_to_unit_sphere(query_lon, query_lat), k=k)
    return idx.reshape(len(query_lon), -1)


def idw_interpolate(
    known_lon: np.ndarray,
    known_lat: np.ndarray,
    known_value: np.ndarray,
    query_lon: np.ndarray | float,
    query_lat: np.ndarray | float,
    power: float = 2.0,
    k: int | None = None,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation on great-circle distances.

    weight_i = 1 / distance_i ** power. A query point that exactly
    coincides with one or more known points returns the (mean of the)
    known value(s) at that location rather than dividing by zero.

    Args:
        known_lon, known_lat, known_value: 1D arrays of equal length --
            the sample locations (degrees) and values.
        query_lon, query_lat: Scalar or 1D array of query locations.
        power: Distance exponent.
        k: If given, use only the k nearest known points per query
            (by great-circle distance); otherwise use all points.

    Returns:
        1D array of interpolated values, one per query point.
    """
    known_lon = np.asarray(known_lon, dtype=float)
    known_lat = np.asarray(known_lat, dtype=float)
    known_value = np.asarray(known_value, dtype=float)
    query_lon = np.atleast_1d(np.asarray(query_lon, dtype=float))
    query_lat = np.atleast_1d(np.asarray(query_lat, dtype=float))

    n_known = known_lon.shape[0]
    if k is None or k >= n_known:
        dist = haversine_km(
            query_lon[:, None], query_lat[:, None], known_lon[None, :], known_lat[None, :]
        )
        values = np.broadcast_to(known_value[None, :], dist.shape)
    else:
        idx = _knn_indices(known_lon, known_lat, query_lon, query_lat, k)
        dist = haversine_km(
            query_lon[:, None], query_lat[:, None], known_lon[idx], known_lat[idx]
        )
        values = known_value[idx]

    with np.errstate(divide="ignore"):
        weights = 1.0 / dist**power
    exact = ~np.isfinite(weights)
    has_exact = exact.any(axis=1)
    weights[exact] = 1.0
    weights[has_exact[:, None] & ~exact] = 0.0
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


def fit_variogram_parameters(
    lon: np.ndarray,
    lat: np.ndarray,
    value: np.ndarray,
    variogram_model: str = DEFAULT_VARIOGRAM_MODEL,
) -> list[float]:
    """Fit a variogram model once on the full point set, for reuse.

    Local (k-nearest-neighbor) kriging fits pass the result back in via
    `variogram_parameters` to skip refitting -- see module docstring.

    Args:
        lon, lat, value: 1D arrays of equal length.
        variogram_model: pykrige variogram model name.

    Returns:
        Fitted parameters, in pykrige's order for `variogram_model`.
    """
    with warnings.catch_warnings():
        # Expected for ~3,500 points: the (n+1)x(n+1) kriging matrix is
        # ill-conditioned, but only the fitted variogram parameters are used.
        warnings.simplefilter("ignore", category=LinAlgWarning)
        ok = OrdinaryKriging(
            np.asarray(lon, dtype=float),
            np.asarray(lat, dtype=float),
            np.asarray(value, dtype=float),
            variogram_model=variogram_model,
            coordinates_type="geographic",
            pseudo_inv=True,
        )
    return [float(p) for p in ok.variogram_model_parameters]


def ordinary_kriging_interpolate(
    known_lon: np.ndarray,
    known_lat: np.ndarray,
    known_value: np.ndarray,
    query_lon: np.ndarray | float,
    query_lat: np.ndarray | float,
    variogram_model: str = DEFAULT_VARIOGRAM_MODEL,
    variogram_parameters: list[float] | None = None,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging on geographic (lon/lat) coordinates.

    With `k` set, fits a separate small kriging system per query point over
    its k nearest known points (see module docstring); without it, fits one
    global system and evaluates all query points against it.

    Args:
        known_lon, known_lat, known_value: 1D arrays of equal length --
            the sample locations (degrees) and values.
        query_lon, query_lat: Scalar or 1D array of query locations.
        variogram_model: pykrige variogram model name.
        variogram_parameters: Pre-fit parameters (see
            :func:`fit_variogram_parameters`); if None, fit per system.
        k: If given, use local kriging over the k nearest known points per
            query point; otherwise fit one global system.

    Returns:
        (predicted_value, predicted_variance), each a 1D array with one
        entry per query point.
    """
    known_lon = np.asarray(known_lon, dtype=float)
    known_lat = np.asarray(known_lat, dtype=float)
    known_value = np.asarray(known_value, dtype=float)
    query_lon = np.atleast_1d(np.asarray(query_lon, dtype=float))
    query_lat = np.atleast_1d(np.asarray(query_lat, dtype=float))
    n_known = known_lon.shape[0]

    if k is None or k >= n_known:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=LinAlgWarning)
            ok = OrdinaryKriging(
                known_lon,
                known_lat,
                known_value,
                variogram_model=variogram_model,
                variogram_parameters=variogram_parameters,
                coordinates_type="geographic",
                pseudo_inv=True,
            )
            z, var = ok.execute("points", query_lon, query_lat)
        return np.asarray(z, dtype=float), np.asarray(var, dtype=float)

    if variogram_parameters is None:
        variogram_parameters = fit_variogram_parameters(
            known_lon, known_lat, known_value, variogram_model=variogram_model
        )

    idx = _knn_indices(known_lon, known_lat, query_lon, query_lat, k)
    z = np.empty(len(query_lon))
    var = np.empty(len(query_lon))
    for i, neighbors in enumerate(idx):
        ok = OrdinaryKriging(
            known_lon[neighbors],
            known_lat[neighbors],
            known_value[neighbors],
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            coordinates_type="geographic",
            pseudo_inv=True,
        )
        zi, vi = ok.execute("points", query_lon[i : i + 1], query_lat[i : i + 1])
        z[i] = zi[0]
        var[i] = vi[0]
    return z, var


def leave_one_out_cv(
    lon: np.ndarray,
    lat: np.ndarray,
    value: np.ndarray,
    k: int = DEFAULT_K_NEIGHBORS,
    idw_power: float = 2.0,
    variogram_model: str = DEFAULT_VARIOGRAM_MODEL,
    exclude_same_coordinate: bool = True,
) -> pd.DataFrame:
    """Leave-one-out CV RMSE/MAE for IDW vs. ordinary kriging.

    Each point is predicted from its k nearest *other* points, using
    :func:`idw_interpolate` and :func:`ordinary_kriging_interpolate` with a
    single globally-fit variogram (see module docstring for why).

    By default each fold is leave-*location*-out: Berkeley Earth city
    coordinates are grid-snapped, so 677 coordinate groups (2,821 of the
    3,510 locations) hold 2+ cities with bit-identical coordinates and
    near-identical trend values. With plain leave-one-row-out, a held-out
    point's same-coordinate twin stays in the neighborhood and leaks the
    answer -- IDW's exact-match handling then reproduces it perfectly
    while kriging's fitted nugget does not, which is a property of the
    duplicates, not of the interpolators. Excluding the whole coordinate
    group per fold removes that leak; set `exclude_same_coordinate=False`
    for the classic per-row variant.

    Args:
        lon, lat, value: 1D arrays of equal length (one per location).
        k: Neighborhood size for both methods. With exclusion enabled, k is
            clamped to n minus the largest coordinate-group size so every
            fold still gets a full neighborhood.
        idw_power: IDW distance exponent.
        variogram_model: pykrige variogram model name.
        exclude_same_coordinate: Exclude all points at the held-out point's
            exact coordinates (not just the point itself) from its fold.

    Returns:
        DataFrame with one row per method ("idw", "kriging") and columns
        `rmse`, `mae`, `n`.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    value = np.asarray(value, dtype=float)
    n = len(lon)

    _, group_id, group_sizes = np.unique(
        np.column_stack([lon, lat]), axis=0, return_inverse=True, return_counts=True
    )
    group_id = group_id.reshape(-1)
    max_excluded = int(group_sizes.max()) if exclude_same_coordinate else 1
    k_eff = min(k, n - max_excluded)
    if k_eff < 1:
        raise ValueError(
            f"no neighbors left to predict from: n={n}, "
            f"largest coordinate group={max_excluded}"
        )

    neighbor_idx = _knn_indices(lon, lat, lon, lat, k_eff + max_excluded)
    variogram_parameters = fit_variogram_parameters(
        lon, lat, value, variogram_model=variogram_model
    )

    idw_pred = np.empty(n)
    ok_pred = np.empty(n)
    for i in range(n):
        candidates = neighbor_idx[i]
        if exclude_same_coordinate:
            kept = candidates[group_id[candidates] != group_id[i]]
        else:
            kept = candidates[candidates != i]
        neighbors = kept[:k_eff]
        idw_pred[i] = idw_interpolate(
            lon[neighbors], lat[neighbors], value[neighbors], lon[i], lat[i], power=idw_power
        )[0]
        z, _ = ordinary_kriging_interpolate(
            lon[neighbors],
            lat[neighbors],
            value[neighbors],
            lon[i],
            lat[i],
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
        )
        ok_pred[i] = z[0]

    def _rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - value) ** 2)))

    def _mae(pred: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - value)))

    return pd.DataFrame(
        {
            "method": ["idw", "kriging"],
            "rmse": [_rmse(idw_pred), _rmse(ok_pred)],
            "mae": [_mae(idw_pred), _mae(ok_pred)],
            "n": [n, n],
        }
    )


def build_grid(
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    lat_min: float = -60.0,
    lat_max: float = 85.0,
    resolution: float = DEFAULT_GRID_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """1D axes of grid-cell centers spaced `resolution` degrees apart.

    Defaults span all longitudes and exclude Antarctica (no Berkeley Earth
    city stations south of about -60 degrees), matching the city-location
    extent used to fit the surfaces.

    Returns:
        (lons, lats): 1D arrays of cell-center coordinates in degrees.
    """
    lons = np.arange(lon_min + resolution / 2.0, lon_max, resolution)
    lats = np.arange(lat_min + resolution / 2.0, lat_max, resolution)
    return lons, lats


def download_land_polygons(dest: Path = LAND_ZIP_PATH, url: str = LAND_URL) -> Path:
    """Download Natural Earth's 110m land polygons (~70 KB) if not cached.

    Args:
        dest: Destination zip path; created (with parents) if missing.
        url: Source URL.

    Returns:
        `dest`.
    """
    if dest.exists():
        logger.info("up to date: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("downloaded %s (%d bytes)", dest.name, len(response.content))
    return dest


def load_land_geometry(zip_path: Path = LAND_ZIP_PATH) -> Geometry:
    """Load Natural Earth land polygons as a single (multi)polygon.

    Raises:
        FileNotFoundError: if `zip_path` is missing -- run
            :func:`download_land_polygons` first.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"no such file: {zip_path}; run download_land_polygons() first"
        )
    gdf = gpd.read_file(f"zip://{zip_path}")
    return gdf.union_all()


def mask_to_land(
    grid_lon: np.ndarray, grid_lat: np.ndarray, values: np.ndarray, land: Geometry
) -> np.ndarray:
    """Set `values` to NaN wherever (grid_lon, grid_lat) falls outside `land`.

    Args:
        grid_lon, grid_lat, values: Arrays of identical shape (e.g. from
            `np.meshgrid`), in degrees / interpolated units.
        land: A (multi)polygon of land area, e.g. from
            :func:`load_land_geometry`.

    Returns:
        Copy of `values` with ocean cells set to NaN.
    """
    on_land = contains_xy(land, grid_lon, grid_lat)
    out = np.array(values, dtype=float, copy=True)
    out[~on_land] = np.nan
    return out


def render_trend_surface(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    values: np.ndarray,
    title: str = "Warming trend (°C/decade)",
) -> go.Figure:
    """Plotly heatmap of an interpolated trend surface.

    Uses a diverging RdBu_r colorscale centered at zero (warming positive
    in red, cooling negative in blue), per README plotting conventions.

    Args:
        grid_lon, grid_lat: 1D axes of grid-cell centers, in degrees.
        values: 2D array of shape (len(grid_lat), len(grid_lon)); NaN
            cells (e.g. ocean, masked by :func:`mask_to_land`) render blank.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    vmax = float(np.nanmax(np.abs(values)))
    fig = go.Figure(
        data=go.Heatmap(
            x=grid_lon,
            y=grid_lat,
            z=values,
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorbar={"title": "°C/decade"},
        )
    )
    fig.update_layout(title=title, xaxis_title="Longitude", yaxis_title="Latitude")
    return fig


def build_interpolated_surface(
    trends_path: Path = DEFAULT_TRENDS_PATH,
    out_dir: Path = OUTPUTS_DIR,
    value_col: str = DEFAULT_VALUE_COL,
    k: int = DEFAULT_K_NEIGHBORS,
    resolution: float = DEFAULT_GRID_RESOLUTION,
) -> dict:
    """Run the Phase 3 pipeline: LOO-CV, pick a winner, render its surface.

    The winner is the method with lower leave-location-out CV RMSE (see
    :func:`leave_one_out_cv`); the classic leave-row-out CV is computed
    alongside to show how much the grid-snapped duplicates inflate it.
    Land cells are interpolated with the winner and written as an HTML
    heatmap to `out_dir`; ocean cells are left NaN.

    Args:
        trends_path: Parquet file from :func:`src.trends.build_city_trends`.
        out_dir: Destination directory for the rendered figure.
        value_col: Column of `trends_path` to interpolate.
        k: Neighborhood size for both LOO-CV and the rendered surface.
        resolution: Grid spacing in degrees (see :func:`build_grid`).

    Returns:
        Dict with keys `cv` (leave-location-out DataFrame, used to pick the
        winner), `cv_leave_row_out` (DataFrame), `winner` (str),
        `figure_path` (Path), `grid_lon`, `grid_lat`, `surface` (the
        masked 2D array).
    """
    trends = pd.read_parquet(trends_path)
    lon = trends["Longitude"].to_numpy()
    lat = trends["Latitude"].to_numpy()
    value = trends[value_col].to_numpy()

    cv = leave_one_out_cv(lon, lat, value, k=k)
    cv_row = leave_one_out_cv(lon, lat, value, k=k, exclude_same_coordinate=False)
    winner = cv.loc[cv["rmse"].idxmin(), "method"]
    logger.info(
        "leave-location-out RMSE: %s",
        dict(zip(cv["method"], cv["rmse"], strict=True)),
    )
    logger.info(
        "leave-row-out RMSE (inflated by duplicate leak): %s",
        dict(zip(cv_row["method"], cv_row["rmse"], strict=True)),
    )
    logger.info("winner (leave-location-out): %s", winner)

    grid_lon_1d, grid_lat_1d = build_grid(resolution=resolution)
    grid_lon, grid_lat = np.meshgrid(grid_lon_1d, grid_lat_1d)

    download_land_polygons()
    land = load_land_geometry()
    on_land = contains_xy(land, grid_lon, grid_lat)

    flat_values = np.full(grid_lon.size, np.nan)
    query_lon = grid_lon.ravel()[on_land.ravel()]
    query_lat = grid_lat.ravel()[on_land.ravel()]
    if winner == "idw":
        flat_values[on_land.ravel()] = idw_interpolate(
            lon, lat, value, query_lon, query_lat, k=k
        )
    else:
        params = fit_variogram_parameters(lon, lat, value)
        z, _ = ordinary_kriging_interpolate(
            lon, lat, value, query_lon, query_lat, variogram_parameters=params, k=k
        )
        flat_values[on_land.ravel()] = z

    surface = flat_values.reshape(grid_lon.shape)
    fig = render_trend_surface(
        grid_lon_1d,
        grid_lat_1d,
        surface,
        title=f"Land warming trend, {winner} interpolation (°C/decade)",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "trend_surface.html"
    fig.write_html(figure_path)
    logger.info("wrote %s", figure_path)

    return {
        "cv": cv,
        "cv_leave_row_out": cv_row,
        "winner": winner,
        "figure_path": figure_path,
        "grid_lon": grid_lon_1d,
        "grid_lat": grid_lat_1d,
        "surface": surface,
    }


def main() -> None:
    """Run the Phase 3 pipeline and print the LOO-CV comparison."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = build_interpolated_surface()
    print("leave-location-out CV (winner selection):")
    print(result["cv"].to_string(index=False))
    print("leave-row-out CV (inflated by grid-snapped duplicate leak):")
    print(result["cv_leave_row_out"].to_string(index=False))
    print(f"winner (lower leave-location-out RMSE): {result['winner']}")
    print(f"figure written to: {result['figure_path']}")


if __name__ == "__main__":
    main()
