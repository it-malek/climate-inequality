"""Generate a placeholder 1-degree global population NetCDF.

This is a **stand-in** for a real coarse population product (e.g. SEDAC GPW v4,
which requires an Earthdata login). It is *not* real data: it is a smooth,
deterministic latitude-banded field that is positive over habitable latitudes
and tapers toward the poles, so the people-weighting pipeline produces non-
trivial, reproducible numbers until the real raster is dropped into
``data/raw/population/`` (the directory is gitignored, mirroring the other raw
grids; only this generator is committed).

    uv run python scripts/make_dummy_population_grid.py

Output: ``data/raw/population/global_pop_1deg.nc`` with variable ``population``
on 1-degree ``lat`` (-89.5..89.5) / ``lon`` (-179.5..179.5) cell centers, signed
[-180,180] longitudes (matches city_trends.parquet; the sampler also normalizes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Allow `uv run python scripts/make_dummy_population_grid.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import RAW_DIR  # noqa: E402
from src.population import POP_VAR  # noqa: E402

OUT_PATH = RAW_DIR / "population" / "global_pop_1deg.nc"


def latitude_band_population(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A smooth, deterministic population-like field (people per cell, arbitrary).

    A mixture of Gaussian latitude bands centered on the real population peaks
    (a strong ~30 deg N band, a tropical band, a weaker southern band), uniform
    in longitude. Values are positive over land-ish latitudes and ~0 at the
    poles; absolute scale is meaningless (only relative weights matter).
    """
    lat2d = lat[:, None] * np.ones((1, lon.size))
    band = (
        1.0 * np.exp(-(((lat2d - 30.0) / 18.0) ** 2))
        + 0.6 * np.exp(-(((lat2d - 8.0) / 14.0) ** 2))
        + 0.35 * np.exp(-(((lat2d + 28.0) / 16.0) ** 2))
    )
    # Per-cell counts, integer-ish; floor tiny values to keep poles ~0.
    counts = np.round(band * 5.0e5).astype("float32")
    return counts


def main() -> None:
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    values = latitude_band_population(lat, lon)
    ds = xr.Dataset(
        {POP_VAR: (("lat", "lon"), values)},
        coords={"lat": lat.astype("float32"), "lon": lon.astype("float32")},
        attrs={
            "title": "PLACEHOLDER 1-degree population grid (NOT real data)",
            "note": "Synthetic latitude-banded stand-in; replace with SEDAC GPW v4.",
        },
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT_PATH)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB), "
          f"var={POP_VAR}, shape={values.shape}, "
          f"max={values.max():.0f}, nonzero cells={int((values > 0).sum())}")


if __name__ == "__main__":
    main()
