#!/usr/bin/env python
"""One-time fetch of ERA5 monthly 2 m temperature for the area-weighted cross-check.

Downloads **ERA5 monthly-averaged 2 m temperature, 1950-2013, regridded server-side
to 1deg x 1deg** from the Copernicus Climate Data Store (CDS) and writes a clean
``t2m(time, latitude, longitude)`` NetCDF to ``data/raw/era5/`` (gitignored), the
file :mod:`src.era5_weighting` reads. The 1deg regrid mirrors the Berkeley grid and
keeps the download ~200 MB (vs ~3 GB at native 0.25deg).

This is **not** run in CI and **not** imported by ``src`` or the tests -- it is a
manual, credentialed, networked step the analyst runs once.

Prerequisites
-------------
1. A free Copernicus CDS account: https://cds.climate.copernicus.eu/
2. Your API key in ``~/.cdsapirc`` (Copernicus shows the exact two lines on your
   profile page), e.g.::

       url: https://cds.climate.copernicus.eu/api
       key: <your-key>

3. Accept the dataset licence once (on the dataset's CDS download page).
4. Install the optional dependency: ``uv sync --extra era5`` (or ``uv add cdsapi``).

Usage
-----
    uv run python scripts/fetch_era5.py            # defaults: 1950-2013, 1deg
    uv run python scripts/fetch_era5.py --grid 0.25 --out data/raw/era5/era5_quarter.nc

The trend window the cross-check actually fits is 1950-01..2013-09
(:data:`src.cleaning.DEFAULT_START`/``DEFAULT_END``); fetching whole years keeps the
request simple and the extra months are sliced out at fit time.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("fetch_era5")

DATASET = "reanalysis-era5-single-levels-monthly-means"
DEFAULT_OUT = Path("data/raw/era5/era5_t2m_monthly_1950_2013_1deg.nc")
DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2013


def build_request(start_year: int, end_year: int, grid: float) -> dict:
    """The CDS request payload for monthly-mean 2 m temperature on a regular grid."""
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(y) for y in range(start_year, end_year + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": [grid, grid],  # regular lat/lon regrid, server-side
    }


def conform_to_t2m(raw_path: Path, out_path: Path) -> Path:
    """Normalize the CDS download to ``t2m(time, latitude, longitude)``.

    Recent CDS deliveries may name the time axis ``valid_time`` and carry singleton
    ``expver`` / ``number`` dims; :mod:`src.era5_weighting` expects exactly
    ``t2m(time, latitude, longitude)``. This squeezes the singletons and renames the
    time coordinate so the produced file drops straight into the cross-check.
    """
    import xarray as xr  # heavy / optional; imported lazily

    ds = xr.open_dataset(raw_path)
    try:
        if "valid_time" in ds.dims or "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        for singleton in ("expver", "number"):
            if singleton in ds.dims and ds.sizes.get(singleton, 1) == 1:
                ds = ds.squeeze(singleton, drop=True)
            elif singleton in ds.coords:
                ds = ds.drop_vars(singleton, errors="ignore")
        if "t2m" not in ds.data_vars:
            raise SystemExit(
                f"downloaded file has no 't2m' variable (found {sorted(ds.data_vars)})"
            )
        da = ds["t2m"].transpose("time", "latitude", "longitude")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        da.to_dataset(name="t2m").to_netcdf(out_path)
    finally:
        ds.close()
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--grid", type=float, default=1.0, help="degrees (1.0 default)")
    args = parser.parse_args()

    try:
        import cdsapi  # optional; imported lazily so --help works without it
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "cdsapi is not installed; run `uv sync --extra era5` (or `uv add cdsapi`)"
        ) from exc

    request = build_request(args.start_year, args.end_year, args.grid)
    logger.info("requesting %s %d-%d at %g deg -> %s",
                DATASET, args.start_year, args.end_year, args.grid, args.out)
    client = cdsapi.Client()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        raw_path = Path(tmp.name)
    try:
        client.retrieve(DATASET, request, str(raw_path))
        conform_to_t2m(raw_path, args.out)
    finally:
        raw_path.unlink(missing_ok=True)
    logger.info("wrote %s -- now run: uv run python -m src.era5_validation", args.out)


if __name__ == "__main__":
    main()
