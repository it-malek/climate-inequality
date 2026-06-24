"""Tests for scripts/fetch_era5.py -- the one-time ERA5 download helper.

Only the offline glue is tested (no CDS, no network): the request payload shape and
``conform_to_t2m``, which normalizes a real CDS delivery (``valid_time`` axis,
singleton ``expver``/``number`` dims) to the ``t2m(time, latitude, longitude)``
layout :mod:`src.era5_weighting` requires.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_era5.py"
_spec = importlib.util.spec_from_file_location("fetch_era5", _SCRIPT)
fetch_era5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_era5)


class TestBuildRequest:
    def test_payload_shape(self):
        req = fetch_era5.build_request(1950, 2013, 1.0)
        assert req["variable"] == ["2m_temperature"]
        assert req["year"][0] == "1950" and req["year"][-1] == "2013"
        assert len(req["year"]) == 64
        assert req["month"] == [f"{m:02d}" for m in range(1, 13)]
        assert req["grid"] == [1.0, 1.0]
        assert req["data_format"] == "netcdf"


class TestConformToT2m:
    def test_renames_valid_time_and_squeezes_expver(self, tmp_path):
        months = pd.date_range("1950-01-01", periods=6, freq="MS")
        # Raw CDS-style: valid_time axis + a singleton expver dim, dims out of order.
        data = np.arange(6 * 1 * 2 * 3, dtype="float32").reshape(6, 1, 2, 3)
        raw = xr.Dataset(
            {"t2m": (("valid_time", "expver", "latitude", "longitude"), data)},
            coords={
                "valid_time": months,
                "expver": [1],
                "latitude": [10.0, 0.0],
                "longitude": [0.0, 10.0, 20.0],
            },
        )
        raw_path = tmp_path / "raw.nc"
        raw.to_netcdf(raw_path)

        out_path = fetch_era5.conform_to_t2m(raw_path, tmp_path / "clean.nc")
        ds = xr.open_dataset(out_path)
        try:
            assert tuple(ds["t2m"].dims) == ("time", "latitude", "longitude")
            assert ds.sizes == {"time": 6, "latitude": 2, "longitude": 3}
        finally:
            ds.close()

    def test_missing_t2m_raises(self, tmp_path):
        raw = xr.Dataset(
            {"sst": (("time", "latitude", "longitude"),
                     np.zeros((2, 2, 2), dtype="float32"))},
            coords={"time": pd.date_range("1950-01-01", periods=2, freq="MS"),
                    "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )
        raw_path = tmp_path / "raw.nc"
        raw.to_netcdf(raw_path)
        with pytest.raises(SystemExit, match="no 't2m' variable"):
            fetch_era5.conform_to_t2m(raw_path, tmp_path / "clean.nc")
