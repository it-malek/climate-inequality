"""Tests for src.population -- people-weighted exposure off the GPW v4 grid.

Synthetic NetCDF grids only (shaped like GPW v4: a ``raster`` band stack over
``latitude``/``longitude`` with the count variable on bands 1-5): no network, no
84 MB raster. Grids are coarse so each city maps to a known cell, making the
weighted arithmetic checkable by hand.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.population import (
    GPW_POP_VAR,
    POP_COVERAGE_COL,
    POP_WEIGHTED_COL,
    latitude_area_weights,
    population_weighted_country_mean,
    sample_population,
)


def write_gpw_grid(path, lat, lon, count_2020, var=GPW_POP_VAR):
    """Write a tiny GPW-v4-shaped NetCDF: 5 count bands over (latitude, longitude).

    Only band 5 (raster value 5 == Population Count 2020) carries `count_2020`;
    the others are zero. Mirrors the real file's dims/var/units so the production
    band-selection + dim-name + descending-latitude path is exercised.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    data = np.zeros((5, lat.size, lon.size), dtype="float32")
    data[4] = np.asarray(count_2020, dtype="float32")
    ds = xr.Dataset(
        {var: (("raster", "latitude", "longitude"), data)},
        coords={"raster": [1, 2, 3, 4, 5], "latitude": lat, "longitude": lon},
    )
    ds[var].attrs["units"] = "Persons"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


@pytest.fixture
def grid(tmp_path):
    """2x2 GPW-like grid, descending latitude; population rises west->east."""
    # lat {10, 0} (descending, like GPW), lon {0, 10}; band-5 count[lat, lon].
    count = np.array([[10.0, 90.0], [10.0, 90.0]])
    return write_gpw_grid(tmp_path / "gpw.nc", [10.0, 0.0], [0.0, 10.0], count)


class TestSamplePopulation:
    def test_nearest_cell_count_from_2020_band(self, grid):
        out = sample_population(np.array([0.0, 9.0]), np.array([0.0, 9.0]), grid)
        assert out.tolist() == [10.0, 90.0]

    def test_missing_grid_raises_informative_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="population grid not found"):
            sample_population(np.array([0.0]), np.array([0.0]), tmp_path / "absent.nc")

    def test_negative_fill_becomes_nan(self, tmp_path):
        # GPW ocean/no-data fill is a large negative -> NaN, not a giant weight.
        count = np.array([[-3.4e38, 50.0], [-3.4e38, 50.0]])
        g = write_gpw_grid(tmp_path / "gpw.nc", [10.0, 0.0], [0.0, 10.0], count)
        out = sample_population(np.array([0.0]), np.array([0.0]), g)
        assert np.isnan(out[0])


class TestPopulationWeightedCountryMean:
    def test_weighted_arithmetic_no_cos_lat(self, grid):
        # Low-pop western city (10, slope 0.1) + high-pop eastern (90, slope 0.5).
        trends = pd.DataFrame(
            {
                "Country": ["A", "A"],
                "Latitude": [0.0, 0.0],
                "Longitude": [0.0, 10.0],
                "slope_c_per_decade": [0.1, 0.5],
            }
        )
        out = population_weighted_country_mean(trends, grid).set_index("Country")
        # (0.1*10 + 0.5*90)/100 = 0.46 (population-weighted; latitude irrelevant).
        assert out.loc["A", POP_WEIGHTED_COL] == pytest.approx(0.46)
        assert out.loc["A", POP_COVERAGE_COL] == pytest.approx(1.0)

    def test_columns_and_one_row_per_country(self, grid):
        trends = pd.DataFrame(
            {
                "Country": ["A", "A", "B"],
                "Latitude": [0.0, 10.0, 0.0],
                "Longitude": [0.0, 10.0, 10.0],
                "slope_c_per_decade": [0.1, 0.2, 0.3],
            }
        )
        out = population_weighted_country_mean(trends, grid)
        assert out.columns.tolist() == ["Country", POP_WEIGHTED_COL, POP_COVERAGE_COL]
        assert out["Country"].tolist() == ["A", "B"]

    def test_zero_population_excluded(self, tmp_path):
        count = np.array([[0.0, 50.0], [0.0, 50.0]])
        g = write_gpw_grid(tmp_path / "gpw.nc", [10.0, 0.0], [0.0, 10.0], count)
        trends = pd.DataFrame(
            {
                "Country": ["A", "A"],
                "Latitude": [0.0, 0.0],
                "Longitude": [0.0, 10.0],
                "slope_c_per_decade": [0.1, 0.5],
            }
        )
        out = population_weighted_country_mean(trends, g).set_index("Country")
        assert out.loc["A", POP_WEIGHTED_COL] == pytest.approx(0.5)
        assert out.loc["A", POP_COVERAGE_COL] == pytest.approx(0.5)

    def test_falls_back_to_unweighted_when_no_positive_weights(self, tmp_path, caplog):
        g = write_gpw_grid(
            tmp_path / "gpw.nc", [10.0, 0.0], [0.0, 10.0], np.zeros((2, 2))
        )
        trends = pd.DataFrame(
            {
                "Country": ["A", "A"],
                "Latitude": [0.0, 0.0],
                "Longitude": [0.0, 10.0],
                "slope_c_per_decade": [0.1, 0.5],
            }
        )
        with caplog.at_level(logging.INFO, logger="src.population"):
            out = population_weighted_country_mean(trends, g).set_index("Country")
        assert out.loc["A", POP_WEIGHTED_COL] == pytest.approx(0.3)
        assert out.loc["A", POP_COVERAGE_COL] == pytest.approx(0.0)
        assert "unweighted mean" in caplog.text


class TestAreaWeightedDiagnostics:
    def test_latitude_weights_cos_and_normalized(self):
        w = latitude_area_weights(np.array([0.0, 60.0]))
        # cos(0)=1, cos(60)=0.5 -> normalized to [2/3, 1/3].
        assert w == pytest.approx([2 / 3, 1 / 3])
        assert w.sum() == pytest.approx(1.0)
