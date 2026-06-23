"""Tests for src.population -- people-weighted warming exposure.

Synthetic NetCDF population grids only: no network, no real raster. The grid is
deliberately coarse so each city-location maps to a known cell, making the
weighted-mean arithmetic checkable by hand.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.population import (
    POP_COVERAGE_COL,
    POP_WEIGHTED_COL,
    population_weighted_country_mean,
    sample_population,
)


def write_population_grid(path, lat, lon, values, var="population"):
    """Write a tiny (lat, lon) population NetCDF (mirrors the static-grid tests)."""
    ds = xr.Dataset(
        {var: (("lat", "lon"), np.asarray(values, dtype=float))},
        coords={"lat": np.asarray(lat, dtype=float), "lon": np.asarray(lon, dtype=float)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


@pytest.fixture
def grid(tmp_path):
    """A 2x2 grid: population rises west->east so coordinates pick known cells."""
    # lat {0, 10}, lon {0, 10}; population[lat_idx, lon_idx].
    values = np.array([[10.0, 90.0], [10.0, 90.0]])
    return write_population_grid(tmp_path / "pop.nc", [0.0, 10.0], [0.0, 10.0], values)


class TestSamplePopulation:
    def test_nearest_cell_values(self, grid):
        # (lat~0, lon~0) -> 10; (lat~9, lon~9) -> 90 (nearest cell is lat 10/lon 10).
        out = sample_population(np.array([0.0, 9.0]), np.array([0.0, 9.0]), grid)
        assert out.tolist() == [10.0, 90.0]


class TestPopulationWeightedCountryMean:
    def test_weighted_arithmetic(self, grid):
        # Country A: a low-pop western city (pop 10, slope 0.1) and a high-pop
        # eastern city (pop 90, slope 0.5). People-weighted mean leans to 0.5.
        trends = pd.DataFrame(
            {
                "Country": ["A", "A"],
                "Latitude": [0.0, 0.0],
                "Longitude": [0.0, 10.0],
                "slope_c_per_decade": [0.1, 0.5],
            }
        )
        out = population_weighted_country_mean(trends, grid).set_index("Country")
        # (0.1*10 + 0.5*90) / 100 = 0.46  (vs unweighted 0.30).
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

    def test_zero_population_excluded_from_weights(self, tmp_path):
        # West cell has zero population; only the eastern city should count.
        grid = write_population_grid(
            tmp_path / "pop.nc", [0.0, 10.0], [0.0, 10.0],
            np.array([[0.0, 50.0], [0.0, 50.0]]),
        )
        trends = pd.DataFrame(
            {
                "Country": ["A", "A"],
                "Latitude": [0.0, 0.0],
                "Longitude": [0.0, 10.0],
                "slope_c_per_decade": [0.1, 0.5],
            }
        )
        out = population_weighted_country_mean(trends, grid).set_index("Country")
        assert out.loc["A", POP_WEIGHTED_COL] == pytest.approx(0.5)
        assert out.loc["A", POP_COVERAGE_COL] == pytest.approx(0.5)

    def test_falls_back_to_unweighted_when_no_positive_weights(self, tmp_path, caplog):
        grid = write_population_grid(
            tmp_path / "pop.nc", [0.0, 10.0], [0.0, 10.0], np.zeros((2, 2)),
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
            out = population_weighted_country_mean(trends, grid).set_index("Country")
        # No positive weights -> unweighted mean 0.3, coverage 0.
        assert out.loc["A", POP_WEIGHTED_COL] == pytest.approx(0.3)
        assert out.loc["A", POP_COVERAGE_COL] == pytest.approx(0.0)
        assert "unweighted mean" in caplog.text
