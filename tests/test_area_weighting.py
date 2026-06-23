"""Tests for src.area_weighting -- area-weighted gridded warming exposure.

Synthetic NetCDF fixtures only (no network, no 199 MB / 84 MB rasters):

- a Berkeley-shaped ``temperature(time, latitude, longitude)`` grid with an
  exactly-linear per-cell trend, so the recovered Theil-Sen slope is known;
- a GPW-shaped raster stack whose band 11 carries numeric national-id codes,
  plus a tab-separated lookup mapping those codes to ISO3.

Grids are coarse so each cell maps to a known country and the cos(lat)
arithmetic is checkable by hand.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.area_weighting import (
    AREA_COLUMNS,
    GPW_POP_VAR,
    area_weighted_country_trends,
    assign_cell_iso3,
    cell_trends,
    load_national_id_lookup,
    world_land_mean,
)
from src.population import latitude_area_weights

START, END = "1950-01-01", "2013-09-01"


def write_berkeley_grid(path, lats, lons, slopes_per_decade, *, start=START, end=END,
                        coverage=None):
    """Berkeley-shaped grid with an exact linear trend ``slope`` per cell.

    ``slopes_per_decade[i, j]`` is the °C/decade slope injected at cell (i, j);
    NaN there yields an all-NaN (ocean) cell. ``coverage`` optionally NaNs out a
    leading fraction of months per cell to exercise the coverage gate.
    """
    months = pd.date_range(start, end, freq="MS")
    t_frac = (months.year + (months.month - 0.5) / 12).to_numpy()
    decades = (months.year + (months.month - 0.5) / 12 - 1950) / 10.0
    decades = decades.to_numpy()
    slopes = np.asarray(slopes_per_decade, dtype=float)
    data = np.empty((months.size, slopes.shape[0], slopes.shape[1]), dtype="float32")
    for i in range(slopes.shape[0]):
        for j in range(slopes.shape[1]):
            if np.isnan(slopes[i, j]):
                data[:, i, j] = np.nan
            else:
                data[:, i, j] = slopes[i, j] * decades
    if coverage is not None:
        keep = int(round(coverage * months.size))
        data[:-keep, :, :] = np.nan  # only the trailing `keep` months are finite
    ds = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), data)},
        coords={
            "time": t_frac.astype(float),
            "latitude": np.asarray(lats, dtype=float),
            "longitude": np.asarray(lons, dtype=float),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def write_natid_grid(path, lats, lons, codes, var=GPW_POP_VAR):
    """GPW-shaped raster stack whose band 11 carries national-id `codes`.

    Bands 1-10 are zero; band 11 (raster value 11) holds the numeric country
    codes; ocean cells use the GPW negative fill. Mirrors the real file's
    dims/var so the production band-11 selection path is exercised.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    data = np.zeros((11, lats.size, lons.size), dtype="float32")
    data[10] = np.asarray(codes, dtype="float32")
    ds = xr.Dataset(
        {var: (("raster", "latitude", "longitude"), data)},
        coords={"raster": list(range(1, 12)), "latitude": lats, "longitude": lons},
    )
    ds[var].attrs["units"] = "Persons"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def write_lookup(path, mapping):
    """Tab-separated GPW national-id lookup (Value/ISOCODE/NAME0) from `mapping`."""
    rows = [{"Value": v, "ISOCODE": iso, "NAME0": iso} for v, iso in mapping.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


# Ocean fill matching GPW's large-negative no-data convention.
FILL = -3.4e38


@pytest.fixture
def lookup_path(tmp_path):
    return write_lookup(tmp_path / "natid_lookup.txt", {10: "AAA", 20: "BBB", 30: "CCC"})


class TestLoadNationalIdLookup:
    def test_parses_tab_separated_to_iso3(self, lookup_path):
        out = load_national_id_lookup(lookup_path)
        assert out == {10: "AAA", 20: "BBB", 30: "CCC"}

    def test_skips_blank_iso(self, tmp_path):
        path = tmp_path / "lk.txt"
        pd.DataFrame(
            {"Value": [1, 2], "ISOCODE": ["AAA", ""], "NAME0": ["A", "Disputed"]}
        ).to_csv(path, sep="\t", index=False)
        assert load_national_id_lookup(path) == {1: "AAA"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="national-identifier lookup"):
            load_national_id_lookup(tmp_path / "absent.txt")


class TestAssignCellIso3:
    def test_maps_codes_and_masks_ocean(self, tmp_path, lookup_path):
        lats, lons = [10.0, 0.0], [0.0, 10.0]
        # NW=AAA(10), NE=ocean(fill), SW=BBB(20), SE=unknown code(99 -> None).
        codes = np.array([[10.0, FILL], [20.0, 99.0]])
        gpw = write_natid_grid(tmp_path / "natid.nc", lats, lons, codes)
        out = assign_cell_iso3(
            np.array(lats), np.array(lons), gpw, load_national_id_lookup(lookup_path)
        )
        assert out[0, 0] == "AAA"
        assert out[0, 1] is None  # ocean fill
        assert out[1, 0] == "BBB"
        assert out[1, 1] is None  # code absent from the lookup
        assert out.shape == (2, 2)


class TestCellTrends:
    def test_recovers_injected_slope_on_land(self, tmp_path):
        lats, lons = [10.0, 0.0], [0.0, 10.0]
        slopes = np.array([[0.20, 0.30], [0.10, np.nan]])  # SE ocean
        nc = write_berkeley_grid(tmp_path / "t.nc", lats, lons, slopes)
        mask = np.array([["AAA", "BBB"], ["CCC", None]], dtype=object)
        _, _, fit = cell_trends(nc, mask)
        assert fit[0, 0] == pytest.approx(0.20, abs=1e-9)
        assert fit[0, 1] == pytest.approx(0.30, abs=1e-9)
        assert fit[1, 0] == pytest.approx(0.10, abs=1e-9)
        assert np.isnan(fit[1, 1])  # ocean cell never fit

    def test_unassigned_cells_not_fit(self, tmp_path):
        lats, lons = [10.0, 0.0], [0.0, 10.0]
        slopes = np.full((2, 2), 0.25)
        nc = write_berkeley_grid(tmp_path / "t.nc", lats, lons, slopes)
        mask = np.array([["AAA", None], [None, None]], dtype=object)
        _, _, fit = cell_trends(nc, mask)
        assert fit[0, 0] == pytest.approx(0.25, abs=1e-9)
        assert np.isnan(fit[0, 1]) and np.isnan(fit[1, 0]) and np.isnan(fit[1, 1])

    def test_low_coverage_cell_gated_out(self, tmp_path):
        lats, lons = [0.0], [0.0]
        nc = write_berkeley_grid(
            tmp_path / "t.nc", lats, lons, np.array([[0.2]]), coverage=0.5
        )
        mask = np.array([["AAA"]], dtype=object)
        _, _, fit = cell_trends(nc, mask, min_coverage=0.9)
        assert np.isnan(fit[0, 0])  # only 50% of months finite, below the 0.9 gate

    def test_streaming_matches_single_band(self, tmp_path):
        lats = [20.0, 10.0, 0.0, -10.0]
        lons = [0.0, 10.0]
        slopes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        nc = write_berkeley_grid(tmp_path / "t.nc", lats, lons, slopes)
        mask = np.full((4, 2), "AAA", dtype=object)
        _, _, one = cell_trends(nc, mask, lat_chunk=99)
        _, _, chunked = cell_trends(nc, mask, lat_chunk=1)
        np.testing.assert_allclose(one, chunked, rtol=0, atol=1e-12)


class TestAreaWeightedCountryTrends:
    def test_cos_latitude_weighting_applied(self, tmp_path, lookup_path):
        # One country (AAA) spanning two latitudes with different slopes: the
        # honest mean weights each cell by cos(lat), NOT a flat average.
        lats, lons = [60.0, 0.0], [0.0]
        slopes = np.array([[0.40], [0.10]])
        codes = np.array([[10.0], [10.0]])  # both cells -> AAA
        nc = write_berkeley_grid(tmp_path / "t.nc", lats, lons, slopes)
        gpw = write_natid_grid(tmp_path / "g.nc", lats, lons, codes)
        out = area_weighted_country_trends(nc, gpw, lookup_path=lookup_path)

        w = latitude_area_weights(np.array([60.0, 0.0]))
        expected = float(np.dot(w, [0.40, 0.10]))
        assert out.columns.tolist() == list(AREA_COLUMNS)
        row = out.set_index("iso3").loc["AAA"]
        assert row["trend_c_per_decade_area_weighted"] == pytest.approx(expected)
        # cos-weighting pulls the mean toward the lower-latitude (cos≈1) cell,
        # away from the flat average of 0.25.
        assert row["trend_c_per_decade_area_weighted"] < 0.25
        assert row["area_cell_coverage"] == pytest.approx(1.0)

    def test_per_country_grouping_and_coverage(self, tmp_path, lookup_path):
        lats, lons = [0.0, 0.0001], [0.0, 10.0]
        # AAA gets two cells (one ocean -> coverage 0.5); BBB one cell.
        slopes = np.array([[0.20, 0.30], [np.nan, 0.50]])
        codes = np.array([[10.0, 20.0], [10.0, 20.0]])
        nc = write_berkeley_grid(tmp_path / "t.nc", lats, lons, slopes)
        gpw = write_natid_grid(tmp_path / "g.nc", lats, lons, codes)
        out = area_weighted_country_trends(nc, gpw, lookup_path=lookup_path).set_index(
            "iso3"
        )
        # AAA: only the finite cell (0.20) enters; one of two assigned cells fit.
        assert out.loc["AAA", "trend_c_per_decade_area_weighted"] == pytest.approx(0.20)
        assert out.loc["AAA", "area_cell_coverage"] == pytest.approx(0.5)
        # BBB: both cells fit (0.30, 0.50) at ~equal latitude -> ~flat mean 0.40.
        assert out.loc["BBB", "trend_c_per_decade_area_weighted"] == pytest.approx(
            0.40, abs=1e-6
        )
        assert out.loc["BBB", "area_cell_coverage"] == pytest.approx(1.0)

    def test_world_land_mean_matches_manual(self, tmp_path, lookup_path):
        lats, lons = [30.0, 0.0], [0.0]
        slopes = np.array([[0.30], [0.15]])
        codes = np.array([[10.0], [20.0]])
        nc = write_berkeley_grid(tmp_path / "t.nc", lats, lons, slopes)
        gpw = write_natid_grid(tmp_path / "g.nc", lats, lons, codes)
        got = world_land_mean(nc, gpw_path=gpw, lookup_path=lookup_path)
        w = latitude_area_weights(np.array([30.0, 0.0]))
        assert got == pytest.approx(float(np.dot(w, [0.30, 0.15])))
