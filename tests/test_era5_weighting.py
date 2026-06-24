"""Tests for src.era5_weighting -- the ERA5 reanalysis area-weighted cross-check.

Synthetic NetCDF fixtures only (no network, no CDS download). The ERA5 fixture
mirrors the real product's quirks the module must handle:

- variable ``t2m`` in **absolute Kelvin** (a large baseline offset that must cancel
  out of a *slope*);
- a **CF datetime** time axis (decoded by xarray, unlike Berkeley's fractional
  years);
- **0-360 longitudes** (normalized to geographic ``[-180, 180)`` for GPW sampling).

The injected per-cell trend is exactly linear in the same decimal-decade axis the
station/Berkeley pipeline fits, so the recovered Theil-Sen slope is known to 1e-9
and ERA5/Berkeley parity is exact on equivalent data.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.area_weighting import (
    AREA_COLUMNS,
    GPW_POP_VAR,
    area_weighted_country_trends,
)
from src.era5_weighting import (
    ERA5_AREA_COL,
    ERA5_COLUMNS,
    ERA5_COVERAGE_COL,
    era5_area_weighted_country_trends,
    era5_cell_iso3,
    era5_time_to_months,
    era5_world_land_mean,
    normalize_longitudes,
)
from src.population import latitude_area_weights

START, END = "1950-01-01", "2013-09-01"
FILL = -3.4e38  # GPW negative no-data fill


def _decade_axis(months: pd.DatetimeIndex) -> np.ndarray:
    """The mid-month decimal-decade axis (matches src.cleaning.to_decimal_decades)."""
    return ((months.year + (months.month - 0.5) / 12) / 10.0).to_numpy()


def write_era5_grid(path, lats, lons, slopes_per_decade, *, start=START, end=END,
                    baseline_k=280.0, coverage=None):
    """ERA5-shaped ``t2m(time, latitude, longitude)`` grid, Kelvin, CF time, 0-360 lon.

    ``data = baseline_k + slope * decimal_decades`` so the recovered slope equals
    the injected ``slopes_per_decade[i, j]`` (the Kelvin offset cancels). NaN slope
    -> all-NaN ocean cell; ``coverage`` NaNs out a leading fraction of months.
    """
    months = pd.date_range(start, end, freq="MS")
    decades = _decade_axis(months)
    slopes = np.asarray(slopes_per_decade, dtype=float)
    data = np.empty((months.size, slopes.shape[0], slopes.shape[1]), dtype="float32")
    for i in range(slopes.shape[0]):
        for j in range(slopes.shape[1]):
            if np.isnan(slopes[i, j]):
                data[:, i, j] = np.nan
            else:
                data[:, i, j] = baseline_k + slopes[i, j] * decades
    if coverage is not None:
        keep = int(round(coverage * months.size))
        data[:-keep, :, :] = np.nan
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), data)},
        coords={
            "time": months,  # datetime64 -> CF-encoded on write, decoded on read
            "latitude": np.asarray(lats, dtype=float),
            "longitude": np.asarray(lons, dtype=float),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def write_berkeley_grid(path, lats, lons, slopes_per_decade, *, start=START, end=END):
    """Berkeley-shaped ``temperature`` grid (fractional-year time) for parity tests."""
    months = pd.date_range(start, end, freq="MS")
    t_frac = (months.year + (months.month - 0.5) / 12).to_numpy()
    decades = _decade_axis(months)
    slopes = np.asarray(slopes_per_decade, dtype=float)
    data = np.empty((months.size, slopes.shape[0], slopes.shape[1]), dtype="float32")
    for i in range(slopes.shape[0]):
        for j in range(slopes.shape[1]):
            data[:, i, j] = (np.nan if np.isnan(slopes[i, j])
                             else slopes[i, j] * decades)
    ds = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), data)},
        coords={"time": t_frac.astype(float),
                "latitude": np.asarray(lats, dtype=float),
                "longitude": np.asarray(lons, dtype=float)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def write_natid_grid(path, lats, lons, codes, var=GPW_POP_VAR):
    """GPW-shaped raster stack whose band 11 carries national-id `codes`."""
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
    rows = [{"Value": v, "ISOCODE": iso, "NAME0": iso} for v, iso in mapping.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def lookup_path(tmp_path):
    return write_lookup(tmp_path / "natid_lookup.txt", {10: "AAA", 20: "BBB", 30: "CCC"})


class TestNormalizeLongitudes:
    def test_maps_0_360_to_signed(self):
        out = normalize_longitudes([0.0, 10.0, 180.0, 190.0, 350.0])
        np.testing.assert_allclose(out, [0.0, 10.0, -180.0, -170.0, -10.0])


class TestEra5TimeToMonths:
    def test_snaps_to_first_of_month(self):
        raw = pd.to_datetime(["1950-01-16", "1950-02-28", "2013-09-15"]).to_numpy()
        out = era5_time_to_months(raw)
        assert list(out) == list(pd.to_datetime(["1950-01-01", "1950-02-01",
                                                 "2013-09-01"]))


class TestEra5CellIso3:
    def test_samples_gpw_at_geographic_longitudes(self, tmp_path, lookup_path):
        # ERA5 ships 0-360 lons; col0=350 (-> geographic -10), col1=10. The natid
        # grid lives on geographic lons, so normalization must route each cell to
        # the right country while keeping native column order.
        era5_lons = [350.0, 10.0]
        lats = [0.0]
        natid = write_natid_grid(
            tmp_path / "natid.nc", lats, [-10.0, 10.0],
            np.array([[20.0, 10.0]]),  # -10deg -> BBB, +10deg -> AAA
        )
        out = era5_cell_iso3(np.array(lats), np.array(era5_lons), natid,
                             {10: "AAA", 20: "BBB"})
        assert out[0, 0] == "BBB"   # native col0 (350 -> -10)
        assert out[0, 1] == "AAA"   # native col1 (+10)


class TestCellTrendsOnEra5:
    def test_recovers_slope_through_kelvin_and_cf_time(self, tmp_path, lookup_path):
        # A single AAA cell; absolute-Kelvin baseline must cancel out of the slope.
        lats, lons = [0.0], [10.0]
        nc = write_era5_grid(tmp_path / "e.nc", lats, lons, np.array([[0.27]]),
                             baseline_k=290.0)
        natid = write_natid_grid(tmp_path / "n.nc", lats, lons, np.array([[10.0]]))
        out = era5_area_weighted_country_trends(nc, natid, lookup_path=lookup_path)
        row = out.set_index("iso3").loc["AAA"]
        assert row[ERA5_AREA_COL] == pytest.approx(0.27, abs=1e-9)
        assert row[ERA5_COVERAGE_COL] == pytest.approx(1.0)
        assert out.columns.tolist() == list(ERA5_COLUMNS)

    def test_low_coverage_cell_omitted(self, tmp_path, lookup_path):
        lats, lons = [0.0], [10.0]
        nc = write_era5_grid(tmp_path / "e.nc", lats, lons, np.array([[0.2]]),
                             coverage=0.5)
        natid = write_natid_grid(tmp_path / "n.nc", lats, lons, np.array([[10.0]]))
        out = era5_area_weighted_country_trends(nc, natid, lookup_path=lookup_path)
        assert out.empty  # the only cell is below the 0.9 coverage gate


class TestEra5AreaWeightedCountryTrends:
    def test_cos_latitude_weighting_applied(self, tmp_path, lookup_path):
        lats, lons = [60.0, 0.0], [10.0]
        slopes = np.array([[0.40], [0.10]])
        codes = np.array([[10.0], [10.0]])  # both -> AAA
        nc = write_era5_grid(tmp_path / "e.nc", lats, lons, slopes)
        natid = write_natid_grid(tmp_path / "n.nc", lats, lons, codes)
        out = era5_area_weighted_country_trends(nc, natid, lookup_path=lookup_path)

        w = latitude_area_weights(np.array([60.0, 0.0]))
        expected = float(np.dot(w, [0.40, 0.10]))
        row = out.set_index("iso3").loc["AAA"]
        assert row[ERA5_AREA_COL] == pytest.approx(expected)
        assert row[ERA5_AREA_COL] < 0.25  # cos-weighting pulls toward the equator

    def test_parity_with_berkeley_on_equivalent_data(self, tmp_path, lookup_path):
        # Same slopes, lats, lons (0..180 agree under normalization) and natid: the
        # ERA5 path must reproduce the Berkeley path bit-for-bit -- only the data
        # source (and its quirks) differ, never the operator/window/weighting.
        lats, lons = [40.0, 0.0], [0.0, 10.0]
        slopes = np.array([[0.30, 0.20], [0.10, 0.25]])
        codes = np.array([[10.0, 20.0], [10.0, 30.0]])
        natid = write_natid_grid(tmp_path / "n.nc", lats, lons, codes)

        berk = write_berkeley_grid(tmp_path / "b.nc", lats, lons, slopes)
        era5 = write_era5_grid(tmp_path / "e.nc", lats, lons, slopes)

        b = area_weighted_country_trends(berk, natid, lookup_path=lookup_path)
        e = era5_area_weighted_country_trends(era5, natid, lookup_path=lookup_path)
        np.testing.assert_allclose(
            b[AREA_COLUMNS[1]].to_numpy(), e[ERA5_AREA_COL].to_numpy(), atol=1e-9
        )
        assert b["iso3"].tolist() == e["iso3"].tolist()


class TestEra5WorldLandMean:
    def test_uniform_slope_recovered(self, tmp_path, lookup_path):
        lats, lons = [30.0, 0.0, -30.0], [0.0, 10.0]
        nc = write_era5_grid(tmp_path / "e.nc", lats, lons, np.full((3, 2), 0.19))
        natid = write_natid_grid(tmp_path / "n.nc", lats, lons,
                                 np.full((3, 2), 10.0))
        assert era5_world_land_mean(nc, natid, lookup_path=lookup_path) == pytest.approx(
            0.19, abs=1e-6  # float32-stored t2m -> ~5e-8 slope round-off
        )
