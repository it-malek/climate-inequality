"""Tests for src.explain -- geo preflight, feature assembly, both models.

Class-based, synthetic fixtures, pytest.approx() -- matching
tests/test_emissions.py / tests/test_validation.py conventions. No network:
every NetCDF/parquet input is a tiny synthetic file written to tmp_path, so
download_file()/prepare_koppen_grid() short-circuit on their idempotent
exists() checks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from src.explain import (
    BOUNDARY_POINTS,
    CITY_MODEL_SPECS,
    CITY_PARTIAL_R2_GROUPS,
    COUNTRY_MODEL_SPECS,
    FEATURES_COLUMNS,
    aggregate_features_by_country,
    add_coast_distance,
    add_latitude_features,
    add_station_density,
    attach_income_group,
    build_city_features,
    build_country_table,
    check_coordinate_orientation,
    check_country_join,
    check_nan_rate,
    check_sampling_determinism,
    coast_distance_km,
    collapse_koppen,
    compare_city_specs,
    fit_city_model,
    fit_country_model,
    load_income_groups,
    normalize_longitudes,
    print_city_sanity_checks,
    sample_static_grid,
)
from src.interpolate import haversine_km
from src.trends import CITY_KEYS


def write_static_grid(path, lat, lon, values, var):
    """Write a tiny (lat, lon) NetCDF grid for sample_static_grid tests."""
    ds = xr.Dataset(
        {var: (("lat", "lon"), np.asarray(values, dtype=float))},
        coords={"lat": np.asarray(lat, dtype=float), "lon": np.asarray(lon, dtype=float)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def make_synthetic_city_features(
    n_countries=8, n_per_country=10, seed=0, beta_lat=0.002, noise_sd=0.002
):
    """City-location features with a planted abs_latitude -> slope effect."""
    rng = np.random.default_rng(seed)
    koppen_classes = ["A", "B", "C", "D"]
    rows = []
    for ci in range(n_countries):
        for j in range(n_per_country):
            lat = rng.uniform(-80.0, 80.0)
            lon = rng.uniform(-180.0, 180.0)
            abs_lat = abs(lat)
            rows.append(
                {
                    "City": f"City{ci}_{j}",
                    "Country": f"Country{ci}",
                    "Latitude": lat,
                    "Longitude": lon,
                    "slope_c_per_decade": 0.05 + beta_lat * abs_lat + rng.normal(0.0, noise_sd),
                    "abs_latitude": abs_lat,
                    "hemisphere": "N" if lat >= 0 else "S",
                    "coast_km": rng.uniform(0.0, 2000.0),
                    "elevation_m": rng.uniform(0.0, 1000.0),
                    "koppen": koppen_classes[(ci + j) % 4],
                    "station_density": rng.uniform(0.0, 20.0),
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_country_table(
    seed=0, n_per_continent=8, beta_emissions=0.03, beta_lat=0.001, noise_sd=0.002
):
    """Country table with planted log10_emissions and mean_abs_lat effects."""
    rng = np.random.default_rng(seed)
    continents = ["Africa", "Europe", "Asia"]
    income_groups = ["Low income", "High income"]
    rows = []
    for ci, continent in enumerate(continents):
        for j in range(n_per_continent):
            log10_em = rng.uniform(-1.0, 4.0)
            mean_abs_lat = rng.uniform(0.0, 70.0)
            trend = (
                0.1
                + beta_emissions * log10_em
                + beta_lat * mean_abs_lat
                + rng.normal(0.0, noise_sd)
            )
            rows.append(
                {
                    "Country": f"C{ci}_{j}",
                    "continent": continent,
                    "income_group": income_groups[j % 2],
                    "log10_emissions": log10_em,
                    "mean_abs_lat": mean_abs_lat,
                    "trend_c_per_decade": trend,
                }
            )
    return pd.DataFrame(rows)


class TestCoordinateOrientation:
    def test_ascending_minus180_180(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc",
            lat=[-10.0, 0.0, 10.0],
            lon=[-180.0, -90.0, 0.0, 90.0],
            values=np.zeros((3, 4)),
            var="z",
        )
        orient = check_coordinate_orientation(path)
        assert orient["lat_ascending"] is True
        assert orient["lon_ascending"] is True
        assert orient["lon_convention"] == "-180-180"
        assert orient["lat_min"] == pytest.approx(-10.0)
        assert orient["lat_max"] == pytest.approx(10.0)

    def test_descending_lat_0_360(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc",
            lat=[10.0, 0.0, -10.0],
            lon=[0.0, 90.0, 180.0, 270.0],
            values=np.zeros((3, 4)),
            var="z",
        )
        orient = check_coordinate_orientation(path)
        assert orient["lat_ascending"] is False
        assert orient["lon_convention"] == "0-360"
        assert orient["lon_max"] == pytest.approx(270.0)

    def test_normalize_longitudes_noop_and_shift(self):
        lons = np.array([-170.0, -10.0, 0.0, 90.0])
        assert np.array_equal(normalize_longitudes(lons, "-180-180"), lons)
        out_360 = normalize_longitudes(lons, "0-360")
        assert out_360.tolist() == pytest.approx([190.0, 350.0, 0.0, 90.0])

    def test_normalize_longitudes_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="unknown longitude convention"):
            normalize_longitudes(np.array([0.0]), "bogus")


class TestCitySanityChecks:
    """Synthetic NYC/Cairo/Reykjavik with planted feature grids.

    Coordinates are arbitrary (not the real Berkeley grid-snapped values) --
    only the relative geometry matters: Cairo sits at the center of a square
    "land" polygon (far from any coastline) while New York and Reykjavik sit
    just inside its edges (near the coastline).
    """

    LAND = box(-50.0, -50.0, 50.0, 50.0)
    GRID_LAT = [-50.0, 0.0, 49.0, 50.0]
    GRID_LON = [-50.0, 0.0, 49.0, 50.0]

    @pytest.fixture
    def grids(self, tmp_path):
        etopo = write_static_grid(
            tmp_path / "etopo.nc", self.GRID_LAT, self.GRID_LON, np.full((4, 4), 50.0), "z"
        )
        koppen_all_b = write_static_grid(
            tmp_path / "koppen_b.nc", self.GRID_LAT, self.GRID_LON, np.full((4, 4), 5.0), "koppen_code"
        )
        koppen_cairo_e = np.full((4, 4), 5.0)
        koppen_cairo_e[1, 1] = 29.0  # Cairo's cell (lat=0, lon=0) -> group E
        koppen_bad = write_static_grid(
            tmp_path / "koppen_e.nc", self.GRID_LAT, self.GRID_LON, koppen_cairo_e, "koppen_code"
        )
        return {"etopo": etopo, "koppen_b": koppen_all_b, "koppen_e": koppen_bad}

    def test_passes_with_plausible_features(self, grids):
        trends = pd.DataFrame(
            {
                "City": ["New York", "Cairo", "Reykjavík"],
                "Country": ["United States", "Egypt", "Iceland"],
                "Latitude": [49.0, 0.0, 0.0],
                "Longitude": [0.0, 0.0, 49.0],
            }
        )
        report = print_city_sanity_checks(
            trends, etopo_path=grids["etopo"], koppen_path=grids["koppen_b"], land=self.LAND
        )
        assert report.columns.tolist() == [
            "City", "Country", "Latitude", "Longitude", "elevation_m", "koppen", "coast_km",
        ]
        by_city = report.set_index("City")
        assert (by_city["elevation_m"] == 50.0).all()
        assert by_city.loc["Cairo", "koppen"] == "B"
        assert by_city.loc["Cairo", "coast_km"] > by_city.loc["New York", "coast_km"]
        assert by_city.loc["Cairo", "coast_km"] > by_city.loc["Reykjavík", "coast_km"]

    def test_wrong_koppen_class_for_cairo_fails(self, grids):
        trends = pd.DataFrame(
            {
                "City": ["New York", "Cairo", "Reykjavík"],
                "Country": ["United States", "Egypt", "Iceland"],
                "Latitude": [49.0, 0.0, 0.0],
                "Longitude": [0.0, 0.0, 49.0],
            }
        )
        with pytest.raises(AssertionError, match="Cairo Koeppen class"):
            print_city_sanity_checks(
                trends, etopo_path=grids["etopo"], koppen_path=grids["koppen_e"], land=self.LAND
            )

    def test_coast_distance_ordering_fails_when_cairo_is_coastal(self, grids):
        # Cairo near the land edge (small coast_km); NYC/Reykjavik at the
        # interior (large coast_km) -- the ordering the preflight guards
        # against, e.g. from a swapped/misaligned coordinate axis.
        trends = pd.DataFrame(
            {
                "City": ["New York", "Cairo", "Reykjavík"],
                "Country": ["United States", "Egypt", "Iceland"],
                "Latitude": [0.0, 49.0, 0.0],
                "Longitude": [0.0, 49.0, 0.0],
            }
        )
        with pytest.raises(AssertionError, match=r"coast_km\[Cairo\]"):
            print_city_sanity_checks(
                trends, etopo_path=grids["etopo"], koppen_path=grids["koppen_b"], land=self.LAND
            )

    def test_missing_sanity_city_raises(self, grids):
        trends = pd.DataFrame(
            {
                "City": ["New York", "Cairo"],
                "Country": ["United States", "Egypt"],
                "Latitude": [49.0, 0.0],
                "Longitude": [0.0, 0.0],
            }
        )
        with pytest.raises(AssertionError, match="sanity cities not found"):
            print_city_sanity_checks(
                trends, etopo_path=grids["etopo"], koppen_path=grids["koppen_b"], land=self.LAND
            )


class TestSamplingDeterminism:
    def test_repeated_sampling_matches(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc",
            lat=[-10.0, 0.0, 10.0],
            lon=[-10.0, 0.0, 10.0],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            var="z",
        )
        lats = np.array([0.0, 10.0, -10.0])
        lons = np.array([0.0, -10.0, 10.0])
        out = check_sampling_determinism(path, "z", lats, lons, n_repeats=3)
        assert out.tolist() == sample_static_grid(path, lats, lons, "z").tolist()

    def test_boundary_points_finite_and_deterministic(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc",
            lat=[-90.0, 0.0, 90.0],
            lon=[-180.0, 0.0, 180.0],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            var="z",
        )
        boundary_lats, boundary_lons = BOUNDARY_POINTS
        out = check_sampling_determinism(path, "z", boundary_lats, boundary_lons, n_repeats=2)
        assert np.all(np.isfinite(out))

    def test_nondeterministic_sampling_raises(self, tmp_path, monkeypatch):
        path = write_static_grid(
            tmp_path / "grid.nc", lat=[0.0, 1.0], lon=[0.0, 1.0], values=np.zeros((2, 2)), var="z"
        )
        calls = iter([np.array([1.0, 2.0]), np.array([1.0, 3.0])])
        monkeypatch.setattr("src.explain.sample_static_grid", lambda *a, **k: next(calls))
        with pytest.raises(AssertionError, match="nondeterministic"):
            check_sampling_determinism(
                path, "z", np.array([0.0, 1.0]), np.array([0.0, 1.0]), n_repeats=2
            )


class TestNanRate:
    def test_nan_fraction_computed_and_logged(self, caplog):
        values = pd.Series([1.0, np.nan, 3.0, np.nan])
        with caplog.at_level(logging.INFO, logger="src.explain"):
            fraction = check_nan_rate(values, "test_label", max_nan_fraction=0.6)
        assert fraction == pytest.approx(0.5)
        assert "test_label" in caplog.text

    def test_raises_when_exceeds_max(self):
        values = np.array([np.nan, np.nan, 1.0])
        with pytest.raises(AssertionError, match="exceeds max"):
            check_nan_rate(values, "test_label", max_nan_fraction=0.5)


class TestCountryJoinIntegrity:
    def test_reports_unmatched_country(self, caplog):
        trends = pd.DataFrame({"Country": ["Alandia", "Burma", "Atlantis"]})
        inequality = pd.DataFrame(
            {"Country": ["Alandia", "Burma"], "owid_country": ["Alandia", "Myanmar"]}
        )
        income = pd.DataFrame({"owid_country": ["Alandia", "Myanmar"]})
        with caplog.at_level(logging.INFO, logger="src.explain"):
            report = check_country_join(
                trends, inequality, income, overrides={"Burma": "Myanmar"}
            )
        assert report["trends_not_in_inequality"] == ["Atlantis"]
        assert report["inequality_not_in_trends"] == []
        assert report["trends_owid_not_in_inequality_owid"] == []
        assert report["inequality_owid_not_in_income"] == []
        assert "Atlantis" in caplog.text

    def test_override_inconsistency_reported(self):
        # override says Burma -> Myanmar, but country_inequality's
        # owid_country was never renamed -- the override is inconsistent.
        trends = pd.DataFrame({"Country": ["Burma"]})
        inequality = pd.DataFrame({"Country": ["Burma"], "owid_country": ["Burma"]})
        income = pd.DataFrame({"owid_country": ["Burma"]})
        report = check_country_join(trends, inequality, income, overrides={"Burma": "Myanmar"})
        assert report["trends_owid_not_in_inequality_owid"] == ["Myanmar"]

    def test_unmatched_income_group_reported(self):
        trends = pd.DataFrame({"Country": ["Alandia"]})
        inequality = pd.DataFrame({"Country": ["Alandia"], "owid_country": ["Alandia"]})
        income = pd.DataFrame({"owid_country": ["Other"]})
        report = check_country_join(trends, inequality, income, overrides={})
        assert report["inequality_owid_not_in_income"] == ["Alandia"]


class TestLatitudeFeatures:
    def test_abs_and_hemisphere(self):
        trends = pd.DataFrame({"Latitude": [10.0, -5.0, 0.0]})
        out = add_latitude_features(trends)
        assert out["abs_latitude"].tolist() == [10.0, 5.0, 0.0]
        assert out["hemisphere"].tolist() == ["N", "S", "N"]


class TestCoastDistance:
    LAND = box(-10.0, -10.0, 10.0, 10.0)

    def test_outside_point_matches_haversine(self):
        out = coast_distance_km(np.array([20.0]), np.array([0.0]), self.LAND)
        assert out[0] == pytest.approx(haversine_km(20.0, 0.0, 10.0, 0.0))

    def test_inside_point_matches_haversine_to_edge(self):
        out = coast_distance_km(np.array([0.0]), np.array([0.0]), self.LAND)
        # (0, 0) is equidistant (10 degrees) from every edge of the box.
        assert out[0] == pytest.approx(haversine_km(0.0, 0.0, 10.0, 0.0))

    def test_add_coast_distance_adds_column(self):
        df = pd.DataFrame({"Longitude": [20.0], "Latitude": [0.0]})
        out = add_coast_distance(df, land=self.LAND)
        assert "coast_km" in out.columns
        assert out["coast_km"].iloc[0] > 0.0


class TestStaticGridSampling:
    VALUES = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    def test_nearest_cell_sampling(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc", lat=[-10.0, 0.0, 10.0], lon=[-10.0, 0.0, 10.0], values=self.VALUES, var="z"
        )
        out = sample_static_grid(path, np.array([0.0, 10.0]), np.array([0.0, -10.0]), "z")
        assert out.tolist() == [5.0, 7.0]

    def test_out_of_range_point_clamps_to_edge(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc", lat=[-10.0, 0.0, 10.0], lon=[-10.0, 0.0, 10.0], values=self.VALUES, var="z"
        )
        out = sample_static_grid(path, np.array([100.0]), np.array([-100.0]), "z")
        assert out.tolist() == [7.0]  # clamps to (lat=10, lon=-10)

    def test_0_360_grid_normalizes_signed_query_longitude(self, tmp_path):
        path = write_static_grid(
            tmp_path / "grid.nc",
            lat=[0.0, 10.0],
            lon=[0.0, 90.0, 180.0, 270.0],
            values=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            var="z",
        )
        # signed lon=-90 normalizes to 270 in this grid's 0-360 convention.
        out = sample_static_grid(path, np.array([0.0]), np.array([-90.0]), "z")
        assert out.tolist() == [4.0]


class TestKoppenCollapse:
    def test_known_codes_map_to_groups(self):
        codes = pd.Series([1, 4, 8, 17, 29, 30])
        assert collapse_koppen(codes).tolist() == ["A", "B", "C", "D", "E", "E"]

    def test_unmapped_codes_are_nan(self):
        codes = pd.Series([0, 31, np.nan])
        assert collapse_koppen(codes).isna().all()


class TestStationDensity:
    def test_dense_cluster_vs_isolated_point(self):
        df = pd.DataFrame(
            {
                "Longitude": [0.0, 0.01, -0.01, 0.0, 100.0],
                "Latitude": [0.0, 0.0, 0.0, 0.01, 0.0],
            }
        )
        out = add_station_density(df, radius_km=50.0, k=10)
        assert out["station_density"].iloc[:4].tolist() == [3.0, 3.0, 3.0, 3.0]
        assert out["station_density"].iloc[4] == 0.0


class TestIncomeGroups:
    def test_load_income_groups_picks_latest_year(self, tmp_path):
        path = tmp_path / "income.csv"
        pd.DataFrame(
            {
                "Entity": ["Alandia", "Alandia", "Burma"],
                "Code": ["ALA", "ALA", "MMR"],
                "Year": [2010, 2020, 2020],
                "World Bank's income classification": [
                    "Lower middle income", "High income", "Low income",
                ],
            }
        ).to_csv(path, index=False)
        out = load_income_groups(path).set_index("owid_country")
        assert out.loc["Alandia", "income_group"] == "High income"
        assert out.loc["Burma", "income_group"] == "Low income"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download"):
            load_income_groups(tmp_path / "absent.csv")

    def test_attach_income_group_applies_override_and_logs_unmatched(self, caplog):
        country_table = pd.DataFrame({"owid_country": ["Burma", "Atlantis"]})
        income = pd.DataFrame(
            {"owid_country": ["Myanmar", "Wonderland"], "income_group": ["Low income", "High income"]}
        )
        with caplog.at_level(logging.WARNING, logger="src.explain"):
            out = attach_income_group(country_table, income, overrides={"Burma": "Myanmar"})
        by_country = out.set_index("owid_country")
        assert by_country.loc["Burma", "income_group"] == "Low income"
        assert pd.isna(by_country.loc["Atlantis", "income_group"])
        assert "Atlantis" in caplog.text


class TestBuildCityFeatures:
    @pytest.fixture
    def inputs(self, tmp_path):
        trends = pd.DataFrame(
            {
                "City": ["A", "B", "C", "D"],
                "Country": ["X", "X", "Y", "Y"],
                "Latitude": [10.0, 20.0, -10.0, -20.0],
                "Longitude": [10.0, 20.0, -10.0, 30.0],
                "slope_c_per_decade": [0.1, 0.2, 0.05, 0.15],
            }
        )
        trends_path = tmp_path / "city_trends.parquet"
        trends.to_parquet(trends_path)

        etopo_path = write_static_grid(
            tmp_path / "etopo.nc", lat=[-90.0, 0.0, 90.0], lon=[-180.0, 0.0, 180.0], values=np.zeros((3, 3)), var="z"
        )
        koppen_path = write_static_grid(
            tmp_path / "koppen.nc", lat=[-90.0, 0.0, 90.0], lon=[-180.0, 0.0, 180.0], values=np.full((3, 3), 5.0), var="koppen_code"
        )
        return {
            "trends_path": trends_path,
            "trends": trends,
            "etopo_path": etopo_path,
            "koppen_path": koppen_path,
            "land": box(-180.0, -90.0, 180.0, 90.0),
        }

    def test_one_row_per_city_key_with_all_columns(self, tmp_path, inputs):
        out_path = tmp_path / "city_features.parquet"
        features = build_city_features(
            trends_path=inputs["trends_path"],
            out_path=out_path,
            land=inputs["land"],
            etopo_path=inputs["etopo_path"],
            koppen_path=inputs["koppen_path"],
            include_density=True,
        )
        assert features.columns.tolist() == FEATURES_COLUMNS
        assert len(features) == len(inputs["trends"])
        assert features.set_index(CITY_KEYS).index.is_unique
        assert (features["koppen"] == "B").all()
        assert features["station_density"].notna().all()
        assert out_path.exists()

    def test_include_density_false_gives_nan_density(self, tmp_path, inputs):
        out_path = tmp_path / "city_features.parquet"
        features = build_city_features(
            trends_path=inputs["trends_path"],
            out_path=out_path,
            land=inputs["land"],
            etopo_path=inputs["etopo_path"],
            koppen_path=inputs["koppen_path"],
            include_density=False,
        )
        assert features["station_density"].isna().all()


class TestCityModel:
    def test_baseline_recovers_latitude_effect(self):
        features = make_synthetic_city_features(seed=1, beta_lat=0.002, noise_sd=0.002)
        result, fit = fit_city_model(features, "baseline", CITY_MODEL_SPECS["baseline"])
        term = next(t for t in result.terms if t.term == "abs_latitude")
        assert term.ci_low < 0.002 < term.ci_high
        assert result.moran_i is not None
        assert 0.0 <= result.moran_p <= 1.0
        assert fit.nobs == len(features)

    def test_compare_city_specs_all_three(self):
        features = make_synthetic_city_features(seed=2)
        results = compare_city_specs(features)
        assert [r.spec_name for r in results] == list(CITY_MODEL_SPECS)
        for r in results:
            assert np.isfinite(r.r2)
            assert r.moran_i is not None
            assert 0.0 <= r.moran_p <= 1.0
            for t in r.terms:
                assert np.isfinite(t.coef)
                assert np.isfinite(t.se)
                assert 0.0 <= t.p_value <= 1.0

        baseline, full, interaction = results
        assert baseline.partial_r2 == {}
        assert set(full.partial_r2) == set(CITY_PARTIAL_R2_GROUPS["full"])
        assert set(interaction.partial_r2) == set(CITY_PARTIAL_R2_GROUPS["interaction"])


class TestCountryTable:
    def test_aggregate_features_by_country(self):
        features = pd.DataFrame({"Country": ["A", "A", "B"], "abs_latitude": [10.0, 20.0, 5.0]})
        out = aggregate_features_by_country(features).set_index("Country")
        assert out.loc["A", "mean_abs_lat"] == pytest.approx(15.0)
        assert out.loc["B", "mean_abs_lat"] == pytest.approx(5.0)

    def test_build_country_table_drops_nonpositive_and_computes_log10(self, caplog):
        features = pd.DataFrame({"Country": ["A", "B"], "abs_latitude": [10.0, 20.0]})
        inequality = pd.DataFrame(
            {
                "Country": ["A", "B"],
                "owid_country": ["A", "B"],
                "cum_co2_t_per_capita": [100.0, 0.0],
            }
        )
        income = pd.DataFrame(
            {"owid_country": ["A", "B"], "income_group": ["High income", "Low income"]}
        )
        with caplog.at_level(logging.WARNING, logger="src.explain"):
            out = build_country_table(features, inequality, income)
        assert out["Country"].tolist() == ["A"]
        assert out["log10_emissions"].iloc[0] == pytest.approx(2.0)
        assert out["mean_abs_lat"].iloc[0] == pytest.approx(10.0)
        assert out["income_group"].iloc[0] == "High income"
        assert "non-positive" in caplog.text


class TestCountryModel:
    def test_all_specs_finite(self):
        table = make_synthetic_country_table(seed=3)
        results = fit_country_model(table)
        assert [r.spec_name for r in results] == list(COUNTRY_MODEL_SPECS)
        for r in results:
            assert np.isfinite(r.r2)
            assert r.n > 0
            for t in r.terms:
                assert np.isfinite(t.coef)
                assert np.isfinite(t.se)
                assert np.isfinite(t.ci_low)
                assert np.isfinite(t.ci_high)
                assert 0.0 <= t.p_value <= 1.0

    def test_lat_continent_recovers_emissions_effect(self):
        # The DGP is trend = 0.1 + 0.03*log10_emissions + 0.001*mean_abs_lat
        # + noise, with no continent effect -- "lat_continent" (the key
        # spec per the plan) matches this exactly, so it should recover
        # beta_emissions=0.03 tightly even though "continent_fe" (which
        # omits mean_abs_lat) can be biased by chance within-continent
        # correlation between the two regressors.
        table = make_synthetic_country_table(seed=4, noise_sd=0.001)
        results = fit_country_model(table)
        lat_continent = next(r for r in results if r.spec_name == "lat_continent")
        term = next(t for t in lat_continent.terms if t.term == "log10_emissions")
        assert term.ci_low < 0.03 < term.ci_high
