"""Tests for src.emissions — aggregation, cumulative per-capita, join, OLS.

No real dataset and no network: synthetic OWID-shaped frames, and tiny
csv/parquet files written to tmp_path for the end-to-end test (downloads
are skipped because download_file is idempotent and the files exist).
"""

import logging

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.emissions import (
    INEQUALITY_COLUMNS,
    InequalityResult,
    aggregate_trends_by_country,
    build_inequality_analysis,
    cumulative_consumption_per_capita,
    cumulative_emissions_per_capita,
    join_country_data,
    load_continents,
    quantify_inequality,
)
from tests.test_population import write_gpw_grid

SLOPE = 0.05  # injected °C/decade per 10x emissions


def make_owid_rows(country, iso_code, years, co2, population, consumption_co2=None):
    """Rows for one country in the OWID long format.

    `consumption_co2` defaults to all-missing (the column is always present so
    :func:`src.emissions.load_owid_co2`'s ``usecols`` finds it, mirroring the
    real OWID csv where the series is sparse).
    """
    return pd.DataFrame(
        {
            "country": country,
            "year": years,
            "iso_code": iso_code,
            "co2": co2,
            "consumption_co2": np.nan if consumption_co2 is None else consumption_co2,
            "population": population,
        }
    )


def make_inequality_frame(noise_sd=0.0, seed=0, specs=None):
    """Country table with trend = intercept + SLOPE * log10(emissions).

    Continent intercepts are correlated with emissions (Europe is both
    higher-emitting and offset warmer), so pooled OLS is confounded while
    continent fixed effects recover SLOPE.
    """
    if specs is None:
        specs = [
            ("Africa", 0.10, [0.0, 0.5, 1.0, 1.5]),
            ("Europe", 0.13, [2.0, 2.5, 3.0]),
        ]
    rng = np.random.default_rng(seed)
    rows = []
    for continent, intercept, log_emissions in specs:
        for lx in log_emissions:
            i = len(rows)
            rows.append(
                {
                    "Country": f"C{i}",
                    "owid_country": f"C{i}",
                    "continent": continent,
                    "n_cities": 5,
                    "trend_c_per_decade": (
                        intercept + SLOPE * lx + rng.normal(0.0, noise_sd)
                    ),
                    "cumulative_co2_mt": 10.0,
                    "population": 1_000_000,
                    "cum_co2_t_per_capita": 10.0**lx,
                    # Consumption lens (window-matched). Positive and monotone
                    # in lx so the v2 wide-registry consumers have valid input.
                    "consumption_start_year": 1990,
                    "cum_consumption_t_per_capita": 10.0**lx,
                    "cum_co2_window_t_per_capita": 10.0**lx,
                    # People-weighted exposure lens.
                    "trend_c_per_decade_pop_weighted": (
                        intercept + SLOPE * lx + rng.normal(0.0, noise_sd)
                    ),
                    "pop_weight_coverage": 1.0,
                }
            )
    return pd.DataFrame(rows)


class TestAggregateTrendsByCountry:
    def test_unweighted_mean_and_city_count(self):
        trends = pd.DataFrame(
            {
                "Country": ["A", "A", "A", "B"],
                "slope_c_per_decade": [0.1, 0.2, 0.3, 0.5],
            }
        )
        out = aggregate_trends_by_country(trends)
        assert out.columns.tolist() == ["Country", "n_cities", "trend_c_per_decade"]
        by_country = out.set_index("Country")
        assert by_country.loc["A", "n_cities"] == 3
        assert by_country.loc["A", "trend_c_per_decade"] == pytest.approx(0.2)
        assert by_country.loc["B", "n_cities"] == 1
        assert by_country.loc["B", "trend_c_per_decade"] == pytest.approx(0.5)


class TestCumulativeEmissionsPerCapita:
    def test_basic_arithmetic(self):
        owid = make_owid_rows("A", "AAA", [2011, 2012, 2013], [1.0, 2.0, 3.0], 1e6)
        out = cumulative_emissions_per_capita(owid, cutoff_year=2013)
        row = out.iloc[0]
        assert row["cumulative_co2_mt"] == pytest.approx(6.0)
        # 6 Mt = 6e6 t over 1e6 people.
        assert row["cum_co2_t_per_capita"] == pytest.approx(6.0)

    def test_respects_cutoff_year(self):
        owid = make_owid_rows(
            "A", "AAA", [2012, 2013, 2014], [1.0, 2.0, 100.0], 1e6
        )
        out = cumulative_emissions_per_capita(owid, cutoff_year=2013)
        assert out.iloc[0]["cumulative_co2_mt"] == pytest.approx(3.0)

    def test_skips_null_years(self):
        owid = make_owid_rows("A", "AAA", [2011, 2012, 2013], [np.nan, 2.0, 4.0], 1e6)
        out = cumulative_emissions_per_capita(owid, cutoff_year=2013)
        assert out.iloc[0]["cumulative_co2_mt"] == pytest.approx(6.0)

    def test_excludes_owid_aggregate_rows(self):
        frames = [
            make_owid_rows("A", "AAA", [2013], [1.0], 1e6),
            make_owid_rows("World", "OWID_WRL", [2013], [999.0], 7e9),
            make_owid_rows("Asia", None, [2013], [500.0], 4e9),
        ]
        out = cumulative_emissions_per_capita(pd.concat(frames), cutoff_year=2013)
        assert out["country"].tolist() == ["A"]

    def test_drops_country_without_cutoff_population(self, caplog):
        frames = [
            make_owid_rows("A", "AAA", [2013], [1.0], 1e6),
            make_owid_rows("B", "BBB", [2013], [1.0], np.nan),
        ]
        with caplog.at_level(logging.INFO, logger="src.emissions"):
            out = cumulative_emissions_per_capita(pd.concat(frames), cutoff_year=2013)
        assert out["country"].tolist() == ["A"]
        assert "B" in caplog.text

    def test_drops_country_with_no_emissions_record(self, caplog):
        frames = [
            make_owid_rows("A", "AAA", [2012, 2013], [1.0, 1.0], 1e6),
            make_owid_rows("B", "BBB", [2012, 2013], [np.nan, np.nan], 1e6),
        ]
        with caplog.at_level(logging.INFO, logger="src.emissions"):
            out = cumulative_emissions_per_capita(pd.concat(frames), cutoff_year=2013)
        assert out["country"].tolist() == ["A"]
        assert "B" in caplog.text


class TestCumulativeConsumptionPerCapita:
    def test_window_matched_arithmetic(self):
        # Consumption begins 2012; the window-matched production cumulative must
        # sum production over that SAME [2012..2013] window, excluding 2011.
        owid = make_owid_rows(
            "A", "AAA", [2011, 2012, 2013],
            co2=[1.0, 2.0, 3.0], population=1e6,
            consumption_co2=[np.nan, 10.0, 20.0],
        )
        out = cumulative_consumption_per_capita(owid, cutoff_year=2013)
        row = out.set_index("country").loc["A"]
        assert row["consumption_start_year"] == 2012
        # consumption 30 Mt over 1e6 people -> 30 t/cap.
        assert row["cum_consumption_t_per_capita"] == pytest.approx(30.0)
        # production over the matched window is 2+3=5 Mt (2011's 1 Mt excluded).
        assert row["cum_co2_window_t_per_capita"] == pytest.approx(5.0)

    def test_respects_cutoff_year(self):
        owid = make_owid_rows(
            "A", "AAA", [2012, 2013, 2014],
            co2=[1.0, 2.0, 99.0], population=1e6,
            consumption_co2=[5.0, 5.0, 99.0],
        )
        out = cumulative_consumption_per_capita(owid, cutoff_year=2013)
        row = out.set_index("country").loc["A"]
        assert row["cum_consumption_t_per_capita"] == pytest.approx(10.0)
        assert row["cum_co2_window_t_per_capita"] == pytest.approx(3.0)

    def test_drops_country_without_any_consumption(self, caplog):
        frames = [
            make_owid_rows("A", "AAA", [2013], [1.0], 1e6, consumption_co2=[2.0]),
            make_owid_rows("B", "BBB", [2013], [1.0], 1e6, consumption_co2=[np.nan]),
        ]
        with caplog.at_level(logging.INFO, logger="src.emissions"):
            out = cumulative_consumption_per_capita(pd.concat(frames), cutoff_year=2013)
        assert out["country"].tolist() == ["A"]
        assert "B" in caplog.text

    def test_drops_country_without_cutoff_population(self, caplog):
        frames = [
            make_owid_rows("A", "AAA", [2013], [1.0], 1e6, consumption_co2=[2.0]),
            make_owid_rows("B", "BBB", [2013], [1.0], np.nan, consumption_co2=[2.0]),
        ]
        with caplog.at_level(logging.INFO, logger="src.emissions"):
            out = cumulative_consumption_per_capita(pd.concat(frames), cutoff_year=2013)
        assert out["country"].tolist() == ["A"]
        assert "B" in caplog.text

    def test_excludes_owid_aggregate_rows(self):
        frames = [
            make_owid_rows("A", "AAA", [2013], [1.0], 1e6, consumption_co2=[2.0]),
            make_owid_rows("World", "OWID_WRL", [2013], [999.0], 7e9, consumption_co2=[9.0]),
        ]
        out = cumulative_consumption_per_capita(pd.concat(frames), cutoff_year=2013)
        assert out["country"].tolist() == ["A"]

    def test_returns_only_consumption_columns(self):
        owid = make_owid_rows(
            "A", "AAA", [2012, 2013], [1.0, 2.0], 1e6, consumption_co2=[3.0, 4.0]
        )
        out = cumulative_consumption_per_capita(owid, cutoff_year=2013)
        assert out.columns.tolist() == [
            "country",
            "consumption_start_year",
            "cum_consumption_t_per_capita",
            "cum_co2_window_t_per_capita",
        ]


class TestJoinCountryData:
    @staticmethod
    def inputs():
        country_trends = pd.DataFrame(
            {
                "Country": ["Alandia", "Burma"],
                "n_cities": [2, 3],
                "trend_c_per_decade": [0.1, 0.2],
                "trend_c_per_decade_pop_weighted": [0.12, 0.18],
                "pop_weight_coverage": [1.0, 1.0],
            }
        )
        emissions = pd.DataFrame(
            {
                "country": ["Alandia", "Myanmar"],
                "cumulative_co2_mt": [10.0, 20.0],
                "population": [1e6, 2e6],
                "cum_co2_t_per_capita": [10.0, 10.0],
                "consumption_start_year": [1990, 1990],
                "cum_consumption_t_per_capita": [8.0, 8.0],
                "cum_co2_window_t_per_capita": [5.0, 5.0],
            }
        )
        continents = pd.DataFrame(
            {"country": ["Alandia", "Myanmar"], "continent": ["Europe", "Asia"]}
        )
        return country_trends, emissions, continents

    def test_exact_and_override_matches(self):
        out = join_country_data(*self.inputs())
        assert out.columns.tolist() == INEQUALITY_COLUMNS
        assert len(out) == 2
        burma = out.set_index("Country").loc["Burma"]
        assert burma["owid_country"] == "Myanmar"  # default BERKELEY_TO_OWID
        assert burma["continent"] == "Asia"
        assert burma["cum_co2_t_per_capita"] == pytest.approx(10.0)

    def test_unmatched_country_dropped_and_logged(self, caplog):
        country_trends, emissions, continents = self.inputs()
        atlantis = pd.DataFrame(
            {"Country": ["Atlantis"], "n_cities": [1], "trend_c_per_decade": [0.3]}
        )
        country_trends = pd.concat([country_trends, atlantis], ignore_index=True)
        with caplog.at_level(logging.WARNING, logger="src.emissions"):
            out = join_country_data(country_trends, emissions, continents)
        assert "Atlantis" not in out["Country"].tolist()
        assert "no OWID emissions match" in caplog.text
        assert "Atlantis" in caplog.text

    def test_missing_continent_kept_but_logged(self, caplog):
        country_trends, emissions, continents = self.inputs()
        continents = continents[continents["country"] != "Myanmar"]
        with caplog.at_level(logging.WARNING, logger="src.emissions"):
            out = join_country_data(country_trends, emissions, continents)
        assert len(out) == 2
        assert out.set_index("Country").loc["Burma", "continent"] is np.nan or (
            pd.isna(out.set_index("Country").loc["Burma", "continent"])
        )
        assert "no continent" in caplog.text


class TestQuantifyInequality:
    def test_fe_recovers_injected_slope_exactly(self):
        result = quantify_inequality(make_inequality_frame())
        assert isinstance(result, InequalityResult)
        assert result.ols_fe.coef == pytest.approx(SLOPE, abs=1e-8)
        assert result.ols_fe.r2 == pytest.approx(1.0, abs=1e-8)

    def test_pooled_biased_when_continents_confound(self):
        result = quantify_inequality(make_inequality_frame())
        # Europe is both higher-emitting and offset warmer, so the pooled
        # slope absorbs the continent difference; FE removes it.
        assert result.ols_pooled.coef > SLOPE + 0.005
        assert abs(result.ols_fe.coef - SLOPE) < 1e-6

    def test_spearman_is_one_for_monotone_construction(self):
        result = quantify_inequality(make_inequality_frame())
        assert result.spearman_rho == pytest.approx(1.0)
        assert result.spearman_p < 1e-6

    def test_counts(self):
        result = quantify_inequality(make_inequality_frame())
        assert result.n_countries == 7
        assert result.n_continents == 2

    def test_ci_brackets_slope_under_noise(self):
        frame = make_inequality_frame(
            noise_sd=0.003,
            seed=1,
            specs=[
                ("Africa", 0.10, [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]),
                ("Europe", 0.13, [1.8, 2.1, 2.4, 2.7, 3.0, 3.3]),
            ],
        )
        result = quantify_inequality(frame)
        assert result.ols_fe.ci_low < SLOPE < result.ols_fe.ci_high
        assert result.ols_fe.p_value < 0.05
        assert result.ols_fe.se > 0

    def test_nonpositive_emissions_dropped(self, caplog):
        frame = make_inequality_frame()
        zero_row = frame.iloc[[0]].assign(cum_co2_t_per_capita=0.0, Country="Z0")
        frame = pd.concat([frame, zero_row], ignore_index=True)
        with caplog.at_level(logging.WARNING, logger="src.emissions"):
            result = quantify_inequality(frame)
        assert result.n_countries == 7
        assert "non-positive" in caplog.text


class TestBuildInequalityAnalysis:
    """End to end on synthetic files; no network (inputs pre-exist)."""

    COUNTRIES = [
        # (berkeley_name, owid_name, iso, continent, log10 per-capita)
        ("Alphaland", "Alphaland", "AAA", "Africa", 0.0),
        ("Betaland", "Betaland", "BBB", "Africa", 1.0),
        ("Gammaland", "Gammaland", "GGG", "Africa", 2.0),
        ("Burma", "Myanmar", "MMR", "Asia", 2.2),
        ("Deltaland", "Deltaland", "DDD", "Asia", 2.6),
        ("Epsilonland", "Epsilonland", "EEE", "Asia", 3.0),
    ]

    @pytest.fixture
    def paths(self, tmp_path):
        population = 1_000_000
        owid_frames = [
            # 0 Mt in 2012 plus the full amount in 2013, so cumulative sums
            # land exactly on 10**lx tonnes per person. Consumption present from
            # 2012, so the consumption window is [2012..2013] (== full record).
            make_owid_rows(
                owid, iso, [2012, 2013], [0.0, 10.0**lx], population,
                consumption_co2=[0.0, (10.0**lx) * 0.8],
            )
            for _, owid, iso, _, lx in self.COUNTRIES
        ]
        owid_frames.append(
            make_owid_rows("World", "OWID_WRL", [2012, 2013], [9e3, 9e3], 7e9)
        )
        co2_path = tmp_path / "owid-co2-data.csv"
        pd.concat(owid_frames, ignore_index=True).to_csv(co2_path, index=False)

        continents_path = tmp_path / "continents.csv"
        pd.DataFrame(
            {
                "Entity": [owid for _, owid, *_ in self.COUNTRIES],
                "Code": [iso for *_, iso, _, _ in self.COUNTRIES],
                "Year": 2023,
                "World region according to OWID": [
                    continent for *_, continent, _ in self.COUNTRIES
                ],
            }
        ).to_csv(continents_path, index=False)

        offsets = {"Africa": 0.10, "Asia": 0.13}
        trends = pd.DataFrame(
            {
                "Country": [name for name, *_ in self.COUNTRIES for _ in range(2)],
                # Distinct in-grid coordinates per country (both rows share one).
                "Latitude": [
                    5.0 + 8.0 * i for i in range(len(self.COUNTRIES)) for _ in range(2)
                ],
                "Longitude": [
                    5.0 + 8.0 * i for i in range(len(self.COUNTRIES)) for _ in range(2)
                ],
                "slope_c_per_decade": [
                    offsets[continent] + SLOPE * lx
                    for _, _, _, continent, lx in self.COUNTRIES
                    for _ in range(2)
                ],
                "analysis_window": "1950-01-01..2013-09-01",
            }
        )
        trends_path = tmp_path / "city_trends.parquet"
        con = duckdb.connect()
        try:
            con.register("_trends", trends)
            con.execute(f"COPY _trends TO '{trends_path}' (FORMAT PARQUET)")
        finally:
            con.close()

        # Uniform GPW-shaped grid over the coordinate range: people-weighting
        # then equals the unweighted mean, an exact end-to-end check.
        pop_grid_path = write_gpw_grid(
            tmp_path / "gpw.nc", [60.0, 0.0], [0.0, 60.0],
            np.full((2, 2), 100.0),
        )
        return {
            "trends_path": trends_path,
            "co2_path": co2_path,
            "continents_path": continents_path,
            "out_dir": tmp_path / "outputs",
            "table_path": tmp_path / "country_inequality.parquet",
            "pop_grid_path": pop_grid_path,
        }

    def test_pipeline_end_to_end(self, paths):
        out = build_inequality_analysis(**paths)
        result = out["result"]
        # cutoff_year=2013 was derived from analysis_window; the injected
        # log-linear relationship survives the whole pipeline exactly.
        assert result.n_countries == 6
        assert result.n_continents == 2
        assert result.ols_fe.coef == pytest.approx(SLOPE, abs=1e-6)
        table = out["table"]
        assert table.columns.tolist() == INEQUALITY_COLUMNS
        assert table.set_index("Country").loc["Burma", "owid_country"] == "Myanmar"
        assert (table["n_cities"] == 2).all()
        # Consumption lens populated additively: window-matched production over
        # [2012..2013] equals the full record here, and consumption is 0.8x it.
        burma = table.set_index("Country").loc["Burma"]
        assert burma["consumption_start_year"] == 2012
        assert burma["cum_co2_window_t_per_capita"] == pytest.approx(10.0**2.2)
        assert burma["cum_consumption_t_per_capita"] == pytest.approx(10.0**2.2 * 0.8)
        # Uniform population grid -> people-weighted mean equals the unweighted
        # country mean, and coverage is full.
        assert table["trend_c_per_decade_pop_weighted"].to_numpy() == pytest.approx(
            table["trend_c_per_decade"].to_numpy()
        )
        assert (table["pop_weight_coverage"] == 1.0).all()
        assert out["figure_path"].exists()
        assert out["table_path"].exists()

    def test_degrades_when_population_grid_absent(self, paths, tmp_path, caplog):
        # Best-effort: an absent population grid leaves the people-weighted
        # columns NULL (the v2 exposure lens degrades) without failing the build.
        paths = {**paths, "pop_grid_path": tmp_path / "absent.nc"}
        with caplog.at_level(logging.WARNING, logger="src.emissions"):
            out = build_inequality_analysis(**paths)
        table = out["table"]
        assert table.columns.tolist() == INEQUALITY_COLUMNS
        assert table["trend_c_per_decade_pop_weighted"].isna().all()
        assert table["pop_weight_coverage"].isna().all()
        assert "population grid absent" in caplog.text

    def test_parquet_types_are_explicit(self, paths):
        build_inequality_analysis(**paths)
        con = duckdb.connect()
        try:
            schema = {
                row[0]: row[1]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(paths["table_path"])],
                ).fetchall()
            }
        finally:
            con.close()
        assert schema["Country"] == "VARCHAR"
        assert schema["continent"] == "VARCHAR"
        assert schema["n_cities"] == "BIGINT"
        assert schema["population"] == "BIGINT"
        assert schema["trend_c_per_decade"] == "DOUBLE"
        assert schema["cum_co2_t_per_capita"] == "DOUBLE"
        assert schema["consumption_start_year"] == "BIGINT"
        assert schema["cum_consumption_t_per_capita"] == "DOUBLE"
        assert schema["cum_co2_window_t_per_capita"] == "DOUBLE"
        assert schema["trend_c_per_decade_pop_weighted"] == "DOUBLE"
        assert schema["pop_weight_coverage"] == "DOUBLE"


class TestLoaders:
    def test_load_continents_renames_real_header(self, tmp_path):
        path = tmp_path / "continents.csv"
        path.write_text(
            "Entity,Code,Year,World region according to OWID\n"
            "Afghanistan,AFG,2023,Asia\n"
        )
        out = load_continents(path)
        assert out.columns.tolist() == ["country", "continent"]
        assert out.iloc[0]["continent"] == "Asia"

    def test_load_continents_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download"):
            load_continents(tmp_path / "absent.csv")
