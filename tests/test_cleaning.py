"""Tests for src.cleaning — coordinate parsing, dates, windowing, coverage."""

import pandas as pd
import pytest

from src.cleaning import (
    add_date_parts,
    coverage_by_city,
    filter_window,
    parse_coordinate,
    parse_coordinates,
    to_decimal_decades,
)


class TestParseCoordinate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("32.95N", 32.95),
            ("32.95S", -32.95),
            ("100.53E", 100.53),
            ("100.53W", -100.53),
            ("0.00N", 0.0),
            (" 41.88N ", 41.88),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_coordinate(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["13.31U", "abcN", "", "42.0"])
    def test_invalid_raises(self, raw):
        with pytest.raises(ValueError):
            parse_coordinate(raw)


class TestParseCoordinatesVectorized:
    def test_matches_scalar(self):
        df = pd.DataFrame(
            {
                "Latitude": ["32.95N", "12.05S"],
                "Longitude": ["100.53W", "77.04E"],
            }
        )
        out = parse_coordinates(df)
        assert out["Latitude"].tolist() == pytest.approx([32.95, -12.05])
        assert out["Longitude"].tolist() == pytest.approx([-100.53, 77.04])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"Latitude": ["1.0N"], "Longitude": ["1.0E"]})
        parse_coordinates(df)
        assert df["Latitude"].iloc[0] == "1.0N"

    def test_out_of_bounds_latitude_raises(self):
        df = pd.DataFrame({"Latitude": ["95.0N"], "Longitude": ["10.0E"]})
        with pytest.raises(ValueError, match="Latitude"):
            parse_coordinates(df)

    def test_bad_suffix_raises(self):
        df = pd.DataFrame({"Latitude": ["13.31U"], "Longitude": ["10.0E"]})
        with pytest.raises(ValueError, match="suffix"):
            parse_coordinates(df)


class TestDatesAndWindow:
    def _df(self):
        return pd.DataFrame(
            {
                "dt": ["1949-12-01", "1950-01-01", "2000-06-01", "2013-10-01"],
                "AverageTemperature": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_add_date_parts(self):
        out = add_date_parts(self._df())
        assert out["Year"].tolist() == [1949, 1950, 2000, 2013]
        assert out["Month"].tolist() == [12, 1, 6, 10]

    def test_filter_window_default(self):
        out = filter_window(self._df())
        assert len(out) == 2  # 1950-01 and 2000-06 survive


class TestToDecimalDecades:
    def test_midmonth_values(self):
        out = to_decimal_decades(pd.Series(["1950-01-01", "2013-09-01"]))
        assert out.iloc[0] == pytest.approx((1950 + 0.5 / 12) / 10)
        assert out.iloc[1] == pytest.approx((2013 + 8.5 / 12) / 10)

    def test_same_month_a_decade_apart_differs_by_one(self):
        out = to_decimal_decades(pd.Series(["1950-06-01", "1960-06-01"]))
        assert out.iloc[1] - out.iloc[0] == pytest.approx(1.0)

    def test_accepts_datetime_series(self):
        out = to_decimal_decades(pd.Series(pd.to_datetime(["2000-01-01"])))
        assert out.iloc[0] == pytest.approx((2000 + 0.5 / 12) / 10)


class TestCoverage:
    def test_keep_flag(self):
        # City A: full coverage for a 3-month window; City B: 1 of 3.
        df = pd.DataFrame(
            {
                "City": ["A", "A", "A", "B"],
                "Country": ["X", "X", "X", "X"],
                "AverageTemperature": [1.0, 2.0, 3.0, 1.0],
            }
        )
        out = coverage_by_city(
            df, min_fraction=0.9, start="2000-01-01", end="2000-03-01"
        )
        keep = dict(zip(out["City"], out["keep"]))
        assert keep["A"] is True or keep["A"] == True  # noqa: E712
        assert not keep["B"]

    def test_group_keys_split_same_named_cities(self):
        # Same (City, Country) at two grid coordinates: one fully covered
        # location, one with 1 of 3 months.
        df = pd.DataFrame(
            {
                "City": ["A"] * 4,
                "Country": ["X"] * 4,
                "Latitude": [30.0, 30.0, 30.0, 45.0],
                "Longitude": [-90.0] * 4,
                "AverageTemperature": [1.0, 2.0, 3.0, 1.0],
            }
        )
        kwargs = dict(min_fraction=0.9, start="2000-01-01", end="2000-03-01")
        pooled = coverage_by_city(df, **kwargs)
        assert len(pooled) == 1
        assert pooled["coverage"].iloc[0] == pytest.approx(4 / 3)  # inflated

        per_location = coverage_by_city(
            df,
            group_keys=("City", "Country", "Latitude", "Longitude"),
            **kwargs,
        )
        assert len(per_location) == 2
        keep = dict(zip(per_location["Latitude"], per_location["keep"]))
        assert keep[30.0] and not keep[45.0]
