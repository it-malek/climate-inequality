"""Tests for src.data_io — download sync logic and DuckDB ingestion.

No network and no real Kaggle download: kagglehub is monkeypatched and the
CSV is a small synthetic fixture mimicking GlobalLandTemperaturesByCity.csv.
"""

import duckdb
import pandas as pd
import pytest

from src import data_io
from src.cleaning import parse_coordinate
from src.data_io import (
    _sync_tree,
    city_csv_path,
    download_raw_data,
    load_city_temperatures,
    safe_identifier,
    write_typed_parquet,
)

HEADER = (
    "dt,AverageTemperature,AverageTemperatureUncertainty,"
    "City,Country,Latitude,Longitude"
)

# Mimics real quirks: null temperatures, S/W hemispheres, a unicode city
# name, and two cities sharing one grid-snapped coordinate pair.
ROWS = [
    "1950-01-01,3.07,0.46,Århus,Denmark,57.05N,10.33E",
    "1950-02-01,,0.50,Århus,Denmark,57.05N,10.33E",
    "1950-01-01,25.66,0.32,São Paulo,Brazil,23.31S,46.31W",
    "1950-02-01,25.10,0.41,São Paulo,Brazil,23.31S,46.31W",
    "1950-01-01,24.95,0.37,Guarulhos,Brazil,23.31S,46.31W",
    "1950-03-01,26.01,0.29,Guarulhos,Brazil,23.31S,46.31W",
]


def write_csv(path, rows=ROWS):
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def csv_path(tmp_path):
    return write_csv(tmp_path / "GlobalLandTemperaturesByCity.csv")


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "climate.duckdb"


class TestLoadCityTemperatures:
    def test_counts_and_dates(self, csv_path, db_path):
        result = load_city_temperatures(csv_path, db_path)
        assert result.n_rows == 6
        assert result.n_cities == 3
        assert str(result.dt_min) == "1950-01-01"
        assert str(result.dt_max) == "1950-03-01"
        assert result.db_path == db_path
        assert db_path.exists()

    def test_coordinates_match_scalar_parser(self, csv_path, db_path):
        load_city_temperatures(csv_path, db_path)
        con = duckdb.connect(str(db_path))
        try:
            rows = con.execute(
                "SELECT DISTINCT City, Latitude, Longitude FROM city_temps"
            ).fetchall()
        finally:
            con.close()
        coords = {city: (lat, lon) for city, lat, lon in rows}
        assert coords["Århus"] == (
            pytest.approx(parse_coordinate("57.05N")),
            pytest.approx(parse_coordinate("10.33E")),
        )
        assert coords["São Paulo"] == (pytest.approx(-23.31), pytest.approx(-46.31))
        # Grid-snapped: distinct cities may legitimately share coordinates.
        assert coords["Guarulhos"] == coords["São Paulo"]

    def test_nulls_preserved(self, csv_path, db_path):
        load_city_temperatures(csv_path, db_path)
        con = duckdb.connect(str(db_path))
        try:
            n_null = con.execute(
                "SELECT count(*) FROM city_temps WHERE AverageTemperature IS NULL"
            ).fetchone()[0]
        finally:
            con.close()
        assert n_null == 1

    def test_schema_types(self, csv_path, db_path):
        load_city_temperatures(csv_path, db_path)
        con = duckdb.connect(str(db_path))
        try:
            schema = {
                row[0]: row[1]
                for row in con.execute("DESCRIBE city_temps").fetchall()
            }
        finally:
            con.close()
        assert schema["dt"] == "DATE"
        assert schema["AverageTemperature"] == "DOUBLE"
        assert schema["Latitude"] == "DOUBLE"
        assert schema["Longitude"] == "DOUBLE"

    def test_reload_is_idempotent(self, csv_path, db_path):
        load_city_temperatures(csv_path, db_path)
        result = load_city_temperatures(csv_path, db_path)
        assert result.n_rows == 6

    def test_missing_csv_raises(self, tmp_path, db_path):
        with pytest.raises(FileNotFoundError):
            load_city_temperatures(tmp_path / "nope.csv", db_path)

    def test_bad_table_name_raises(self, csv_path, db_path):
        with pytest.raises(ValueError, match="identifier"):
            load_city_temperatures(csv_path, db_path, table="bad; DROP TABLE x")

    def test_bad_hemisphere_suffix_raises(self, tmp_path, db_path):
        csv = write_csv(
            tmp_path / "bad.csv",
            ["1950-01-01,1.0,0.1,A,X,13.31U,10.00E"],
        )
        with pytest.raises(ValueError, match="suffix"):
            load_city_temperatures(csv, db_path)

    def test_out_of_bounds_coordinate_raises(self, tmp_path, db_path):
        csv = write_csv(
            tmp_path / "oob.csv",
            ["1950-01-01,1.0,0.1,A,X,95.00N,10.00E"],
        )
        with pytest.raises(ValueError, match="exceed"):
            load_city_temperatures(csv, db_path)


class TestSafeIdentifier:
    @pytest.mark.parametrize("name", ["city_temps", "_df", "Table1", "DOUBLE"])
    def test_valid_passes_through(self, name):
        assert safe_identifier(name) == name

    @pytest.mark.parametrize("name", ["", "1abc", "a-b", "x; DROP TABLE y", "a b"])
    def test_invalid_raises(self, name):
        with pytest.raises(ValueError, match="identifier"):
            safe_identifier(name)


class TestWriteTypedParquet:
    SCHEMA = {"name": "VARCHAR", "value": "DOUBLE", "count": "BIGINT"}

    def _df(self):
        return pd.DataFrame(
            {"name": ["b", "a"], "value": [2, 1], "count": [20.0, 10.0]}
        )

    def test_casts_types_and_orders_rows(self, tmp_path):
        out = tmp_path / "out.parquet"
        write_typed_parquet(self._df(), out, self.SCHEMA, order_by=("name",))
        back = pd.read_parquet(out)
        assert back.columns.tolist() == list(self.SCHEMA)
        assert back["name"].tolist() == ["a", "b"]  # sorted, not input order
        assert back["value"].dtype == "float64"  # int input cast to DOUBLE
        assert back["count"].dtype == "int64"  # float input cast to BIGINT

    def test_overwrites_deterministically(self, tmp_path):
        out = tmp_path / "out.parquet"
        write_typed_parquet(self._df(), out, self.SCHEMA, order_by=("name",))
        first = out.read_bytes()
        write_typed_parquet(self._df(), out, self.SCHEMA, order_by=("name",))
        assert out.read_bytes() == first

    def test_rejects_unsafe_column_name(self, tmp_path):
        df = self._df().rename(columns={"name": "name; DROP"})
        with pytest.raises(ValueError, match="identifier"):
            write_typed_parquet(
                df, tmp_path / "x.parquet", {"name; DROP": "VARCHAR"}, ("value",)
            )


class TestSyncTree:
    def test_copies_nested_files(self, tmp_path):
        src = tmp_path / "src"
        (src / "sub").mkdir(parents=True)
        (src / "a.csv").write_text("aaa")
        (src / "sub" / "b.csv").write_text("bbb")
        dest = _sync_tree(src, tmp_path / "dest")
        assert (dest / "a.csv").read_text() == "aaa"
        assert (dest / "sub" / "b.csv").read_text() == "bbb"

    def test_skips_same_size_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.csv").write_text("aaa")
        dest = _sync_tree(src, tmp_path / "dest")
        # Same size, different content: must be treated as up to date.
        (dest / "a.csv").write_text("zzz")
        _sync_tree(src, dest)
        assert (dest / "a.csv").read_text() == "zzz"

    def test_recopies_when_size_differs(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.csv").write_text("aaaa")
        dest = _sync_tree(src, tmp_path / "dest")
        (dest / "a.csv").write_text("z")
        _sync_tree(src, dest)
        assert (dest / "a.csv").read_text() == "aaaa"


class TestDownloadRawData:
    def test_downloads_into_one_dir_per_dataset(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(handle):
            calls.append(handle)
            cache = tmp_path / "cache" / handle.replace("/", "_")
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "data.csv").write_text(handle)
            return str(cache)

        monkeypatch.setattr(
            data_io.kagglehub, "dataset_download", fake_download
        )
        dest = tmp_path / "raw"
        out = download_raw_data(dest, handles=("owner/ds-one", "owner/ds-two"))
        assert calls == ["owner/ds-one", "owner/ds-two"]
        assert set(out) == {"ds-one", "ds-two"}
        assert (dest / "ds-one" / "data.csv").read_text() == "owner/ds-one"
        assert (dest / "ds-two" / "data.csv").read_text() == "owner/ds-two"


class TestCityCsvPath:
    def test_finds_nested_csv(self, tmp_path):
        nested = tmp_path / "some-dataset-dir"
        nested.mkdir()
        target = write_csv(nested / "GlobalLandTemperaturesByCity.csv")
        assert city_csv_path(tmp_path) == target

    def test_missing_raises_with_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_raw_data"):
            city_csv_path(tmp_path)
