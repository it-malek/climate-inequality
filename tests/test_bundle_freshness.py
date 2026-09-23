"""A replaced snapshot must not reuse cached values from its predecessor."""
import json

import pandas as pd
import pytest
import streamlit as st

from app import loaders


def test_stats_change_at_same_path_without_cache_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
    st.cache_data.clear()
    path = tmp_path / "stats.json"
    path.write_text('{"n": 151}')
    assert loaders.load_stats() == {"n": 151}
    assert loaders.load_stats() == {"n": 151}
    path.write_text('{"n": 152}')
    assert loaders.load_stats() == {"n": 152}
    path.unlink()
    with pytest.raises(FileNotFoundError):
        loaders.load_stats()


def test_derived_city_series_changes_without_cache_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
    st.cache_data.clear()
    path = tmp_path / "city_anomalies.parquet"
    pd.DataFrame({"city_id": [1], "dt": ["2000-01-01"], "anomaly": [0.1]}).to_parquet(path)
    assert loaders.load_city_series(1).anomaly.tolist() == [0.1]
    pd.DataFrame({"city_id": [1], "dt": ["2000-01-01"], "anomaly": [0.2]}).to_parquet(path)
    assert loaders.load_city_series(1).anomaly.tolist() == [0.2]


def test_optional_missing_then_available(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
    st.cache_data.clear()
    assert loaders.load_residual_summary() is None
    (tmp_path / "residual_structure_summary.json").write_text(json.dumps({"static": {"n": 151}}))
    assert loaders.load_residual_summary() == {"static": {"n": 151}}


def test_manifest_rejects_changed_missing_and_incompatible_bundle(tmp_path, monkeypatch):
    from src.snapshot import write_manifest

    monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
    path = tmp_path / "stats.json"
    path.write_text('{"n": 151}')
    write_manifest(tmp_path)
    assert loaders.load_stats() == {"n": 151}
    path.write_text('{"n": 152}')
    with pytest.raises(ValueError, match="mismatched"):
        loaders.load_stats()
    write_manifest(tmp_path, version="test-replacement")
    assert loaders.load_stats() == {"n": 152}
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        loaders.load_stats()
    manifest = tmp_path / "release.json"
    manifest.write_text('{"schema_version": 999}')
    with pytest.raises(ValueError, match="schema"):
        loaders.load_stats()


def test_entrypoint_fails_closed_after_warm_bundle_loses_stats(synthetic_bundle, tmp_path, monkeypatch):
    import shutil
    from streamlit.testing.v1 import AppTest

    directory = tmp_path / "published"
    shutil.copytree(synthetic_bundle["bundle_dir"], directory)
    monkeypatch.setattr(loaders, "APP_DATA_DIR", directory)
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=15).run()
    assert not at.exception
    assert len(at.metric) == 4
    (directory / "stats.json").unlink()
    at.run()
    assert not at.exception
    assert len(at.metric) == 0
    assert "unavailable" in at.error[0].value
