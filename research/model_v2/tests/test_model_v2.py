"""Tests for the Model V2 research scaffolding.

The synthetic tests run anywhere; the M0 verification test needs the local
``data/processed`` artifacts and is skipped without them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import cv, regions
from research.model_v2.spatial import (
    contiguity_weights,
    distance_band_weights,
    haversine_matrix,
    knn_weights,
    local_morans,
    morans_i,
)
from src.decomposition import OUTCOME_COL
from src.emissions import DEFAULT_INEQUALITY_PATH
from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH


def _grid_points(n_side: int = 8) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = np.meshgrid(np.linspace(-40, 40, n_side), np.linspace(-30, 30, n_side))
    return lon.ravel(), lat.ravel()


def test_haversine_known_distance() -> None:
    d = haversine_matrix(np.array([0.0, 0.0]), np.array([0.0, 90.0]))
    assert d[0, 1] == pytest.approx(10007.5, rel=1e-3)


def test_knn_weights_row_standardised_and_deterministic() -> None:
    lon, lat = _grid_points()
    w = knn_weights(haversine_matrix(lon, lat), 4)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all(np.diag(w) == 0)
    assert np.array_equal(w, knn_weights(haversine_matrix(lon, lat), 4))


def test_distance_band_isolates_get_nearest() -> None:
    dist = haversine_matrix(np.array([0.0, 1.0, 50.0]), np.array([0.0, 0.0, 0.0]))
    w, n_iso = distance_band_weights(dist, 500)
    assert n_iso == 1
    assert np.allclose(w.sum(axis=1), 1.0)


def test_contiguity_weights_island_fallback() -> None:
    ids = np.array(["A", "B", "C"])
    pairs = pd.DataFrame({"iso3_a": ["A"], "iso3_b": ["B"]})
    dist = haversine_matrix(np.array([0.0, 1.0, 30.0]), np.array([0.0, 0.0, 0.0]))
    w, n_fallback = contiguity_weights(ids, pairs, dist)
    assert n_fallback == 1
    assert w[2].sum() == pytest.approx(1.0)


def test_morans_i_detects_smooth_field_and_not_noise() -> None:
    lon, lat = _grid_points()
    w = knn_weights(haversine_matrix(lon, lat), 6)
    smooth = np.sin(np.radians(lon) * 3) + np.cos(np.radians(lat) * 3)
    assert morans_i(smooth, w, 199, 0).statistic > 0.5
    noise = np.random.default_rng(1).normal(size=len(lon))
    assert abs(morans_i(noise, w, 199, 0).statistic) < 0.25


def test_local_morans_labels_are_quadrant_consistent() -> None:
    lon, lat = _grid_points()
    w = knn_weights(haversine_matrix(lon, lat), 6)
    values = np.where(lon > 0, 1.0, -1.0) + np.random.default_rng(0).normal(scale=0.1, size=len(lon))
    local = local_morans(values, w, 99, 0)
    assert set(local["label"]).issubset({"HH", "LL", "HL", "LH", "ns"})
    assert (local.loc[local["label"] == "HH", "z"] > 0).all()


def test_m49_covers_each_iso_once() -> None:
    table = regions.m49_table()
    assert table["iso3"].is_unique
    assert set(table["m49_region"]) == {"Africa", "Asia", "Europe", "Americas", "Oceania"}


def _synthetic_design(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(0, 60, n)
    frame = pd.DataFrame(
        {
            "Country": [f"c{i}" for i in range(n)],
            "cum_co2_per_capita": rng.lognormal(3, 1, n),
            "cum_co2_total": rng.lognormal(6, 1, n),
            "abs_latitude": lat,
            "elevation": rng.uniform(0, 2000, n),
            "continentality": rng.uniform(0, 1000, n),
            "climate_zone": rng.choice(["A", "B", "C"], n),
            "hemisphere": rng.choice(["N", "S"], n),
            "spatial_block": rng.choice(["X", "Y", "Z"], n),
            "income_group": rng.choice(["High", "Low"], n),
            "population": rng.lognormal(15, 1, n),
            "station_density": rng.uniform(0, 5, n),
        }
    )
    frame[OUTCOME_COL] = 0.1 + 0.002 * lat + rng.normal(scale=0.02, size=n)
    return frame


def test_v1_design_reproduces_ols_in_sample() -> None:
    design = _synthetic_design()
    yhat = cv.V1Design().fit(design).predict(design)
    assert cv._r2(design[OUTCOME_COL].to_numpy(), yhat) > 0.5


def test_unseen_level_rule_mean_effect_vs_reference() -> None:
    design = _synthetic_design()
    train = design[design["spatial_block"] != "Z"]
    test = design[design["spatial_block"] == "Z"]
    mean_effect = cv.fit_predict_v1(train, test, unseen="mean_effect")
    reference = cv.fit_predict_v1(train, test, unseen="reference")
    enc = cv.V1Design().fit(train)
    effects = [0.0] + [enc.coef[enc.columns.index(f"spatial_block={lv}")] for lv in enc.levels["spatial_block"][1:]]
    assert np.allclose(mean_effect - reference, np.mean(effects))


def test_buffer_removes_near_training_rows() -> None:
    design = _synthetic_design(n=40)
    lon, lat = _grid_points(n_side=7)
    lon, lat = lon[:40], lat[:40]
    dist = haversine_matrix(lon, lat)
    res = cv.cross_validate(design, np.arange(40), dist=dist, buffer_km=1500)
    assert np.all(res.nearest_train_km > 1500)
    assert np.all(res.n_train < 40)


def test_kmeans_folds_deterministic_and_complete() -> None:
    lon, lat = _grid_points()
    a = cv.folds_kmeans(lon, lat, 5, seed=0)
    b = cv.folds_kmeans(lon, lat, 5, seed=0)
    assert np.array_equal(a, b)
    assert set(a) == set(range(5))


def test_scorecard_keys_present() -> None:
    design = _synthetic_design()
    lon, lat = _grid_points()
    lon, lat = lon[: len(design)], lat[: len(design)]
    w = knn_weights(haversine_matrix(lon, lat), 4)
    fitted = cv.V1Design().fit(design).predict(design)
    res = cv.cross_validate(design, cv.folds_random(len(design), 5, seed=0))
    card = cv.scorecard(design, fitted, res, w, n_permutations=19, shares=True)
    for key in ("in_sample_r2", "cv_r2", "cv_rmse", "cv_mae", "calibration_slope", "residual_morans_i_cv", "share_geography", "residual_share"):
        assert key in card
    assert card["share_geography"] + card["share_emissions"] + card["share_socioeconomic"] + card["share_population"] + card["residual_share"] == pytest.approx(1.0)


@pytest.mark.skipif(
    not (DEFAULT_INEQUALITY_PATH.exists() and DEFAULT_FEATURES_PATH.exists() and INCOME_PATH.exists()),
    reason="needs the local data/processed artifacts",
)
def test_m0_refit_matches_tagged_bundle() -> None:
    from research.model_v2.m0 import verify_against_bundle

    report = verify_against_bundle()
    assert report["all_bundle_values_match_reference"]
    for key in ("in_sample_r2", "share_geography", "share_emissions", "residual_morans_i"):
        assert abs(report[key]["refit_minus_bundle"]) < 1e-9
