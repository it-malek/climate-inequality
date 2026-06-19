"""Tests for the perturbation-stability layer (src.stability).

Synthetic designs only (reusing tests.test_decomposition.make_country_design)
plus synthetic country centroids — no real data, no network. The layer must be
deterministic from a seed and byte-stable through round_floats, mirroring the
decomposition's own contract.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.feature_schema import INTERPRETATION_NOTE
from src.stability import (
    RESIDUAL_KEY,
    TOP_INFLUENCE,
    build_stability_summary,
    summary_payload,
)
from tests.test_decomposition import make_country_design


def _centroids(design: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Synthetic per-country centroid table (Country, Longitude, Latitude)."""
    countries = pd.unique(design["Country"])
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Country": countries,
            "Longitude": rng.uniform(-180.0, 180.0, len(countries)),
            "Latitude": rng.uniform(-60.0, 70.0, len(countries)),
        }
    )


def _summary(*, n=80, seed=0, n_boot=200, k=5, n_perm=49, **design_kwargs):
    design = make_country_design(n=n, seed=seed, **design_kwargs)
    return build_stability_summary(
        design, _centroids(design, seed=seed),
        n_boot=n_boot, seed=1, k=k, n_permutations=n_perm,
    )


def test_check_invariant_holds():
    # build_stability_summary calls result.check() internally; an explicit call
    # documents the contract (bootstrap-mean shares + residual sum to 1).
    result = _summary()
    result.check()
    groups = result.share_stability["groups"]
    assert sum(g["mean"] for g in groups.values()) == pytest.approx(1.0, abs=1e-9)


def test_groups_cover_named_axes_and_residual():
    groups = _summary().share_stability["groups"]
    # make_country_design plants emissions + geography signal only.
    assert {"emissions", "geography", RESIDUAL_KEY} <= set(groups)


def test_cis_are_ordered_and_in_unit_interval():
    groups = _summary().share_stability["groups"]
    for key, g in groups.items():
        assert g["ci_low"] <= g["ci_high"], key
        assert g["ci_low"] <= g["mean"] <= g["ci_high"], key
        assert -1e-9 <= g["ci_low"] and g["ci_high"] <= 1.0 + 1e-9, key


def test_probabilities_in_unit_interval():
    share = _summary().share_stability
    assert 0.0 <= share["p_geography_largest"] <= 1.0
    assert 0.0 <= share["p_emissions_positive"] <= 1.0


def test_block_bootstrap_present_alongside_iid():
    block = _summary().share_stability["block_bootstrap"]
    assert block["by"] == "spatial_block"
    assert {"emissions", "geography", RESIDUAL_KEY} <= set(block["groups"])
    for g in block["groups"].values():
        assert g["ci_low"] <= g["ci_high"]


def test_geography_dominance_tracks_planted_signal():
    # Geography carries the signal -> it should be the largest axis in (nearly)
    # every resample; emissions is independent noise.
    share = _summary(beta_emissions=0.0, beta_lat=0.06, noise_sd=0.004).share_stability
    assert share["groups"]["geography"]["mean"] > share["groups"]["emissions"]["mean"]
    assert share["p_geography_largest"] > 0.8


def test_influence_is_ranked_top_n_per_group():
    influence = _summary().influence
    assert influence["method"] == "leave_one_country_out"
    by_group = influence["by_group"]
    assert {"emissions", "geography"} <= set(by_group)
    for group, rows in by_group.items():
        assert len(rows) <= TOP_INFLUENCE
        magnitudes = [abs(delta) for _, delta in rows]
        assert magnitudes == sorted(magnitudes, reverse=True), group
        for country, delta in rows:
            assert isinstance(country, str)
            assert isinstance(delta, float)


def test_residual_morans_i_in_range_with_pvalue():
    rs = _summary().residual_spatial
    assert -1.0 <= rs["morans_i"] <= 1.0
    assert 0.0 < rs["p_value"] <= 1.0
    assert rs["k_neighbors"] == 5
    assert rs["method"] == "centroid kNN"
    assert rs["n"] > rs["k_neighbors"]


def test_morans_i_degrades_when_centroids_too_few():
    # Fewer countries with centroids than k+1 -> Moran's I cannot be computed,
    # but the layer must degrade to None rather than raise.
    design = make_country_design(n=40, seed=2)
    centroids = _centroids(design).iloc[:3]  # only 3 of 40 have a centroid
    result = build_stability_summary(
        design, centroids, n_boot=50, seed=1, k=8, n_permutations=19
    )
    assert result.residual_spatial["morans_i"] is None
    assert result.residual_spatial["p_value"] is None


def test_summary_payload_carries_interpretation():
    payload = summary_payload(_summary(n_boot=50, n_perm=19))
    assert payload["interpretation"] == INTERPRETATION_NOTE
    assert "share_stability" in payload and "residual_spatial" in payload


def test_deterministic_and_byte_stable():
    # Same inputs + seed must reproduce the serialized summary exactly.
    first = json.dumps(summary_payload(_summary(n_boot=120, n_perm=29)))
    second = json.dumps(summary_payload(_summary(n_boot=120, n_perm=29)))
    assert first == second
