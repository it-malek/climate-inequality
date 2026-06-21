"""Validation of the committed shipped-artifact bundle.

The per-layer suites (``test_pcs`` / ``test_projections`` / ``test_decomposition`` /
``test_stability`` / ``test_coupling``) exercise the *producing functions* on synthetic
in-memory inputs. This module instead validates the *artifacts on disk* -- both the
committed ``app/data/`` bundle and the synthetic bundle built by ``conftest`` -- asserting
value-free invariants and cross-file consistency that would catch a stale or internally
inconsistent bundle (e.g. a summary that no longer matches its table, or projections that
diverged from their upstream source columns).

No new dependencies, no golden values: every assertion is an invariant that holds for any
valid bundle, so the suite survives data regeneration. Float comparisons use tolerances and
byte-equality is asserted only on ``round_floats``-serialized JSON (the project's only
cross-platform byte-stable surface; ``*.parquet`` floats and Moran's I drift Mac vs Linux).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import app_assets
from src.coupling import (
    IMPACT_INDEX,
    RESPONSIBILITY_INDEX,
    _inequality_coefficient,
    _rank_desc,
    _zscore,
)
from src.coupling import compute_coupling
from src.coupling import summary_payload as coupling_summary_payload
from src.physical_model import DRIVERS, TRAJECTORY_SCHEMA
from src.projections import ID_COL, PCS_V1_BINDING
from src.stability import RESIDUAL_KEY

ABS = 1e-9


# --------------------------------------------------------------------------- #
# Bundle selection: run every check against both shipped sources.
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["committed", "synthetic"])
def bundle(request, synthetic_bundle) -> Path:
    """A directory holding a shipped artifact bundle.

    ``committed`` is the git-tracked ``app/data/``; ``synthetic`` is the
    session-built, network-free bundle from ``conftest``. Individual checks skip
    when a particular artifact is absent from the chosen bundle.
    """
    if request.param == "committed":
        committed = app_assets.APP_DATA_DIR
        if not committed.exists():
            pytest.skip("no committed app/data bundle")
        return committed
    return Path(synthetic_bundle["bundle_dir"])


def _json(bundle: Path, name: str):
    path = bundle / name
    if not path.exists():
        pytest.skip(f"{name} absent in {bundle.name} bundle")
    return json.loads(path.read_text(encoding="utf-8"))


def _parquet(bundle: Path, name: str) -> pd.DataFrame:
    path = bundle / name
    if not path.exists():
        pytest.skip(f"{name} absent in {bundle.name} bundle")
    return pd.read_parquet(path)


def _coupling_summary_json(result) -> str:
    """Serialize a CouplingResult the way the pipeline commits it (byte-stable)."""
    return json.dumps(coupling_summary_payload(result), indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Projections -- validated through the committed coupling table + upstream join
# (``projections_v1.parquet`` is not part of the committed bundle).
# --------------------------------------------------------------------------- #
class TestProjections:
    def test_responsibility_non_negative_and_finite(self, bundle):
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        r = table[RESPONSIBILITY_INDEX].to_numpy(dtype=float)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)  # cumulative per-capita CO2 cannot be negative

    def test_impact_finite(self, bundle):
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        assert np.all(np.isfinite(table[IMPACT_INDEX].to_numpy(dtype=float)))

    def test_one_row_per_country_no_nan(self, bundle):
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        assert table[ID_COL].is_unique
        assert not table[[RESPONSIBILITY_INDEX, IMPACT_INDEX]].isna().any().any()

    def test_pcs_binding_passthrough_identity(self, bundle):
        """The shipped projections equal their upstream source columns (the PCS
        v1 identity binding), verified end-to-end on disk."""
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        inequality = _parquet(bundle, app_assets.INEQUALITY_ASSET)
        merged = table.merge(inequality, on=ID_COL, how="left", validate="one_to_one")
        assert not merged[list(PCS_V1_BINDING.values())].isna().any().any()
        for pcs_name, source_col in PCS_V1_BINDING.items():
            assert merged[pcs_name].to_numpy() == pytest.approx(
                merged[source_col].to_numpy(), abs=ABS
            ), pcs_name


# --------------------------------------------------------------------------- #
# Decomposition summary
# --------------------------------------------------------------------------- #
class TestDecomposition:
    def test_shares_partition_unit_interval(self, bundle):
        d = _json(bundle, app_assets.DECOMPOSITION_SUMMARY_ASSET)
        for group, share in d["shares"].items():
            assert -ABS <= share <= 1.0 + ABS, group
        assert 0.0 - ABS <= d["total_r2"] <= 1.0 + ABS
        assert sum(d["shares"].values()) + d["residual_share"] == pytest.approx(
            1.0, abs=ABS
        )
        assert d["residual_share"] == pytest.approx(1.0 - d["total_r2"], abs=ABS)

    def test_carries_interpretation(self, bundle):
        d = _json(bundle, app_assets.DECOMPOSITION_SUMMARY_ASSET)
        assert d["interpretation"].strip()


# --------------------------------------------------------------------------- #
# Stability summary
# --------------------------------------------------------------------------- #
class TestStability:
    def test_bootstrap_means_partition_and_cis_ordered(self, bundle):
        groups = _json(bundle, app_assets.STABILITY_SUMMARY_ASSET)["share_stability"][
            "groups"
        ]
        assert sum(g["mean"] for g in groups.values()) == pytest.approx(1.0, abs=ABS)
        for key, g in groups.items():
            assert g["ci_low"] <= g["mean"] <= g["ci_high"], key
            assert -ABS <= g["ci_low"] and g["ci_high"] <= 1.0 + ABS, key

    def test_probabilities_and_block_bootstrap(self, bundle):
        share = _json(bundle, app_assets.STABILITY_SUMMARY_ASSET)["share_stability"]
        assert 0.0 <= share["p_geography_largest"] <= 1.0
        assert 0.0 <= share["p_emissions_positive"] <= 1.0
        for key, g in share["block_bootstrap"]["groups"].items():
            assert g["ci_low"] <= g["ci_high"], key

    def test_morans_i_and_seed(self, bundle):
        summary = _json(bundle, app_assets.STABILITY_SUMMARY_ASSET)
        assert isinstance(summary["seed"], int)
        rs = summary["residual_spatial"]
        if rs["morans_i"] is not None:
            assert -1.0 <= rs["morans_i"] <= 1.0
            assert 0.0 < rs["p_value"] <= 1.0

    def test_point_shares_match_decomposition(self, bundle):
        """The stability layer perturbs the decomposition, so its point estimates
        must equal the decomposition shares it was built from."""
        groups = _json(bundle, app_assets.STABILITY_SUMMARY_ASSET)["share_stability"][
            "groups"
        ]
        decomp = _json(bundle, app_assets.DECOMPOSITION_SUMMARY_ASSET)
        for group, share in decomp["shares"].items():
            assert groups[group]["point"] == pytest.approx(share, abs=ABS), group
        assert groups[RESIDUAL_KEY]["point"] == pytest.approx(
            decomp["residual_share"], abs=ABS
        )


# --------------------------------------------------------------------------- #
# Coupling table + summary
# --------------------------------------------------------------------------- #
class TestCoupling:
    def test_table_internal_consistency(self, bundle):
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        r = table[RESPONSIBILITY_INDEX].to_numpy(dtype=float)
        im = table[IMPACT_INDEX].to_numpy(dtype=float)
        # ranks recomputed from the values (descending, ties = min).
        assert table["responsibility_rank"].to_numpy() == pytest.approx(_rank_desc(r))
        assert table["impact_rank"].to_numpy() == pytest.approx(_rank_desc(im))
        # rank_gap is exactly impact_rank - responsibility_rank. (It need NOT sum
        # to zero: with tied values, method="min" ranks compress unequally between
        # the two projections; z_gap below is the tie-robust zero-sum analogue.)
        assert table["rank_gap"].to_numpy() == pytest.approx(
            table["impact_rank"].to_numpy() - table["responsibility_rank"].to_numpy()
        )
        # z_gap consistent with the z-score operator; zero mean by construction.
        assert table["z_gap"].to_numpy() == pytest.approx(_zscore(im) - _zscore(r), abs=ABS)
        assert table["z_gap"].mean() == pytest.approx(0.0, abs=ABS)

    def test_summary_matches_table(self, bundle):
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        summary = _json(bundle, app_assets.COUPLING_SUMMARY_ASSET)
        r = table[RESPONSIBILITY_INDEX].to_numpy(dtype=float)
        im = table[IMPACT_INDEX].to_numpy(dtype=float)
        assert summary["spearman_rho"] == pytest.approx(
            float(stats.spearmanr(r, im)[0]), abs=ABS
        )
        assert summary["inequality_coefficient"] == pytest.approx(
            _inequality_coefficient(r, im), abs=ABS
        )
        assert summary["n_high_impact_low_responsibility"] == int(
            (table["z_gap"].to_numpy() > 0.0).sum()
        )

    def test_summary_bounds_and_closure(self, bundle):
        summary = _json(bundle, app_assets.COUPLING_SUMMARY_ASSET)
        assert 0.0 - ABS <= summary["inequality_coefficient"] <= 1.0 + ABS
        assert -1.0 <= summary["spearman_rho"] <= 1.0
        assert "interpretation" not in summary  # L3 summary carries no narrative

    def test_top_lists_consistent_with_table(self, bundle):
        """Each leader's reported z_gap matches the table's z_gap for that country
        (validates the lists without depending on cross-platform sort order)."""
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        summary = _json(bundle, app_assets.COUPLING_SUMMARY_ASSET)
        z_by_country = dict(zip(table[ID_COL], table["z_gap"].to_numpy(dtype=float)))
        for key in ("top_suffer_least_cause", "top_cause_least_suffer"):
            for country, z in summary[key]:
                assert country in z_by_country, country
                assert z == pytest.approx(z_by_country[country], abs=ABS), country


# --------------------------------------------------------------------------- #
# Layer 1 physical-model trajectory + summary
# --------------------------------------------------------------------------- #
class TestPhysical:
    def test_trajectory_schema_and_contiguous(self, bundle):
        traj = _parquet(bundle, app_assets.PHYSICAL_TRAJECTORY_ASSET)
        assert list(traj.columns) == list(TRAJECTORY_SCHEMA)
        # Schema is about identity *and* type: a year drifting to float or
        # in_sample to int would pass a column-name check but break consumers.
        assert str(traj["year"].dtype).startswith("int")
        assert traj["in_sample"].dtype == bool
        years = traj["year"].to_numpy()
        assert np.all(np.diff(years) == 1)  # sorted and contiguous

    def test_band_brackets_mean(self, bundle):
        traj = _parquet(bundle, app_assets.PHYSICAL_TRAJECTORY_ASSET)
        mean = traj["predicted_mean"].to_numpy()
        lower = traj["lower95"].to_numpy()
        upper = traj["upper95"].to_numpy()
        assert np.all(lower <= mean + ABS)
        assert np.all(mean <= upper + ABS)
        # Strictly positive width: a degenerate zero-width band (e.g. a vanished
        # predictive variance) would satisfy lower <= mean <= upper but is invalid.
        assert np.all(upper > lower)

    def test_in_sample_flag_matches_train_end(self, bundle):
        traj = _parquet(bundle, app_assets.PHYSICAL_TRAJECTORY_ASSET)
        summary = _json(bundle, app_assets.PHYSICAL_SUMMARY_ASSET)
        expected = traj["year"].to_numpy() <= summary["train_end"]
        assert np.array_equal(traj["in_sample"].to_numpy().astype(bool), expected)

    def test_summary_matches_trajectory(self, bundle):
        traj = _parquet(bundle, app_assets.PHYSICAL_TRAJECTORY_ASSET)
        summary = _json(bundle, app_assets.PHYSICAL_SUMMARY_ASSET)
        assert summary["n_years"] == len(traj)
        n_test = int((traj["year"].to_numpy() > summary["train_end"]).sum())
        assert summary["hindcast"]["n_test"] == n_test

    def test_summary_carries_forcings_hash(self, bundle):
        # Provenance anchor: the committed L1 summary records the SHA-256 of the
        # forcings table it was built from (value-free -- assert shape, not value).
        summary = _json(bundle, app_assets.PHYSICAL_SUMMARY_ASSET)
        forcings_hash = summary.get("forcings_hash")
        assert isinstance(forcings_hash, str)
        assert len(forcings_hash) == 64
        assert all(c in "0123456789abcdef" for c in forcings_hash)

    def test_summary_keys_and_bounds(self, bundle):
        summary = _json(bundle, app_assets.PHYSICAL_SUMMARY_ASSET)
        assert {
            "interpretation", "outcome", "n_years", "train_end",
            "ar1_rho", "lags", "sensitivity", "hindcast",
        } <= set(summary)
        assert set(summary["sensitivity"]) == set(DRIVERS)
        assert set(summary["lags"]) == set(DRIVERS)
        assert abs(summary["ar1_rho"]) < 1.0  # stationary AR(1)
        assert 0.0 - ABS <= summary["hindcast"]["test_band_coverage"] <= 1.0 + ABS
        for sens in summary["sensitivity"].values():
            assert sens["ci_low"] <= sens["mean"] <= sens["ci_high"]
        assert summary["interpretation"].strip()


# --------------------------------------------------------------------------- #
# Determinism (in-process, same platform -> byte-identical JSON)
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_coupling_summary_recomputes_byte_stable(self, bundle):
        table = _parquet(bundle, app_assets.COUPLING_ASSET)
        proj = table[[ID_COL, RESPONSIBILITY_INDEX, IMPACT_INDEX]].copy()
        _, first = compute_coupling(proj)
        _, second = compute_coupling(proj)
        assert _coupling_summary_json(first) == _coupling_summary_json(second)
