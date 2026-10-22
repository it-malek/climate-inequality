"""Synthetic contracts for explicit V1 bootstrap sampling populations."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import bootstrap_population as bp, stability
from src.decomposition import COUNTRY_COL, OUTCOME_COL
from tests.test_decomposition import make_country_design


def design():
    frame = make_country_design(n=24)
    frame.loc[0, OUTCOME_COL] = np.nan
    return frame


def test_population_audit_reports_exact_missing_field_and_complete_order():
    frame = design()
    audit = bp.population_audit(frame)
    assert audit["sampling_rows_legacy"] == 24
    assert audit["sampling_rows_complete_case"] == 23
    assert audit["excluded"] == [
        {"Country": frame.loc[0, COUNTRY_COL], "missing_fields": [OUTCOME_COL]}
    ]
    assert audit["complete_country_order"] == frame.loc[1:, COUNTRY_COL].tolist()


def test_original_country_and_block_sampler_parity_including_rng_state():
    frame = design()
    for block, original in [
        (False, stability._resample_countries),
        (True, stability._resample_blocks),
    ]:
        first, second = np.random.default_rng(1), np.random.default_rng(1)
        for _ in range(8):
            actual, _, _ = bp.sample_draw(frame, first, block=block)
            pd.testing.assert_frame_equal(actual, original(frame, second))
        assert first.random() == second.random()


def test_country_complete_case_population_fixes_retained_n_without_replacement():
    frame = design()
    legacy = [row for _, _, row in bp.reconstruct_draws(frame, n=30)]
    complete = [
        row for _, _, row in bp.reconstruct_draws(bp.complete_design(frame), n=30)
    ]
    assert len({row["retained_n"] for row in legacy}) > 1
    assert {row["retained_n"] for row in complete} == {23}
    # Replacement duplicates are retained as frequency weights.
    assert any(row["unique_retained_countries"] < 23 for row in complete)


def test_blocks_exact_when_no_complete_block_disappears():
    rows = []
    result = bp.block_equivalence(design(), n=30, record=rows.append)
    assert result["exact_all_retained_inputs_identities_weights"]
    assert len(rows) == 30
    assert all(
        row["legacy"]["ordered_retained_countries"]
        == row["complete_case"]["ordered_retained_countries"]
        for row in rows
    )


def test_block_equivalence_fails_if_missingness_removes_an_entire_block():
    frame = design()
    frame.loc[0, stability.SPATIAL_BLOCK_COL] = "only_incomplete"
    assert not bp.block_equivalence(frame, n=30)[
        "exact_all_retained_inputs_identities_weights"
    ]


def test_kish_multiplicity_ess_and_ordered_identities():
    frame = bp.complete_design(design())
    sample = frame.iloc[[0, 0, 0, 1]].reset_index(drop=True)
    record = bp.draw_diagnostics(sample, np.array([0, 0, 0, 1]))
    assert record["ordered_retained_countries"] == sample[COUNTRY_COL].tolist()
    assert record["unique_retained_countries"] == 2
    assert record["kish_multiplicity_ess"] == 1.6
    assert record["per_retained_row_weight"] == 1
    assert record["per_retained_row_normalized_weight"] == 0.25


def test_failed_fits_logged_without_replacement_or_retry(monkeypatch):
    attempted, records = [], []

    def fit(sample, *args, **kwargs):
        attempted.append(sample)
        if len(attempted) == 2:
            raise ValueError("synthetic unusable draw")
        return SimpleNamespace(
            n=len(bp.complete_design(sample)),
            shares={"geography": 0.6},
            residual_share=0.4,
        )

    monkeypatch.setattr(bp, "group_lmg_shares", fit)
    rows, failed = bp.evaluate_draws(design(), n=3, record=records.append)
    assert len(attempted) == 3
    assert len(rows) == 2 and failed == 1
    assert [row["draw"] for row in records] == [0, 1, 2]
    assert records[1]["status"] == "failed"
    assert records[1]["failure_reason"] == "synthetic unusable draw"
