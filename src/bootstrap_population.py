"""Explicit population audits for fixed V1 descriptive bootstrap diagnostics.

The estimator is unchanged. These helpers distinguish sampling the original
incomplete rows from sampling the fixed observed complete-country population.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import hashlib
import json

import numpy as np
import pandas as pd

from src import stability
from src.decomposition import COUNTRY_COL, OUTCOME_COL, group_lmg_shares
from src.feature_schema import SCHEMA_V1, STATUS_AVAILABLE

LEGACY_POLICY = "resample-rows-then-complete-case-v1"
COMPLETE_CASE_POLICY = "complete-case-country-bootstrap-v1"


def required_columns(design):
    groups = stability._used_features(design.columns, SCHEMA_V1, STATUS_AVAILABLE)
    return [OUTCOME_COL, *[name for names in groups.values() for name in names]]


def complete_design(design):
    return stability._complete_case(design, SCHEMA_V1, STATUS_AVAILABLE)


def population_audit(design):
    if design[COUNTRY_COL].duplicated().any():
        raise ValueError("source country identities must be unique")
    columns = required_columns(design)
    complete = complete_design(design)
    excluded = []
    for _, row in design.loc[design[columns].isna().any(axis=1)].iterrows():
        excluded.append(
            {
                "Country": str(row[COUNTRY_COL]),
                "missing_fields": [
                    column for column in columns if pd.isna(row[column])
                ],
            }
        )
    block = stability.SPATIAL_BLOCK_COL
    return {
        "sampling_rows_legacy": len(design),
        "sampling_rows_complete_case": len(complete),
        "complete_country_order": complete[COUNTRY_COL].astype(str).tolist(),
        "required_fields": columns,
        "excluded": excluded,
        "legacy_block_order": pd.unique(design[block].dropna()).tolist(),
        "complete_block_order": pd.unique(complete[block].dropna()).tolist(),
    }


def sample_draw(design, rng, *, block=False):
    """Same RNG calls, positions and row order as the original resamplers."""
    column = stability.SPATIAL_BLOCK_COL
    drawn_blocks = None
    if block and column in design.columns and len(pd.unique(design[column].dropna())):
        blocks = pd.unique(design[column].dropna())
        drawn_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        positions = np.concatenate(
            [
                np.flatnonzero(design[column].to_numpy() == value)
                for value in drawn_blocks
            ]
        )
    else:
        positions = rng.integers(0, len(design), len(design))
    return design.iloc[positions].reset_index(drop=True), positions, drawn_blocks


def draw_diagnostics(sample, positions, drawn_blocks=None):
    complete_mask = sample[required_columns(sample)].notna().all(axis=1).to_numpy()
    retained = complete_design(sample)
    identities = retained[COUNTRY_COL].astype(str).tolist()
    counts = Counter(identities)
    n = len(identities)
    block = stability.SPATIAL_BLOCK_COL
    block_counts = (
        Counter(retained[block].dropna().astype(str))
        if block in retained
        else Counter()
    )
    return {
        "sampled_n": len(sample),
        "retained_n": n,
        "unique_retained_countries": len(counts),
        "kish_multiplicity_ess": float(
            n * n / sum(value * value for value in counts.values())
        )
        if n
        else 0.0,
        "retained_block_count": len(block_counts),
        "retained_block_row_counts": dict(block_counts),
        "drawn_blocks": [str(value) for value in drawn_blocks]
        if drawn_blocks is not None
        else None,
        "source_row_positions": np.asarray(positions).tolist(),
        "ordered_retained_source_positions": np.asarray(positions)[
            complete_mask
        ].tolist(),
        "ordered_retained_countries": identities,
        "country_multiplicities": dict(counts),
        "per_retained_row_weight": 1.0,
        "per_retained_row_normalized_weight": 1 / n if n else None,
    }


def reconstruct_draws(design, *, n=2000, seed=0, block=False):
    rng = np.random.default_rng(seed)
    for draw in range(n):
        sample, positions, blocks = sample_draw(design, rng, block=block)
        yield draw, sample, draw_diagnostics(sample, positions, blocks)


def evaluate_draws(
    design, *, n=2000, seed=0, block=False, record: Callable | None = None
):
    """Fit every predeclared draw; preserve original failures without replacement."""
    shares, failures = [], 0
    for draw, sample, diagnostics in reconstruct_draws(
        design, n=n, seed=seed, block=block
    ):
        row = {"draw": draw, **diagnostics}
        try:
            fit = group_lmg_shares(sample, SCHEMA_V1, status=STATUS_AVAILABLE)
            if fit.n != diagnostics["retained_n"]:
                raise RuntimeError(
                    "estimator sample differs from recorded retained identities"
                )
            values = {**fit.shares, stability.RESIDUAL_KEY: fit.residual_share}
            shares.append(values)
            row.update(status="success", shares=values, estimator_n=fit.n)
        except (ValueError, np.linalg.LinAlgError) as error:
            failures += 1
            row.update(
                status="failed",
                failure_type=type(error).__name__,
                failure_reason=str(error),
            )
        if record is not None:
            record(row)
    return shares, failures


def block_equivalence(design, *, n=2000, seed=1, record: Callable | None = None):
    """Prove every retained input row, identity and multiplicity is exactly equal."""
    complete = complete_design(design)
    legacy = reconstruct_draws(design, n=n, seed=seed, block=True)
    conditional = reconstruct_draws(complete, n=n, seed=seed, block=True)
    identities = hashlib.sha256()
    all_equal = True
    for (draw, sample_a, a), (_, sample_b, b) in zip(legacy, conditional):
        a_frame, b_frame = complete_design(sample_a), complete_design(sample_b)
        equal = (
            a["ordered_retained_countries"] == b["ordered_retained_countries"]
            and a["country_multiplicities"] == b["country_multiplicities"]
            and a["drawn_blocks"] == b["drawn_blocks"]
        )
        try:
            pd.testing.assert_frame_equal(a_frame, b_frame, check_exact=True)
        except AssertionError:
            equal = False
        all_equal &= equal
        identities.update(
            json.dumps(
                {
                    "draw": draw,
                    "countries": a["ordered_retained_countries"],
                    "multiplicities": a["country_multiplicities"],
                },
                sort_keys=True,
            ).encode()
        )
        if record is not None:
            record(
                {
                    "draw": draw,
                    "exact_retained_inputs_identities_weights": bool(equal),
                    "legacy": a,
                    "complete_case": b,
                }
            )
    return {
        "n_verified_draws": n,
        "exact_all_retained_inputs_identities_weights": bool(all_equal),
        "ordered_identity_multiplicity_sha256": identities.hexdigest(),
        "legacy_population": len(design),
        "complete_population": len(complete),
    }


def diagnostics_summary(records):
    """Descriptive sample sizes and frequency-weight ESS, not spatial ESS."""
    out = {"n_draws": len(records)}
    for name in [
        "sampled_n",
        "retained_n",
        "unique_retained_countries",
        "kish_multiplicity_ess",
        "retained_block_count",
    ]:
        values = np.array([record[name] for record in records], dtype=float)
        out[name] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return out
