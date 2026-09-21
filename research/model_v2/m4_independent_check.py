"""Independent reconstruction of the M4 final assessment from its saved evidence.

A second calculation path, written from the M4 evaluation rules and the evaluator's artifact formats. It imports nothing from ``m4_evaluate``,
``m3_evaluate``, ``m3_spatial``, ``m2_evaluate``, ``m2_conditional``, ``m2_feasibility``, ``m2_latitude_basis``,
``cv``, ``spatial``, ``run_territory_correction``, ``src.decomposition`` or ``src.stability``: NumPy, pandas, the
standard library and, for the permutation Shapley average and the paired resampling interval only,
``m3_independent_math`` (itself a second path that reads no file). Given a completed ``m4_final`` package it:

1. verifies every SHA-256 digest of ``m4_result_manifest.json`` before reading any other artifact (a mismatch or
   an unlisted artifact is a refusal) and binds the manifest's naming block and completeness flags to the record;
2. rebuilds the retained static design inputs independently -- the canonical ISO3 order and the four categorical
   labels from ``m0_countries.csv``, the exact numerics from the M0* rows of ``m1b_design_matrices.csv`` per
   representation, the Berkeley outcome from ``rank_audit_country_predictions.csv`` (``drop_per_capita``
   ``observed``) -- regenerates the rows of every draw (country: ``default_rng(0)``, ``integers(0, n, n)`` per
   draw, rows in drawn order; block: ``default_rng(1)``, ``choice`` over ``pandas.unique`` of ``spatial_block``
   in canonical order, the rows of each chosen block in canonical order, concatenated in chosen order),
   re-encodes each draw and refits all 16 group coalitions of **every** draw (all draws x 2 kinds x 2
   representations, no subset), comparing shares, row counts and the usable and rank-deficient flags;
3. recomputes every bootstrap summary (mean, SD with ddof 1, ``numpy.percentile`` 2.5/97.5 linear,
   ``p_geography_largest`` with ties to the earlier group, the 2.5th percentile of the geography margin, counts)
   from the saved draws, S's point shares from a full-sample refit, and the material-change flags;
4. recomputes the buffer-sensitivity CV R2, RMSE and MAE and the paired intervals from the saved predictions,
   and M0*'s scores against the committed M0 sensitivity cards (``m0_scorecard_territory_corrected.json``);
5. recomputes the product-check CV RMSE (and R2, MAE) of S and of each computed extension from the saved
   predictions; S's in-sample R2 and LMG shares by its own OLS on the rebuilt full-sample design with each ERA5
   outcome (aligned: ``product_stability_aligned_era5_trends.csv``; legacy: ``era5_area_trends.parquet``), the
   M0.5 ``r2.era5`` anchors when S is M0*, and the residual agreement (Pearson; Spearman as the Pearson
   correlation of average ranks; sign-agreement count) between product and Berkeley in-sample OLS residuals;
6. checks the final record against the committed M2 manifest and scorecard and the M3 station manifest, the
   naming rule (no qualifying extension -> S; one -> that extension; two -> null), the inventory status
   (I1..I27, each 'verified evidence' or 'permitted disposition') and its internal cross-references;
7. runs four negative controls on in-memory copies (never on the files) and requires each to be detected.

Per-draw design. Intercept; the six exact numerics of the representation; for each of climate_zone,
hemisphere, spatial_block and income_group, 0/1 columns for every level of the draw's own sorted levels after
the first. For M2, knots from the draw's ``abs_latitude``: minimum, ``numpy.quantile`` at 1/3 and 2/3
(``method='linear'``), maximum; the draw is degenerate when any consecutive knot gap is <= 1e-6. Then

    d_k(a) = [(a - xi_k)_+^3 - (a - xi_4)_+^3] / (xi_4 - xi_k),   N1 = d_1 - d_3,   N2 = d_2 - d_3,

and I(hemisphere == 'S') * abs_latitude, all three in geography. Groups: emissions = cum_co2_*; geography =
abs_latitude, elevation, continentality, climate_zone, hemisphere, spatial_block and the M2 columns;
socioeconomic = income_group; population = population, station_density. A group without a column in a draw
is a null player (share 0).

Coalition R2 (projection R2 tolerant of rank deficiency). For the intercept plus the columns of a coalition's
groups, the singular values s are computed (``numpy.linalg.svd``) and the numerical rank is the count of
s > s_max * max(n, p) * eps, the ``lstsq(rcond=None)`` / ``matrix_rank`` cut-off. A full-column-rank
coalition is projected by its reduced QR, fitted = Q Q'y; a rank-deficient one by the leading ``rank`` left
singular vectors, fitted = U_r U_r'y, which is the minimum-norm least-squares (pseudo-inverse) projection.
R2 = 1 - RSS/TSS with TSS about the draw's own mean. Shapley shares are averages of marginal R2 over all 24
orderings of the four groups. A draw is usable when its design builds and every share and the residual share
are finite; its rank-deficient flag is rank(full design) < columns. The same projection gives S's full-sample
points, product in-sample R2 and shares, and the residuals behind the agreement statistics.

Tolerances. Draw shares against the saved draws: 1e-9 (QR/SVD against the evaluator's lstsq per draw). Point
shares, product in-sample R2 and shares, the M0.5 anchors and residual agreement: 1e-10. Summaries,
sensitivity scores, paired intervals and product CV scores recomputed from the same round-trip parsed values:
1e-12. M0* sensitivity anchor: 1e-9. Labels, counts, flags, draw rows, identifiers, naming fields
and manifest digests: exact.

Format points resolved here (none changes a number):
* the sign-agreement count is exact except at numerically-zero in-sample residuals (|r| <= 1e-10 max|r|, e.g. the
  one country of a one-country level such as climate zone E, whose residual is +-1e-17 and whose ``numpy.sign``
  is arbitrary in either fit): the saved count must lie in [firm agreements, firm + such positions];
* the M0* anchor is required for every sensitivity cell in either branch; a non-computable M0* cell fails it
  (refusal otherwise);
* a named group with no column in a draw (e.g. a single income level) is a null player with share 0 here; the
  evaluator's share lookup would stop on such a draw, so no written package carries one;
* summaries are recomputed from the saved draws (after every saved draw is matched by the refit), points from
  the full-sample refit; the rank-deficient flag of usable draws is compared exactly.

Not independent: the inputs are the same saved artifacts and pinned tables the evaluator reads; the numerics,
the outcome, the ERA5 outcomes and every saved out-of-fold prediction are taken as given. The buffer and
product CV predictions are not refitted (the territorial and centroid memberships, M49 and random folds are
not re-derived), spatial-extension fits, theta, accounting parts, Moran statistics and ``arm_conditions`` are not
recomputed, the consolidated table and provenance are read but not re-derived, and the branch-B aligned-ERA5
anchor against the M2 conditional arm is not re-checked. Written before any M3 or M4 result existed;
executed after the result.

Run: ``uv run python -m research.model_v2.m4_independent_check [--package DIR] [--out FILE]``
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.model_v2.m3_independent_math import paired_rmse_interval, permutation_shapley

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'research' / 'model_v2' / 'outputs'
PACKAGE_DIR = OUTPUTS / 'm4_final'
OUTPUT_PATH = OUTPUTS / 'm4_final_verification' / 'm4_independent_check.json'

MANIFEST = 'm4_result_manifest.json'
ARTIFACTS = ('m4_bootstrap_draws.csv', 'm4_bootstrap_summary.json', 'm4_sensitivities.json',
             'm4_sensitivity_predictions.csv', 'm4_products.json', 'm4_product_predictions.csv',
             'm4_consolidated_table.json', 'm4_consolidated_table.csv', 'm4_inventory_status.json',
             'm4_final_specification.json', 'm4_provenance.json')

REPRESENTATIONS = ('primary_total_co2', 'per_capita')
PRIMARY_REPRESENTATION = 'primary_total_co2'
NUMERIC = {'primary_total_co2': ('cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'population',
                                 'station_density'),
           'per_capita': ('cum_co2_per_capita', 'abs_latitude', 'elevation', 'continentality', 'population',
                          'station_density')}
CATEGORICALS = ('climate_zone', 'hemisphere', 'spatial_block', 'income_group')
HEMISPHERES = ('N', 'S')
ADDED = ('abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude')
GROUP_OF = {'cum_co2_total': 'emissions', 'cum_co2_per_capita': 'emissions', 'abs_latitude': 'geography',
            'elevation': 'geography', 'continentality': 'geography', 'climate_zone': 'geography',
            'hemisphere': 'geography', 'spatial_block': 'geography', 'income_group': 'socioeconomic',
            'population': 'population', 'station_density': 'population'}
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')     # named order; ties go to the earlier
SHARE_COLUMNS = (*GROUPS, 'residual')
STATIC_MODEL = {'M0*': 'M0star', 'M2': 'M2'}
COMPARATOR = 'M0star'
EXTENSION_FAMILIES = ('SEM', 'SAR')

KINDS = ('country', 'block')
SEEDS = {'country': 0, 'block': 1}
KNOT_GAP = 1e-6
QUANTILE_PROBABILITIES = (1 / 3, 2 / 3)
MATERIAL_RESPONSIBILITY = 0.10
EPS = float(np.finfo(float).eps)

SENSITIVITY_PROTOCOLS = ('buffer_territory_1000km', 'buffer_centroid_1500km')
SENSITIVITY_ANCHOR = {'buffer_territory_1000km': 'sensitivity_territory_1000km',
                      'buffer_centroid_1500km': 'sensitivity_centroid_1500km'}
PRODUCTS = ('aligned_era5', 'legacy_era5')
ERA5_COLUMN = 'trend_c_per_decade_era5_area'
CARD_PATH = {'primary_loco': (), 'm49_subregion_lo': ('secondary',), 'random10': ('random_reference_only',)}
RESAMPLES, RESAMPLE_SEED = 2000, 0

INVENTORY_IDS = tuple(f'I{i}' for i in range(1, 28))
INVENTORY_STATUSES = ('verified evidence', 'permitted disposition')
NAMING_FIELDS = ('retained_static_specification', 'qualifying_spatial_extensions', 'final_primary_predictive_model')

# The real-data structure; synthetic tests pass their own.
EXPECTED = {'n': 151, 'n_boot': 2000, 'full_columns': {'M0*': 20, 'M2': 23}}

DRAW_SHARE_TOL = 1e-9
POINT_TOL = 1e-10
SUMMARY_TOL = 1e-12
SENSITIVITY_TOL = 1e-12
ANCHOR_TOL = 1e-9
PRODUCT_CV_TOL = 1e-12
IN_SAMPLE_TOL = 1e-10
AGREEMENT_TOL = 1e-10
CONTROL_SHIFT = 1e-6
ZERO_RESIDUAL_RELATIVE = 1e-10

TOLERANCES = {
    'bootstrap_draw_shares_vs_saved': DRAW_SHARE_TOL,
    'point_shares_full_sample_refit': POINT_TOL,
    'bootstrap_summaries_from_saved_draws': SUMMARY_TOL,
    'sensitivity_scores_and_paired_intervals': SENSITIVITY_TOL,
    'm0star_sensitivity_anchor_vs_committed_m0_cards': ANCHOR_TOL,
    'product_cv_scores_from_saved_predictions': PRODUCT_CV_TOL,
    'product_in_sample_r2_shares_and_m0_5_anchor': IN_SAMPLE_TOL,
    'product_residual_agreement': AGREEMENT_TOL,
    'product_sign_agreement_count': 'exact outside numerically-zero in-sample residuals (|r| <= 1e-10 max|r|, '
                                    'leverage-one rows such as a one-country level), whose numpy.sign is arbitrary; '
                                    'the saved count must lie in [firm, firm + zero positions]',
    'exact': 'labels, counts, draw rows, usable / degenerate / rank-deficient flags, material-change flags, '
             'observed columns, naming fields, M2 label, stopping indicator, inventory, manifest digests',
}
SCOPE = {
    'independent': [
        'the calculation path: row regeneration of every bootstrap draw, per-draw encoding and latitude knots, '
        'natural cubic columns, coalition projection R2 (QR / truncated SVD), permutation Shapley shares, '
        'rank-deficiency flags, percentile summaries and material-change flags, sensitivity and product CV scores, '
        'paired intervals, full-sample OLS in-sample R2 and shares under each ERA5 product, residual agreement, '
        'final-record consistency and the naming rule',
        'imports nothing from m4_evaluate, m3_evaluate, m3_spatial, m2_evaluate, m2_conditional, m2_feasibility, '
        'm2_latitude_basis, cv, spatial, run_territory_correction, src.decomposition or src.stability; '
        'm3_independent_math is used for permutation Shapley and the paired interval only',
    ],
    'not_independent': [
        'same inputs: the saved M4 artifacts, the pinned country label table, the committed M1b encoded design '
        '(numerics taken as given), the rank-audit outcome, the ERA5 outcome files and the committed M0, M0.5, M2 '
        'and M3 station records',
        'sensitivity and product out-of-fold predictions are taken as given: the 1000 km territorial and 1500 km '
        'centroid memberships, the M49 and random folds and the refits behind the predictions are not re-derived',
        'spatial-extension fits, theta, accounting parts, filtered shares, Moran statistics, arm_conditions and '
        'extension paired blocks are not recomputed; the consolidated table and provenance are read, not re-derived',
        'the branch-B aligned-ERA5 anchor against the committed M2 conditional arm is not re-checked',
        'written in the same session as the M4 evaluator, before any M3 or M4 result existed; executed after the '
        'result',
    ],
}


class EvidenceRefused(ValueError):
    """The saved evidence cannot be reconstructed; nothing is compared."""

    def __init__(self, message, records=()):
        super().__init__(message)
        self.records = list(records)


class DigestRefused(EvidenceRefused):
    """The saved evidence does not match its manifest."""


class DegenerateDraw(ValueError):
    """A bootstrap draw on which the design cannot be built or the shares are not finite."""


@dataclasses.dataclass(frozen=True)
class Inputs:
    """Every non-package file the check reads (defaults: the committed repository records)."""

    countries: Path = OUTPUTS / 'm0_countries.csv'
    design: Path = OUTPUTS / 'm1b_primary' / 'm1b_design_matrices.csv'
    outcome: Path = OUTPUTS / 'rank_audit_country_predictions.csv'
    m0_sensitivity: Path = OUTPUTS / 'm0_scorecard_territory_corrected.json'
    aligned_era5: Path = OUTPUTS / 'product_stability_aligned_era5_trends.csv'
    aligned_summary: Path = OUTPUTS / 'product_stability_aligned_summary.json'
    legacy_era5: Path = ROOT / 'app' / 'data' / 'era5_area_trends.parquet'
    legacy_summary: Path = OUTPUTS / 'product_stability_summary.json'
    m2_manifest: Path = OUTPUTS / 'm2_primary' / 'm2_result_manifest.json'
    m2_scorecard: Path = OUTPUTS / 'm2_primary' / 'm2_scorecard.json'
    m3_station_manifest: Path = OUTPUTS / 'm3_station' / 'm3_result_manifest.json'


# ---------------------------------------------------------------------
# Comparison records
# ---------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def numeric(name, recomputed, reported, tol) -> dict:
    """One numeric comparison; a missing, Boolean or non-finite value never passes."""
    try:
        if isinstance(recomputed, (bool, np.bool_)) or isinstance(reported, (bool, np.bool_)):
            raise TypeError('a Boolean is not a number here')
        diff = abs(float(recomputed) - float(reported))
        passes = math.isfinite(diff) and diff <= tol
    except (TypeError, ValueError):
        diff, passes = float('nan'), False
    return {'check': name, 'recomputed': _jsonable(recomputed), 'reported': _jsonable(reported),
            'abs_diff': _jsonable(diff), 'tolerance': tol, 'passes': bool(passes)}


def exact(name, recomputed, reported) -> dict:
    return {'check': name, 'recomputed': _jsonable(recomputed), 'reported': _jsonable(reported),
            'abs_diff': None, 'tolerance': 'exact', 'passes': bool(type_strict_equal(recomputed, reported))}


def type_strict_equal(a, b) -> bool:
    """Equality that does not let True == 1 or 0 == False pass, recursively through lists and dicts."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)) and bool(a) == bool(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(type_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(type_strict_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


def sequence(name, recomputed, reported) -> dict:
    """Exact comparison of a long sequence, recorded as lengths, digests and the number of differing positions."""
    recomputed, reported = list(recomputed), list(reported)
    digest = lambda values: hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:16]  # noqa: E731
    differing = [i for i, (a, b) in enumerate(itertools.zip_longest(recomputed, reported, fillvalue=object()))
                 if not type_strict_equal(a, b)]
    return {'check': name, 'recomputed': {'n': len(recomputed), 'sha256_16': digest(recomputed)},
            'reported': {'n': len(reported), 'sha256_16': digest(reported)}, 'abs_diff': None, 'tolerance': 'exact',
            'n_differing': len(differing), 'first_differing_positions': differing[:5], 'passes': not differing}


def array_close(name, recomputed, reported, tol, labels=None) -> dict:
    """Element-wise |recomputed - reported| <= tol over equal-shaped arrays, recorded as one comparison."""
    a, b = np.asarray(recomputed, dtype=float), np.asarray(reported, dtype=float)
    if a.shape != b.shape:
        return {'check': name, 'n': int(a.size), 'abs_diff': 'nan', 'tolerance': tol, 'passes': False,
                'reason': f'shape {a.shape} vs {b.shape}'}
    diff = np.abs(a - b).ravel()
    bad = ~(np.isfinite(diff) & (diff <= tol))
    finite = diff[np.isfinite(diff)]
    worst = float(finite.max()) if finite.size and finite.size == diff.size else (float('nan') if diff.size else 0.0)
    positions = np.flatnonzero(bad)[:5].tolist()
    return {'check': name, 'n': int(diff.size), 'n_failed': int(bad.sum()), 'abs_diff': _jsonable(worst),
            'tolerance': tol, 'passes': not bad.any(),
            'failed_examples': [labels[i] for i in positions] if labels is not None else positions}


def section(records) -> dict:
    diffs = [r.get('abs_diff') for r in records]
    finite = [d for d in diffs if isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d)]
    failed = [r['check'] for r in records if not r['passes']]
    return {'n_comparisons': len(records), 'n_failed': len(failed), 'max_abs_diff': max(finite, default=0.0),
            'passes': bool(records) and not failed, 'failed_checks': failed, 'comparisons': records}


def dig(node, *keys):
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


# ---------------------------------------------------------------------
# Independent mathematics
# ---------------------------------------------------------------------


def natural_cubic_columns(knots, abs_latitude) -> np.ndarray:
    """(n, 2) natural cubic columns N1 = d_1 - d_3 and N2 = d_2 - d_3 on four knots."""
    a = np.asarray(abs_latitude, dtype=float)
    xi = [float(k) for k in knots]
    tail = np.maximum(a - xi[3], 0.0) ** 3
    d = [(np.maximum(a - xi[k], 0.0) ** 3 - tail) / (xi[3] - xi[k]) for k in range(3)]
    return np.column_stack([d[0] - d[2], d[1] - d[2]])


def draw_knots(abs_latitude) -> tuple[float, float, float, float]:
    """Minimum, linear 1/3 and 2/3 quantiles, maximum; degenerate on a consecutive gap <= 1e-6."""
    a = np.asarray(abs_latitude, dtype=float)
    if a.ndim != 1 or len(a) < 4 or not np.isfinite(a).all() or (a < 0).any():
        raise DegenerateDraw(f'latitudes cannot carry four knots (n = {len(a)})')
    lower, upper = np.quantile(a, list(QUANTILE_PROBABILITIES), method='linear')
    knots = (float(a.min()), float(lower), float(upper), float(a.max()))
    if not all(b - k > KNOT_GAP for k, b in zip(knots, knots[1:])):
        raise DegenerateDraw(f'consecutive knot gap <= {KNOT_GAP}: {knots}')
    return knots


def group_of(name: str) -> str:
    if name == 'intercept':
        return 'intercept'
    if name in ADDED:
        return 'geography'
    return GROUP_OF[name.split('=')[0]]


def projection(z, y, tss) -> dict:
    """Projection of y on span(z): reduced QR at full column rank, leading left singular vectors otherwise."""
    z = np.asarray(z, dtype=float)
    singular = np.linalg.svd(z, compute_uv=False)
    cutoff = float(singular[0]) * max(z.shape) * EPS if singular.size else 0.0
    rank = int(np.count_nonzero(singular > cutoff))
    if rank == z.shape[1]:
        basis, _ = np.linalg.qr(z, mode='reduced')
    else:
        basis = np.linalg.svd(z, full_matrices=False)[0][:, :rank]
    residual = y - basis @ (basis.T @ y)
    return {'r2': 1.0 - float(residual @ residual) / tss, 'rank': rank, 'columns': int(z.shape[1]),
            'residual': residual}


def group_allocation(x, groups, y) -> dict:
    """16 coalition projection R2 values and permutation Shapley shares over the four named groups."""
    y = np.asarray(y, dtype=float)
    centred = y - y.mean()
    tss = float(centred @ centred)
    if not (math.isfinite(tss) and tss > 0):
        raise DegenerateDraw('the outcome has no variance in these rows')
    labels = np.asarray(groups)
    members = {g: labels == g for g in GROUPS}
    cache: dict[frozenset, dict] = {}

    def fit(coalition) -> dict:
        key = frozenset(coalition)
        if key not in cache:
            keep = labels == 'intercept'
            for g in key:
                keep = keep | members[g]
            cache[key] = projection(x[:, keep], y, tss)
        return cache[key]

    shares = permutation_shapley(GROUPS, lambda coalition: fit(coalition)['r2'])
    full = fit(GROUPS)
    return {'shares': shares, 'r2': full['r2'], 'residual_share': 1.0 - full['r2'], 'rank': full['rank'],
            'columns': full['columns'], 'residual': full['residual']}


def average_ranks(values) -> np.ndarray:
    """1-based ranks with tied values sharing the mean of their positions."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - math.fsum(a.tolist()) / len(a), b - math.fsum(b.tolist()) / len(b)
    return math.fsum((a * b).tolist()) / math.sqrt(math.fsum((a * a).tolist()) * math.fsum((b * b).tolist()))


def sign_agreement(name, a, b, reported) -> dict:
    """Sign-agreement count, exact outside numerically-zero residual positions.

    A row that is the only member of a categorical level (leverage one) has an in-sample residual that is zero up to
    rounding (about 1e-17), so its ``numpy.sign`` in either fit is arbitrary. Such positions (|r| <= 1e-10 max|r|)
    are counted separately: the saved count must lie between the firm agreements and the firm agreements plus the
    number of numerically-zero positions; everywhere else the comparison is exact.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    zero = (np.abs(a) <= ZERO_RESIDUAL_RELATIVE * np.max(np.abs(a))) | (np.abs(b) <= ZERO_RESIDUAL_RELATIVE
                                                                         * np.max(np.abs(b)))
    firm = int(np.sum((np.sign(a) == np.sign(b)) & ~zero))
    ambiguous = int(zero.sum())
    valid = isinstance(reported, int) and not isinstance(reported, bool)
    return {'check': name, 'recomputed': {'firm_agreements': firm, 'numerically_zero_positions': ambiguous,
                                          'numpy_sign_count': int(np.sum(np.sign(a) == np.sign(b)))},
            'reported': _jsonable(reported), 'abs_diff': None,
            'tolerance': 'exact outside numerically-zero residual positions',
            'passes': bool(valid and firm <= reported <= firm + ambiguous)}


def scores(observed, prediction) -> tuple[float, float, float]:
    """R2 = 1 - SSE/SST, RMSE and MAE."""
    y = np.asarray(observed, dtype=float)
    e = y - np.asarray(prediction, dtype=float)
    centred = y - math.fsum(y.tolist()) / len(y)
    sse = math.fsum((e * e).tolist())
    return (1.0 - sse / math.fsum((centred * centred).tolist()), math.sqrt(sse / len(e)),
            math.fsum(np.abs(e).tolist()) / len(e))


def naming_rule(static, qualifying):
    """None qualifying -> the static specification; one -> that extension; two -> null (no selection rule)."""
    if not qualifying:
        return static
    return qualifying[0] if len(qualifying) == 1 else None


# ---------------------------------------------------------------------
# Evidence and inputs
# ---------------------------------------------------------------------


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, text=()) -> pd.DataFrame:
    """Exact float64 round trip; text columns kept as strings, no NA inference."""
    try:
        return pd.read_csv(path, float_precision='round_trip', keep_default_na=False,
                           dtype=dict.fromkeys(text, str))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(text) + ['observed', 'prediction'])


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, float_precision='round_trip', dtype={'iso3': str}, keep_default_na=False, na_values=[''])


def verify_manifest(package) -> tuple[dict, list]:
    """Every manifest digest against the bytes on disk, before anything else is read."""
    package = Path(package)
    path = package / MANIFEST
    if not path.is_file():
        raise DigestRefused(f'{path} does not exist')
    manifest = json.loads(path.read_text())
    recorded = manifest.get('artifact_sha256', {}) if isinstance(manifest, dict) else {}
    records = [exact(f'manifest/{name}/listed', name in recorded, True) for name in ARTIFACTS]
    for name, digest in sorted(recorded.items()):
        target = package / name
        inside = target.resolve().parent == package.resolve()
        actual = sha256(target) if inside and target.is_file() else 'missing'
        records.append(exact(f'manifest/{name}/sha256', actual, digest))
    failed = [r['check'] for r in records if not r['passes']]
    if failed:
        raise DigestRefused(f'manifest verification failed: {failed}', records)
    return manifest, records


BOOLEAN_TEXT = {'True': True, 'False': False}
DRAW_TEXT = ('representation', 'model', 'bootstrap', 'draw', 'n_rows', 'usable', 'failure', 'rank_deficient',
             *SHARE_COLUMNS)


def parse_draws(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed draws from all-text columns: ints, Booleans (None where blank or unparseable) and float shares."""
    missing = [c for c in DRAW_TEXT if c not in raw.columns]
    if missing:
        raise EvidenceRefused(f'm4_bootstrap_draws.csv lacks columns {missing}')

    def integer(text):
        try:
            return int(text)
        except ValueError:
            return None

    out = pd.DataFrame({c: raw[c].astype(str).tolist() for c in ('representation', 'model', 'bootstrap', 'failure')})
    out['draw'] = [integer(v) for v in raw['draw']]
    out['n_rows'] = [integer(v) for v in raw['n_rows']]
    out['usable'] = pd.Series([BOOLEAN_TEXT.get(v) for v in raw['usable']], dtype=object)
    out['rank_deficient'] = pd.Series([BOOLEAN_TEXT.get(v) for v in raw['rank_deficient']], dtype=object)
    for g in SHARE_COLUMNS:
        out[g] = [float(v) if v != '' else math.nan for v in raw[g]]
    return out


@dataclasses.dataclass(frozen=True)
class Evidence:
    manifest: dict
    manifest_records: list
    draws: pd.DataFrame
    summary: dict
    sensitivities: dict
    sensitivity_predictions: pd.DataFrame
    products: dict
    product_predictions: pd.DataFrame
    consolidated: dict
    consolidated_rows: pd.DataFrame
    inventory: dict
    final: dict
    provenance: dict


def load_evidence(package=PACKAGE_DIR) -> Evidence:
    package = Path(package)
    manifest, records = verify_manifest(package)
    load = lambda name: json.loads((package / name).read_text())  # noqa: E731
    return Evidence(
        manifest=manifest, manifest_records=records,
        draws=parse_draws(pd.read_csv(package / 'm4_bootstrap_draws.csv', dtype=str, keep_default_na=False)),
        summary=load('m4_bootstrap_summary.json'), sensitivities=load('m4_sensitivities.json'),
        sensitivity_predictions=read_csv(package / 'm4_sensitivity_predictions.csv',
                                         text=('protocol', 'representation', 'model', 'iso3')),
        products=load('m4_products.json'),
        product_predictions=read_csv(package / 'm4_product_predictions.csv',
                                     text=('product', 'representation', 'model', 'protocol', 'iso3')),
        consolidated=load('m4_consolidated_table.json'),
        consolidated_rows=pd.read_csv(package / 'm4_consolidated_table.csv', dtype=str, keep_default_na=False),
        inventory=load('m4_inventory_status.json'), final=load('m4_final_specification.json'),
        provenance=load('m4_provenance.json'))


@dataclasses.dataclass(frozen=True)
class Context:
    order: list
    labels: dict            # categorical -> str array in canonical order
    numerics: dict          # representation -> (n, 6) exact numerics in canonical order
    frozen_design: dict     # representation -> full-sample M0* design pivot (canonical order)
    y: np.ndarray           # Berkeley outcome
    products: dict          # product -> outcome in canonical order (NaN where missing)
    product_anchor: dict    # product -> M0.5 r2.era5
    m0_cards: dict          # sensitivity protocol -> committed M0 card
    m2_manifest: dict
    m2_scorecard: dict
    station_manifest: dict
    input_sha256: dict


def load_context(inputs: Inputs) -> Context:
    labels = pd.read_csv(inputs.countries, usecols=['iso3', *CATEGORICALS], dtype=str, keep_default_na=False)
    order = labels.iso3.tolist()
    design = read_csv(inputs.design, text=('representation', 'model', 'iso3', 'column', 'group'))
    numerics, frozen = {}, {}
    for representation in REPRESENTATIONS:
        rows = design[(design.representation == representation) & (design.model == COMPARATOR)]
        pivot = rows.pivot(index='iso3', columns='column', values='value').reindex(index=order)
        frozen[representation] = pivot
        numerics[representation] = pivot.reindex(columns=list(NUMERIC[representation])).to_numpy(dtype=float)
    outcome = read_csv(inputs.outcome, text=('variant', 'iso3', 'Country'))
    y = (outcome[outcome.variant == 'drop_per_capita'].set_index('iso3').reindex(order)['observed']
         .to_numpy(dtype=float))
    products, anchors = {}, {}
    for product, table_path, summary_path in (('aligned_era5', inputs.aligned_era5, inputs.aligned_summary),
                                              ('legacy_era5', inputs.legacy_era5, inputs.legacy_summary)):
        table = read_table(table_path)
        products[product] = pd.to_numeric(table.set_index('iso3').reindex(order)[ERA5_COLUMN]).to_numpy(dtype=float)
        anchors[product] = dig(json.loads(Path(summary_path).read_text()), 'r2', 'era5')
    m0 = json.loads(Path(inputs.m0_sensitivity).read_text())
    station_path = Path(inputs.m3_station_manifest)
    return Context(
        order=order, labels={c: labels[c].to_numpy(dtype=object) for c in CATEGORICALS}, numerics=numerics,
        frozen_design=frozen, y=y, products=products, product_anchor=anchors,
        m0_cards={p: m0.get(SENSITIVITY_ANCHOR[p]) for p in SENSITIVITY_PROTOCOLS},
        m2_manifest=json.loads(Path(inputs.m2_manifest).read_text()),
        m2_scorecard=json.loads(Path(inputs.m2_scorecard).read_text()),
        station_manifest=json.loads(station_path.read_text()) if station_path.is_file() else {},
        input_sha256={field.name: (sha256(getattr(inputs, field.name)) if Path(getattr(inputs, field.name)).is_file()
                                   else 'missing') for field in dataclasses.fields(inputs)})


# ---------------------------------------------------------------------
# Static share uncertainty: every draw recomputed
# ---------------------------------------------------------------------


def design_for_rows(context, representation, static, rows) -> tuple[np.ndarray, list[str], list[str]]:
    """The S design on the given rows, encoded from those rows alone (levels, and knots for M2)."""
    rows = np.asarray(rows, dtype=int)
    numerics = context.numerics[representation][rows]
    columns = [np.ones(len(rows))] + [numerics[:, j] for j in range(numerics.shape[1])]
    names = ['intercept', *NUMERIC[representation]]
    for c in CATEGORICALS:
        values = context.labels[c][rows]
        for level in sorted({str(v) for v in values})[1:]:
            columns.append((values == level).astype(float))
            names.append(f'{c}={level}')
    if static == 'M2':
        a = numerics[:, NUMERIC[representation].index('abs_latitude')]
        knots = draw_knots(a)
        cubic = natural_cubic_columns(knots, a)
        columns += [cubic[:, 0], cubic[:, 1], (context.labels['hemisphere'][rows] == 'S').astype(float) * a]
        names += list(ADDED)
    return np.column_stack(columns), names, [group_of(n) for n in names]


def draw_row_sets(context, kind, n_boot) -> list[np.ndarray]:
    """The regenerated rows of every draw of one bootstrap kind."""
    n = len(context.order)
    rng = np.random.default_rng(SEEDS[kind])
    if kind == 'country':
        return [rng.integers(0, n, n) for _ in range(n_boot)]
    labels = pd.Series(list(context.labels['spatial_block']))
    blocks = pd.unique(labels)
    members = {str(block): np.flatnonzero(labels.to_numpy() == block) for block in blocks}
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        out.append(np.concatenate([members[str(block)] for block in chosen]))
    return out


def recompute_draw(context, representation, static, rows) -> dict:
    try:
        x, _, groups = design_for_rows(context, representation, static, rows)
        allocation = group_allocation(x, groups, context.y[rows])
        values = [allocation['shares'][g] for g in GROUPS] + [allocation['residual_share']]
        if not all(math.isfinite(v) for v in values):
            raise DegenerateDraw('non-finite share')
    except (DegenerateDraw, np.linalg.LinAlgError) as error:
        return {'n_rows': len(rows), 'usable': False, 'rank_deficient': None, 'failure': str(error),
                'shares': [math.nan] * len(SHARE_COLUMNS)}
    return {'n_rows': len(rows), 'usable': True, 'rank_deficient': bool(allocation['rank'] < allocation['columns']),
            'failure': '', 'shares': values}


def recompute_bootstrap(context, static, n_boot) -> dict:
    out = {}
    for representation in REPRESENTATIONS:
        for kind in KINDS:
            draws = [recompute_draw(context, representation, static, rows)
                     for rows in draw_row_sets(context, kind, n_boot)]
            out[(representation, kind)] = {
                'n_rows': [d['n_rows'] for d in draws], 'usable': [d['usable'] for d in draws],
                'rank_deficient': [d['rank_deficient'] for d in draws],
                'failure': [d['failure'] for d in draws],
                'shares': np.array([d['shares'] for d in draws], dtype=float).reshape(len(draws), len(SHARE_COLUMNS))}
    return out


def full_sample_points(context, static) -> dict:
    points = {}
    for representation in REPRESENTATIONS:
        x, _, groups = design_for_rows(context, representation, static, np.arange(len(context.order)))
        allocation = group_allocation(x, groups, context.y)
        points[representation] = {**allocation['shares'], 'residual': allocation['residual_share']}
    return points


def draws_of(draws, representation, kind) -> pd.DataFrame:
    return draws[(draws.representation == representation) & (draws.bootstrap == kind)]


def input_records(context, static, expected) -> list[dict]:
    records = [exact('inputs/n_countries', len(context.order), expected['n']),
               exact('inputs/iso3_unique', len(set(context.order)) == len(context.order), True),
               exact('inputs/labels_non_empty', int(sum((context.labels[c] == '').sum() for c in CATEGORICALS)), 0),
               exact('inputs/hemisphere_levels_within_N_S', set(context.labels['hemisphere'].tolist()) <= set(HEMISPHERES),
                     True),
               exact('inputs/outcome_finite', bool(np.isfinite(context.y).all()), True)]
    full = np.arange(len(context.order))
    for representation in REPRESENTATIONS:
        stem = f'inputs/{representation}'
        records.append(exact(f'{stem}/numerics_finite', bool(np.isfinite(context.numerics[representation]).all()), True))
        pivot = context.frozen_design[representation]
        x, names, _ = design_for_rows(context, representation, 'M0*', full)
        records.append(exact(f'{stem}/frozen_m0star_column_set', sorted(pivot.columns), sorted(names)))
        if set(pivot.columns) == set(names):
            records.append(array_close(f'{stem}/frozen_m0star_design_rebuilt_from_labels', x,
                                       pivot.reindex(columns=names).to_numpy(dtype=float), 0.0))
        try:
            xs, names_s, _ = design_for_rows(context, representation, static, full)
            singular = np.linalg.svd(xs, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * max(xs.shape) * EPS))
            records.append(exact(f'{stem}/S_full_sample_columns', len(names_s), expected['full_columns'][static]))
            records.append(exact(f'{stem}/S_full_sample_rank', rank, expected['full_columns'][static]))
        except DegenerateDraw as error:
            records.append(exact(f'{stem}/S_full_sample_design', str(error), 'buildable'))
    return records


def bootstrap_records(evidence, context, recomputed, static, n_boot) -> list[dict]:
    draws = evidence.draws
    records = [exact('draws/representations', sorted(set(draws.representation)), sorted(REPRESENTATIONS)),
               exact('draws/kinds', sorted(set(draws.bootstrap)), sorted(KINDS)),
               exact('draws/model_is_S', sorted(set(draws.model)), [STATIC_MODEL[static]])]
    for representation in REPRESENTATIONS:
        for kind in KINDS:
            saved, mine = draws_of(draws, representation, kind), recomputed[(representation, kind)]
            stem = f'{representation}/{kind}'
            records += [sequence(f'{stem}/draw_index', saved.draw.tolist(), list(range(n_boot))),
                        sequence(f'{stem}/n_rows', saved.n_rows.tolist(), mine['n_rows']),
                        sequence(f'{stem}/usable', saved.usable.tolist(), mine['usable']),
                        sequence(f'{stem}/rank_deficient', saved.rank_deficient.tolist(), mine['rank_deficient']),
                        exact(f'{stem}/failure_text_iff_unusable',
                              int(sum((f != '') == (u is True) for f, u in zip(saved.failure, saved.usable))), 0)]
            if len(saved) != n_boot:
                continue
            usable = np.array([u is True for u in saved.usable]) & np.array(mine['usable'], dtype=bool)
            index = np.flatnonzero(usable)
            labels = [f'draw {i}' for i in index]
            for j, g in enumerate(SHARE_COLUMNS):
                records.append(array_close(f'{stem}/shares/{g}', mine['shares'][index, j],
                                           saved[g].to_numpy(dtype=float)[index], DRAW_SHARE_TOL, labels))
            unusable = saved[[u is not True for u in saved.usable]]
            records.append(exact(f'{stem}/unusable_draws_carry_no_share',
                                 int(np.isfinite(unusable[list(SHARE_COLUMNS)].to_numpy(dtype=float)).sum()), 0))
    return records


# ---------------------------------------------------------------------
# Bootstrap summaries and material change, from the saved draws
# ---------------------------------------------------------------------


def summary_records(evidence, context, points, static) -> list[dict]:
    records, margins = [], {}
    for representation in REPRESENTATIONS:
        block = evidence.summary.get(representation, {})
        point = points[representation]
        for kind in KINDS:
            saved = draws_of(evidence.draws, representation, kind)
            usable = saved[[u is True for u in saved.usable]]
            node = dig(block, kind) or {}
            stem = f'{representation}/{kind}'
            records += [exact(f'{stem}/n_draws', int(len(saved)), node.get('n_draws')),
                        exact(f'{stem}/usable', int(len(usable)), node.get('usable')),
                        exact(f'{stem}/degenerate', int(len(saved) - len(usable)), node.get('degenerate')),
                        exact(f'{stem}/rank_deficient_usable', int(sum(v is True for v in usable.rank_deficient)),
                              node.get('rank_deficient_usable')),
                        exact(f'{stem}/seed', SEEDS[kind], node.get('seed'))]
            for g in SHARE_COLUMNS:
                values = usable[g].to_numpy(dtype=float)
                reported = dig(node, 'groups', g) or {}
                records.append(numeric(f'{stem}/{g}/point_vs_full_sample_refit', point[g], reported.get('point'),
                                       POINT_TOL))
                if len(values) < 2:
                    records.append(exact(f'{stem}/{g}/at_least_two_usable_draws', len(values), '>= 2'))
                    continue
                mean = math.fsum(values.tolist()) / len(values)
                sd = math.sqrt(math.fsum(((values - mean) ** 2).tolist()) / (len(values) - 1))
                low, high = np.percentile(values, [2.5, 97.5])
                records += [numeric(f'{stem}/{g}/mean', mean, reported.get('mean'), SUMMARY_TOL),
                            numeric(f'{stem}/{g}/sd', sd, reported.get('sd'), SUMMARY_TOL),
                            numeric(f'{stem}/{g}/ci_low', float(low), reported.get('ci_low'), SUMMARY_TOL),
                            numeric(f'{stem}/{g}/ci_high', float(high), reported.get('ci_high'), SUMMARY_TOL)]
            if len(usable) == 0:
                margins[(representation, kind)] = math.nan
                continue
            share = {g: usable[g].to_numpy(dtype=float) for g in GROUPS}
            largest = ((share['geography'] > share['emissions']) & (share['geography'] >= share['socioeconomic'])
                       & (share['geography'] >= share['population']))
            margin = share['geography'] - np.maximum.reduce([share['emissions'], share['socioeconomic'],
                                                             share['population']])
            margins[(representation, kind)] = float(np.percentile(margin, 2.5))
            records += [numeric(f'{stem}/p_geography_largest', int(largest.sum()) / len(largest),
                                node.get('p_geography_largest'), SUMMARY_TOL),
                        numeric(f'{stem}/geography_margin_percentile_2_5', margins[(representation, kind)],
                                node.get('geography_margin_percentile_2_5'), SUMMARY_TOL)]
        country_emissions = draws_of(evidence.draws, representation, 'country')
        emissions = country_emissions[[u is True for u in country_emissions.usable]].emissions.to_numpy(dtype=float)
        upper = float(np.percentile(emissions, 97.5)) if len(emissions) else math.nan
        flags = {
            'point_geography_not_largest': not all(point['geography'] > point[g] for g in GROUPS if g != 'geography'),
            'interval_geography_not_established_country': bool(margins[(representation, 'country')] <= 0),
            'interval_geography_not_established_block': bool(margins[(representation, 'block')] <= 0),
            'responsibility_exceeds_0_10': bool(point['emissions'] > MATERIAL_RESPONSIBILITY),
            'responsibility_upper_country_ci_exceeds_0_10': bool(upper > MATERIAL_RESPONSIBILITY),
        }
        flags['material_change_to_v1_conclusion'] = bool(flags['point_geography_not_largest']
                                                         or flags['interval_geography_not_established_country']
                                                         or flags['responsibility_exceeds_0_10'])
        saved_flags = dig(block, 'material_change') or {}
        records.append(exact(f'{representation}/material_change/keys', sorted(saved_flags), sorted(flags)))
        records += [exact(f'{representation}/material_change/{k}', v, saved_flags.get(k)) for k, v in flags.items()]
    records.append(exact('summary/representations', sorted(evidence.summary), sorted(REPRESENTATIONS)))
    return records


# ---------------------------------------------------------------------
# Buffer sensitivities, from the saved predictions
# ---------------------------------------------------------------------


def ordered_predictions(frame, order, **match) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in match.items():
        mask &= (frame[column] == value).to_numpy()
    return frame[mask]


def prediction_vectors(rows, order) -> tuple[bool, np.ndarray, np.ndarray]:
    complete = len(rows) == len(order) and sorted(rows.iso3) == sorted(order)
    indexed = rows.drop_duplicates('iso3').set_index('iso3').reindex(order)
    return (complete, pd.to_numeric(indexed.observed).to_numpy(dtype=float),
            pd.to_numeric(indexed.prediction).to_numpy(dtype=float))


def sensitivity_records(evidence, context, static, qualifying) -> list[dict]:
    static_model = STATIC_MODEL[static]
    models = [static_model] + ([COMPARATOR] if static_model != COMPARATOR else []) + list(qualifying)
    predictions, saved = evidence.sensitivity_predictions, evidence.sensitivities
    expected_entries = sorted(f'{p}/{r}' for p in SENSITIVITY_PROTOCOLS for r in REPRESENTATIONS)
    records = [exact('sensitivity/entries', sorted(k for k in saved if k != 'scope'), expected_entries)]
    seen = set()
    for protocol in SENSITIVITY_PROTOCOLS:
        for representation in REPRESENTATIONS:
            entry = saved.get(f'{protocol}/{representation}', {})
            stem = f'{protocol}/{representation}'
            records.append(exact(f'{stem}/models', sorted(k for k in entry if not k.startswith('paired_')), sorted(models)))
            computed = {}
            for model in models:
                node = entry.get(model) or {}
                rows = ordered_predictions(predictions, context.order, protocol=protocol,
                                           representation=representation, model=model)
                seen.add((protocol, representation, model))
                if node.get('status') != 'computed':
                    records += [exact(f'{stem}/{model}/status', node.get('status'), 'non_computable'),
                                exact(f'{stem}/{model}/no_predictions_when_not_computed', len(rows), 0)]
                    if model == COMPARATOR:
                        records.append(exact(f'{stem}/{model}/anchor_requires_computed_m0star', node.get('status'),
                                             'computed'))
                    continue
                complete, observed, prediction = prediction_vectors(rows, context.order)
                records += [exact(f'{stem}/{model}/one_prediction_per_country', complete, True),
                            array_close(f'{stem}/{model}/observed_is_berkeley_outcome', observed, context.y, 0.0)]
                r2, rmse, mae = scores(context.y, prediction)
                records += [numeric(f'{stem}/{model}/cv_r2', r2, node.get('cv_r2'), SENSITIVITY_TOL),
                            numeric(f'{stem}/{model}/cv_rmse', rmse, node.get('cv_rmse'), SENSITIVITY_TOL),
                            numeric(f'{stem}/{model}/cv_mae', mae, node.get('cv_mae'), SENSITIVITY_TOL)]
                computed[model] = prediction
                if model == COMPARATOR:
                    card = context.m0_cards.get(protocol) or {}
                    for metric, value in (('cv_r2', r2), ('cv_rmse', rmse), ('cv_mae', mae)):
                        records.append(numeric(f'{stem}/M0star_anchor_vs_committed_m0_card/{metric}', value,
                                               card.get(metric), ANCHOR_TOL))
            pairs = {}
            if static_model != COMPARATOR and static_model in computed and COMPARATOR in computed:
                pairs['paired_S_minus_M0star'] = (static_model, COMPARATOR)
            for name in qualifying:
                if name in computed and static_model in computed:
                    pairs[f'paired_{name}_minus_S'] = (name, static_model)
            records.append(exact(f'{stem}/paired_blocks', sorted(k for k in entry if k.startswith('paired_')),
                                 sorted(pairs)))
            for key, (new, reference) in pairs.items():
                interval = paired_rmse_interval(context.y, computed[new], computed[reference])
                node = entry.get(key) or {}
                reported = node.get('country_bootstrap_95_interval') or [None, None]
                records += [numeric(f'{stem}/{key}/delta_rmse', interval['delta_rmse'], node.get('delta_rmse'),
                                    SENSITIVITY_TOL),
                            numeric(f'{stem}/{key}/interval_low', interval['interval'][0], reported[0], SENSITIVITY_TOL),
                            numeric(f'{stem}/{key}/interval_high', interval['interval'][-1], reported[-1],
                                    SENSITIVITY_TOL),
                            exact(f'{stem}/{key}/resamples', RESAMPLES, node.get('resamples')),
                            exact(f'{stem}/{key}/seed', RESAMPLE_SEED, node.get('seed'))]
    combos = set(zip(predictions.protocol, predictions.representation, predictions.model))
    records.append(exact('sensitivity/no_unexpected_prediction_rows', sorted(combos - seen), []))
    return records


# ---------------------------------------------------------------------
# Product check, from the saved predictions and an own OLS
# ---------------------------------------------------------------------


def product_records(evidence, context, static, qualifying) -> list[dict]:
    static_model = STATIC_MODEL[static]
    predictions, saved = evidence.product_predictions, evidence.products
    full = np.arange(len(context.order))
    records, berkeley = [], {}
    for product in PRODUCTS:
        y = context.products[product]
        finite = bool(np.isfinite(y).all())
        if not finite:
            records += [exact(f'{product}/non_computable', dig(saved, product, 'status'), 'non_computable'),
                        exact(f'{product}/no_predictions', int((predictions['product'] == product).sum()), 0)]
            continue
        records.append(exact(f'{product}/not_marked_non_computable', product in saved, False))
        for representation in REPRESENTATIONS:
            stem = f'{product}/{representation}'
            entry = saved.get(stem) or {}
            s = entry.get('S') or {}
            card = s.get('card') or {}
            records.append(exact(f'{stem}/S/model', s.get('model'), static_model))
            for protocol, path in CARD_PATH.items():
                rows = ordered_predictions(predictions, context.order, product=product, representation=representation,
                                           model=static_model, protocol=protocol)
                complete, observed, prediction = prediction_vectors(rows, context.order)
                node = dig(card, *path) if path else card
                node = node if isinstance(node, dict) else {}
                r2, rmse, mae = scores(y, prediction)
                records += [exact(f'{stem}/S/{protocol}/one_prediction_per_country', complete, True),
                            array_close(f'{stem}/S/{protocol}/observed_is_product_outcome', observed, y, 0.0),
                            numeric(f'{stem}/S/{protocol}/cv_rmse', rmse, node.get('cv_rmse'), PRODUCT_CV_TOL),
                            numeric(f'{stem}/S/{protocol}/cv_r2', r2, node.get('cv_r2'), PRODUCT_CV_TOL),
                            numeric(f'{stem}/S/{protocol}/cv_mae', mae, node.get('cv_mae'), PRODUCT_CV_TOL)]
            x, _, groups = design_for_rows(context, representation, static, full)
            fit = group_allocation(x, groups, y)
            if representation not in berkeley:
                berkeley[representation] = group_allocation(x, groups, context.y)
            records += [exact(f'{stem}/S/full_design_full_rank', fit['rank'], fit['columns']),
                        numeric(f'{stem}/S/in_sample_r2', fit['r2'], card.get('in_sample_r2'), IN_SAMPLE_TOL),
                        numeric(f'{stem}/S/residual_share', fit['residual_share'], card.get('residual_share'),
                                IN_SAMPLE_TOL)]
            records += [numeric(f'{stem}/S/share_{g}', fit['shares'][g], card.get(f'share_{g}'), IN_SAMPLE_TOL)
                        for g in GROUPS]
            if static == 'M0*':
                records.append(numeric(f'{stem}/S/in_sample_r2_vs_m0_5_r2_era5', fit['r2'],
                                       context.product_anchor[product], IN_SAMPLE_TOL))
            shares = fit['shares']
            flags = {'point_geography_not_largest': not all(shares['geography'] > shares[g] for g in GROUPS
                                                            if g != 'geography'),
                     'responsibility_exceeds_0_10': bool(shares['emissions'] > MATERIAL_RESPONSIBILITY)}
            records.append(exact(f'{stem}/S/material_change_point', flags, s.get('material_change_point')))
            product_residual, berkeley_residual = fit['residual'], berkeley[representation]['residual']
            agreement = s.get('residual_agreement_with_berkeley') or {}
            records += [numeric(f'{stem}/S/residual_agreement/pearson', pearson(product_residual, berkeley_residual),
                                agreement.get('pearson'), AGREEMENT_TOL),
                        numeric(f'{stem}/S/residual_agreement/spearman',
                                pearson(average_ranks(product_residual), average_ranks(berkeley_residual)),
                                agreement.get('spearman'), AGREEMENT_TOL),
                        sign_agreement(f'{stem}/S/residual_agreement/sign_agreement_n', product_residual,
                                       berkeley_residual, agreement.get('sign_agreement_n'))]
            for name in qualifying:
                node = entry.get(name) or {}
                if node.get('status') != 'computed':
                    records.append(exact(f'{stem}/{name}/no_predictions_when_not_computed',
                                         len(ordered_predictions(predictions, context.order, product=product,
                                                                 representation=representation, model=name)), 0))
                    continue
                for protocol, path in CARD_PATH.items():
                    rows = ordered_predictions(predictions, context.order, product=product,
                                               representation=representation, model=name, protocol=protocol)
                    complete, observed, prediction = prediction_vectors(rows, context.order)
                    reported = dig(node, 'card', *path)
                    reported = reported if isinstance(reported, dict) else {}
                    records += [exact(f'{stem}/{name}/{protocol}/one_prediction_per_country', complete, True),
                                array_close(f'{stem}/{name}/{protocol}/observed_is_product_outcome', observed, y, 0.0),
                                numeric(f'{stem}/{name}/{protocol}/cv_rmse', scores(y, prediction)[1],
                                        reported.get('cv_rmse'), PRODUCT_CV_TOL)]
    for name in qualifying:
        arm = dig(saved, f'aligned_era5/{PRIMARY_REPRESENTATION}', name) or {}
        expected = None if arm.get('status') != 'computed' else not dig(arm, 'arm_conditions', 'qualifies')
        records.append(exact(f'product_sensitive/{name}', expected, saved.get(f'product_sensitive/{name}', 'missing')))
    return records


# ---------------------------------------------------------------------
# The final record
# ---------------------------------------------------------------------


def final_records(evidence, context, points) -> list[dict]:
    final, manifest = evidence.final, evidence.manifest
    static = final.get('retained_static_specification')
    qualifying = final.get('qualifying_spatial_extensions')
    naming = context.station_manifest.get('final_naming') or {}
    label = dig(context.m2_scorecard, 'verdict', 'verdict')
    records = [
        exact('final/retained_static_specification_vs_m2_manifest', static,
              context.m2_manifest.get('retained_static_specification')),
        exact('final/m2_manifest_static_consistent_with_label', context.m2_manifest.get('retained_static_specification'),
              'M2' if label == 'supported and promoted' else 'M0*'),
        exact('final/qualifying_spatial_extensions_vs_m3_station_naming', qualifying,
              naming.get('qualifying_spatial_extensions')),
        exact('final/final_primary_predictive_model_vs_m3_station_naming', final.get('final_primary_predictive_model'),
              naming.get('final_primary_predictive_model')),
        exact('final/m3_station_naming_static_vs_m2_manifest', naming.get('retained_static_specification'),
              context.m2_manifest.get('retained_static_specification')),
        exact('final/qualifying_is_a_list_of_distinct_registered_names',
              isinstance(qualifying, list) and len(set(qualifying)) == len(qualifying)
              and all(q in [f'M3-{f}({static})' for f in EXTENSION_FAMILIES] for q in qualifying), True),
        exact('final/naming_rule', naming_rule(static, qualifying if isinstance(qualifying, list) else []),
              final.get('final_primary_predictive_model')),
        exact('final/m2_label_vs_m2_scorecard', final.get('m2_label'), label),
        exact('final/static_stopping_indicator_vs_m2_scorecard', final.get('static_stopping_indicator'),
              context.m2_scorecard.get('static_stopping_indicator')),
        exact('final/manifest_naming_block', manifest.get('final'), {k: final.get(k) for k in NAMING_FIELDS}),
        exact('final/manifest_inventory_complete', manifest.get('inventory_complete'), True),
        exact('final/manifest_integrity_passed', manifest.get('integrity_passed'), True),
        exact('final/material_change_vs_bootstrap_summary', final.get('material_change_to_v1_conclusion'),
              {r: dig(evidence.summary, r, 'material_change') for r in REPRESENTATIONS}),
    ]
    items = evidence.inventory.get('items') or []
    records += [
        exact('inventory/ids', [i.get('id') for i in items], list(INVENTORY_IDS)),
        exact('inventory/statuses_permitted', [i.get('id') for i in items if i.get('status') not in INVENTORY_STATUSES],
              []),
        exact('inventory/retained_static_specification', evidence.inventory.get('retained_static_specification'), static),
        exact('inventory/qualifying_spatial_extensions', evidence.inventory.get('qualifying_spatial_extensions'),
              qualifying),
    ]
    three = dig(final, 'three_numbers_per_retained_stage') or {}
    records.append(exact('final/three_numbers_stages', sorted(three),
                         sorted([static, *(qualifying if isinstance(qualifying, list) else [])])))
    geography = dig(evidence.summary, PRIMARY_REPRESENTATION, 'country', 'groups', 'geography') or {}
    stage = three.get(static) or {}
    records += [exact('final/three_numbers/S/geography_share_vs_summary', stage.get('geography_share'),
                      geography.get('point')),
                exact('final/three_numbers/S/geography_interval_vs_summary',
                      stage.get('geography_share_country_bootstrap_95'),
                      [geography.get('ci_low'), geography.get('ci_high')]),
                numeric('final/three_numbers/S/geography_share_vs_full_sample_refit', points[PRIMARY_REPRESENTATION]
                        ['geography'], stage.get('geography_share'), POINT_TOL)]
    records += [exact('package/consolidated_csv_rows_vs_json_rows', len(evidence.consolidated_rows),
                      len(evidence.consolidated.get('rows') or []))]
    return records


# ---------------------------------------------------------------------
# Negative controls (in-memory copies only)
# ---------------------------------------------------------------------


def _failed(records) -> list[str]:
    return [r['check'] for r in records if not r['passes']]


def negative_controls(evidence, context, recomputed, points, static, qualifying, n_boot) -> dict:
    controls = {}

    baseline = _failed(bootstrap_records(evidence, context, recomputed, static, n_boot))
    draws = evidence.draws.copy()
    target = [i for i, (r, k, u) in enumerate(zip(draws.representation, draws.bootstrap, draws.usable))
              if r == PRIMARY_REPRESENTATION and k == 'country' and u is True]
    if not target:
        controls['a_perturbed_draw_share'] = {'description': 'no usable primary country draw', 'detected': False}
    else:
        position = draws.index[target[0]]
        draws.loc[position, 'geography'] = draws.loc[position, 'geography'] + CONTROL_SHIFT
        altered = dataclasses.replace(evidence, draws=draws)
        failed = _failed(bootstrap_records(altered, context, recomputed, static, n_boot))
        controls['a_perturbed_draw_share'] = {
            'description': 'the geography share of the first usable primary country draw increased by 1e-6 in an '
                           'in-memory copy of m4_bootstrap_draws.csv',
            'target': f'{PRIMARY_REPRESENTATION}/country/draw {int(draws.loc[position, "draw"])}',
            'baseline_failed_checks': baseline, 'failed_checks': failed,
            'also_failed_summary_checks': _failed(summary_records(altered, context, points, static)),
            'detected': bool(not baseline and f'{PRIMARY_REPRESENTATION}/country/shares/geography' in failed)}

    baseline = _failed(summary_records(evidence, context, points, static))
    summary = copy.deepcopy(evidence.summary)
    node = dig(summary, PRIMARY_REPRESENTATION, 'country', 'groups', 'geography')
    if not isinstance(node, dict) or not isinstance(node.get('ci_low'), (int, float)):
        controls['b_shifted_percentile'] = {'description': 'no geography ci_low to shift', 'detected': False}
    else:
        node['ci_low'] = node['ci_low'] + CONTROL_SHIFT
        failed = _failed(summary_records(dataclasses.replace(evidence, summary=summary), context, points, static))
        controls['b_shifted_percentile'] = {
            'description': 'the primary country-bootstrap geography ci_low increased by 1e-6 in an in-memory copy of '
                           'm4_bootstrap_summary.json',
            'baseline_failed_checks': baseline, 'failed_checks': failed,
            'detected': bool(not baseline and f'{PRIMARY_REPRESENTATION}/country/geography/ci_low' in failed)}

    baseline = _failed(final_records(evidence, context, points))
    final = copy.deepcopy(evidence.final)
    original = final.get('final_primary_predictive_model')
    final['final_primary_predictive_model'] = static if original is None else None
    failed = _failed(final_records(dataclasses.replace(evidence, final=final), context, points))
    controls['c_changed_naming_field'] = {
        'description': 'final_primary_predictive_model changed in an in-memory copy of m4_final_specification.json',
        'original': original, 'changed_to': final['final_primary_predictive_model'],
        'baseline_failed_checks': baseline, 'failed_checks': failed,
        'detected': bool(not baseline and 'final/naming_rule' in failed)}

    baseline = _failed(sensitivity_records(evidence, context, static, qualifying))
    frame = evidence.sensitivity_predictions.copy()
    candidates = np.flatnonzero((frame.model == COMPARATOR).to_numpy())
    if not len(candidates):
        controls['d_perturbed_sensitivity_prediction'] = {'description': 'no M0* sensitivity prediction',
                                                          'detected': False}
    else:
        position = frame.index[candidates[0]]
        frame.loc[position, 'prediction'] = float(frame.loc[position, 'prediction']) + CONTROL_SHIFT
        failed = _failed(sensitivity_records(dataclasses.replace(evidence, sensitivity_predictions=frame), context,
                                             static, qualifying))
        stem = f'{frame.loc[position, "protocol"]}/{frame.loc[position, "representation"]}/{COMPARATOR}'
        controls['d_perturbed_sensitivity_prediction'] = {
            'description': 'one saved M0* sensitivity prediction increased by 1e-6 in an in-memory copy of '
                           'm4_sensitivity_predictions.csv',
            'target': f'{stem}/{frame.loc[position, "iso3"]}', 'baseline_failed_checks': baseline,
            'failed_checks': failed, 'detected': bool(not baseline and f'{stem}/cv_rmse' in failed)}
    return controls


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


def reconstruct(evidence: Evidence, context: Context, expected=None) -> dict:
    expected = {**EXPECTED, **(expected or {})}
    static = evidence.final.get('retained_static_specification')
    qualifying = evidence.final.get('qualifying_spatial_extensions')
    if static not in STATIC_MODEL or not isinstance(qualifying, list):
        raise EvidenceRefused(f'the final record names no usable retained static specification ({static!r}) or '
                              f'qualifying list ({qualifying!r})')
    n_boot = int(expected['n_boot'])
    started = time.perf_counter()
    recomputed = recompute_bootstrap(context, static, n_boot)
    bootstrap_seconds = time.perf_counter() - started
    points = full_sample_points(context, static)
    checks = {
        '1_manifest_and_package': evidence.manifest_records,
        '2_bootstrap_draws': input_records(context, static, expected)
        + bootstrap_records(evidence, context, recomputed, static, n_boot),
        '3_bootstrap_summaries_and_material_change': summary_records(evidence, context, points, static),
        '4_buffer_sensitivities': sensitivity_records(evidence, context, static, qualifying),
        '5_product_check': product_records(evidence, context, static, qualifying),
        '6_final_record': final_records(evidence, context, points),
    }
    sections = {name: section(records) for name, records in checks.items()}
    failures = [f'{name}: {check}' for name, block in sections.items() for check in block['failed_checks']]
    controls = negative_controls(evidence, context, recomputed, points, static, qualifying, n_boot)
    counts = {f'{r}/{k}': {'usable': int(sum(recomputed[(r, k)]['usable'])),
                           'rank_deficient_usable': int(sum(v is True for v in recomputed[(r, k)]['rank_deficient']))}
              for r in REPRESENTATIONS for k in KINDS}
    return {
        'what_this_is': 'independent reconstruction of the M4 final assessment from its saved evidence '
                        '(M4_EVALUATOR_SPEC.md §4)',
        'implementation': 'research/model_v2/m4_independent_check.py',
        'scope': SCOPE, 'tolerances': TOLERANCES, 'expected_structure': expected,
        'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying,
        'inputs': {'artifact_sha256': evidence.manifest.get('artifact_sha256'), 'auxiliary_sha256': context.input_sha256},
        'bootstrap_recomputation': {'draws_recomputed': n_boot * len(KINDS) * len(REPRESENTATIONS),
                                    'recomputed_counts': counts, 'seconds': round(bootstrap_seconds, 1),
                                    'note': 'every draw is regenerated and refitted; seconds is not deterministic'},
        'section_summary': {name: {k: b[k] for k in ('passes', 'n_comparisons', 'n_failed', 'max_abs_diff')}
                            for name, b in sections.items()},
        'all_checks_pass': not failures, 'failed_checks': failures,
        'negative_controls': controls,
        'negative_controls_all_detected': all(c['detected'] for c in controls.values()),
        'sections': sections,
    }


def run(package=PACKAGE_DIR, inputs: Inputs | None = None, expected=None) -> dict:
    evidence = load_evidence(package)
    return reconstruct(evidence, load_context(inputs or Inputs()), expected)


def main(argv=None, inputs: Inputs | None = None, expected=None) -> int:
    parser = argparse.ArgumentParser(description='Independently reconstruct the M4 final assessment.')
    parser.add_argument('--package', default=str(PACKAGE_DIR), help='completed m4_final output directory')
    parser.add_argument('--out', default=str(OUTPUT_PATH), help='where to write the verification JSON')
    args = parser.parse_args(argv)
    try:
        report = run(args.package, inputs, expected)
    except EvidenceRefused as refusal:
        report = {'what_this_is': 'independent reconstruction of the M4 final assessment (refused)',
                  'scope': SCOPE, 'refused': str(refusal), 'manifest': refusal.records, 'all_checks_pass': False,
                  'negative_controls': None, 'negative_controls_all_detected': False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + '\n')
    passed = bool(report['all_checks_pass'] and report['negative_controls_all_detected'])
    if 'refused' in report:
        print(f"REFUSED: {report['refused']}")
    else:
        print(f"bootstrap draws recomputed: {report['bootstrap_recomputation']['draws_recomputed']} "
              f"in {report['bootstrap_recomputation']['seconds']} s")
        for name, block in report['section_summary'].items():
            print(f"  {name}: {block['n_comparisons'] - block['n_failed']}/{block['n_comparisons']} "
                  f"(max abs diff {block['max_abs_diff']:.3g})")
        for name, control in report['negative_controls'].items():
            print(f"  control {name}: detected={control['detected']}")
        for failure in report['failed_checks']:
            print(f'  FAIL {failure}')
    print(f'all_checks_pass={report["all_checks_pass"]} '
          f'negative_controls_all_detected={report["negative_controls_all_detected"]} -> exit {0 if passed else 3}')
    return 0 if passed else 3


if __name__ == '__main__':
    sys.exit(main())
