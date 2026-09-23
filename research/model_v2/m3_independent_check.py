"""Independent reconstruction of an M3 package from its saved evidence.

A second calculation path for one weights arm (``outputs/m3_station`` or ``outputs/m3_land_centroid``). It
imports nothing from ``m3_spatial``, ``m3_evaluate``, ``m2_evaluate``, ``m2_feasibility``,
``m2_latitude_basis``, ``cv``, ``spatial``, ``run_territory_correction``, ``src.decomposition`` or
``src.stability``. Its mathematics is ``m3_independent_math.py`` (eigenvalue log-determinants, QR regressions,
loop-based lags and predictions, permutation Shapley, the paired interval) plus NumPy, pandas and the standard
library. Given a completed package and the committed M2 primary result it:

1. verifies every digest of ``m3_result_manifest.json`` (and the two M2 artifacts it reads against
   ``m2_result_manifest.json``) before anything else is parsed; a mismatch is a refusal;
2. rebuilds the inputs: canonical ISO3 order and the four categorical labels of ``m0_countries.csv``, the exact
   numeric predictors of the ``M0star`` rows of ``m1b_design_matrices.csv``, the outcome (the package's
   ``observed`` column, required to equal the rank-audit ``drop_per_capita`` vector bit for bit), great-circle
   distances from its own haversine (radius 6371.0088 km) on station or land centroids, and the full-sample
   station and land kNN8 matrices (digests against ``m3_provenance.json``);
3. for every row of ``m3_fits.csv``: the static design from the saved column order, the saved training levels
   (drop-first, the first saved level is the reference) and, for M2 designs, the natural cubic columns from
   the saved hex knots,

       d_k(a) = [(a - xi_k)_+^3 - (a - xi_4)_+^3] / (xi_4 - xi_k),   N1 = d_1 - d_3,   N2 = d_2 - d_3,

   with I(hemisphere == 'S') * abs_latitude; the training graph W from the saved neighbour lists (1/8 per
   listed neighbour), each list checked against a kNN8 recomputation (distance, then canonical index) and each
   held-out attachment list against the sink-rule recomputation;
4. the concentrated log-likelihood at the saved theta, local optimality on a fine grid (one-sided at the
   domain bound), the global-basin scan, beta by QR at the saved theta, sigma2, ``at_domain_bound``; for the
   full-sample fits the whole saved 199-point grid;
5. every held-out prediction by explicit loops over the saved attachments from training outcomes only, with the
   mean-effect unseen-level adjustment, and the in-sample one-step predictions and trends;
6. per family x representation x protocol, RMSE, MAE, R2, fold summaries, calibration, per-M49-region errors,
   each model's own worst region (n >= 3) and residual Moran statistics; the paired interval against the static
   model's committed primary out-of-fold predictions; Q1-Q5 and the final naming, or ``arm_conditions``;
7. the accounting parts and the filtered-trend group shares (Shapley by averaging over all orderings);
8. the representation identity (theta 1e-5; predictions, scores, accounting parts 1e-6);
9. five negative controls on in-memory copies (perturbed theta, swapped neighbour, perturbed prediction,
   shifted veto threshold, perturbed beta element), each of which must be detected.

Rules chosen here. **Global basin**: the reference scan evaluates l_c on ``linspace(-0.99, 0.99, 397)``; the
saved theta must lie between the scan points on either side of the scan argmax (earliest maximum) and attain
the scan maximum up to 1e-8 (``m3_independent_math.global_scan``). **Local optimality**: 41 points over
theta +/- 1e-3 clipped to the domain (one-sided at a bound), l_c(theta) >= every grid value - 1e-8.
**Worst region**: the largest RMSE among regions with n >= 3; an exact tie goes to the first region name in
sorted order. **Full-sample fits** carry no saved levels or latitude state (the evaluator writes them empty):
levels are the sorted labels of all rows and the M2 state is the committed M2 scorecard's
``latitude_state_full_sample``; both are re-derived here and compared.

Tolerances. log-likelihood 1e-8 absolute; beta |QR - saved| <= 1e-8 * max(1, |saved|); sigma2 1e-8 relative;
predictions, adjustments and trends 1e-9; the two innovation expressions and mean(innovation^2) vs sigma2 1e-10;
knots re-derived from training latitudes 1e-10 degrees; metrics, paired intervals and qualification inputs
1e-12; calibration 1e-9 * max(1, |saved|); Moran statistics 1e-10; graph distances 1e-6 km; accounting parts
and filtered shares 1e-9; the recomputed representation-identity differences against the saved report 1e-9;
labels, counts, identifiers, neighbour lists, layouts, levels, Booleans, names and digests exact.

Not independent: the inputs are the same saved artifacts and pinned tables the evaluator reads; fold ids,
training memberships (the 500 km territorial bounds) and the static model's predictions and LMG shares are taken
as given (they belong to the M2 result and its own check); nearest-training distances, permutation p-values,
the expected-information standard errors, the overfitting diagnostics and the reason text of a non-computable
disposition are not recomputed. The check is pre-specified: its rules and tolerances are fixed by the
evaluator's artifact formats, not by any result.

Run: ``uv run python -m research.model_v2.m3_independent_check --package research/model_v2/outputs/m3_station``
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.model_v2 import m3_independent_math as mi

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'research' / 'model_v2' / 'outputs'
M2_DIR = OUTPUTS / 'm2_primary'
COUNTRIES_PATH = OUTPUTS / 'm0_countries.csv'
DESIGN_PATH = OUTPUTS / 'm1b_primary' / 'm1b_design_matrices.csv'
GEOMETRY_PATH = OUTPUTS / 'country_geometry.csv'
RANK_AUDIT_PATH = OUTPUTS / 'rank_audit_country_predictions.csv'
INPUT_PIN_KEYS = {
    'countries': 'research/model_v2/outputs/m0_countries.csv',
    'design': 'research/model_v2/outputs/m1b_primary/m1b_design_matrices.csv',
    'geometry': 'research/model_v2/outputs/country_geometry.csv',
    'rank_audit': 'research/model_v2/outputs/rank_audit_country_predictions.csv',
}

MANIFEST = 'm3_result_manifest.json'
ARTIFACTS = ('m3_scorecard.json', 'm3_country_predictions.csv', 'm3_fits.csv', 'm3_graphs.csv',
             'm3_full_sample_grid.csv', 'm3_provenance.json')
M2_MANIFEST = 'm2_result_manifest.json'
M2_ARTIFACTS = ('m2_scorecard.json', 'm2_country_predictions.csv')

FAMILIES = ('sem', 'sar')
FAMILY_NAME = {'sem': 'SEM', 'sar': 'SAR'}
PRIMARY_REPRESENTATION = 'primary_total_co2'
REPRESENTATIONS = (PRIMARY_REPRESENTATION, 'per_capita')
PRIMARY = 'primary_loco'
PROTOCOLS = (PRIMARY, 'm49_subregion_lo', 'random10')
CARD_PATH = {PRIMARY: (), 'm49_subregion_lo': ('secondary',), 'random10': ('random_reference_only',)}
CATEGORICALS = ('climate_zone', 'hemisphere', 'spatial_block', 'income_group')
NUMERIC = {
    'primary_total_co2': ('cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'population',
                          'station_density'),
    'per_capita': ('cum_co2_per_capita', 'abs_latitude', 'elevation', 'continentality', 'population',
                   'station_density'),
}
ADDED = ('abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude')
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')
STATIC_MODEL = {'M0*': 'M0star', 'M2': 'M2'}
BASIS_VERSION = 'm2-latitude-basis-v1'
KNOT_GAP = 1e-6
ARMS = ('station', 'land')
WEIGHT_DIGEST_KEY = {'station': 'station_weights_sha256', 'land': 'area_weights_sha256'}
EARTH_RADIUS_KM = 6371.0088
BOUND_MARGIN = 1e-6
EXPECTED_EVALUATIONS = mi.GRID_POINTS + 2 + mi.GOLDEN_ITERATIONS
VETO = 0.001
MIN_REGION_N = 3
EQUIVALENT_SCORES = ('in_sample_one_step_r2', 'trend_r2', 'cv_r2', 'cv_rmse', 'cv_mae',
                     'residual_morans_i_in_sample', 'residual_morans_i_cv', 'worst_region.rmse',
                     'secondary.cv_rmse', 'random_reference_only.cv_rmse')
ACCOUNTING_PARTS = ('a_static', 'a_dependence', 'a_innovation', 'absorbed_fraction_of_static_residual')
# The real-data structure; synthetic tests pass their own.
EXPECTED = {'n': 151, 'folds': {PRIMARY: 151, 'm49_subregion_lo': 20, 'random10': 10}}

LOGLIK_TOL = 1e-8
BETA_TOL = 1e-8
SIGMA2_TOL = 1e-8
PREDICTION_TOL = 1e-9
QUANTITY_IDENTITY_TOL = 1e-10
KNOT_TOL = 1e-10
METRIC_TOL = 1e-12
CALIBRATION_TOL = 1e-9
MORAN_TOL = 1e-10
DISTANCE_TOL_KM = 1e-6
GRID_THETA_TOL = 1e-15
ACCOUNTING_TOL = 1e-9
IDENTITY_THETA_TOL = 1e-5            # representation-identity tolerance for theta
IDENTITY_TOL = 1e-6                  # predictions, non-allocation scores, accounting parts
IDENTITY_REPORT_TOL = 1e-9
CONTROL_THETA_SHIFT = 1e-3
CONTROL_SHIFT = 1e-6

TOLERANCES = {
    'loglik_absolute': LOGLIK_TOL, 'beta': 'abs(QR - saved) <= 1e-8 * max(1, abs(saved))',
    'sigma2_relative': SIGMA2_TOL, 'predictions_adjustments_trends': PREDICTION_TOL,
    'innovation_expressions_and_sigma2_gap': QUANTITY_IDENTITY_TOL, 'knots_degrees': KNOT_TOL,
    'metrics_paired_qualification_inputs': METRIC_TOL, 'calibration': 'abs diff <= 1e-9 * max(1, abs(saved))',
    'moran_statistics': MORAN_TOL, 'graph_distances_km': DISTANCE_TOL_KM, 'grid_theta_values': GRID_THETA_TOL,
    'accounting_and_filtered_shares': ACCOUNTING_TOL,
    'identity_amendment_a1': {'theta': IDENTITY_THETA_TOL, 'predictions_scores_accounting': IDENTITY_TOL},
    'identity_report_values': IDENTITY_REPORT_TOL,
    'local_optimality': {'half_width': mi.LOCAL_HALF_WIDTH, 'points': mi.LOCAL_POINTS, 'atol': mi.OPTIMALITY_ATOL,
                         'domain_bound': 'grid clipped to [-0.99, 0.99], so the check is one-sided at a bound'},
    'global_basin': {'scan_points': mi.SCAN_POINTS, 'rule': 'saved theta between the scan points either side of '
                     'the scan argmax (earliest maximum) and l_c(theta) >= scan maximum - 1e-8'},
    'exact': 'labels, counts, identifiers, neighbour lists, layouts, levels, Booleans, names, digests',
}
SCOPE = {
    'independent': [
        'the calculation path: haversine distances, kNN8 and sink attachments, design rebuild with natural cubic '
        'columns, eigenvalue log-determinant likelihood, QR coefficients, loop-based predictions and in-sample '
        'quantities, metrics, worst regions, Moran statistics, paired intervals, Q1-Q5, final naming, accounting '
        'parts, permutation-averaged filtered shares and the representation identity',
        'imports only numpy, pandas, the standard library and research.model_v2.m3_independent_math',
    ],
    'same_inputs': [
        'the saved M3 artifacts, the committed M2 primary predictions, scorecard and manifest, and the pinned '
        'country, design, geometry and rank-audit tables; fold ids and training memberships are taken as given',
    ],
    'not_recomputed': [
        'the 500 km territorial exclusions, nearest-training distances, permutation p-values, expected-information '
        'standard errors, overfitting diagnostics, the static model predictions and LMG shares (M2 result), and '
        'the failure reason of a non-computable family',
    ],
    'timing': 'written before any M3 result existed, after reading the evaluator artifact formats; run after',
}


class DigestRefused(ValueError):
    """The saved evidence does not match its manifest; nothing is reconstructed."""

    def __init__(self, message, records):
        super().__init__(message)
        self.records = records


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


MISSING = '<missing>'


def node(tree, *path):
    """tree[path[0]][path[1]]...; MISSING when any key is absent."""
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return MISSING
        tree = tree[key]
    return tree


def dotted(tree, name):
    return node(tree, *name.split('.'))


def close(check, label, recomputed, reported, tol, scale=False) -> dict:
    """|recomputed - reported| <= tol (divided by max(1, |reported|) when ``scale``); missing never passes."""
    try:
        r, s = float(recomputed), float(reported)
        raw = abs(r - s)
        diff = raw / max(1.0, abs(s)) if scale else raw
        passes = math.isfinite(diff) and diff <= tol
    except (TypeError, ValueError):
        r, s, raw, diff, passes = recomputed, reported, math.nan, math.nan, False
    return {'check': check, 'label': label, 'recomputed': r, 'reported': s, 'diff': diff, 'raw': raw, 'tol': tol,
            'passes': bool(passes)}


def gap(check, label, value, tol, detail=None) -> dict:
    """A recomputed discrepancy (already a difference) must be finite and at most tol."""
    try:
        value = float(value)
        passes = math.isfinite(value) and value <= tol
    except (TypeError, ValueError):
        value, passes = math.nan, False
    return {'check': check, 'label': label, 'diff': value, 'raw': value, 'tol': tol, 'passes': bool(passes),
            'detail': detail}


def arrays(check, label, recomputed, reported, tol) -> dict:
    r, s = np.asarray(recomputed, dtype=float), np.asarray(reported, dtype=float)
    if r.shape != s.shape or r.size == 0:
        return {'check': check, 'label': label, 'diff': math.nan, 'raw': math.nan, 'tol': tol, 'passes': False,
                'detail': f'shapes {r.shape} vs {s.shape}'}
    return gap(check, label, float(np.max(np.abs(r - s))), tol)


def same(check, label, recomputed, reported) -> dict:
    passes = (not (isinstance(reported, str) and reported == MISSING)) and recomputed == reported
    return {'check': check, 'label': label, 'recomputed': recomputed, 'reported': reported, 'passes': bool(passes)}


def flag(check, label, passes, detail=None) -> dict:
    return {'check': check, 'label': label, 'passes': bool(passes), 'detail': detail}


def _digest(values) -> str:
    return hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:16]


def sequence(check, label, recomputed, reported) -> dict:
    recomputed, reported = [str(v) for v in recomputed], [str(v) for v in reported]
    differing = sum(1 for a, b in zip(recomputed, reported) if a != b) + abs(len(recomputed) - len(reported))
    return {'check': check, 'label': label, 'passes': recomputed == reported, 'n_differing': differing,
            'recomputed': {'n': len(recomputed), 'sha256_16': _digest(recomputed)},
            'reported': {'n': len(reported), 'sha256_16': _digest(reported)}}


def summarise(check, items) -> dict:
    failed = [i for i in items if not i['passes']]
    diffs = [i['raw'] for i in items if isinstance(i.get('raw'), float) and math.isfinite(i['raw'])]
    record = {'check': check, 'n': len(items), 'n_failed': len(failed), 'passes': bool(items) and not failed,
              'max_abs_diff': max(diffs) if diffs else None,
              'tolerance': items[0].get('tol', 'exact') if items else None}
    scaled = [i['diff'] for i in items if 'diff' in i and i['diff'] != i.get('raw') and math.isfinite(i['diff'])]
    if scaled:
        record['max_scaled_diff'] = max(scaled)
    record['failed_examples'] = [{k: v for k, v in i.items() if k not in ('check', 'passes', 'section')}
                                 for i in failed[:5]]
    return record


def section(items) -> dict:
    by_check: dict[str, list] = {}
    for item in items:
        by_check.setdefault(item['check'], []).append(item)
    records = [summarise(check, group) for check, group in by_check.items()]
    diffs = [r['max_abs_diff'] for r in records if r['max_abs_diff'] is not None]
    failed = [r['check'] for r in records if not r['passes']]
    return {'n_checks': len(records), 'n_comparisons': len(items), 'n_failed_comparisons': sum(r['n_failed'] for r in records),
            'max_abs_diff': max(diffs) if diffs else None, 'passes': not failed, 'empty': not records,
            'failed_checks': failed, 'checks': records}


# ---------------------------------------------------------------------
# Independent arithmetic (reference mathematics for everything spatial)
# ---------------------------------------------------------------------


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()


def haversine_km(lon, lat) -> np.ndarray:
    """Great-circle distances by the haversine formula, one row at a time."""
    phi, lam = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    out = np.empty((len(phi), len(phi)))
    for i in range(len(phi)):
        s_phi, s_lam = np.sin((phi[i] - phi) / 2.0), np.sin((lam[i] - lam) / 2.0)
        h = s_phi * s_phi + np.cos(phi[i]) * np.cos(phi) * s_lam * s_lam
        out[i] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return out


def weights_from_lists(lists, size) -> np.ndarray:
    """W with 1/8 on each listed position, built entry by entry."""
    w = np.zeros((len(lists), size))
    for i, row in enumerate(lists):
        for j in row:
            w[i, j] += mi.LINK_WEIGHT
    return w


def linear_quantile(values, probability) -> float:
    """Type-7 quantile: x[k] + (h - k)(x[k+1] - x[k]) with h = (n - 1) p."""
    x = sorted(float(v) for v in values)
    h = (len(x) - 1) * probability
    k = math.floor(h)
    return x[k] if k + 1 >= len(x) else x[k] + (h - k) * (x[k + 1] - x[k])


def knots_from(abs_latitude) -> list[float]:
    values = [float(v) for v in abs_latitude]
    return [min(values), linear_quantile(values, 1 / 3), linear_quantile(values, 2 / 3), max(values)]


def natural_cubic(knots, a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    last = float(knots[3])

    def cube(t):
        return t * t * t

    def d(k):
        xi = float(knots[k])
        return (cube(np.maximum(a - xi, 0.0)) - cube(np.maximum(a - last, 0.0))) / (last - xi)
    return np.column_stack([d(0) - d(2), d(1) - d(2)])


def column_group(name) -> str:
    if name == 'intercept':
        return 'intercept'
    if name.startswith('cum_co2_'):
        return 'emissions'
    if (name.startswith('abs_latitude') or name in ('elevation', 'continentality')
            or name.startswith(('climate_zone=', 'hemisphere', 'spatial_block='))):
        return 'geography'
    if name.startswith('income_group='):
        return 'socioeconomic'
    if name in ('population', 'station_density'):
        return 'population'
    raise ValueError(f'column {name!r} has no named group')


def region_table(regions, errors) -> dict[str, dict]:
    table: dict[str, list] = {}
    for region, e in zip(regions, errors):
        table.setdefault(str(region), []).append(float(e))
    return {r: {'n': len(v), 'rmse': math.sqrt(math.fsum(x * x for x in v) / len(v)),
                'mae': math.fsum(abs(x) for x in v) / len(v), 'bias': math.fsum(v) / len(v)}
            for r, v in sorted(table.items())}


def worst_region(table) -> tuple[str | None, float, int]:
    best = None
    for region, stats in table.items():   # sorted names: the first of an exact tie wins
        if stats['n'] >= MIN_REGION_N and (best is None or stats['rmse'] > table[best]['rmse']):
            best = region
    return (None, math.nan, 0) if best is None else (best, table[best]['rmse'], table[best]['n'])


def qualification(delta, low, high, m49_change, worst_change, computable, veto=VETO) -> dict:
    if not computable:
        return {'qualifies': False, 'Q5_computable': False}
    if not all(math.isfinite(v) for v in (delta, low, high, m49_change, worst_change)):
        return {'qualifies': None}
    q = {'Q1_delta_below_zero': delta < 0, 'Q2_interval_upper_below_zero': high < 0,
         'Q3_m49_veto_passed': m49_change <= veto, 'Q4_worst_region_veto_passed': worst_change <= veto,
         'Q5_computable': True}
    return {**q, 'qualifies': all(q.values()), 'worsened_generalization': delta > 0 and low > 0}


def final_naming(static, qualifies: dict) -> dict:
    """The final naming rule from the recomputed station-arm primary-representation Booleans."""
    qualifying = [f'M3-{FAMILY_NAME[f]}({static})' for f in FAMILIES if qualifies.get(f) is True]
    final = static if not qualifying else qualifying[0] if len(qualifying) == 1 else None
    return {'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying,
            'final_primary_predictive_model': final}


# ---------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------


def read_csv(path, **kwargs) -> pd.DataFrame:
    """Exact float64 round trip; pandas' default parser is only accurate to about 1e-14 relative."""
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, **kwargs)


def verify_manifest(directory, manifest_name, required) -> tuple[dict, list]:
    """Every listed digest against the bytes on disk; the required artifacts must be listed."""
    directory = Path(directory)
    path = directory / manifest_name
    if not path.is_file():
        raise DigestRefused(f'{path} is missing', [flag('manifest_present', str(path), False)])
    manifest = json.loads(path.read_text())
    recorded = manifest.get('artifact_sha256', {})
    records = [same('artifact_listed', f'{manifest_name}/{name}', True, name in recorded) for name in required]
    for name, digest in sorted(recorded.items()):
        target = directory / name
        inside = target.resolve().parent == directory.resolve()
        actual = sha256(target) if inside and target.is_file() else 'missing'
        records.append(same('artifact_sha256', f'{manifest_name}/{name}', actual, digest))
    failed = [f"{r['label']}" for r in records if not r['passes']]
    if failed:
        raise DigestRefused(f'manifest verification failed: {failed}', records)
    return manifest, records


@dataclasses.dataclass
class Evidence:
    package: Path
    manifest: dict
    scorecard: dict
    provenance: dict
    predictions: pd.DataFrame
    fits: pd.DataFrame
    graphs: pd.DataFrame
    grid: pd.DataFrame
    m2_manifest: dict
    m2_manifest_sha256: str
    m2_scorecard: dict
    m2_predictions: pd.DataFrame
    countries: pd.DataFrame
    design: pd.DataFrame
    geometry: pd.DataFrame
    rank_audit: pd.DataFrame
    manifest_records: list
    input_sha256: dict


def load_evidence(package, m2=M2_DIR, countries=COUNTRIES_PATH, design=DESIGN_PATH, geometry=GEOMETRY_PATH,
                  rank_audit=RANK_AUDIT_PATH) -> Evidence:
    package, m2 = Path(package), Path(m2)
    manifest, records = verify_manifest(package, MANIFEST, ARTIFACTS)
    m2_manifest, m2_records = verify_manifest(m2, M2_MANIFEST, M2_ARTIFACTS)
    text = dict.fromkeys(('weights', 'family', 'representation', 'protocol', 'fit', 'iso3', 'm49_subregion'), str)
    predictions = read_csv(package / 'm3_country_predictions.csv', dtype=text,
                           na_values={'trend': [''], 'nearest_train_km': ['']})
    fits = read_csv(package / 'm3_fits.csv', dtype={**text, 'columns': str, 'beta': str, 'training_iso3': str,
                                                    'levels': str, 'latitude_state': str})
    inputs = {'countries': countries, 'design': design, 'geometry': geometry, 'rank_audit': rank_audit}
    return Evidence(
        package=package, manifest=manifest,
        scorecard=json.loads((package / 'm3_scorecard.json').read_text()),
        provenance=json.loads((package / 'm3_provenance.json').read_text()),
        predictions=predictions, fits=fits,
        graphs=read_csv(package / 'm3_graphs.csv', dtype=str),
        grid=read_csv(package / 'm3_full_sample_grid.csv', dtype=dict.fromkeys(('weights', 'family', 'representation'), str)),
        m2_manifest=m2_manifest, m2_manifest_sha256=sha256(m2 / M2_MANIFEST),
        m2_scorecard=json.loads((m2 / 'm2_scorecard.json').read_text()),
        m2_predictions=read_csv(m2 / 'm2_country_predictions.csv',
                                dtype={'representation': str, 'model': str, 'protocol': str, 'iso3': str,
                                       'm49_subregion': str, 'Country': str, 'extrapolation': str},
                                na_values={'nearest_train_km': ['']}),
        countries=read_csv(countries, usecols=['iso3', *CATEGORICALS, 'station_lon', 'station_lat'],
                           dtype=dict.fromkeys(('iso3', *CATEGORICALS), str)),
        design=read_csv(design, usecols=['representation', 'model', 'iso3', 'column', 'value'],
                        dtype={'representation': str, 'model': str, 'iso3': str, 'column': str}),
        geometry=read_csv(geometry, usecols=['iso3', 'centroid_lon', 'centroid_lat'], dtype={'iso3': str}),
        rank_audit=read_csv(rank_audit, usecols=['variant', 'iso3', 'observed'], dtype={'variant': str, 'iso3': str}),
        manifest_records=records + m2_records,
        input_sha256={role: {'path': str(path), 'sha256': sha256(path)} for role, path in inputs.items()})


# ---------------------------------------------------------------------
# Context: inputs rebuilt independently
# ---------------------------------------------------------------------


@dataclasses.dataclass
class Context:
    arm: str
    static: str
    static_model: str
    order: list
    pos: dict
    labels: dict
    numerics: dict
    y: np.ndarray
    regions: np.ndarray
    dist: np.ndarray
    station_w: np.ndarray
    land_w: np.ndarray
    full_states: dict
    fold_ids: dict
    fold_members: dict
    required: dict
    expected: dict
    grids: dict


def _static_rows(frame, representation, model, protocol, order) -> pd.DataFrame:
    rows = frame[(frame.representation == representation) & (frame.model == model) & (frame.protocol == protocol)]
    return rows.set_index('iso3').reindex(order)


def build_context(ev: Evidence, expected) -> tuple[Context, list]:
    items = list(ev.manifest_records)
    order = ev.countries.iso3.tolist()
    pos = {iso: i for i, iso in enumerate(order)}
    n = len(order)
    items.append(same('country_iso3_unique_and_expected_count', 'm0_countries', (n, len(pos)), (expected['n'], n)))
    labels = {c: ev.countries[c].to_numpy(dtype=str) for c in CATEGORICALS}

    arm = ev.manifest.get('weights')
    static = ev.scorecard.get('retained_static_specification')
    items.append(same('arm_is_station_or_land', 'manifest', True, arm in ARMS))
    items.append(same('retained_static_specification_is_M0*_or_M2', 'scorecard', True, static in STATIC_MODEL))
    if arm not in ARMS or static not in STATIC_MODEL:
        raise DigestRefused(f'unreadable arm {arm!r} or static specification {static!r}', items)
    static_model = STATIC_MODEL[static]

    numerics = {}
    for representation in REPRESENTATIONS:
        rows = ev.design[(ev.design.representation == representation) & (ev.design.model == 'M0star')]
        names = [c for c in dict.fromkeys(rows.column) if c != 'intercept' and '=' not in c]
        wide = rows.pivot(index='iso3', columns='column', values='value')
        items.append(same('design_numeric_columns', representation, sorted(NUMERIC[representation]), sorted(names)))
        covered = set(order) <= set(wide.index)
        items.append(flag('design_covers_every_country', representation, covered))
        numerics[representation] = {c: wide.reindex(order)[c].to_numpy(dtype=float) if c in wide else np.full(n, np.nan)
                                    for c in NUMERIC[representation]}
        items.append(flag('design_numerics_finite', representation,
                          all(np.isfinite(v).all() for v in numerics[representation].values())))

    audit = ev.rank_audit[ev.rank_audit.variant == 'drop_per_capita'].set_index('iso3').reindex(order)
    y_audit = audit.observed.to_numpy(dtype=float)
    in_sample = ev.predictions[ev.predictions.protocol == 'in_sample']
    if len(in_sample):
        first = in_sample[(in_sample.representation == in_sample.representation.iloc[0])
                          & (in_sample.family == in_sample.family.iloc[0])]
        y = first.set_index('iso3').reindex(order).observed.to_numpy(dtype=float)
        source = 'package in_sample observed'
    else:
        y, source = y_audit, 'rank audit (no computable run wrote in-sample rows)'
    items.append(flag('outcome_equals_rank_audit_drop_per_capita_bit_for_bit', source,
                      bool(np.isfinite(y).all() and np.array_equal(y, y_audit))))
    observed = ev.predictions.set_index('iso3').observed
    items.append(flag('every_package_observed_value_equals_the_outcome', 'm3_country_predictions',
                      bool(len(observed) == 0 or np.array_equal(observed.to_numpy(dtype=float),
                                                                y[[pos.get(i, 0) for i in observed.index]]))
                      and set(observed.index) <= set(pos)))
    region_of = dict(zip(ev.predictions.iso3, ev.predictions.m49_subregion))
    region_of.update({k: v for k, v in zip(ev.m2_predictions.iso3, ev.m2_predictions.m49_subregion) if k not in region_of})
    regions = np.array([region_of.get(iso, '') for iso in order], dtype=object)
    pairs = pd.concat([ev.predictions[['iso3', 'm49_subregion']], ev.m2_predictions[['iso3', 'm49_subregion']]])
    items.append(flag('one_m49_label_per_country', 'predictions', bool((pairs.groupby('iso3').m49_subregion.nunique() == 1).all())
                      and bool((regions != '').all())))

    station = ev.countries.set_index('iso3').reindex(order)
    station_dist = haversine_km(station.station_lon.to_numpy(dtype=float), station.station_lat.to_numpy(dtype=float))
    geometry = ev.geometry.set_index('iso3').reindex(order)
    items.append(flag('geometry_covers_every_country', 'country_geometry', bool(geometry.centroid_lon.notna().all())))
    land_dist = haversine_km(geometry.centroid_lon.to_numpy(dtype=float), geometry.centroid_lat.to_numpy(dtype=float))
    station_w = weights_from_lists(mi.knn_neighbours(station_dist, range(n)), n)
    land_w = weights_from_lists(mi.knn_neighbours(land_dist, range(n)), n)
    identity = ev.provenance.get('identity', {})
    items.append(same('full_sample_station_knn8_sha256', 'm3_provenance identity', array_sha256(station_w),
                      identity.get('station_weights_sha256', MISSING)))
    items.append(same('full_sample_land_knn8_sha256', 'm3_provenance identity', array_sha256(land_w),
                      identity.get('area_weights_sha256', MISSING)))

    full_states = {}
    if static_model == 'M2':
        for representation in REPRESENTATIONS:
            full_states[representation] = node(ev.m2_scorecard, 'representations', representation, 'M2',
                                               'latitude_state_full_sample')

    fold_ids, fold_members, required = {}, {}, {}
    for protocol in PROTOCOLS:
        rows = _static_rows(ev.m2_predictions, PRIMARY_REPRESENTATION, static_model, protocol, order)
        ids = rows.fold_id.to_numpy()
        ok = bool(rows.prediction.notna().all())
        items.append(flag('static_predictions_cover_every_country', protocol, ok))
        ids = np.where(pd.isna(ids), -1, ids).astype(int)
        fold_ids[protocol] = ids
        folds = sorted(set(ids.tolist()))
        items.append(same('fold_count', protocol, len(folds), expected['folds'][protocol]))
        if protocol == PRIMARY:
            items.append(flag('primary_fold_id_is_canonical_index', protocol, bool(np.array_equal(ids, np.arange(n)))))
            labels_of = {f: order[f] for f in folds if 0 <= f < n}
        else:
            labels_of = {f: str(f) for f in folds}
        fold_members[protocol] = {labels_of[f]: [i for i in range(n) if ids[i] == f] for f in labels_of}
        required[protocol] = [labels_of[f] for f in folds if f in labels_of]
    context = Context(arm=arm, static=static, static_model=static_model, order=order, pos=pos, labels=labels,
                      numerics=numerics, y=y, regions=regions, dist=station_dist if arm == 'station' else land_dist,
                      station_w=station_w, land_w=land_w, full_states=full_states, fold_ids=fold_ids,
                      fold_members=fold_members, required=required, expected=expected,
                      grids={(r, f): (rows.theta.to_numpy(dtype=float), rows.loglik.to_numpy(dtype=float))
                             for (r, f), rows in ev.grid.groupby(['representation', 'family'])})
    return context, items


# ---------------------------------------------------------------------
# One fit
# ---------------------------------------------------------------------


@dataclasses.dataclass
class FitResult:
    key: tuple
    items: list
    computable: bool = False
    train: list | None = None
    test: list | None = None
    attachments: list | None = None
    theta: float = math.nan
    at_bound: bool = False
    full: dict | None = None


def as_bool(value) -> bool:
    """A saved Boolean cell: a parsed bool or the literal text True/False (never the truthiness of a string)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value) in ('True', 'False'):
        return str(value) == 'True'
    raise ValueError(f'not a Boolean cell: {value!r}')


def parse_state(state) -> tuple[list[float], int]:
    record = json.loads(state) if isinstance(state, str) else dict(state)
    frozen = {'version': BASIS_VERSION, 'quantile_method': 'linear', 'probabilities': ['1/3', '2/3']}
    for key, value in frozen.items():
        if record.get(key) != value:
            raise ValueError(f'latitude state has {key}={record.get(key)!r}, not {value!r}')
    knots = [float.fromhex(h) for h in record['knots_hex']]
    if [float(k) for k in record['knots']] != knots:
        raise ValueError('decimal knots disagree with their hex serialization')
    if len(knots) != 4 or not all(b - a > KNOT_GAP for a, b in zip(knots, knots[1:])):
        raise ValueError(f'knots are not four strictly increasing values: {knots}')
    return knots, int(record['n_train'])


def design_matrix(ctx: Context, representation, columns, rows, knots) -> np.ndarray:
    rows = list(rows)
    a = ctx.numerics[representation]['abs_latitude'][rows]
    out = np.empty((len(rows), len(columns)))
    for j, name in enumerate(columns):
        if name == 'intercept':
            out[:, j] = 1.0
        elif name in NUMERIC[representation]:
            out[:, j] = ctx.numerics[representation][name][rows]
        elif name in ('abs_latitude_ns1', 'abs_latitude_ns2'):
            out[:, j] = natural_cubic(knots, a)[:, 0 if name.endswith('1') else 1]
        elif name == 'hemisphere=S:abs_latitude':
            out[:, j] = (ctx.labels['hemisphere'][rows] == 'S').astype(float) * a
        elif '=' in name and name.split('=', 1)[0] in CATEGORICALS:
            c, level = name.split('=', 1)
            out[:, j] = (ctx.labels[c][rows] == level).astype(float)
        else:
            raise ValueError(f'unknown design column {name!r}')
    return out


def graph_lookup(graphs: pd.DataFrame) -> dict:
    lookup: dict = {}
    for row in graphs.itertuples(index=False):
        entry = lookup.setdefault((row.protocol, row.fit), {'training': [], 'held_out': [], 'weights': set()})
        entry.setdefault(row.role, []).append((row.iso3, row.neighbours.split('|') if row.neighbours else []))
        entry['weights'].add(row.weights)
    return lookup


def prediction_lookup(predictions: pd.DataFrame) -> dict:
    return {key: rows.set_index('iso3') for key, rows in predictions.groupby(['representation', 'family', 'protocol'])}


def check_fit(ctx: Context, row: dict, graphs: dict, predictions: dict) -> FitResult:
    """Structure, graph, likelihood, optimality, coefficients and predictions of one saved fit; an unexpected
    error in the saved evidence is recorded as a failed comparison, never raised."""
    try:
        return _check_fit(ctx, row, graphs, predictions)
    except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
        key = tuple(str(row.get(k)) for k in ('representation', 'family', 'protocol', 'fit'))
        return FitResult(key, [{**flag('fit_check_completes', '/'.join(key), False, repr(error)), 'section': 'structure'}])


def _check_fit(ctx: Context, row: dict, graphs: dict, predictions: dict) -> FitResult:
    items: list = []
    family_code, representation, protocol, label = (str(row.get(k)) for k in ('family', 'representation', 'protocol', 'fit'))
    tag = f'{representation}/{family_code}/{protocol}/{label}'
    result = FitResult((representation, family_code, protocol, label), items)

    def add(section_name, item):
        items.append({**item, 'section': section_name})

    try:
        family = FAMILY_NAME[family_code]
        if representation not in REPRESENTATIONS or protocol not in ('full_sample', *PROTOCOLS):
            raise ValueError(f'unknown representation {representation!r} or protocol {protocol!r}')
        columns = str(row['columns']).split('|')
        beta = np.array(json.loads(row['beta']), dtype=float)
        train_iso = str(row['training_iso3']).split('|')
        train = [ctx.pos[i] for i in train_iso]
        theta = float(row['theta'])
    except (KeyError, ValueError, TypeError) as error:
        add('structure', flag('fit_row_parses', tag, False, repr(error)))
        return result
    result.train, result.theta = train, theta
    n_train, p = len(train), len(columns)
    add('structure', same('weights_column_is_the_arm', tag, ctx.arm, str(row.get('weights'))))
    add('structure', flag('training_iso3_strictly_canonical', tag, all(b > a for a, b in zip(train, train[1:]))))
    add('structure', same('n_and_p', tag, (n_train, p, p), (int(row['n']), int(row['p']), len(beta))))
    if protocol == 'full_sample':
        add('structure', same('full_sample_uses_every_country', tag, ctx.order, train_iso))

    derived = {c: sorted(set(ctx.labels[c][train].tolist())) for c in CATEGORICALS}
    text = str(row.get('levels') or '')
    if text:
        try:
            levels = {str(c): [str(v) for v in values] for c, values in json.loads(text).items()}
        except (ValueError, AttributeError) as error:
            add('structure', flag('levels_parse', tag, False, repr(error)))
            return result
        add('structure', same('levels_are_the_sorted_training_labels', tag, derived, levels))
    elif protocol == 'full_sample':
        levels = derived
    else:
        add('structure', flag('training_levels_saved', tag, False))
        return result

    added = [c for c in columns if c in ADDED]
    numerics = [c for c in columns if c != 'intercept' and '=' not in c and c not in ADDED]
    dummies = sorted(c for c in columns if '=' in c and c not in ADDED)
    expected_dummies = sorted(f'{c}={v}' for c in levels for v in levels[c][1:])
    add('structure', same('column_layout', tag,
                          (sorted(NUMERIC[representation]), expected_dummies, True, True),
                          (sorted(numerics), dummies, columns[:1] == ['intercept'], len(set(columns)) == p)))
    add('structure', same('static_added_block', tag, list(ADDED) if ctx.static_model == 'M2' else [],
                          columns[-3:] if added else []))

    knots = None
    if ctx.static_model == 'M2':
        state = str(row.get('latitude_state') or '')
        if not state and protocol == 'full_sample':
            state = ctx.full_states.get(representation, MISSING)
        try:
            if state == MISSING or not state:
                raise ValueError('no latitude state')
            knots, state_n = parse_state(state)
        except (ValueError, KeyError, TypeError) as error:
            add('structure', flag('latitude_state_parses', tag, False, repr(error)))
            return result
        own = knots_from(ctx.numerics[representation]['abs_latitude'][train])
        add('structure', gap('knots_rederived_from_training_latitudes', tag,
                             max(abs(a - b) for a, b in zip(own, knots)), KNOT_TOL))
        add('structure', same('latitude_state_n_train', tag, n_train, state_n))
    else:
        add('structure', same('no_latitude_state_for_M0star', tag, '', str(row.get('latitude_state') or '')))

    try:
        x_train = design_matrix(ctx, representation, columns, train, knots)
    except (ValueError, KeyError, IndexError) as error:
        add('structure', flag('design_rebuilds', tag, False, repr(error)))
        return result
    rank = int(np.linalg.matrix_rank(x_train))
    add('structure', same('design_rank_equals_p', tag, p, rank))
    add('structure', flag('training_nodes_at_least_9', tag, n_train >= mi.MIN_TRAINING_NODES))

    graph = graphs.get((protocol, label))
    if graph is None:
        add('graphs', flag('graph_rows_present', tag, False))
        return result
    add('graphs', same('graph_rows_are_the_arm', tag, {ctx.arm}, graph['weights']))
    add('graphs', sequence('graph_training_nodes_equal_the_fit_training_set', tag, train_iso,
                           [iso for iso, _ in graph['training']]))
    saved_lists = [nb for _, nb in graph['training']]
    local = {iso: k for k, iso in enumerate(train_iso)}
    valid_lists = (len(saved_lists) == n_train and all(
        len(nb) == mi.K_NEIGHBOURS and len(set(nb)) == mi.K_NEIGHBOURS and all(j in local for j in nb)
        and train_iso[i] not in nb for i, nb in enumerate(saved_lists)))
    add('graphs', flag('saved_training_lists_are_eight_distinct_training_nodes', tag, valid_lists))
    recomputed = [[ctx.order[j] for j in nb] for nb in mi.knn_neighbours(ctx.dist, train)] if n_train >= 9 else []
    add('graphs', sequence('training_neighbour_lists_equal_knn8_recomputation', tag,
                           ['|'.join(nb) for nb in recomputed], ['|'.join(nb) for nb in saved_lists]))
    if not valid_lists:
        return result
    local_lists = [tuple(local[j] for j in nb) for nb in saved_lists]
    w = weights_from_lists(local_lists, n_train)

    test_iso = [iso for iso, _ in graph['held_out']]
    attachments = None
    if protocol == 'full_sample':
        add('graphs', same('full_sample_has_no_held_out_nodes', tag, 0, len(test_iso)))
    else:
        members = [ctx.order[i] for i in ctx.fold_members[protocol].get(label, [])]
        add('graphs', sequence('held_out_nodes_equal_the_fold_members', tag, members, test_iso))
        try:
            test = [ctx.pos[i] for i in test_iso]
            again = mi.attachment_neighbours(ctx.dist, train, test)
            saved_attach = [nb for _, nb in graph['held_out']]
            add('graphs', sequence('held_out_attachments_equal_sink_rule_recomputation', tag,
                                   ['|'.join(ctx.order[j] for j in nb) for nb in again],
                                   ['|'.join(nb) for nb in saved_attach]))
            attachments = [tuple(local[j] for j in nb) for nb in saved_attach]
            mi.dense_weights(attachments, n_train)
            result.test, result.attachments = test, [[ctx.pos[j] for j in nb] for nb in saved_attach]
        except (KeyError, ValueError, mi.StructuralFailure) as error:
            add('graphs', flag('held_out_attachments_valid', tag, False, repr(error)))
            return result

    y_train = ctx.y[train]
    eigenvalues = mi.weight_eigenvalues(w)
    at = mi.profile(family, y_train, x_train, w, theta, eigenvalues)
    add('likelihood', same('evaluation_valid_at_saved_theta', tag, None, at.failure))
    add('likelihood', close('loglik', tag, at.loglik, row['loglik'], LOGLIK_TOL))
    sigma2 = float(row['sigma2'])
    add('likelihood', gap('sigma2_relative', tag, abs(at.sigma2 - sigma2) / abs(sigma2) if sigma2 else math.nan,
                          SIGMA2_TOL))
    add('likelihood', same('at_domain_bound', tag, bool(abs(theta) > mi.DOMAIN[1] - BOUND_MARGIN),
                           as_bool(row['at_domain_bound'])))
    add('likelihood', flag('theta_in_domain', tag, mi.DOMAIN[0] <= theta <= mi.DOMAIN[1]))
    add('likelihood', same('evaluations', tag, EXPECTED_EVALUATIONS, int(row['evaluations'])))
    add('coefficients', same('beta_length', tag, p, len(beta)))
    if len(beta) == p:
        scaled = np.abs(at.beta - beta) / np.maximum(1.0, np.abs(beta))
        add('coefficients', {**gap('beta_qr_at_saved_theta', tag, float(np.max(scaled)), BETA_TOL),
                             'raw': float(np.max(np.abs(at.beta - beta)))})

    local_check = mi.local_optimality_check(family, y_train, x_train, w, theta)
    add('optimality', {**flag('local_maximum_on_fine_grid', tag, local_check['passes'],
                              {'shortfall': local_check['shortfall'], 'grid': [local_check['grid_min'],
                                                                                local_check['grid_max']]})})
    scan = mi.global_scan(family, y_train, x_train, w, theta)
    add('optimality', flag('in_global_grid_basin', tag, scan.get('in_global_basin') is True,
                           {'theta_argmax': scan.get('theta_argmax'), 'basin': scan.get('basin'),
                            'failed_points': len(scan.get('failed_points', []))}))

    finite = bool(math.isfinite(theta) and np.isfinite(beta).all() and math.isfinite(sigma2))
    result.computable = bool(at.valid and finite and rank == p and n_train >= mi.MIN_TRAINING_NODES
                             and len(beta) == p)
    result.at_bound = as_bool(row['at_domain_bound'])

    if protocol == 'full_sample':
        grid = ctx.grids.get((representation, family_code))
        if grid is None:
            add('likelihood', flag('full_sample_grid_present', tag, False))
        else:
            thetas, values = grid
            add('likelihood', arrays('grid_theta_values', tag, thetas,
                                     np.linspace(mi.DOMAIN[0], mi.DOMAIN[1], mi.GRID_POINTS), GRID_THETA_TOL))
            again = [mi.profile(family, y_train, x_train, w, t, eigenvalues).loglik for t in thetas]
            add('likelihood', arrays('grid_loglik', tag, again, values, LOGLIK_TOL))
            add('likelihood', same('grid_local_maxima', tag, mi.strict_local_maxima(values),
                                   int(row['grid_local_maxima'])))
            add('likelihood', gap('estimate_at_least_grid_maximum', tag, max(0.0, max(values) - float(row['loglik'])),
                                  LOGLIK_TOL))
        rows = predictions.get((representation, family_code, 'in_sample'))
        if rows is not None and len(beta) == p:
            q = mi.in_sample_quantities(family, ctx.y, x_train, local_lists, theta, beta, sigma2)
            saved = rows.reindex(ctx.order)
            add('predictions', arrays('in_sample_one_step', tag, q['one_step'], saved.prediction, PREDICTION_TOL))
            add('predictions', arrays('in_sample_trend', tag, q['trend'], saved.trend, PREDICTION_TOL))
            add('predictions', gap('innovation_expressions_agree', tag, q['innovation_expression_gap'],
                                   QUANTITY_IDENTITY_TOL))
            add('predictions', gap('mean_squared_innovation_equals_sigma2', tag, q['sigma2_relative_gap'],
                                   QUANTITY_IDENTITY_TOL))
            add('predictions', same('in_sample_row_fields', tag, ([-1] * len(train), [len(train)] * len(train),
                                                                 [0] * len(train), [0.0] * len(train)),
                                    (saved.fold_id.tolist(), saved.n_train.tolist(), saved.unseen_levels.tolist(),
                                     saved.adjustment.astype(float).tolist())))
            result.full = {'family': family, 'x': x_train, 'w': w, 'q': q, 'columns': columns, 'beta': beta,
                           'theta': theta}
            result.computable = result.computable and bool(np.isfinite(q['trend']).all())
        elif len(beta) == p:
            q = mi.in_sample_quantities(family, ctx.y, x_train, local_lists, theta, beta, sigma2)
            result.full = {'family': family, 'x': x_train, 'w': w, 'q': q, 'columns': columns, 'beta': beta,
                           'theta': theta}
        return result

    rows = predictions.get((representation, family_code, protocol))
    if rows is None or len(beta) != p:
        return result
    try:
        x_test = design_matrix(ctx, representation, columns, result.test, knots)
        names = [f'added:{c}' if c in ADDED else c for c in columns]   # keep the interaction out of 'hemisphere='
        test_labels = {c: ctx.labels[c][result.test] for c in levels}
        adjustment = mi.unseen_level_adjustment(beta, names, levels, test_labels, len(result.test))
        prediction = mi.held_out_predictions(family, theta, beta, x_train, y_train, x_test, attachments, adjustment)
    except (ValueError, KeyError) as error:
        add('predictions', flag('held_out_prediction_rebuilds', tag, False, repr(error)))
        return result
    saved = rows.reindex(test_iso)
    add('predictions', arrays('held_out_prediction', tag, prediction, saved.prediction, PREDICTION_TOL))
    add('predictions', arrays('held_out_adjustment', tag, adjustment, saved.adjustment, PREDICTION_TOL))
    unseen = [sum(1 for c in levels if test_labels[c][k] not in set(levels[c])) for k in range(len(test_iso))]
    fold = ctx.fold_ids[protocol][result.test].tolist()
    add('predictions', same('held_out_row_fields', tag, (unseen, fold, [n_train] * len(test_iso)),
                            (saved.unseen_levels.tolist(), saved.fold_id.tolist(), saved.n_train.tolist())))
    result.computable = result.computable and bool(np.isfinite(prediction).all())
    return result


# ---------------------------------------------------------------------
# Cards, paired comparisons, qualification, naming
# ---------------------------------------------------------------------


def protocol_values(ctx: Context, rows: pd.DataFrame, one_step: np.ndarray) -> dict:
    saved = rows.reindex(ctx.order)
    y, prediction = saved.observed.to_numpy(dtype=float), saved.prediction.to_numpy(dtype=float)
    folds = saved.fold_id.to_numpy()
    e = y - prediction
    ids = sorted(set(folds.tolist()))
    fold_rmse = [math.sqrt(math.fsum(v * v for v in e[folds == f]) / int((folds == f).sum())) for f in ids]
    worst_fold = ids[int(np.argmax(fold_rmse))]
    absolute = np.abs(e)
    calibration = mi.calibration(y, prediction)
    table = region_table(ctx.regions, e)
    region, region_rmse, region_n = worst_region(table)
    return {
        'n': len(y), 'cv_r2': mi.r2(y, prediction), 'cv_rmse': mi.rmse(y, prediction), 'cv_mae': mi.mae(y, prediction),
        'in_sample_one_step_r2': mi.r2(ctx.y, one_step), 'in_sample_one_step_rmse': mi.rmse(ctx.y, one_step),
        'n_folds': len(ids), 'fold_rmse_median': float(np.median(fold_rmse)), 'fold_rmse_min': min(fold_rmse),
        'fold_rmse_max': max(fold_rmse), 'worst_fold': int(worst_fold),
        'fold_error_iqr': float(np.quantile(absolute, 0.75) - np.quantile(absolute, 0.25)),
        'calibration_slope': calibration['slope'], 'calibration_intercept': calibration['intercept'],
        'mean_n_train': math.fsum(saved.n_train.astype(float)) / len(y),
        'rows_with_unseen_level': int((saved.unseen_levels.to_numpy() > 0).sum()),
        'residual_morans_i_in_sample': mi.moran_statistic(ctx.y - one_step, ctx.station_w),
        'residual_morans_i_cv': mi.moran_statistic(e, ctx.station_w),
        'recorded_land_centroid_moran_cv': mi.moran_statistic(e, ctx.land_w),
        'worst_region': {'region': region, 'rmse': region_rmse, 'n': region_n},
        'region_rmse': {r: s['rmse'] for r, s in table.items()}, 'region_mae': {r: s['mae'] for r, s in table.items()},
        'region_bias': {r: s['bias'] for r, s in table.items()},
        'prediction': prediction, 'errors': e,
    }


EXACT_CARD = ('n', 'n_folds', 'worst_fold', 'rows_with_unseen_level')
METRIC_CARD = ('cv_r2', 'cv_rmse', 'cv_mae', 'in_sample_one_step_r2', 'in_sample_one_step_rmse', 'fold_rmse_median',
               'fold_rmse_min', 'fold_rmse_max', 'fold_error_iqr', 'mean_n_train')
MORAN_CARD = ('residual_morans_i_in_sample', 'residual_morans_i_cv', 'recorded_land_centroid_moran_cv')


def card_items(label, values, card) -> list:
    items = [same(f'card.{k}', label, values[k], node(card, k)) for k in EXACT_CARD]
    items += [close(f'card.{k}', label, values[k], node(card, k), METRIC_TOL) for k in METRIC_CARD]
    items += [close(f'card.{k}', label, values[k], node(card, k), CALIBRATION_TOL, scale=True)
              for k in ('calibration_slope', 'calibration_intercept')]
    items += [close(f'card.{k}', label, values[k], node(card, k), MORAN_TOL) for k in MORAN_CARD]
    items.append(same('card.worst_region.region', label, values['worst_region']['region'],
                      node(card, 'worst_region', 'region')))
    items.append(same('card.worst_region.n', label, values['worst_region']['n'], node(card, 'worst_region', 'n')))
    items.append(close('card.worst_region.rmse', label, values['worst_region']['rmse'],
                       node(card, 'worst_region', 'rmse'), METRIC_TOL))
    for key in ('region_rmse', 'region_mae', 'region_bias'):
        saved = node(card, key)
        items.append(same(f'card.{key}.regions', label, sorted(values[key]), sorted(saved) if isinstance(saved, dict) else saved))
        items += [close(f'card.{key}', f'{label}/{r}', v, node(card, key, r), METRIC_TOL) for r, v in values[key].items()]
    return items


@dataclasses.dataclass
class Recomputed:
    computable: dict
    values: dict        # (rep, fam) -> {protocol: values}, plus 'trend_r2'
    paired: dict        # (rep, fam) -> dict
    conditions: dict    # (rep, fam) -> qualification inputs
    accounting: dict    # (rep, fam) -> dict


def entry_of(ev: Evidence, representation, family):
    return node(ev.scorecard, 'families', f'{representation}/{family}')


def metric_and_paired_items(ev: Evidence, ctx: Context, results: dict, predictions: dict, rec: Recomputed) -> list:
    items = []
    for representation in REPRESENTATIONS:
        static_rows = {p: _static_rows(ev.m2_predictions, representation, ctx.static_model, p, ctx.order)
                       for p in PROTOCOLS}
        for family in FAMILIES:
            label = f'{representation}/{family}'
            entry = entry_of(ev, representation, family)
            if entry == MISSING:
                items.append(flag('scorecard_entry_present', label, False))
                continue
            status = entry.get('status')
            items.append(same('status_equals_recomputed_computability', label,
                              'computable' if rec.computable.get((representation, family)) else 'non_computable', status))
            if status != 'computable':
                items.append(flag('non_computable_has_no_prediction_rows', label,
                                  all((representation, family, p) not in predictions for p in ('in_sample', *PROTOCOLS))))
                continue
            full = results.get((representation, family, 'full_sample', 'all'))
            in_rows = predictions.get((representation, family, 'in_sample'))
            if full is None or full.full is None or in_rows is None or any(
                    (representation, family, p) not in predictions for p in PROTOCOLS):
                items.append(flag('computable_entry_has_full_fit_and_all_prediction_rows', label, False))
                continue
            one_step = in_rows.reindex(ctx.order).prediction.to_numpy(dtype=float)
            card = entry.get('card', {})
            values = {}
            for protocol in PROTOCOLS:
                rows = predictions[(representation, family, protocol)]
                items.append(same('prediction_rows_cover_the_order_once', f'{label}/{protocol}', sorted(ctx.order),
                                  sorted(rows.index.tolist())))
                items.append(same('prediction_fold_ids_equal_static_folds', f'{label}/{protocol}',
                                  ctx.fold_ids[protocol].tolist(), rows.reindex(ctx.order).fold_id.tolist()))
                saved = rows.reindex(ctx.order)
                items.append(arrays('saved_error_equals_observed_minus_prediction', f'{label}/{protocol}',
                                    saved.observed - saved.prediction, saved.error, METRIC_TOL))
                values[protocol] = protocol_values(ctx, rows, one_step)
                items += card_items(f'{label}/{protocol}', values[protocol], node(card, *CARD_PATH[protocol]))
            trend = in_rows.reindex(ctx.order).trend.to_numpy(dtype=float)
            values['trend_r2'] = mi.r2(ctx.y, trend)
            values['structural_residual_moran_in_sample'] = mi.moran_statistic(ctx.y - trend, ctx.station_w)
            items.append(close('card.trend_r2', label, values['trend_r2'], node(card, 'trend_r2'), METRIC_TOL))
            items.append(close('card.structural_residual_moran_in_sample', label,
                               values['structural_residual_moran_in_sample'],
                               node(card, 'structural_residual_moran_in_sample'), MORAN_TOL))
            p = len(full.full['columns'])
            items.append(same('card.effective_degrees_of_freedom', label, p + 1, node(card, 'effective_degrees_of_freedom')))
            sigma2 = ev.fits[(ev.fits.representation == representation) & (ev.fits.family == family)
                             & (ev.fits.protocol == 'full_sample')].sigma2
            items.append(close('card.sigma2', label, sigma2.iloc[0] if len(sigma2) else math.nan, node(card, 'sigma2'),
                               METRIC_TOL, scale=True))
            rec.values[(representation, family)] = values

            y = ctx.y
            static = {p_: static_rows[p_].prediction.to_numpy(dtype=float) for p_ in PROTOCOLS}
            items.append(flag('static_observed_equals_the_outcome', label, all(
                np.array_equal(static_rows[p_].observed.to_numpy(dtype=float), y) for p_ in PROTOCOLS)))
            new = values[PRIMARY]['prediction']
            pair = mi.paired_rmse_interval(y, new, static[PRIMARY])
            saved = node(entry, 'paired', 'vs_static')
            items.append(close('paired.vs_static.delta_rmse', label, pair['delta_rmse'], node(saved, 'delta_rmse'), METRIC_TOL))
            for k, name in enumerate(('low', 'high')):
                items.append(close(f'paired.vs_static.interval_{name}', label, pair['interval'][k],
                                   node(saved, 'country_bootstrap_95_interval')[k]
                                   if isinstance(node(saved, 'country_bootstrap_95_interval'), list) else MISSING,
                                   METRIC_TOL))
            items.append(same('paired.vs_static.resamples_and_seed', label, (mi.RESAMPLES, mi.RESAMPLE_SEED),
                              (node(saved, 'resamples'), node(saved, 'seed'))))
            change = np.abs(y - new) - np.abs(y - static[PRIMARY])
            items.append(same('paired.countries', label,
                              {'improved': int((change < 0).sum()), 'worsened': int((change > 0).sum()),
                               'tied': int((change == 0).sum())}, node(entry, 'paired', 'countries')))
            for protocol in ('m49_subregion_lo', 'random10'):
                items.append(close(f'paired.{protocol}_rmse_change_vs_static', label,
                                   mi.rmse(y, values[protocol]['prediction']) - mi.rmse(y, static[protocol]),
                                   node(entry, 'paired', f'{protocol}_rmse_change_vs_static'), METRIC_TOL))
            if ctx.static_model == 'M2':
                m0 = _static_rows(ev.m2_predictions, representation, 'M0star', PRIMARY, ctx.order)
                descriptive = mi.paired_rmse_interval(y, new, m0.prediction.to_numpy(dtype=float))
                saved = node(entry, 'paired', 'vs_m0star_descriptive')
                bounds = node(saved, 'country_bootstrap_95_interval')
                items.append(close('paired.vs_m0star_descriptive.delta_rmse', label, descriptive['delta_rmse'],
                                   node(saved, 'delta_rmse'), METRIC_TOL))
                items.append(gap('paired.vs_m0star_descriptive.interval', label,
                                 max(abs(a - b) for a, b in zip(descriptive['interval'], bounds))
                                 if isinstance(bounds, list) and len(bounds) == 2 else math.nan, METRIC_TOL))
            static_m49 = mi.rmse(y, static['m49_subregion_lo'])
            static_worst = worst_region(region_table(ctx.regions, y - static[PRIMARY]))[1]
            rec.paired[(representation, family)] = pair
            rec.conditions[(representation, family)] = {
                'delta': pair['delta_rmse'], 'low': pair['interval'][0], 'high': pair['interval'][1],
                'm49_change': values['m49_subregion_lo']['cv_rmse'] - static_m49,
                'worst_change': values[PRIMARY]['worst_region']['rmse'] - static_worst}
    return items


def qualification_items(ev: Evidence, ctx: Context, rec: Recomputed, veto=VETO) -> tuple[list, dict]:
    items, qualifies = [], {}
    for representation in REPRESENTATIONS:
        for family in FAMILIES:
            label = f'{representation}/{family}'
            entry = entry_of(ev, representation, family)
            if entry == MISSING:
                continue
            deciding = ctx.arm == 'station' and representation == PRIMARY_REPRESENTATION
            key = 'qualification' if deciding else 'arm_conditions'
            saved = node(entry, key)
            computable = entry.get('status') == 'computable' and rec.computable.get((representation, family), False)
            inputs = rec.conditions.get((representation, family))
            if computable and inputs is not None:
                q = qualification(inputs['delta'], inputs['low'], inputs['high'], inputs['m49_change'],
                                  inputs['worst_change'], True, veto)
            else:
                q = qualification(math.nan, math.nan, math.nan, math.nan, math.nan, False, veto)
            for name, value in q.items():
                items.append(same(f'{key}.{name}', label, value, node(saved, name)))
            if not deciding:
                items.append(same(f'{key}.descriptive_only', label, True, node(saved, 'descriptive_only')))
            if computable and inputs is not None and q.get('qualifies') is not None:
                items.append(close(f'{key}.delta_rmse', label, inputs['delta'], node(saved, 'delta_rmse'), METRIC_TOL))
                interval = node(saved, 'interval')
                items.append(gap(f'{key}.interval', label, max(abs(inputs['low'] - interval[0]), abs(inputs['high'] - interval[1]))
                                 if isinstance(interval, list) and len(interval) == 2 else math.nan, METRIC_TOL))
                items.append(close(f'{key}.m49_rmse_change', label, inputs['m49_change'], node(saved, 'm49_rmse_change'),
                                   METRIC_TOL))
                items.append(close(f'{key}.worst_region_rmse_change', label, inputs['worst_change'],
                                   node(saved, 'worst_region_rmse_change'), METRIC_TOL))
                items.append(same(f'{key}.veto_threshold', label, VETO, node(saved, 'veto_threshold')))
            if deciding:
                qualifies[family] = q.get('qualifies')
    return items, qualifies


def naming_items(ev: Evidence, ctx: Context, qualifies: dict, integrity: bool) -> list:
    saved = ev.scorecard.get('final_naming', MISSING)
    if ctx.arm != 'station' or not integrity:
        return [same('final_naming_is_null', f'{ctx.arm} arm, recomputed integrity {integrity}', None, saved)]
    expected = final_naming(ctx.static, qualifies)
    return [same(f'final_naming.{k}', 'station arm', v, node(saved, k)) for k, v in expected.items()]


# ---------------------------------------------------------------------
# Dependence summaries, graph descriptors, accounting, identity
# ---------------------------------------------------------------------


def dependence_and_descriptor_items(ev: Evidence, ctx: Context, results: dict) -> list:
    items = []
    neighbours = mi.knn_neighbours(ctx.dist, range(len(ctx.order)))
    for representation in REPRESENTATIONS:
        for family in FAMILIES:
            label = f'{representation}/{family}'
            entry = entry_of(ev, representation, family)
            full = results.get((representation, family, 'full_sample', 'all'))
            if entry == MISSING:
                continue
            if full is not None and 'dependence_parameter_full_sample' in entry:
                saved = entry['dependence_parameter_full_sample']
                items.append(same('dependence_parameter_full_sample.theta_and_bound', label,
                                  (full.theta, full.at_bound), (node(saved, 'theta'), node(saved, 'at_domain_bound'))))
            if entry.get('status') != 'computable' or full is None:
                continue
            saved = node(entry, 'dependence_parameter')
            items.append(same('dependence_parameter.full_sample', label, (full.theta, full.at_bound),
                              (node(saved, 'full_sample', 'theta'), node(saved, 'full_sample', 'at_domain_bound'))))
            for protocol in PROTOCOLS:
                fits = [results.get((representation, family, protocol, lab)) for lab in ctx.required[protocol]]
                if any(f is None for f in fits):
                    items.append(flag('dependence_parameter.fits_present', f'{label}/{protocol}', False))
                    continue
                thetas = np.array([f.theta for f in fits])
                block = node(saved, protocol)
                items.append(same('dependence_parameter.counts', f'{label}/{protocol}',
                                  (len(thetas), int((thetas > 0).sum()), sum(f.at_bound for f in fits)),
                                  (node(block, 'n_fits'), node(block, 'strictly_positive_n'), node(block, 'at_domain_bound_n'))))
                for name, value in (('median', float(np.median(thetas))), ('min', float(thetas.min())),
                                    ('max', float(thetas.max()))):
                    items.append(close(f'dependence_parameter.{name}', f'{label}/{protocol}', value, node(block, name),
                                       METRIC_TOL))
                if protocol == PRIMARY:
                    items.append(close('dependence_parameter.same_sign_as_full_fraction_primary', label,
                                       float(np.mean(np.sign(thetas) == np.sign(full.theta))),
                                       node(saved, 'same_sign_as_full_fraction_primary'), METRIC_TOL))
            descriptors = node(entry, 'graph_descriptors')
            first = [ctx.dist[i, nb[0]] for i, nb in enumerate(neighbours)]
            eighth = [ctx.dist[i, nb[-1]] for i, nb in enumerate(neighbours)]
            for name, value in (('first_neighbour_km_median', np.median(first)), ('first_neighbour_km_max', max(first)),
                                ('eighth_neighbour_km_median', np.median(eighth)), ('eighth_neighbour_km_max', max(eighth))):
                items.append(close(f'graph_descriptors.full_sample.{name}', label, float(value),
                                   node(descriptors, 'full_sample', name), DISTANCE_TOL_KM))
            for protocol in PROTOCOLS:
                first, eighth = [], []
                for lab in ctx.required[protocol]:
                    fit = results.get((representation, family, protocol, lab))
                    if fit is None or fit.attachments is None:
                        continue
                    for o, nb in zip(fit.test, fit.attachments):
                        first.append(ctx.dist[o, nb[0]])
                        eighth.append(ctx.dist[o, nb[-1]])
                if not first:
                    items.append(flag('graph_descriptors.attachments_present', f'{label}/{protocol}', False))
                    continue
                for stem, values in (('first', first), ('eighth', eighth)):
                    for stat, fn in (('min', np.min), ('median', np.median), ('max', np.max)):
                        name = f'attachment_{stem}_km_{stat}'
                        items.append(close(f'graph_descriptors.{name}', f'{label}/{protocol}', float(fn(values)),
                                           node(descriptors, protocol, name), DISTANCE_TOL_KM))
    return items


def accounting_items(ev: Evidence, ctx: Context, results: dict, rec: Recomputed) -> list:
    items = []
    for representation in REPRESENTATIONS:
        for family in FAMILIES:
            label = f'{representation}/{family}'
            entry = entry_of(ev, representation, family)
            full = results.get((representation, family, 'full_sample', 'all'))
            if entry == MISSING:
                continue
            has_full = full is not None and full.full is not None and full.computable
            items.append(same('accounting_reported_iff_valid_full_sample_fit', label, has_full, 'accounting' in entry))
            if not has_full or 'accounting' not in entry:
                continue
            saved = entry['accounting']
            fitted = full.full
            y, x, q = ctx.y, fitted['x'], fitted['q']
            shares = {g: node(ev.m2_scorecard, 'representations', representation, ctx.static_model, f'share_{g}')
                      for g in GROUPS}
            saved_shares = node(saved, 'static_lmg_shares')
            for g in GROUPS:
                items.append(close('accounting.static_lmg_shares_equal_committed_m2_shares', f'{label}/{g}',
                                   node(saved_shares, g), shares[g], METRIC_TOL))
            try:
                parts = mi.accounting_parts(y, x, q['innovation'], {g: float(v) for g, v in saved_shares.items()})
            except (ValueError, AttributeError, TypeError) as error:
                items.append(flag('accounting.parts_rebuild', label, False, repr(error)))
                continue
            values = {'a_static': parts['static_component'], 'a_dependence': parts['represented_spatial_dependence'],
                      'a_innovation': parts['remaining_innovation'],
                      'absorbed_fraction_of_static_residual': parts['absorbed_fraction_of_static_residual'],
                      'identity_sum': parts['static_component'] + parts['represented_spatial_dependence']
                      + parts['remaining_innovation'],
                      'tss': parts['tss'], 'rss_static': parts['rss_static'], 'rss_innovation': parts['rss_innovation'],
                      'one_step_r2': 1.0 - parts['rss_innovation'] / parts['tss'],
                      'trend_r2': mi.r2(y, q['trend'])}
            for name, value in values.items():
                items.append(close(f'accounting.estimands.{name}', label, value, node(saved, 'estimands', name),
                                   ACCOUNTING_TOL, scale=name in ('tss', 'rss_static', 'rss_innovation')))
            items.append(gap('accounting.identity_sum_minus_one', label, abs(parts['identity_gap']), ACCOUNTING_TOL))
            items.append(gap('accounting.static_share_sum_minus_a_static', label, abs(parts['static_shares_sum_gap']),
                             ACCOUNTING_TOL))
            items.append(same('accounting.static_lmg_share_sum_equals_a_static', label, True,
                              node(saved, 'static_lmg_share_sum_equals_a_static')))
            static_r2 = math.fsum(float(v) for v in saved_shares.values())
            for g, v in saved_shares.items():
                items.append(close('accounting.composition.static', f'{label}/{g}', float(v) / static_r2,
                                   node(saved, 'composition', 'static', g), ACCOUNTING_TOL))

            columns = fitted['columns']
            owner = [column_group(c) for c in columns]
            groups = {g: x[:, [j for j, h in enumerate(owner) if h == g]] for g in GROUPS if g in owner}
            filtered = mi.filtered_decomposition(fitted['family'], y, groups, fitted['w'], fitted['theta'])
            block = node(saved, 'filtered_trend_shares')
            for g, v in filtered['shares'].items():
                items.append(close('accounting.filtered_trend_shares.shares', f'{label}/{g}', v, node(block, 'shares', g),
                                   ACCOUNTING_TOL))
                items.append(close('accounting.composition.filtered', f'{label}/{g}', filtered['composition'][g],
                                   node(saved, 'composition', 'filtered', g), ACCOUNTING_TOL))
            items.append(same('accounting.filtered_trend_shares.groups', label, sorted(filtered['shares']),
                              sorted(node(block, 'shares')) if isinstance(node(block, 'shares'), dict) else MISSING))
            items.append(close('accounting.filtered_trend_shares.r2_full', label, filtered['r2_full'],
                               node(block, 'r2_full'), ACCOUNTING_TOL))
            items.append(close('accounting.filtered_trend_shares.residual_share', label, filtered['residual'],
                               node(block, 'residual_share'), ACCOUNTING_TOL))
            items.append(same('accounting.filtered_trend_shares.n_coalitions', label, 2 ** len(groups),
                              node(block, 'n_coalitions')))
            items.append(gap('accounting.filtered_shares_sum_to_r2_full', label, abs(filtered['shares_sum_gap']),
                             ACCOUNTING_TOL))
            m = np.eye(len(y)) - fitted['theta'] * fitted['w']
            target = (m @ x if fitted['family'] == 'SEM' else x) @ fitted['beta']
            items.append(gap('accounting.full_coalition_reproduces_the_ml_transformed_fit', label,
                             float(np.max(np.abs(filtered['full_fitted'] - target))), ACCOUNTING_TOL))
            material = filtered.get('material_change', {}).get('material_change')
            items.append(same('accounting.material_change_of_non_spatial_part', label, material,
                              node(saved, 'material_change_of_non_spatial_part')))
            rec.accounting[(representation, family)] = {**values, 'identity_gap': abs(parts['identity_gap'])}
    return items


def identity_items(ev: Evidence, ctx: Context, results: dict, predictions: dict, rec: Recomputed) -> tuple[list, bool]:
    items = []
    saved = ev.scorecard.get('representation_identity', MISSING)
    statuses = {(r, f): node(entry_of(ev, r, f), 'status') for r in REPRESENTATIONS for f in FAMILIES}
    overall = True
    for family in FAMILIES:
        a, b = statuses[(PRIMARY_REPRESENTATION, family)], statuses[('per_capita', family)]
        if a != b:
            items.append(same('identity.passes_when_statuses_differ', family, False, node(saved, 'passes')))
            return items, False
    for family in FAMILIES:
        status = statuses[(PRIMARY_REPRESENTATION, family)]
        if status != 'computable':
            items.append(same('identity.non_computable_record', family, {'status': status}, node(saved, family)))
            continue
        keys = [(r, family) for r in REPRESENTATIONS]
        if not all(k in rec.values and k in rec.accounting and k in rec.paired for k in keys):
            items.append(flag('identity.recomputed_inputs_present', family, False))
            overall = False
            continue
        report = node(saved, family)
        diffs = {}
        rows = {r: {p: predictions[(r, family, p)].reindex(ctx.order).prediction.to_numpy(dtype=float)
                    for p in ('in_sample', *PROTOCOLS)} for r in REPRESENTATIONS}
        diffs['in_sample_one_step'] = float(np.max(np.abs(rows[REPRESENTATIONS[0]]['in_sample']
                                                          - rows[REPRESENTATIONS[1]]['in_sample'])))
        for protocol in PROTOCOLS:
            diffs[protocol] = float(np.max(np.abs(rows[REPRESENTATIONS[0]][protocol] - rows[REPRESENTATIONS[1]][protocol])))
        for name, value in diffs.items():
            items.append(close('identity.predictions', f'{family}/{name}', value, node(report, 'predictions', name),
                               IDENTITY_REPORT_TOL))
        thetas = []
        for protocol, labels in (('full_sample', ['all']), (PRIMARY, ctx.required[PRIMARY])):
            for lab in labels:
                fa, fb = (results.get((r, family, protocol, lab)) for r in REPRESENTATIONS)
                thetas.append(abs(fa.theta - fb.theta) if fa is not None and fb is not None else math.nan)
        theta_max = max(thetas) if thetas and all(math.isfinite(t) for t in thetas) else math.nan
        items.append(close('identity.theta_max', family, theta_max, node(report, 'theta_max'), IDENTITY_REPORT_TOL))
        va, vb = (rec.values[(r, family)] for r in REPRESENTATIONS)

        def score(values, name):
            if name in ('trend_r2',):
                return values['trend_r2']
            if name == 'worst_region.rmse':
                return values[PRIMARY]['worst_region']['rmse']
            if name.startswith('secondary.'):
                return values['m49_subregion_lo'][name.split('.', 1)[1]]
            if name.startswith('random_reference_only.'):
                return values['random10'][name.split('.', 1)[1]]
            return values[PRIMARY][name]
        scores = {name: abs(score(va, name) - score(vb, name)) for name in EQUIVALENT_SCORES}
        pa, pb = (rec.paired[(r, family)] for r in REPRESENTATIONS)
        scores['delta_rmse'] = abs(pa['delta_rmse'] - pb['delta_rmse'])
        scores['interval'] = max(abs(x - z) for x, z in zip(pa['interval'], pb['interval']))
        for name, value in scores.items():
            items.append(close('identity.scores', f'{family}/{name}', value, node(report, 'scores', name),
                               IDENTITY_REPORT_TOL))
        parts = {name: abs(rec.accounting[keys[0]][name] - rec.accounting[keys[1]][name]) for name in ACCOUNTING_PARTS}
        for name, value in parts.items():
            items.append(close('identity.accounting_parts', f'{family}/{name}', value,
                               node(report, 'accounting_parts', name), IDENTITY_REPORT_TOL))
        ok = (all(v <= IDENTITY_TOL for v in diffs.values()) and math.isfinite(theta_max)
              and theta_max <= IDENTITY_THETA_TOL and all(v <= IDENTITY_TOL for v in scores.values())
              and all(v <= IDENTITY_TOL for v in parts.values()))
        items.append(same('identity.family_passes', family, bool(ok), node(report, 'passes')))
        overall &= bool(ok)
    items.append(same('identity.passes', 'overall', bool(overall), node(saved, 'passes')))
    items.append(same('identity.tolerances', 'amendment A1', {'predictions': IDENTITY_TOL, 'theta': IDENTITY_THETA_TOL,
                                                              'scores_and_accounting': IDENTITY_TOL},
                      node(saved, 'tolerances')))
    return items, bool(overall)


def coverage_items(ev: Evidence, ctx: Context, results: dict, duplicates: list) -> tuple[list, dict]:
    items = [same('fit_rows_unique', 'm3_fits', [], duplicates)]
    computable = {}
    for representation in REPRESENTATIONS:
        for family in FAMILIES:
            label = f'{representation}/{family}'
            required = [('full_sample', 'all')] + [(p, lab) for p in PROTOCOLS for lab in ctx.required[p]]
            present = [key[2:] for key in results if key[:2] == (representation, family)]
            items.append(flag('fit_rows_are_required_fits', label, set(present) <= set(required),
                              sorted(set(present) - set(required))[:5]))
            computable[(representation, family)] = bool(
                all((representation, family, *k) in results for k in required)
                and all(results[(representation, family, *k)].computable for k in required))
    grid_keys = {(r, f) for r, f in zip(ev.grid.representation, ev.grid.family)}
    full_keys = {k[:2] for k in results if k[2] == 'full_sample'}
    items.append(same('full_sample_grid_present_for_every_full_sample_fit', 'm3_full_sample_grid', sorted(full_keys),
                      sorted(grid_keys)))
    return items, computable


def consistency_items(ev: Evidence, ctx: Context, identity_ok: bool, rec: Recomputed) -> tuple[list, bool]:
    items = []
    manifest, scorecard = ev.manifest, ev.scorecard
    families = scorecard.get('families', {})
    items.append(same('weights_agree', 'manifest/scorecard/fits', (ctx.arm, ctx.arm, {ctx.arm}),
                      (manifest.get('weights'), scorecard.get('weights'), set(ev.fits.weights) or {ctx.arm})))
    items.append(same('retained_static_specification_agrees', 'manifest/scorecard/m2 manifest', (ctx.static, ctx.static),
                      (manifest.get('retained_static_specification'), ev.m2_manifest.get('retained_static_specification'))))
    items.append(same('branch', 'scorecard', 'B' if ctx.static == 'M2' else 'A', scorecard.get('branch')))
    items.append(same('manifest_family_status', 'manifest', {k: v.get('status') for k, v in families.items()},
                      manifest.get('family_status')))
    items.append(same('manifest_final_naming', 'manifest', scorecard.get('final_naming', MISSING),
                      manifest.get('final_naming', MISSING)))
    items.append(same('manifest_integrity_passed', 'manifest', scorecard.get('integrity_passed', MISSING),
                      manifest.get('integrity_passed', MISSING)))
    items.append(same('provenance_binds_the_m2_manifest', 'm3_provenance', ev.m2_manifest_sha256,
                      ev.provenance.get('m2_result_manifest_sha256', MISSING)))
    pinned = ev.provenance.get('inputs', {})
    for role, key in INPUT_PIN_KEYS.items():
        record = pinned.get(key, {})
        items.append(same('provenance_pins_the_input_read_here', role, ev.input_sha256[role]['sha256'],
                          record.get('sha256', MISSING) if isinstance(record, dict) else MISSING))
    accounting_ok = all(v['identity_gap'] <= ACCOUNTING_TOL for v in rec.accounting.values())
    finite = all(all(math.isfinite(v) for v in c.values()) for c in rec.conditions.values())
    integrity = bool(identity_ok and accounting_ok and finite)
    items.append(same('integrity_passed_equals_recomputed_integrity', 'scorecard', integrity,
                      scorecard.get('integrity_passed', MISSING)))
    items.append(same('integrity_failures_empty_iff_passed', 'scorecard', integrity,
                      scorecard.get('integrity_failures', MISSING) == []))
    return items, integrity


# ---------------------------------------------------------------------
# Negative controls (in-memory copies only)
# ---------------------------------------------------------------------


def _failed(items) -> list[str]:
    return sorted({i['check'] for i in items if not i['passes']})


def negative_controls(ev: Evidence, ctx: Context, graphs: dict, predictions: dict, rec: Recomputed,
                      results: dict) -> dict:
    controls = {}
    rows = [r for r in ev.fits.to_dict('records') if r['protocol'] == PRIMARY]
    target = next((r for r in rows if r['representation'] == PRIMARY_REPRESENTATION
                   and (r['representation'], r['family'], 'in_sample') in predictions), rows[0] if rows else None)
    if target is None:
        for name in ('a_perturbed_theta', 'b_swapped_neighbour', 'c_perturbed_prediction', 'e_perturbed_beta'):
            controls[name] = {'description': 'no primary training fit to perturb', 'applicable': False,
                              'detected': False}
    else:
        key = f"{target['representation']}/{target['family']}/{PRIMARY}/{target['fit']}"
        baseline = _failed(check_fit(ctx, target, graphs, predictions).items)

        row = dict(target)
        shift = CONTROL_THETA_SHIFT if float(row['theta']) <= 0 else -CONTROL_THETA_SHIFT
        row['theta'] = float(row['theta']) + shift
        failed = _failed(check_fit(ctx, row, graphs, predictions).items)
        controls['a_perturbed_theta'] = {
            'description': f'saved theta moved by {shift:+g} toward the interior in an in-memory copy of the fit row',
            'target': key, 'baseline_failed_checks': baseline, 'failed_checks': failed,
            'detected': bool(not baseline and {'loglik', 'local_maximum_on_fine_grid', 'beta_qr_at_saved_theta',
                                               'held_out_prediction'} & set(failed))}

        altered = copy.deepcopy(graphs)
        entry = altered[(PRIMARY, str(target['fit']))]
        node_iso, neighbours = entry['training'][0]
        replacement = next(i for i, _ in entry['training'] if i != node_iso and i not in neighbours)
        entry['training'][0] = (node_iso, [*neighbours[:-1], replacement])
        failed = _failed(check_fit(ctx, target, altered, predictions).items)
        controls['b_swapped_neighbour'] = {
            'description': 'the eighth neighbour of the first training node replaced by a non-neighbouring training node',
            'target': f'{key}/{node_iso}', 'removed': neighbours[-1], 'inserted': replacement,
            'baseline_failed_checks': baseline, 'failed_checks': failed,
            'detected': bool(not baseline and 'training_neighbour_lists_equal_knn8_recomputation' in failed)}

        pkey = (target['representation'], target['family'], PRIMARY)
        held = graphs[(PRIMARY, str(target['fit']))]['held_out']
        if pkey in predictions and held:
            copied = dict(predictions)
            frame = predictions[pkey].copy()
            frame.loc[held[0][0], 'prediction'] = frame.loc[held[0][0], 'prediction'] + CONTROL_SHIFT
            copied[pkey] = frame
            failed = _failed(check_fit(ctx, target, graphs, copied).items)
            local_rec = Recomputed(dict(rec.computable), {}, {}, {}, {})
            metric_failed = _failed(metric_and_paired_items(ev, ctx, results, copied, local_rec))
            controls['c_perturbed_prediction'] = {
                'description': 'one saved held-out prediction increased by 1e-6 in an in-memory copy',
                'target': f'{key}/{held[0][0]}', 'baseline_failed_checks': baseline, 'failed_checks': failed,
                'also_failed_metric_checks': metric_failed,
                'detected': bool(not baseline and 'held_out_prediction' in failed)}
        else:
            controls['c_perturbed_prediction'] = {'description': 'no computable primary-representation family: the '
                                                  'package holds no held-out prediction to perturb',
                                                  'applicable': False, 'detected': False}

        row = dict(target)
        beta = json.loads(row['beta'])
        index = int(np.argmin(np.abs(beta)))
        beta[index] += CONTROL_SHIFT
        row['beta'] = json.dumps(beta)
        failed = _failed(check_fit(ctx, row, graphs, predictions).items)
        controls['e_perturbed_beta'] = {
            'description': 'the smallest-magnitude beta element increased by 1e-6 in an in-memory copy',
            'target': f"{key}/{str(target['columns']).split('|')[index]}", 'baseline_failed_checks': baseline,
            'failed_checks': failed, 'detected': bool(not baseline and 'beta_qr_at_saved_theta' in failed)}

    baseline = _failed(qualification_items(ev, ctx, rec)[0])
    chosen = next(((r, f) for r in REPRESENTATIONS for f in FAMILIES
                   if (r, f) in rec.conditions and rec.computable.get((r, f))
                   and (ctx.arm != 'station' or r == PRIMARY_REPRESENTATION)), None)
    if chosen is None:
        altered = copy.deepcopy(ev.scorecard)
        label = f'{REPRESENTATIONS[0]}/{FAMILIES[0]}'
        entry = altered.get('families', {}).get(label, {})
        block = 'qualification' if 'qualification' in entry else 'arm_conditions'
        saved = entry.get(block, {})
        saved['qualifies'] = not saved.get('qualifies')
        failed = _failed(qualification_items(dataclasses.replace(ev, scorecard=altered), ctx, rec)[0])
        controls['d_shifted_veto_threshold'] = {
            'description': 'no computable family has qualification inputs, so no threshold can flip a veto; the saved '
                           f'{block}.qualifies Boolean of {label} is flipped in an in-memory copy instead',
            'baseline_failed_checks': baseline, 'failed_checks': failed,
            'detected': bool(not baseline and f'{block}.qualifies' in failed)}
    else:
        deciding = ctx.arm == 'station' and chosen[0] == PRIMARY_REPRESENTATION
        block = 'qualification' if deciding else 'arm_conditions'
        variants = {}
        for condition, value_key in (('Q3_m49_veto_passed', 'm49_change'), ('Q4_worst_region_veto_passed', 'worst_change')):
            value = rec.conditions[chosen][value_key]
            threshold = float(np.nextafter(value, -np.inf)) if value <= VETO else float(value)
            failed = _failed(qualification_items(ev, ctx, rec, veto=threshold)[0])
            variants[condition] = {'input': value, 'shifted_threshold': threshold, 'failed_checks': failed,
                                   'flip_flagged': f'{block}.{condition}' in failed}
        controls['d_shifted_veto_threshold'] = {
            'description': 'the 0.001 veto replaced by the nearest threshold that flips Q3 (then Q4) at the recomputed '
                           'input; the saved Boolean must then disagree', 'target': '/'.join(chosen),
            'baseline_failed_checks': baseline, 'variants': variants,
            'detected': bool(not baseline and all(v['flip_flagged'] for v in variants.values()))}
    return dict(sorted(controls.items()))


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


def guarded(stage, empty, fn, *args):
    """Run one reconstruction stage; an error raised by malformed saved evidence becomes a failed comparison."""
    try:
        return fn(*args)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, np.linalg.LinAlgError) as error:
        failure = [flag(f'{stage}_stage_completes', stage, False, repr(error))]
        return (failure, *empty[1:]) if isinstance(empty, tuple) else failure


SECTION_NAMES = {
    'inputs': '1_manifest_and_inputs', 'structure': '2_fit_structure_designs_states', 'graphs': '3_graphs',
    'likelihood': '4_likelihood', 'optimality': '5_optimality', 'coefficients': '6_coefficients',
    'predictions': '7_predictions', 'coverage': '8_coverage_and_computability', 'metrics': '9_metrics_and_paired',
    'qualification': '10_qualification_and_naming', 'accounting': '11_accounting_and_filtered_shares',
    'identity': '12_representation_identity', 'dependence': '13_dependence_and_graph_descriptors',
    'consistency': '14_package_consistency',
}


def reconstruct(ev: Evidence, expected=None, controls=True) -> dict:
    expected = expected or EXPECTED
    ctx, input_list = build_context(ev, expected)
    buckets: dict[str, list] = {name: [] for name in SECTION_NAMES}
    buckets['inputs'] = input_list
    graphs = graph_lookup(ev.graphs)
    predictions = prediction_lookup(ev.predictions)
    results, duplicates = {}, []
    for row in ev.fits.to_dict('records'):
        result = check_fit(ctx, row, graphs, predictions)
        if result.key in results:
            duplicates.append('/'.join(result.key))
        results[result.key] = result
        for item in result.items:
            buckets[item['section']].append(item)
    coverage, computable = coverage_items(ev, ctx, results, duplicates)
    buckets['coverage'] = coverage
    rec = Recomputed(computable, {}, {}, {}, {})
    buckets['metrics'] = guarded('metrics', [], metric_and_paired_items, ev, ctx, results, predictions, rec)
    buckets['accounting'] = guarded('accounting', [], accounting_items, ev, ctx, results, rec)
    buckets['dependence'] = guarded('dependence', [], dependence_and_descriptor_items, ev, ctx, results)
    buckets['identity'], identity_ok = guarded('identity', ([], False), identity_items, ev, ctx, results, predictions,
                                               rec)
    consistency, integrity = consistency_items(ev, ctx, identity_ok, rec)
    buckets['consistency'] = consistency
    qualification_list, qualifies = guarded('qualification', ([], {}), qualification_items, ev, ctx, rec)
    buckets['qualification'] = qualification_list + naming_items(ev, ctx, qualifies, integrity)
    control_record = negative_controls(ev, ctx, graphs, predictions, rec, results) if controls else {}
    applicable = {k: c for k, c in control_record.items() if c.get('applicable', True)}
    sections = {SECTION_NAMES[name]: section(items) for name, items in buckets.items()}
    failures = [f'{name}: {check}' for name, block in sections.items() for check in block['failed_checks']]
    return {
        'what_this_is': 'independent reconstruction of an M3 package from its saved evidence (M3_EVALUATOR_SPEC.md §4)',
        'implementation': 'research/model_v2/m3_independent_check.py with research/model_v2/m3_independent_math.py',
        'scope': SCOPE, 'tolerances': TOLERANCES, 'expected_structure': expected,
        'package': str(ev.package), 'arm': ctx.arm, 'retained_static_specification': ctx.static,
        'inputs': {'package_artifact_sha256': ev.manifest.get('artifact_sha256'),
                   'm2_artifact_sha256': ev.m2_manifest.get('artifact_sha256'),
                   'm2_manifest_sha256': ev.m2_manifest_sha256, 'pinned_inputs': ev.input_sha256,
                   'reference_math_version': mi.VERSION},
        'fits_checked': len(results),
        'section_summary': {name: {k: block[k] for k in ('passes', 'n_checks', 'n_comparisons', 'n_failed_comparisons',
                                                         'max_abs_diff')} for name, block in sections.items()},
        'all_checks_pass': not failures, 'failed_checks': failures,
        'negative_controls': control_record,
        'negative_controls_not_applicable': sorted(set(control_record) - set(applicable)),
        'negative_controls_all_detected': bool(applicable) and all(c['detected'] for c in applicable.values()),
        'sections': sections,
    }


def run(package, m2=M2_DIR, countries=COUNTRIES_PATH, design=DESIGN_PATH, geometry=GEOMETRY_PATH,
        rank_audit=RANK_AUDIT_PATH, expected=None, controls=True) -> dict:
    return reconstruct(load_evidence(package, m2, countries, design, geometry, rank_audit), expected, controls)


def default_out(package) -> Path:
    package = Path(package)
    return package.parent / f'{package.name}_verification' / 'm3_independent_check.json'


def main(argv=None, expected=None) -> int:
    parser = argparse.ArgumentParser(description='Independently reconstruct one M3 package.')
    parser.add_argument('--package', required=True, help='M3 package (outputs/m3_station or outputs/m3_land_centroid)')
    parser.add_argument('--m2', default=str(M2_DIR), help='committed M2 primary result')
    parser.add_argument('--out', help='verification JSON (default <package>/../<package name>_verification/...)')
    parser.add_argument('--countries', default=str(COUNTRIES_PATH))
    parser.add_argument('--design', default=str(DESIGN_PATH))
    parser.add_argument('--geometry', default=str(GEOMETRY_PATH))
    parser.add_argument('--rank-audit', default=str(RANK_AUDIT_PATH))
    args = parser.parse_args(argv)
    try:
        report = run(args.package, args.m2, args.countries, args.design, args.geometry, args.rank_audit, expected)
    except DigestRefused as refusal:
        report = {'what_this_is': 'independent reconstruction of an M3 package (refused)', 'refused': str(refusal),
                  'manifest': refusal.records, 'all_checks_pass': False, 'negative_controls': None,
                  'negative_controls_all_detected': False}
    out = Path(args.out) if args.out else default_out(args.package)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + '\n')
    passed = bool(report['all_checks_pass'] and report['negative_controls_all_detected'])
    if 'refused' in report:
        print(f"REFUSED: {report['refused']}")
    else:
        for name, block in report['section_summary'].items():
            diff = block['max_abs_diff']
            print(f"  {name}: {block['n_comparisons'] - block['n_failed_comparisons']}/{block['n_comparisons']}"
                  f" (max abs diff {diff if diff is None else f'{diff:.3g}'})")
        for name, control in report['negative_controls'].items():
            print(f"  control {name}: detected={control['detected']}")
        for failure in report['failed_checks']:
            print(f'  FAIL {failure}')
    print(f"all_checks_pass={report['all_checks_pass']} "
          f"negative_controls_all_detected={report['negative_controls_all_detected']} -> exit {0 if passed else 3}")
    return 0 if passed else 3


if __name__ == '__main__':
    sys.exit(main())
