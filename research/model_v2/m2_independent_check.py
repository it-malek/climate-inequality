"""Independent reconstruction of the M2 primary result from its saved evidence.

A second calculation path, derived from the evaluation rules and the evaluator's artifact formats,
that shares no code with the evaluator: it imports nothing from
``m2_evaluate``, ``m2_conditional``, ``m2_feasibility``, ``m2_latitude_basis``, ``cv``, ``spatial``,
``run_territory_correction``, ``src.decomposition``, ``src.stability`` or ``m3_*`` (NumPy, pandas and the
standard library only). Given a completed primary output directory it:

1. verifies every SHA-256 digest of ``m2_result_manifest.json`` before reading any other artifact (a mismatch
   is a refusal), and binds the label table and the two committed stage scorecards to ``m2_provenance.json``;
2. rebuilds the encoded design of every saved fit (full sample and every primary, M49 and random training
   fit; both representations; M0* and M2) from the numeric columns of the saved full-sample M0* design, the
   categorical labels of ``m0_countries.csv`` (``iso3`` and the four labels only), the fit's saved training
   levels and column order and, for M2, the natural cubic columns from the saved hex knots,

       d_k(a) = [(a - xi_k)_+^3 - (a - xi_4)_+^3] / (xi_4 - xi_k),   N1 = d_1 - d_3,   N2 = d_2 - d_3,

   with the interaction I(hemisphere == 'S') * abs_latitude; memberships come from ``m2_cv_folds.csv``; the
   knots are also re-derived from the training latitudes (type-7 linear quantiles, implemented here);
3. refits every fit by reduced QR (``numpy.linalg.qr``) and explicit back substitution on the saved
   ``observed`` outcome, and compares coefficients and matrix ranks;
4. recomputes every in-sample fitted value and out-of-fold prediction, adding for each categorical whose
   held-out level is not a training level the mean of [0] + [beta of each non-reference training level];
5. recomputes RMSE, MAE, R2 = 1 - SSE/SST, per-M49-region RMSE and each model's own worst region (n >= 3);
6. re-implements the paired interval (``default_rng(0)``, ``integers(0, n, size=(2000, n))``, percentiles
   2.5/97.5 of the resampled RMSE difference);
7. rebuilds P1, P2, R1-R3 (R4 is the saved integrity flag, cross-checked against the recomputed structure),
   the label, ``worsened_generalization`` and the static stopping indicator;
8. summarises gamma from the saved coefficient table;
9. refits all 16 coalitions of the saved full-sample design by QR and averages marginal R2 over all 24
   orderings of the four groups (Shapley by permutations), with the share accounting;
10. checks the representation identity of predictions and latitude coefficients;
11. runs five negative controls on in-memory copies (never on the files) and requires each to be detected.

Tolerances. Coefficients: |refit - saved| <= 1e-8 * max(1, |saved|), i.e. 1e-8 absolute for coefficients of
magnitude below one and 1e-8 relative above (QR against the evaluator's SVD least squares). Predictions and
fitted values: 1e-9. Knots re-derived from training latitudes: 1e-10 degrees. Rebuilt full-sample added
columns against the saved design: 1e-12 * max(1, |saved|) (power against product cubes). Metrics, paired
interval, gamma summary and stopping indicator: 1e-12, since they are recomputed from the same round-trip
parsed float64 values. Coalition R2 and shares: 1e-10; share accounting: 1e-9. Representation identity:
predictions 1e-9, latitude coefficients 1e-8. Labels, counts, identifiers, dummy columns and hex knots: exact.

Not independent: the inputs are the same saved artifacts and pinned tables the evaluator reads; numeric
predictors, the outcome, fold ids and training memberships are taken as given (the 500 km territorial bounds
are not re-derived); nearest-training distances, Moran statistics, calibration, fold summaries and the
diagnostics block are not recomputed. The check is pre-specified: its rules and tolerances are fixed by the
evaluator's artifact formats, not by any result.

Run: ``uv run python -m research.model_v2.m2_independent_check [--primary DIR] [--out FILE]``
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'research' / 'model_v2' / 'outputs'
PRIMARY_DIR = OUTPUTS / 'm2_primary'
OUTPUT_PATH = OUTPUTS / 'm2_primary_verification' / 'm2_independent_check.json'
COUNTRIES_PATH = OUTPUTS / 'm0_countries.csv'
M1A_SCORECARD = OUTPUTS / 'm1a_scorecard.json'
M1B_SCORECARD = OUTPUTS / 'm1b_primary' / 'm1b_scorecard.json'

MANIFEST = 'm2_result_manifest.json'
ARTIFACTS = ('m2_scorecard.json', 'm2_country_predictions.csv', 'm2_fit_states.csv', 'm2_coefficients.csv',
             'm2_design_matrices.csv', 'm2_shapley_coalitions.csv', 'm2_cv_folds.csv', 'm2_provenance.json')
CATEGORICALS = ('climate_zone', 'hemisphere', 'spatial_block', 'income_group')
LABEL_COLUMNS = ['iso3', *CATEGORICALS]
PRIMARY_REPRESENTATION = 'primary_total_co2'
REPRESENTATIONS = (PRIMARY_REPRESENTATION, 'per_capita')
COMPARATOR, CANDIDATE = 'M0star', 'M2'
MODELS = (COMPARATOR, CANDIDATE)
PROTOCOLS = ('primary_loco', 'm49_subregion_lo', 'random10')
CARD_PATH = {'primary_loco': (), 'm49_subregion_lo': ('secondary',), 'random10': ('random_reference_only',)}
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')
GAMMA = 'hemisphere=S:abs_latitude'
ADDED = ('abs_latitude_ns1', 'abs_latitude_ns2', GAMMA)
LATITUDE_COEFFICIENTS = ('abs_latitude_ns1', 'abs_latitude_ns2', GAMMA, 'abs_latitude')
BASIS_VERSION = 'm2-latitude-basis-v1'
KNOT_GAP = 1e-6
PRACTICAL, VETO = -0.002, 0.001
MIN_REGION_N = 3
RESAMPLES, SEED = 2000, 0
LABEL_PROMOTED, LABEL_SUPPORT, LABEL_NONE = ('supported and promoted', 'predictive support, not promoted',
                                             'not supported')
VERDICT_KEYS = {'P1': 'P1_delta_below_zero', 'P2': 'P2_interval_upper_below_zero', 'R1': 'R1_delta_le_minus_0.002',
                'R2': 'R2_m49_veto_passed', 'R3': 'R3_worst_region_veto_passed', 'R4': 'R4_structural_integrity',
                'worsened_generalization': 'worsened_generalization'}
# The real-data structure; synthetic tests pass their own.
EXPECTED = {'n': 151, 'full_columns': {COMPARATOR: 20, CANDIDATE: 23}}

COEF_TOL = 1e-8
PREDICTION_TOL = 1e-9
KNOT_TOL = 1e-10
DESIGN_TOL = 1e-12
METRIC_TOL = 1e-12
SHARE_TOL = 1e-10
ACCOUNTING_TOL = 1e-9
IDENTITY_PREDICTION_TOL = 1e-9
IDENTITY_COEF_TOL = 1e-8
CONTROL_SHIFT = 1e-6

TOLERANCES = {
    'coefficients': 'abs(refit - saved) <= 1e-8 * max(1, abs(saved)): absolute below magnitude one, relative above',
    'predictions_and_fitted_values': PREDICTION_TOL, 'knots_from_training_latitudes_degrees': KNOT_TOL,
    'full_sample_added_columns': 'abs(rebuilt - saved) <= 1e-12 * max(1, abs(saved))',
    'metrics_paired_interval_gamma_stopping': METRIC_TOL, 'coalition_r2_and_shares': SHARE_TOL,
    'share_accounting': ACCOUNTING_TOL, 'identity_predictions': IDENTITY_PREDICTION_TOL,
    'identity_latitude_coefficients': IDENTITY_COEF_TOL,
    'exact': 'labels, counts, identifiers, column layouts, levels, dummy columns, hex knots, manifest digests',
}
SCOPE = {
    'independent': [
        'the calculation path: design rebuild, natural cubic columns, knot derivation, QR refits, predictions '
        'with the unseen-level rule, metrics, worst regions, paired interval, verdict Booleans and label, '
        'stopping indicator, gamma summary, coalition R2 and permutation Shapley shares, representation identity',
        'imports only numpy, pandas and the standard library; nothing from m2_evaluate, m2_conditional, '
        'm2_feasibility, m2_latitude_basis, cv, spatial, run_territory_correction, src.decomposition, '
        'src.stability or m3_*',
    ],
    'not_independent': [
        'same inputs: the saved M2 artifacts, the pinned country label table and the committed M1a/M1b '
        'scorecards; numeric predictors, the observed outcome, fold ids and training memberships are taken as given',
        'the 500 km territorial bounds, the M49 labelling and the random folds are not re-derived',
        'nearest-training distances, residual Moran statistics, calibration, fold summaries, diagnostics and the '
        'extrapolation disclosure rows are not recomputed',
        'written in the same session as the evaluator and its specification, before any M2 result existed, '
        'after reading the evaluator artifact formats; executed after the result was written',
    ],
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


def numeric(name, recomputed, reported, tol) -> dict:
    """One numeric comparison; a missing or non-finite value never passes."""
    try:
        diff = abs(float(recomputed) - float(reported))
        passes = math.isfinite(diff) and diff <= tol
    except (TypeError, ValueError):
        diff, passes = float('nan'), False
    return {'check': name, 'recomputed': _jsonable(recomputed), 'reported': _jsonable(reported),
            'abs_diff': _jsonable(diff), 'tolerance': tol, 'passes': bool(passes)}


def exact(name, recomputed, reported) -> dict:
    return {'check': name, 'recomputed': _jsonable(recomputed), 'reported': _jsonable(reported),
            'abs_diff': None, 'tolerance': 'exact', 'passes': bool(recomputed == reported)}


def sequence(name, recomputed, reported) -> dict:
    """Exact comparison of a long sequence, recorded as lengths, digests and the number of differing positions."""
    recomputed, reported = list(recomputed), list(reported)
    digest = lambda values: hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:16]  # noqa: E731
    return {'check': name, 'recomputed': {'n': len(recomputed), 'sha256_16': digest(recomputed)},
            'reported': {'n': len(reported), 'sha256_16': digest(reported)}, 'abs_diff': None, 'tolerance': 'exact',
            'n_differing': sum(1 for a, b in itertools.zip_longest(recomputed, reported, fillvalue=object()) if a != b),
            'passes': recomputed == reported}


def same(check, label, recomputed, reported) -> dict:
    return {'check': check, 'label': label, 'recomputed': recomputed, 'reported': reported}


def close(check, label, diff, tol, raw=None) -> dict:
    return {'check': check, 'label': label, 'diff': diff, 'tol': tol, 'raw': raw}


def item_passes(item) -> bool:
    if 'diff' in item:
        return bool(math.isfinite(item['diff']) and item['diff'] <= item['tol'])
    return bool(item['recomputed'] == item['reported'])


def summarise(name, items) -> dict:
    """One record for many element comparisons of the same kind."""
    failed = [i for i in items if not item_passes(i)]
    record = {'check': name, 'n': len(items), 'n_failed': len(failed), 'passes': bool(items) and not failed}
    if items and 'diff' in items[0]:
        diffs = [float(i['diff']) for i in items]
        raw = [float(i['raw']) for i in items if i['raw'] is not None]
        record.update({'max_abs_diff': _jsonable(max(raw) if raw else max(diffs)), 'tolerance': items[0]['tol'],
                       'failed_examples': [i['label'] for i in failed[:10]]})
        if raw:
            record.update({'max_scaled_diff': _jsonable(max(diffs)), 'tolerance_rule': 'diff / max(1, |reported|)'})
    else:
        record.update({'max_abs_diff': None, 'tolerance': 'exact', 'failed_examples': [
            {'label': i['label'], 'recomputed': _jsonable(i['recomputed']), 'reported': _jsonable(i['reported'])}
            for i in failed[:5]]})
    return record


def section(records) -> dict:
    diffs = [r['abs_diff'] if 'abs_diff' in r else r.get('max_abs_diff') for r in records]
    finite = [d for d in diffs if isinstance(d, (int, float)) and math.isfinite(d)]
    failed = [r['check'] for r in records if not r['passes']]
    return {'n_comparisons': len(records), 'n_failed': len(failed), 'max_abs_diff': max(finite, default=0.0),
            'passes': bool(records) and not failed, 'failed_checks': failed, 'comparisons': records}


# ---------------------------------------------------------------------
# Independent mathematics
# ---------------------------------------------------------------------


def natural_cubic_columns(knots, abs_latitude) -> np.ndarray:
    """(n, 2) natural cubic columns N1 = d_1 - d_3 and N2 = d_2 - d_3 on four knots."""
    a = np.asarray(abs_latitude, dtype=float)
    xi = [float(k) for k in knots]
    tail = np.clip(a - xi[3], 0.0, None) ** 3
    d = [(np.clip(a - xi[k], 0.0, None) ** 3 - tail) / (xi[3] - xi[k]) for k in range(3)]
    return np.column_stack([d[0] - d[2], d[1] - d[2]])


def linear_quantile(values, probability) -> float:
    """Type-7 sample quantile: x[j] + (h - j)(x[j+1] - x[j]) with h = (n - 1) p on sorted x."""
    x = sorted(float(v) for v in values)
    h = (len(x) - 1) * probability
    j = int(math.floor(h))
    return x[j] if j >= len(x) - 1 else x[j] + (h - j) * (x[j + 1] - x[j])


def knots_from_training(abs_latitude) -> list[float]:
    values = [float(v) for v in abs_latitude]
    return [min(values), linear_quantile(values, 1 / 3), linear_quantile(values, 2 / 3), max(values)]


def qr_coefficients(x, y) -> np.ndarray:
    """Least squares by reduced QR and explicit back substitution on R beta = Q'y."""
    q, r = np.linalg.qr(np.asarray(x, dtype=float), mode='reduced')
    rhs = q.T @ np.asarray(y, dtype=float)
    beta = np.zeros(r.shape[1])
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for i in range(r.shape[1] - 1, -1, -1):
            beta[i] = (rhs[i] - r[i, i + 1:] @ beta[i + 1:]) / r[i, i]
    return beta


def shapley_by_permutations(r2, groups) -> dict[str, float]:
    """Mean marginal R2 of each group over every ordering of the groups."""
    shares = dict.fromkeys(groups, 0.0)
    orderings = list(itertools.permutations(groups))
    for ordering in orderings:
        before = frozenset()
        for group in ordering:
            shares[group] += r2[before | {group}] - r2[before]
            before = before | {group}
    return {g: v / len(orderings) for g, v in shares.items()}


def rmse(error) -> float:
    return float(np.sqrt(np.mean(np.asarray(error, dtype=float) ** 2)))


def scores(observed, prediction) -> tuple[float, float, float]:
    """R2 = 1 - SSE/SST, RMSE and MAE."""
    y, e = np.asarray(observed, dtype=float), np.asarray(observed, dtype=float) - np.asarray(prediction, dtype=float)
    return 1.0 - float(np.sum(e ** 2)) / float(np.sum((y - y.mean()) ** 2)), rmse(e), float(np.mean(np.abs(e)))


def paired_interval(observed, candidate, comparator) -> dict[str, float]:
    """Paired country resampling of fixed out-of-fold errors: RMSE(candidate) - RMSE(comparator)."""
    y = np.asarray(observed, dtype=float)
    new, old = y - np.asarray(candidate, dtype=float), y - np.asarray(comparator, dtype=float)
    index = np.random.default_rng(SEED).integers(0, len(y), size=(RESAMPLES, len(y)))
    statistic = np.sqrt((new[index] ** 2).mean(axis=1)) - np.sqrt((old[index] ** 2).mean(axis=1))
    low, high = np.quantile(statistic, [0.025, 0.975])
    return {'delta_rmse': rmse(new) - rmse(old), 'low': float(low), 'high': float(high)}


def region_rmse(frame) -> pd.DataFrame:
    grouped = frame.assign(e=frame.observed - frame.prediction).groupby('m49_subregion')['e']
    return pd.DataFrame({'rmse': grouped.apply(lambda e: rmse(e.to_numpy())), 'n': grouped.size()})


def worst_region(frame) -> tuple[str | None, float, int]:
    regions = region_rmse(frame)
    eligible = regions[regions.n >= MIN_REGION_N]
    if eligible.empty:
        return None, float('nan'), 0
    region = eligible.rmse.idxmax()
    return str(region), float(eligible.loc[region, 'rmse']), int(eligible.loc[region, 'n'])


def verdict_conditions(delta, low, high, m49_change, worst_change, integrity, practical=PRACTICAL) -> dict:
    """The pre-specified verdict rules; a non-finite input yields no label."""
    finite = all(math.isfinite(float(v)) for v in (delta, low, high, m49_change, worst_change))
    out = {'P1': bool(delta < 0), 'P2': bool(high < 0), 'R1': bool(delta <= practical),
           'R2': bool(m49_change <= VETO), 'R3': bool(worst_change <= VETO), 'R4': bool(integrity),
           'worsened_generalization': bool(delta > 0 and low > 0), 'finite': finite}
    support = out['P1'] and out['P2']
    if not (out['R4'] and finite):
        out['label'] = None
    elif support and out['R1'] and out['R2'] and out['R3']:
        out['label'] = LABEL_PROMOTED
    else:
        out['label'] = LABEL_SUPPORT if support else LABEL_NONE
    return out


# ---------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, **kwargs) -> pd.DataFrame:
    """Exact float64 round trip; pandas' default parser is only accurate to about 1e-14 relative."""
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, **kwargs)


def verify_manifest(primary_dir) -> tuple[dict, list]:
    """Every manifest digest against the bytes on disk, before anything else is read."""
    primary_dir = Path(primary_dir)
    manifest = json.loads((primary_dir / MANIFEST).read_text())
    recorded = manifest.get('artifact_sha256', {})
    records = [exact(f'manifest/{name}/listed', name in recorded, True) for name in ARTIFACTS]
    for name, digest in sorted(recorded.items()):
        path = primary_dir / name
        inside = path.resolve().parent == primary_dir.resolve()
        actual = sha256(path) if inside and path.is_file() else 'missing'
        records.append(exact(f'manifest/{name}/sha256', actual, digest))
    failed = [r['check'] for r in records if not r['passes']]
    if failed:
        raise DigestRefused(f'manifest verification failed: {failed}', records)
    return manifest, records


@dataclasses.dataclass(frozen=True)
class Evidence:
    manifest: dict
    manifest_records: list
    scorecard: dict
    provenance: dict
    predictions: pd.DataFrame
    states: pd.DataFrame
    coefficients: pd.DataFrame
    designs: pd.DataFrame
    coalitions: pd.DataFrame
    folds: pd.DataFrame
    labels: pd.DataFrame
    stage_deltas: dict
    auxiliary: dict


def load_evidence(primary_dir=PRIMARY_DIR, countries=COUNTRIES_PATH, m1a_scorecard=M1A_SCORECARD,
                  m1b_scorecard=M1B_SCORECARD) -> Evidence:
    primary_dir = Path(primary_dir)
    manifest, manifest_records = verify_manifest(primary_dir)
    text = dict.fromkeys(('representation', 'model', 'protocol', 'fit', 'iso3', 'column', 'group', 'coalition'), str)
    predictions = read_csv(primary_dir / 'm2_country_predictions.csv',
                           dtype={**text, 'm49_subregion': str, 'extrapolation': str, 'nearest_train_km': str})
    states = read_csv(primary_dir / 'm2_fit_states.csv', dtype=str)
    labels = read_csv(countries, usecols=LABEL_COLUMNS, dtype=str)[LABEL_COLUMNS]
    m1a, m1b = json.loads(Path(m1a_scorecard).read_text()), json.loads(Path(m1b_scorecard).read_text())
    return Evidence(
        manifest=manifest, manifest_records=manifest_records,
        scorecard=json.loads((primary_dir / 'm2_scorecard.json').read_text()),
        provenance=json.loads((primary_dir / 'm2_provenance.json').read_text()),
        predictions=predictions, states=states,
        coefficients=read_csv(primary_dir / 'm2_coefficients.csv', dtype=text),
        designs=read_csv(primary_dir / 'm2_design_matrices.csv', dtype=text),
        coalitions=read_csv(primary_dir / 'm2_shapley_coalitions.csv', dtype=text),
        folds=read_csv(primary_dir / 'm2_cv_folds.csv', dtype={'iso3': str, 'Country': str, 'm49_subregion': str,
                                                               'primary_training_iso3': str}),
        labels=labels.set_index('iso3', drop=False),
        stage_deltas={'M1a': m1a['primary']['M1a']['paired_delta_vs_m0']['delta_rmse'],
                      'M1b': m1b['verdict']['delta_rmse']},
        auxiliary={role: {'path': str(path), 'sha256': sha256(path)} for role, path in
                   (('countries', countries), ('m1a_scorecard', m1a_scorecard), ('m1b_scorecard', m1b_scorecard))})


@dataclasses.dataclass(frozen=True)
class Context:
    order: list
    pos: dict
    y: np.ndarray
    labels: dict
    numerics: dict
    numeric_columns: dict
    categorical_order: list
    full: dict
    training: dict
    fold_ids: dict


def build_context(evidence: Evidence) -> Context:
    folds = evidence.folds
    order = folds.iso3.tolist()
    pos = {iso: i for i, iso in enumerate(order)}
    labels = {c: evidence.labels.reindex(order)[c].fillna('').to_numpy(dtype=str) for c in CATEGORICALS}
    p = evidence.predictions
    reference = p[(p.representation == PRIMARY_REPRESENTATION) & (p.model == COMPARATOR) & (p.protocol == 'in_sample')]
    y = reference.set_index('iso3').reindex(order).observed.to_numpy(dtype=float)
    full = {}
    for (representation, model), rows in evidence.designs.groupby(['representation', 'model'], sort=False):
        columns = list(dict.fromkeys(rows.column))
        values = rows.pivot(index='iso3', columns='column', values='value').reindex(index=order, columns=columns)
        groups = rows.drop_duplicates('column').set_index('column').group.reindex(columns).tolist()
        full[(representation, model)] = {'columns': columns, 'groups': groups, 'values': values.to_numpy(dtype=float),
                                         'group_constant': bool((rows.groupby('column').group.nunique() == 1).all())}
    numeric_columns, numerics = {}, {}
    for representation in REPRESENTATIONS:
        base = full[(representation, COMPARATOR)]
        numeric_columns[representation] = [c for c in base['columns']
                                           if c != 'intercept' and '=' not in c and c not in ADDED]
        numerics[representation] = {c: base['values'][:, j] for j, c in enumerate(base['columns'])
                                    if c in numeric_columns[representation]}
    dummies = [c.split('=', 1)[0] for c in full[(PRIMARY_REPRESENTATION, COMPARATOR)]['columns'] if '=' in c]
    return Context(order=order, pos=pos, y=y, labels=labels, numerics=numerics, numeric_columns=numeric_columns,
                   categorical_order=list(dict.fromkeys([*dummies, *CATEGORICALS])), full=full,
                   training={iso: s.split('|') if s else [] for iso, s in zip(order, folds.primary_training_iso3)},
                   fold_ids={'m49_subregion_lo': folds.m49_fold.to_numpy(dtype=int),
                             'random10': folds.random10_fold.to_numpy(dtype=int)})


def prediction_slice(predictions, representation, model, protocol, order) -> pd.DataFrame:
    frame = predictions[(predictions.representation == representation) & (predictions.model == model)
                        & (predictions.protocol == protocol)]
    return frame.set_index('iso3').reindex(order)


def prediction_lookup(predictions) -> dict:
    columns = ['representation', 'model', 'protocol', 'iso3', 'observed', 'prediction', 'error', 'fold_id', 'n_train',
               'unseen_levels', 'extrapolation']
    return {tuple(r[:4]): {'observed': float(r[4]), 'prediction': float(r[5]), 'error': float(r[6]),
                           'fold_id': int(r[7]), 'n_train': int(r[8]), 'unseen_levels': int(r[9]),
                           'extrapolation': str(r[10])}
            for r in predictions[columns].itertuples(index=False, name=None)}


def coefficient_lookup(coefficients) -> dict:
    lookup = {}
    for r in coefficients[['representation', 'model', 'protocol', 'fit', 'column', 'coefficient']].itertuples(
            index=False, name=None):
        lookup.setdefault(tuple(r[:4]), []).append((r[4], float(r[5])))
    return lookup


# ---------------------------------------------------------------------
# Designs, refits and predictions
# ---------------------------------------------------------------------


def encoded(context, representation, columns, rows, knots) -> np.ndarray:
    """One design, by saved column name: numerics, labels, drop-first dummies and the M2 columns."""
    rows = np.asarray(rows, dtype=int)
    numerics = context.numerics[representation]
    out = np.empty((len(rows), len(columns)))
    spline = None
    for j, name in enumerate(columns):
        if name == 'intercept':
            out[:, j] = 1.0
        elif name == GAMMA:
            out[:, j] = (context.labels['hemisphere'][rows] == 'S') * numerics['abs_latitude'][rows]
        elif name in ADDED:
            if spline is None:
                spline = natural_cubic_columns(knots, numerics['abs_latitude'][rows])
            out[:, j] = spline[:, ADDED.index(name)]
        elif '=' in name:
            feature, level = name.split('=', 1)
            out[:, j] = context.labels[feature][rows] == level
        else:
            out[:, j] = numerics[name][rows]
    return out


def membership(context, protocol, fit) -> tuple[np.ndarray, np.ndarray]:
    everyone = np.arange(len(context.order))
    if protocol == 'full_sample':
        return everyone, np.array([], dtype=int)
    if protocol == 'primary_loco':
        return np.array(sorted(context.pos[i] for i in context.training[fit]), dtype=int), np.array([context.pos[fit]])
    ids = context.fold_ids[protocol]
    return everyone[ids != int(fit)], everyone[ids == int(fit)]


def refit(context, row, knots=None, training=None) -> dict:
    """Rebuild, refit and predict one saved fit; ``knots`` or ``training`` replace the saved ones (controls)."""
    columns, levels = row.columns.split('|'), json.loads(row.levels)
    state = json.loads(row.latitude_state) if row.latitude_state else None
    if knots is None and state is not None:
        knots = [float.fromhex(h) for h in state['knots_hex']]
    train, test = membership(context, row.protocol, row.fit)
    if training is not None:
        train = np.array(sorted(training), dtype=int)
    x = encoded(context, row.representation, columns, train, knots)
    beta = qr_coefficients(x, context.y[train])
    rows = train if row.protocol == 'full_sample' else test
    prediction = encoded(context, row.representation, columns, rows, knots) @ beta
    unseen = np.zeros(len(rows), dtype=int)
    for feature, kept in levels.items():
        effects = [0.0] + [float(beta[columns.index(f'{feature}={level}')]) for level in kept[1:]]
        missing = ~np.isin(context.labels[feature][rows], kept)
        prediction = prediction + missing * (sum(effects) / len(effects))
        unseen += missing
    baseline = x[:, [j for j, c in enumerate(columns) if c not in ADDED]]
    return {'columns': columns, 'levels': levels, 'state': state, 'knots': knots, 'train': train, 'test': test,
            'rows': rows, 'beta': beta, 'prediction': prediction, 'unseen': unseen,
            'matrix_columns': x.shape[1], 'matrix_rank': int(np.linalg.matrix_rank(x)),
            'baseline_columns': baseline.shape[1], 'baseline_rank': int(np.linalg.matrix_rank(baseline))}


def structurally_sound(fitted, model) -> bool:
    return bool(fitted['matrix_rank'] == fitted['matrix_columns']
                and fitted['baseline_rank'] == fitted['baseline_columns']
                and (model == COMPARATOR or fitted['matrix_rank'] - fitted['baseline_rank'] == 3))


def layout(context, representation, model, levels) -> list[str]:
    dummies = [f'{c}={level}' for c in context.categorical_order for level in levels.get(c, [])[1:]]
    return ['intercept', *context.numeric_columns[representation], *dummies, *(ADDED if model == CANDIDATE else ())]


SECTION_OF = {'coefficient_columns': 'refit', 'coefficients': 'refit', 'predictions': 'predictions',
              'error_is_observed_minus_prediction': 'predictions', 'fold_id': 'predictions',
              'prediction_n_train': 'predictions', 'unseen_levels': 'predictions', 'extrapolation_flag': 'predictions',
              'prediction_row_present': 'predictions'}


def fit_items(context, lookups, row, fitted) -> list[dict]:
    """Every comparison of one rebuilt fit against its saved state, coefficients and predictions."""
    representation, model, protocol, fit = row.representation, row.model, row.protocol, row.fit
    order, label, train, test = context.order, f'{protocol}:{fit}', fitted['train'], fitted['test']
    items = [
        same('held_out_iso3', label, [order[i] for i in test],
             row.held_out_iso3.split('|') if row.held_out_iso3 else []),
        same('n_train', label, len(train), int(row.n_train)), same('n_test', label, len(test), int(row.n_test)),
        same('levels_are_training_labels', label,
             {c: sorted(set(context.labels[c][train].tolist())) for c in CATEGORICALS}, fitted['levels']),
        same('column_layout', label, layout(context, representation, model, fitted['levels']), fitted['columns']),
        same('matrix_columns', label, fitted['matrix_columns'], int(row.matrix_columns)),
        same('matrix_rank', label, fitted['matrix_rank'], int(row.matrix_rank)),
        same('baseline_columns', label, fitted['baseline_columns'], int(row.baseline_columns)),
        same('baseline_rank', label, fitted['baseline_rank'], int(row.baseline_rank)),
        same('structurally_full_rank', label, structurally_sound(fitted, model), True),
        same('n_train_sh', label, int((context.labels['hemisphere'][train] == 'S').sum()), int(row.n_train_sh)),
    ]
    boundary = ('', '')
    if model == CANDIDATE:
        state, knots = fitted['state'] or {}, [float(k) for k in fitted['knots'] or []]
        derived = knots_from_training(context.numerics[representation]['abs_latitude'][train])
        items += [same('state_version', label, state.get('version'), BASIS_VERSION),
                  same('state_n_train', label, state.get('n_train'), len(train)),
                  same('state_hex_equals_decimal', label, [float.fromhex(h) for h in state.get('knots_hex', [])],
                       [float(k) for k in state.get('knots', [])]),
                  same('four_strictly_increasing_knots', label,
                       len(knots) == 4 and all(b - a > KNOT_GAP for a, b in zip(knots, knots[1:])), True),
                  close('knots_from_training_latitudes', label,
                        max(abs(a - b) for a, b in zip(derived, knots)) if len(knots) == 4 else float('nan'), KNOT_TOL)]
        if len(test) and len(knots) == 4:
            held = context.numerics[representation]['abs_latitude'][test]
            boundary = (str(int((held < knots[0]).sum())), str(int((held > knots[3]).sum())))
    else:
        items.append(same('no_latitude_state', label, row.latitude_state, ''))
    items.append(same('test_boundary_counts', label, boundary,
                      (row.test_below_lower_boundary, row.test_above_upper_boundary)))

    saved = lookups['coefficients'].get((representation, model, protocol, fit), [])
    items.append(same('coefficient_columns', label, [c for c, _ in saved], fitted['columns']))
    for (column, value), estimate in zip(saved, fitted['beta']):
        diff = abs(float(estimate) - value)
        items.append(close('coefficients', f'{label}/{column}', diff / max(1.0, abs(value)), COEF_TOL, raw=diff))

    name = 'in_sample' if protocol == 'full_sample' else protocol
    knots = fitted['knots']
    for i, value, unseen in zip(fitted['rows'], fitted['prediction'], fitted['unseen']):
        tag, saved_row = f'{label}/{order[i]}', lookups['predictions'].get((representation, model, name, order[i]))
        if saved_row is None:
            items.append(same('prediction_row_present', tag, False, True))
            continue
        a = context.numerics[representation]['abs_latitude'][i]
        flag = ('' if model == COMPARATOR or name == 'in_sample' else
                'below' if a < knots[0] else 'above' if a > knots[3] else '')
        items += [close('predictions', tag, abs(float(value) - saved_row['prediction']), PREDICTION_TOL),
                  close('error_is_observed_minus_prediction', tag,
                        abs(saved_row['error'] - (saved_row['observed'] - saved_row['prediction'])), METRIC_TOL),
                  same('fold_id', tag, -1 if name == 'in_sample' else int(i) if name == 'primary_loco' else int(fit),
                       saved_row['fold_id']),
                  same('prediction_n_train', tag, len(train), saved_row['n_train']),
                  same('unseen_levels', tag, int(unseen), saved_row['unseen_levels']),
                  same('extrapolation_flag', tag, flag, saved_row['extrapolation'])]
    return items


def reconstruct_fits(evidence, context) -> dict:
    return {(r.representation, r.model, r.protocol, r.fit): refit(context, r)
            for r in evidence.states.itertuples(index=False)}


def fit_sections(evidence, context, fits) -> dict[str, list]:
    lookups = {'predictions': prediction_lookup(evidence.predictions),
               'coefficients': coefficient_lookup(evidence.coefficients)}
    grouped, covered = {}, {}
    for row in evidence.states.itertuples(index=False):
        fitted = fits[(row.representation, row.model, row.protocol, row.fit)]
        for item in fit_items(context, lookups, row, fitted):
            stem = f'{row.representation}/{row.model}/{row.protocol}/{item["check"]}'
            grouped.setdefault((SECTION_OF.get(item['check'], 'designs'), stem), []).append(item)
        name = 'in_sample' if row.protocol == 'full_sample' else row.protocol
        covered.setdefault((row.representation, row.model, name), []).extend(context.order[i] for i in fitted['rows'])
    sections = {'designs': [], 'refit': [], 'predictions': []}
    for (name, stem), items in grouped.items():
        sections[name].append(summarise(stem, items))
    for representation, model, protocol in itertools.product(REPRESENTATIONS, MODELS, ('in_sample', *PROTOCOLS)):
        rows = covered.get((representation, model, protocol), [])
        sections['predictions'].append(sequence(f'{representation}/{model}/{protocol}/every_country_predicted_once',
                                                sorted(rows), sorted(context.order)))
    return sections


def full_design_items(context, representation, model, knots) -> list[dict]:
    """The saved full-sample design against the one rebuilt from labels, numerics and the full-sample state."""
    full = context.full[(representation, model)]
    saved, columns = full['values'], full['columns']
    rebuilt = encoded(context, representation, columns, np.arange(len(context.order)), knots)
    items = []
    for j, column in enumerate(columns):
        label = f'{representation}/{model}/{column}'
        if column in ADDED:
            raw = np.abs(rebuilt[:, j] - saved[:, j])
            scaled = float(np.max(raw / np.maximum(1.0, np.abs(saved[:, j]))))
            items.append(close('full_design_added_columns', label, scaled, DESIGN_TOL, float(np.max(raw))))
        else:
            items.append(same('full_design_encoded_columns_bit_identical', label,
                              bool(np.array_equal(rebuilt[:, j], saved[:, j])), True))
    return items


def design_records(evidence, context, fits, expected) -> list[dict]:
    records = []
    for representation in REPRESENTATIONS:
        base = context.full[(representation, COMPARATOR)]
        for model in MODELS:
            full, key = context.full[(representation, model)], (representation, model, 'full_sample', 'all')
            stem = f'{representation}/{model}/full_sample_design'
            records.append(summarise(stem + '/values', full_design_items(context, representation, model,
                                                                          fits[key]['knots'] if key in fits else None)))
            records.append(exact(stem + '/columns_equal_full_fit_columns', full['columns'],
                                 fits.get(key, {}).get('columns')))
            records.append(exact(stem + '/columns_and_rank',
                                 [len(full['columns']), int(np.linalg.matrix_rank(full['values']))],
                                 [expected['full_columns'][model]] * 2))
            owner = dict(zip(base['columns'], base['groups']))
            wanted = ['intercept' if c == 'intercept' else 'geography' if c in ADDED else owner.get(c)
                      for c in full['columns']]
            records.append(exact(stem + '/groups', full['groups'], wanted))
            features = {}
            for column, group in zip(full['columns'], full['groups']):
                if '=' in column and column not in ADDED:
                    features.setdefault(column.split('=', 1)[0], set()).add(group)
            consistency = [full['group_constant'], all(len(g) == 1 for g in features.values()),
                           set(full['groups']) <= {'intercept', *GROUPS}]
            records.append(exact(stem + '/group_consistency', consistency, [True, True, True]))
        card = evidence.scorecard['representations'][representation][CANDIDATE]
        saved_state = card.get('latitude_state_full_sample', {})
        fitted_state = fits.get((representation, CANDIDATE, 'full_sample', 'all'), {}).get('state') or {}
        records.append(exact(f'{representation}/M2/scorecard_full_sample_state_hex', saved_state.get('knots_hex'),
                             fitted_state.get('knots_hex')))
    return records


# ---------------------------------------------------------------------
# Evidence consistency, metrics, paired interval and verdict
# ---------------------------------------------------------------------


def fold_counts(context) -> dict:
    return {'full_sample': 1, 'primary_loco': len(context.order),
            'm49_subregion_lo': len(set(context.fold_ids['m49_subregion_lo'])),
            'random10': len(set(context.fold_ids['random10']))}


def consistency_records(evidence, context, expected) -> list[dict]:
    order, folds, p = context.order, evidence.folds, evidence.predictions
    n = len(order)
    records = [exact('n_countries', n, expected['n']), exact('iso3_unique', len(set(order)), n),
               exact('labels_cover_countries', sorted(set(order) - set(evidence.labels.index)), []),
               sequence('primary_fold_is_row_index', folds.primary_fold.tolist(), list(range(n))),
               sequence('primary_n_train_is_membership_size', folds.primary_n_train.tolist(),
                        [len(context.training[i]) for i in order]),
               exact('held_out_never_in_own_training', sum(i in context.training[i] for i in order), 0),
               exact('training_members_are_countries',
                     sorted(set().union(*map(set, context.training.values())) - set(order)), [])]
    by_label = folds.groupby('m49_subregion').m49_fold
    records.append(exact('m49_fold_constant_within_and_distinct_across_labels',
                         [bool((by_label.nunique() == 1).all()), by_label.first().nunique()], [True, by_label.ngroups]))
    positions = p.iso3.map(context.pos)
    records.append(exact('observed_identical_in_every_prediction_row', bool(positions.notna().all() and np.array_equal(
        p.observed.to_numpy(dtype=float), context.y[positions.fillna(0).astype(int).to_numpy()])), True))
    records.append(exact('observed_finite', bool(np.isfinite(context.y).all()), True))
    records.append(exact('prediction_row_count', len(p), len(REPRESENTATIONS) * len(MODELS) * (1 + len(PROTOCOLS)) * n))
    m49 = folds.set_index('iso3').m49_subregion
    records.append(exact('prediction_m49_labels_match_folds',
                         bool((p.m49_subregion.to_numpy() == m49.reindex(p.iso3).to_numpy()).all()), True))
    for representation, model, protocol in itertools.product(REPRESENTATIONS, MODELS, ('in_sample', *PROTOCOLS)):
        rows = p[(p.representation == representation) & (p.model == model) & (p.protocol == protocol)]
        records.append(sequence(f'{representation}/{model}/{protocol}/prediction_rows_in_canonical_order',
                                rows.iso3.tolist(), order))
    states, keys = evidence.states, fold_counts(context)
    for representation, model in itertools.product(REPRESENTATIONS, MODELS):
        subset = states[(states.representation == representation) & (states.model == model)]
        records.append(exact(f'{representation}/{model}/fit_counts', subset.protocol.value_counts().to_dict(), keys))
    fit_keys = sorted(map(tuple, states[['representation', 'model', 'protocol', 'fit']].to_numpy().tolist()))
    coefficient_keys = sorted(set(map(tuple, evidence.coefficients[['representation', 'model', 'protocol', 'fit']]
                                      .to_numpy().tolist())))
    records.append(exact('fit_states_unique_and_equal_to_coefficient_fits',
                         [len(set(fit_keys)) == len(fit_keys), fit_keys == coefficient_keys], [True, True]))
    return records


def input_records(evidence) -> list[dict]:
    records = list(evidence.manifest_records)
    inputs = evidence.provenance.get('inputs', {})
    for role, entry in evidence.auxiliary.items():
        name = Path(entry['path']).name
        bound = [v.get('sha256') for k, v in inputs.items() if Path(k).name == name]
        records.append(exact(f'provenance_binding/{role}', entry['sha256'],
                             bound[0] if len(bound) == 1 else f'{len(bound)} provenance entries named {name}'))
    return records


def card_node(card, path):
    for key in path:
        card = card.get(key, {}) if isinstance(card, dict) else {}
    return card


def metric_records(evidence, context) -> list[dict]:
    records, p = [], evidence.predictions
    for representation, model in itertools.product(REPRESENTATIONS, MODELS):
        card, stem = evidence.scorecard['representations'][representation][model], f'{representation}/{model}'
        frame = prediction_slice(p, representation, model, 'in_sample', context.order)
        r2, root, _ = scores(frame.observed, frame.prediction)
        records += [numeric(f'{stem}/in_sample_r2', r2, card.get('in_sample_r2'), METRIC_TOL),
                    numeric(f'{stem}/in_sample_rmse', root, card.get('in_sample_rmse'), METRIC_TOL),
                    exact(f'{stem}/n', len(frame), card.get('n'))]
        for protocol, path in CARD_PATH.items():
            block, frame = card_node(card, path), prediction_slice(p, representation, model, protocol, context.order)
            r2, root, mae = scores(frame.observed, frame.prediction)
            label = f'{stem}/{".".join(path) or "primary"}'
            records += [numeric(f'{label}.cv_r2', r2, block.get('cv_r2'), METRIC_TOL),
                        numeric(f'{label}.cv_rmse', root, block.get('cv_rmse'), METRIC_TOL),
                        numeric(f'{label}.cv_mae', mae, block.get('cv_mae'), METRIC_TOL)]
        frame = prediction_slice(p, representation, model, 'primary_loco', context.order)
        region, value, size = worst_region(frame)
        worst = card.get('worst_region', {})
        records += [exact(f'{stem}/worst_region.region', region, worst.get('region')),
                    numeric(f'{stem}/worst_region.rmse', value, worst.get('rmse'), METRIC_TOL),
                    exact(f'{stem}/worst_region.n', size, worst.get('n'))]
        regions, reported = region_rmse(frame), card.get('region_rmse', {})
        records.append(exact(f'{stem}/region_set', sorted(regions.index), sorted(reported)))
        records.append(summarise(f'{stem}/region_rmse', [
            close('region_rmse', str(k), abs(v - float(reported.get(k, 'nan'))), METRIC_TOL)
            for k, v in regions.rmse.items()]))
        full = context.full[(representation, model)]
        records.append(exact(f'{stem}/matrix_columns_and_rank', [card.get('matrix_columns'), card.get('matrix_rank')],
                             [len(full['columns']), int(np.linalg.matrix_rank(full['values']))]))
    return records


def paired_records(evidence, context) -> tuple[list[dict], dict]:
    records, intervals, p = [], {}, evidence.predictions
    for representation in REPRESENTATIONS:
        frames = {m: prediction_slice(p, representation, m, 'primary_loco', context.order) for m in MODELS}
        interval = paired_interval(frames[CANDIDATE].observed, frames[CANDIDATE].prediction,
                                   frames[COMPARATOR].prediction)
        intervals[representation] = interval
        saved = evidence.scorecard['representations'][representation].get('paired_primary_comparison', {})
        bounds = list(saved.get('country_bootstrap_95_interval', [None, None])) + [None, None]
        stem = f'{representation}/paired'
        records += [numeric(f'{stem}/delta_rmse', interval['delta_rmse'], saved.get('delta_rmse'), METRIC_TOL),
                    numeric(f'{stem}/interval_low', interval['low'], bounds[0], METRIC_TOL),
                    numeric(f'{stem}/interval_high', interval['high'], bounds[1], METRIC_TOL),
                    exact(f'{stem}/resamples_and_seed', [RESAMPLES, SEED], [saved.get('resamples'), saved.get('seed')])]
    return records, intervals


def structural_integrity(context, fits, representation, expected) -> bool:
    counts = fold_counts(context)
    for model in MODELS:
        mine = {k: f for k, f in fits.items() if k[:2] == (representation, model)}
        full = mine.get((representation, model, 'full_sample', 'all'))
        if (full is None or not full['matrix_rank'] == full['matrix_columns'] == expected['full_columns'][model]
                or any(sum(k[2] == p for k in mine) != c for p, c in counts.items())
                or not all(structurally_sound(f, model) for f in mine.values())):
            return False
    return True


def verdict_records(evidence, context, fits, intervals, expected, practical=PRACTICAL) -> list[dict]:
    """Verdict Booleans and label for both representations (only the primary decides), and the stopping rule."""
    scorecard, p, records = evidence.scorecard, evidence.predictions, []
    withheld = scorecard.get('integrity_passed') is not True
    for representation in REPRESENTATIONS:
        decides = representation == PRIMARY_REPRESENTATION
        block = scorecard['representations'][representation]
        entry = block.get('verdict' if decides else 'representation_conditions', {})
        frames = {(m, q): prediction_slice(p, representation, m, q, context.order) for m in MODELS
                  for q in ('primary_loco', 'm49_subregion_lo')}
        secondary = {m: rmse(frames[(m, 'm49_subregion_lo')].observed - frames[(m, 'm49_subregion_lo')].prediction)
                     for m in MODELS}
        m49 = secondary[CANDIDATE] - secondary[COMPARATOR]
        worst = worst_region(frames[(CANDIDATE, 'primary_loco')])[1] \
            - worst_region(frames[(COMPARATOR, 'primary_loco')])[1]
        interval = intervals[representation]
        conditions = verdict_conditions(interval['delta_rmse'], interval['low'], interval['high'], m49, worst,
                                        entry.get('R4_structural_integrity') is True, practical)
        label = None if withheld else conditions['label']
        support = None if withheld else bool(conditions['P1'] and conditions['P2'] and conditions['R4'])
        stem = f'{representation}/verdict'
        records += [exact(f'{stem}/{key}', conditions[name], entry.get(key)) for name, key in VERDICT_KEYS.items()]
        records += [exact(f'{stem}/inputs_finite', conditions['finite'], True),
                    exact(f'{stem}/label', label, entry.get('verdict' if decides else 'label_descriptive_only')),
                    exact(f'{stem}/predictive_support', support, entry.get('predictive_support')),
                    exact(f'{stem}/promoted', None if withheld else label == LABEL_PROMOTED, entry.get('promoted')),
                    numeric(f'{stem}/delta_rmse', interval['delta_rmse'], entry.get('delta_rmse'), METRIC_TOL),
                    numeric(f'{stem}/interval_low', interval['low'], (entry.get('interval') or [None])[0], METRIC_TOL),
                    numeric(f'{stem}/interval_high', interval['high'], (entry.get('interval') or [None, None])[-1],
                            METRIC_TOL),
                    numeric(f'{stem}/m49_rmse_change', m49, entry.get('m49_rmse_change'), METRIC_TOL),
                    numeric(f'{stem}/worst_region_rmse_change', worst, entry.get('worst_region_rmse_change'),
                            METRIC_TOL),
                    exact(f'{stem}/R4_equals_recomputed_structural_integrity',
                          structural_integrity(context, fits, representation, expected),
                          entry.get('R4_structural_integrity'))]
        if decides:
            retained = None if withheld else 'M2' if label == LABEL_PROMOTED else 'M0*'
            manifest = evidence.manifest
            records += [exact('scorecard/top_level_verdict', label, scorecard.get('verdict', {}).get('verdict')),
                        exact('scorecard/retained_static_specification', retained,
                              scorecard.get('retained_static_specification')),
                        exact('manifest/verdict', label, manifest.get('verdict')),
                        exact('manifest/predictive_support', support, manifest.get('predictive_support')),
                        exact('manifest/retained_static_specification', retained,
                              manifest.get('retained_static_specification')),
                        exact('manifest/integrity_passed', scorecard.get('integrity_passed'),
                              manifest.get('integrity_passed'))]
    stopping = scorecard.get('static_stopping_indicator', {})
    deltas = {**{k: float(v) for k, v in evidence.stage_deltas.items()},
              'M2': intervals[PRIMARY_REPRESENTATION]['delta_rmse']}
    saved = stopping.get('point_delta_rmse_vs_m0star', {})
    best = min(deltas.values())
    records += [numeric(f'stopping/delta_{k}', v, saved.get(k), METRIC_TOL) for k, v in deltas.items()]
    records += [numeric('stopping/best', best, stopping.get('best'), METRIC_TOL),
                exact('stopping/fires', bool(best > PRACTICAL), stopping.get('fires')),
                exact('stopping/m2_decides', bool(deltas['M1a'] > 0 and deltas['M1b'] > 0), stopping.get('m2_decides')),
                exact('manifest/static_stopping_indicator_fires', bool(best > PRACTICAL),
                      evidence.manifest.get('static_stopping_indicator_fires'))]
    return records


# ---------------------------------------------------------------------
# Gamma, shares and representation identity
# ---------------------------------------------------------------------


def latitude_coefficients(coefficients, representation, column, protocol) -> pd.Series:
    rows = coefficients[(coefficients.representation == representation) & (coefficients.model == CANDIDATE)
                        & (coefficients.column == column) & (coefficients.protocol == protocol)]
    return rows.set_index('fit').coefficient.astype(float)


def gamma_records(evidence, context) -> list[dict]:
    records = []
    for representation in REPRESENTATIONS:
        full = latitude_coefficients(evidence.coefficients, representation, GAMMA, 'full_sample').to_numpy()
        values = latitude_coefficients(evidence.coefficients, representation, GAMMA, 'primary_loco').to_numpy()
        saved = evidence.scorecard['representations'][representation][CANDIDATE].get('gamma', {})
        stem = f'{representation}/gamma'
        full_gamma = float(full[0]) if len(full) == 1 else float('nan')
        empty = len(values) == 0
        records += [exact(f'{stem}/one_full_fit', len(full), 1),
                    exact(f'{stem}/n_primary_training_fits', len(values), saved.get('n_primary_training_fits')),
                    exact(f'{stem}/n_primary_training_fits_is_n', len(values), len(context.order)),
                    numeric(f'{stem}/full', full_gamma, saved.get('full'), METRIC_TOL),
                    exact(f'{stem}/strictly_negative_n', int((values < 0).sum()), saved.get('strictly_negative_n')),
                    numeric(f'{stem}/strictly_negative_fraction', float('nan') if empty else float((values < 0).mean()),
                            saved.get('strictly_negative_fraction'), METRIC_TOL),
                    exact(f'{stem}/zero_n', int((values == 0).sum()), saved.get('zero_n')),
                    numeric(f'{stem}/same_sign_as_full_fraction',
                            float('nan') if empty else float(np.mean(np.sign(values) == np.sign(full_gamma))),
                            saved.get('same_sign_as_full_fraction'), METRIC_TOL)]
        for name, reducer in (('median', np.median), ('min', np.min), ('max', np.max)):
            value = float('nan') if empty else float(reducer(values))
            records.append(numeric(f'{stem}/{name}', value, saved.get(name), METRIC_TOL))
    return records


def share_records(evidence, context) -> list[dict]:
    records, y = [], context.y
    tss = float(np.sum((y - y.mean()) ** 2))
    saved = {(r.representation, r.model, r.coalition): (int(r.n_groups), float(r.r2))
             for r in evidence.coalitions.itertuples(index=False)}
    for representation, model in itertools.product(REPRESENTATIONS, MODELS):
        full, stem = context.full[(representation, model)], f'{representation}/{model}/shares'
        x, groups = full['values'], np.array(full['groups'])
        present = [g for g in GROUPS if (groups == g).any()]
        r2, items = {}, []
        for size in range(len(present) + 1):
            for coalition in itertools.combinations(present, size):
                keep = (groups == 'intercept') | np.isin(groups, coalition)
                fitted = x[:, keep] @ qr_coefficients(x[:, keep], y)
                r2[frozenset(coalition)] = 1.0 - float(np.sum((y - fitted) ** 2)) / tss
                name = '+'.join(coalition) or 'intercept_only'
                reported = saved.get((representation, model, name), (None, float('nan')))
                items += [close('coalition_r2', name, abs(r2[frozenset(coalition)] - reported[1]), SHARE_TOL),
                          same('coalition_n_groups', name, size, reported[0])]
        for check in ('coalition_r2', 'coalition_n_groups'):
            records.append(summarise(f'{stem}/{check}', [i for i in items if i['check'] == check]))
        records.append(exact(f'{stem}/n_coalitions', sum(1 for k in saved if k[:2] == (representation, model)),
                             2 ** len(present)))
        shares = shapley_by_permutations(r2, present)
        card = evidence.scorecard['representations'][representation][model]
        records += [numeric(f'{stem}/share_{g}', shares.get(g, 0.0), card.get(f'share_{g}'), SHARE_TOL) for g in GROUPS]
        total = sum(shares.values())
        records += [numeric(f'{stem}/residual_share', 1.0 - r2[frozenset(present)], card.get('residual_share'),
                            SHARE_TOL),
                    numeric(f'{stem}/named_share_sum_equals_in_sample_r2', total, card.get('in_sample_r2'),
                            ACCOUNTING_TOL),
                    numeric(f'{stem}/named_plus_reported_residual_equals_one',
                            total + float(card.get('residual_share', 'nan')), 1.0, ACCOUNTING_TOL)]
    return records


def identity_records(evidence, context, intervals) -> list[dict]:
    records, p = [], evidence.predictions
    reported = evidence.scorecard.get('representation_equivalence', {})
    other = REPRESENTATIONS[1]
    for model, protocol in itertools.product(MODELS, ('in_sample', *PROTOCOLS)):
        a, b = (prediction_slice(p, r, model, protocol, context.order).prediction.to_numpy()
                for r in (PRIMARY_REPRESENTATION, other))
        gap, stem = float(np.max(np.abs(a - b))), f'identity/predictions/{model}/{protocol}'
        records += [numeric(f'{stem}/within_tolerance', gap, 0.0, IDENTITY_PREDICTION_TOL),
                    numeric(f'{stem}/equals_reported', gap,
                            reported.get('predictions', {}).get(model, {}).get(protocol), METRIC_TOL)]
    for column in LATITUDE_COEFFICIENTS:
        stem, saved = f'identity/coefficients/{column}', reported.get('coefficients', {}).get(column, {})
        full = [latitude_coefficients(evidence.coefficients, r, column, 'full_sample') for r in REPRESENTATIONS]
        folds = [latitude_coefficients(evidence.coefficients, r, column, 'primary_loco').reindex(context.order)
                 for r in REPRESENTATIONS]
        full_gap = float(abs(full[0].iloc[0] - full[1].iloc[0])) if len(full[0]) == len(full[1]) == 1 else float('nan')
        fold_gap = float(np.max(np.abs(folds[0].to_numpy() - folds[1].to_numpy())))
        records += [numeric(f'{stem}/full_within_tolerance', full_gap, 0.0, IDENTITY_COEF_TOL),
                    numeric(f'{stem}/primary_training_max_within_tolerance', fold_gap, 0.0, IDENTITY_COEF_TOL),
                    numeric(f'{stem}/full_equals_reported', full_gap, saved.get('full'), METRIC_TOL),
                    numeric(f'{stem}/primary_training_max_equals_reported', fold_gap,
                            saved.get('primary_training_max'), METRIC_TOL),
                    exact(f'{stem}/training_fits_compared', int(folds[0].notna().sum()),
                          saved.get('training_fits_compared'))]
    paired = reported.get('scores', {}).get('paired', {})
    a, b = intervals[PRIMARY_REPRESENTATION], intervals[other]
    bounds = list(paired.get('interval') or []) + [None, None]
    for name, gap, saved in (('delta_rmse', abs(a['delta_rmse'] - b['delta_rmse']), paired.get('delta_rmse')),
                             ('interval_low', abs(a['low'] - b['low']), bounds[0]),
                             ('interval_high', abs(a['high'] - b['high']), bounds[1])):
        records += [numeric(f'identity/paired/{name}/within_tolerance', gap, 0.0, IDENTITY_PREDICTION_TOL),
                    numeric(f'identity/paired/{name}/equals_reported', gap, saved, METRIC_TOL)]
    records.append(exact('identity/reported_passes', reported.get('passes'), True))
    return records


# ---------------------------------------------------------------------
# Negative controls (in-memory copies only)
# ---------------------------------------------------------------------


def _failed(items) -> list[str]:
    return sorted({i['check'] for i in items if not item_passes(i)})


def negative_controls(evidence, context, fits, intervals, expected) -> dict:
    rows = {(r.representation, r.model, r.protocol, r.fit): r for r in evidence.states.itertuples(index=False)}
    lookups = {'predictions': prediction_lookup(evidence.predictions),
               'coefficients': coefficient_lookup(evidence.coefficients)}
    controls = {}

    key = (PRIMARY_REPRESENTATION, CANDIDATE, 'primary_loco', context.order[0])
    baseline = _failed(fit_items(context, lookups, rows[key], fits[key]))
    altered = evidence.predictions.copy()
    mask = ((altered.representation == key[0]) & (altered.model == key[1]) & (altered.protocol == key[2])
            & (altered.iso3 == key[3]))
    altered.loc[mask, 'prediction'] = altered.loc[mask, 'prediction'] + CONTROL_SHIFT
    failed = _failed(fit_items(context, {**lookups, 'predictions': prediction_lookup(altered)}, rows[key], fits[key]))
    metric_failed = [r['check'] for r in metric_records(dataclasses.replace(evidence, predictions=altered), context)
                     if not r['passes']]
    controls['a_perturbed_prediction'] = {
        'description': 'one saved out-of-fold prediction increased by 1e-6 in an in-memory copy',
        'target': '/'.join(key), 'baseline_failed_checks': baseline, 'failed_checks': failed,
        'also_failed_metric_checks': metric_failed, 'detected': bool(not baseline and 'predictions' in failed)}

    saved = evidence.scorecard['representations'][PRIMARY_REPRESENTATION].get('verdict', {})
    delta = intervals[PRIMARY_REPRESENTATION]['delta_rmse']
    r1 = delta <= PRACTICAL
    baseline = [r['check'] for r in verdict_records(evidence, context, fits, intervals, expected) if not r['passes']]
    variants = {}
    for name, threshold in (('boundary', float(np.nextafter(delta, -np.inf)) if r1 else float(delta)),
                            ('unit', -1.0 if r1 else 1.0)):
        records = verdict_records(evidence, context, fits, intervals, expected, practical=threshold)
        failed = [r['check'] for r in records if not r['passes']]
        shifted = next(r['recomputed'] for r in records if r['check'] == f'{PRIMARY_REPRESENTATION}/verdict/label')
        variants[name] = {'threshold': threshold, 'failed_checks': failed, 'label_under_shift': shifted,
                          'saved_label': saved.get('verdict'), 'label_changed': shifted != saved.get('verdict'),
                          'R1_comparison_failed': f'{PRIMARY_REPRESENTATION}/verdict/R1_delta_le_minus_0.002' in failed}
    controls['b_shifted_promotion_threshold'] = {
        'description': 'R1 threshold -0.002 replaced by the nearest value that flips R1 at the saved delta (boundary) '
                       'and by -1 or +1 (unit); the recomputed R1 and, where P1, P2, R2-R4 allow, the label must '
                       'disagree with the saved verdict',
        'saved_delta_rmse': delta, 'saved_R1': saved.get(VERDICT_KEYS['R1']), 'baseline_failed_checks': baseline,
        'variants': variants,
        'detected': bool(not baseline and all(v['R1_comparison_failed'] for v in variants.values()))}

    key = (PRIMARY_REPRESENTATION, CANDIDATE, 'full_sample', 'all')
    knots = [float(k) for k in fits[key]['knots']]
    knots[1] += CONTROL_SHIFT
    shifted = refit(context, rows[key], knots=knots)
    baseline = _failed(fit_items(context, lookups, rows[key], fits[key])
                       + full_design_items(context, key[0], key[1], fits[key]['knots']))
    failed = _failed(fit_items(context, lookups, rows[key], shifted)
                     + full_design_items(context, key[0], key[1], knots))
    channels = ('knots_from_training_latitudes', 'full_design_added_columns', 'coefficients', 'predictions')
    gap = (fits[key]['knots'][2] - fits[key]['knots'][1]) / 2
    resolution = {}
    for shift in (s for s in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if s < gap):
        moved = refit(context, rows[key], knots=[k + shift * (j == 1) for j, k in enumerate(fits[key]['knots'])])
        resolution[repr(shift)] = float(np.max(np.abs(moved['prediction'] - fits[key]['prediction'])))
    controls['c_altered_knot'] = {
        'description': 'interior knot xi_2 of the full-sample M2 state (primary representation) moved by +1e-6 '
                       'degrees; design, refit and fitted values rebuilt from the altered state',
        'target': '/'.join(key), 'baseline_failed_checks': baseline, 'failed_checks': failed,
        'channels_failed': {c: c in failed for c in channels},
        'max_fitted_value_change': float(np.max(np.abs(shifted['prediction'] - fits[key]['prediction']))),
        'max_coefficient_change': float(np.max(np.abs(shifted['beta'] - fits[key]['beta']))),
        'detected_by_rebuilt_predictions': 'predictions' in failed,
        'fitted_value_change_by_knot_shift': resolution,
        'smallest_tested_shift_resolved_by_predictions': next(
            (float(s) for s, change in resolution.items() if change > PREDICTION_TOL), None),
        'note': 'a 1e-6 degree knot error moves fitted values by far less than the 1e-9 prediction tolerance on '
                'realistic designs; it is detected through the knot re-derivation and the full-design identity, and '
                'the prediction channel\'s resolution is recorded above',
        'detected': bool(not baseline and set(failed) & set(channels))}

    target = None
    for iso in context.order:
        key = (PRIMARY_REPRESENTATION, CANDIDATE, 'primary_loco', iso)
        train = fits[key]['train'].tolist()
        excluded = sorted(set(range(len(context.order))) - set(train) - {context.pos[iso]})
        if train and excluded:
            target = (key, train, excluded)
            break
    if target is None:
        controls['d_swapped_training_membership'] = {'description': 'no primary fit has a buffer-excluded country',
                                                     'detected': False}
    else:
        key, train, excluded = target
        swapped = refit(context, rows[key], training=[*train[1:], excluded[0]])
        baseline = _failed(fit_items(context, lookups, rows[key], fits[key]))
        failed = _failed(fit_items(context, lookups, rows[key], swapped))
        controls['d_swapped_training_membership'] = {
            'description': 'one primary training membership swapped: the first training country dropped and the first '
                           'buffer-excluded country added, then refitted',
            'target': '/'.join(key), 'dropped': context.order[train[0]], 'added': context.order[excluded[0]],
            'baseline_failed_checks': baseline, 'failed_checks': failed,
            'max_coefficient_change': float(np.max(np.abs(swapped['beta'] - fits[key]['beta']))),
            'detected': bool(not baseline and {'coefficients', 'predictions'} & set(failed))}

    key = (PRIMARY_REPRESENTATION, CANDIDATE, 'full_sample', 'all')
    altered = evidence.coefficients.copy()
    mask = ((altered.representation == key[0]) & (altered.model == key[1]) & (altered.protocol == key[2])
            & (altered.fit == key[3]) & (altered.column == GAMMA))
    altered.loc[mask, 'coefficient'] = altered.loc[mask, 'coefficient'] + CONTROL_SHIFT
    baseline = _failed(fit_items(context, lookups, rows[key], fits[key]))
    failed = _failed(fit_items(context, {**lookups, 'coefficients': coefficient_lookup(altered)}, rows[key], fits[key]))
    gamma_failed = [r['check'] for r in gamma_records(dataclasses.replace(evidence, coefficients=altered), context)
                    if not r['passes']]
    controls['e_altered_coefficient'] = {
        'description': 'the saved full-sample gamma coefficient (primary representation) increased by 1e-6 in an '
                       'in-memory copy', 'target': '/'.join(key) + '/' + GAMMA, 'baseline_failed_checks': baseline,
        'failed_checks': failed, 'also_failed_gamma_checks': gamma_failed,
        'detected': bool(not baseline and 'coefficients' in failed)}
    return controls


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


def reconstruct(evidence: Evidence, expected=None) -> dict:
    expected = expected or EXPECTED
    context = build_context(evidence)
    fits = reconstruct_fits(evidence, context)
    by_fit = fit_sections(evidence, context, fits)
    paired, intervals = paired_records(evidence, context)
    checks = {
        '1_manifest_and_inputs': input_records(evidence),
        '2_evidence_consistency': consistency_records(evidence, context, expected),
        '3_designs_and_states': design_records(evidence, context, fits, expected) + by_fit['designs'],
        '4_refit_coefficients': by_fit['refit'],
        '5_predictions': by_fit['predictions'],
        '6_metrics': metric_records(evidence, context),
        '7_paired_interval': paired,
        '8_verdict_and_stopping': verdict_records(evidence, context, fits, intervals, expected),
        '9_gamma_summary': gamma_records(evidence, context),
        '10_shares': share_records(evidence, context),
        '11_representation_identity': identity_records(evidence, context, intervals),
    }
    sections = {name: section(records) for name, records in checks.items()}
    failures = [f'{name}: {check}' for name, block in sections.items() for check in block['failed_checks']]
    controls = negative_controls(evidence, context, fits, intervals, expected)
    return {
        'what_this_is': 'independent reconstruction of the M2 primary result from its saved evidence '
                        '(M2_EVALUATOR_SPEC.md §9)',
        'implementation': 'research/model_v2/m2_independent_check.py',
        'scope': SCOPE, 'tolerances': TOLERANCES, 'expected_structure': expected,
        'inputs': {'artifact_sha256': evidence.manifest.get('artifact_sha256'), 'auxiliary': evidence.auxiliary},
        'fits_reconstructed': len(fits),
        'section_summary': {name: {k: b[k] for k in ('passes', 'n_comparisons', 'n_failed', 'max_abs_diff')}
                            for name, b in sections.items()},
        'all_checks_pass': not failures, 'failed_checks': failures,
        'negative_controls': controls,
        'negative_controls_all_detected': all(c['detected'] for c in controls.values()),
        'sections': sections,
    }


def run(primary_dir=PRIMARY_DIR, countries=COUNTRIES_PATH, m1a_scorecard=M1A_SCORECARD, m1b_scorecard=M1B_SCORECARD,
        expected=None) -> dict:
    return reconstruct(load_evidence(primary_dir, countries, m1a_scorecard, m1b_scorecard), expected)


def main(argv=None, expected=None) -> int:
    parser = argparse.ArgumentParser(description='Independently reconstruct the M2 primary result.')
    parser.add_argument('--primary', default=str(PRIMARY_DIR), help='completed M2 primary output directory')
    parser.add_argument('--out', default=str(OUTPUT_PATH), help='where to write the verification JSON')
    parser.add_argument('--countries', default=str(COUNTRIES_PATH), help='country label table (labels only)')
    parser.add_argument('--m1a-scorecard', default=str(M1A_SCORECARD))
    parser.add_argument('--m1b-scorecard', default=str(M1B_SCORECARD))
    args = parser.parse_args(argv)
    try:
        report = run(args.primary, args.countries, args.m1a_scorecard, args.m1b_scorecard, expected)
    except DigestRefused as refusal:
        report = {'what_this_is': 'independent reconstruction of the M2 primary result (refused)',
                  'refused': str(refusal), 'manifest': refusal.records, 'all_checks_pass': False,
                  'negative_controls': None, 'negative_controls_all_detected': False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + '\n')
    passed = bool(report['all_checks_pass'] and report['negative_controls_all_detected'])
    if 'refused' in report:
        print(f"REFUSED: {report['refused']}")
    else:
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
