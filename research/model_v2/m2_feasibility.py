"""M2 predictor-only structural feasibility audit (M2_FEASIBILITY_AUDIT.md).

Builds the frozen M0* design and the M2 candidate design (M0* plus the two natural cubic latitude
columns and I(SH) * abs_latitude) for the full sample, every primary 500 km training fit, every M49
training fit and the random reference folds, in both registered representations, and reports ranks,
added identifiable dimensions, hemisphere support, knots, held-out extrapolation and conditioning. The
same is done for the conditional M1a area-measurement arm (M2_EVALUATION_CONTRACT.md §10).

It never reads an outcome and never estimates anything.
* Numeric predictors are the exact scoring values in the pinned, outcome-free M1b encoded design
  (M0* rows; log10 already applied where the schema requires it).
* Categorical labels and identifiers come from the pinned country table, because that table's
  numeric columns are rounded copies.
* The M1a arm replaces the five remeasured geography features from the pinned M1a measurement file.
* Fold memberships come from the pinned M1b fold artifact.
* Every read is an explicit column allow-list screened for outcome tokens.

A structural failure is recorded, written and returned as a failure. Nothing is dropped, merged,
regularized or substituted. The canonical record is written only from committed, pushed code.

    uv run python -m research.model_v2.m2_feasibility
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v2 import cv
from research.model_v2 import m2_latitude_basis as lb
from src import feature_schema as fs
from src.decomposition import CATEGORICAL_FEATURES, LOG10_FEATURES
from src.stability import _used_features

ROOT = Path(__file__).resolve().parents[2]
COUNTRY_TABLE = 'research/model_v2/outputs/m0_countries.csv'
FOLDS = 'research/model_v2/outputs/m1b_primary/m1b_cv_folds.csv'
ENCODED_DESIGN = 'research/model_v2/outputs/m1b_primary/m1b_design_matrices.csv'
M1A_FEATURES = 'research/model_v2/outputs/m1a_geography_features.csv'
M1B_MANIFEST = 'research/model_v2/outputs/m1b_primary/m1b_result_manifest.json'
PINS = {
    COUNTRY_TABLE: '9e99b3798e91f2b7e40250921edd5e5a788e4191df3e9027dceb98fe76b43536',
    FOLDS: '6a9e3976f51e60a66c6b70fd65cc56626df1eec28a41e774c39260d1a2a459f1',
    ENCODED_DESIGN: 'bbdc8b1df1314b55c409f6ff59db03d39a9109d0a6827bef9ec58ce8a4caa982',
    M1A_FEATURES: 'a62d04d478b4500656c83e28d4fb4fb2d5307388b8ea5142a0f8de8dc059c4b3',
}
# Every local file whose bytes shape the record; all must be committed and pushed for a canonical run.
CODE = ('research/model_v2/m2_feasibility.py', 'research/model_v2/m2_latitude_basis.py',
        'research/model_v2/cv.py', 'src/feature_schema.py', 'src/decomposition.py', 'src/stability.py')
OUT = ROOT / 'research/model_v2/outputs/m2_feasibility'
RECORD_FILES = ('m2_feasibility_fits.csv', 'm2_feasibility_coalitions.csv', 'm2_feasibility_summary.json')

# The country table also carries outcome, fitted, residual and diagnostic columns; only these are read.
LABEL_COLUMNS = ['Country', 'iso3', 'm49_subregion', 'climate_zone', 'hemisphere', 'spatial_block',
                 'income_group']
# Read only to measure how far the table's rounded numeric copies sit from the exact scoring values.
TABLE_NUMERIC_COLUMNS = ['cum_co2_per_capita', 'cum_co2_total', 'abs_latitude', 'elevation', 'continentality',
                         'population', 'station_density']
NUMERIC_COLUMNS = {
    'primary_total_co2': ['cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'population',
                          'station_density'],
    'per_capita': ['cum_co2_per_capita', 'abs_latitude', 'elevation', 'continentality', 'population',
                   'station_density'],
}
FOLD_COLUMNS = ['iso3', 'primary_n_train', 'primary_training_iso3', 'm49_fold', 'random10_fold']
ENCODED_COLUMNS = ['representation', 'model', 'iso3', 'column', 'group', 'value']
M1A_REMEASURED = ['abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere']
M1A_COLUMNS = ['iso3', *M1A_REMEASURED, 'spatial_block']
# Substrings that mark an outcome-bearing or model-derived column; none may enter a read.
NOT_PREDICTORS = ('trend', 'resid', 'fitted', 'lisa', 'leverage', 'studentized', 'cooks', 'sq_share',
                  'era5', 'cell_', 'station_minus', 'people_', 'warming', 'outcome', 'predict', 'error')
# Registered representations and the responsibility column each one drops.
REPRESENTATIONS = {'primary_total_co2': 'cum_co2_per_capita', 'per_capita': 'cum_co2_total'}
MEASUREMENTS = ('station', 'm1a_area')
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')
EXPECTED = {'full_sample_baseline_rank': 20, 'full_sample_candidate_rank': 23, 'added_dimensions': 3,
            'primary_fits': 151, 'm49_fits': 20, 'random10_fits': 10,
            'min_primary_n_train': 124, 'min_m49_n_train': 136}
KNOT_FIELDS = ('knot_lower_boundary', 'knot_interior_1', 'knot_interior_2', 'knot_upper_boundary')


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_digest(path, expected_sha256):
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f'{path} does not match its pinned digest {expected_sha256}')


def guard_columns(columns):
    bad = [c for c in columns if any(token in str(c).lower() for token in NOT_PREDICTORS)]
    if bad:
        raise ValueError(f'{bad} outside the predictor allow-list')


def read_table_columns(path, expected_sha256, columns) -> pd.DataFrame:
    """Only allow-listed, token-screened columns of a pinned CSV."""
    _check_digest(path, expected_sha256)
    guard_columns(columns)
    frame = pd.read_csv(path, usecols=columns, float_precision='round_trip')[columns]
    if list(frame.columns) != list(columns):
        raise ValueError(f'{path} did not yield exactly the allow-listed columns')
    return frame


def encoded_numerics(encoded: pd.DataFrame, representation: str, iso3) -> pd.DataFrame:
    """Exact transformed numeric predictors of M0* for one representation, in ``iso3`` order."""
    rows = encoded[(encoded.representation == representation) & (encoded.model == 'M0star')]
    guard_columns(rows.column.unique())
    wide = rows.pivot(index='iso3', columns='column', values='value')
    return wide.loc[list(iso3), NUMERIC_COLUMNS[representation]].reset_index()


def transform_numerics(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the schema's log10 exactly as the V1 encoder does (for synthetic raw frames)."""
    out = raw.copy()
    for column in LOG10_FEATURES & set(out.columns):
        out[column] = np.log10(out[column].to_numpy(dtype=float))
    return out


def representation_frame(predictors: pd.DataFrame, representation: str) -> pd.DataFrame:
    return predictors.drop(columns=REPRESENTATIONS[representation]).reset_index(drop=True)


def m1a_frame(frame: pd.DataFrame, m1a: pd.DataFrame) -> pd.DataFrame:
    """The frame with the five M1a area remeasurements in place of the station measurements."""
    aligned = m1a.set_index('iso3').loc[frame.iso3]
    if not (aligned.spatial_block.astype(str).to_numpy() == frame.spatial_block.astype(str).to_numpy()).all():
        raise ValueError('the M1a measurement file disagrees with the frozen spatial_block labels')
    out = frame.copy()
    for column in M1A_REMEASURED:
        if aligned[column].isna().any():
            raise ValueError(f'missing M1a {column}')
        out[column] = aligned[column].to_numpy()
    return out


@dataclass(frozen=True)
class EncoderState:
    """M0*'s fitted-encoder state: V1 feature order and train-only categorical levels."""

    numeric: list
    categorical: list
    levels: dict


def baseline_encoder(train: pd.DataFrame) -> EncoderState:
    """The state ``cv.V1Design.fit`` would set, derived without any estimation."""
    features = cv.V1Design()._features(train.columns)
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    return EncoderState(numeric=[f for f in features if f not in CATEGORICAL_FEATURES], categorical=categorical,
                        levels={c: sorted(train[c].astype(str).unique().tolist()) for c in categorical})


def encode(frame: pd.DataFrame, state: EncoderState) -> tuple[np.ndarray, list[str]]:
    """``cv.V1Design._matrix`` on already-transformed numerics: same columns, order and coding."""
    blocks, names = [np.ones((len(frame), 1))], ['intercept']
    for f in state.numeric:
        blocks.append(frame[f].to_numpy(dtype=float).reshape(-1, 1))
        names.append(f)
    for c in state.categorical:
        values = frame[c].astype(str).to_numpy()
        for level in state.levels[c][1:]:
            blocks.append((values == level).astype(float).reshape(-1, 1))
            names.append(f'{c}={level}')
    return np.hstack(blocks), names


def _rank(matrix) -> int:
    return int(np.linalg.matrix_rank(matrix))


def _unit_columns(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / np.where(norms > 0, norms, 1.0)


def _conditioning(matrix, prefix) -> dict:
    raw = np.linalg.svd(matrix, compute_uv=False)
    unit = np.linalg.svd(_unit_columns(matrix), compute_uv=False)
    return {f'{prefix}_singular_max': float(raw.max()), f'{prefix}_singular_min': float(raw.min()),
            f'{prefix}_condition_raw': float(raw.max() / raw.min()) if raw.min() > 0 else float('inf'),
            f'{prefix}_condition_unit_columns': float(unit.max() / unit.min()) if unit.min() > 0
            else float('inf')}


CANDIDATE_FIELDS = (*KNOT_FIELDS, 'test_below_lower_boundary', 'test_above_upper_boundary',
                    'test_lat_min', 'test_lat_max', 'candidate_columns', 'candidate_rank',
                    'candidate_singular_max', 'candidate_singular_min', 'candidate_condition_raw',
                    'candidate_condition_unit_columns', 'added_min_singular_value_unit_columns',
                    'added_identifiable_dimensions')


def fit_record(frame: pd.DataFrame, train_idx, test_idx) -> dict:
    """Structure of one fit: M0* on its training rows, and the M2 candidate under that fit's state."""
    train, test = frame.iloc[np.asarray(train_idx, int)], frame.iloc[np.asarray(test_idx, int)]
    encoder = baseline_encoder(train)
    baseline, _names = encode(train, encoder)
    southern = train.hemisphere.astype(str) == lb.SOUTHERN
    unseen = np.zeros(len(test), bool)
    for c in encoder.categorical:
        unseen |= ~test[c].astype(str).isin(encoder.levels[c]).to_numpy()
    record = {
        'n_train': len(train), 'n_test': len(test),
        'n_train_sh': int(southern.sum()), 'n_train_nh': int((~southern).sum()),
        'train_lat_min': float(train.abs_latitude.min()), 'train_lat_max': float(train.abs_latitude.max()),
        'train_sh_lat_min': float(train.abs_latitude[southern].min()) if southern.any() else float('nan'),
        'train_sh_lat_max': float(train.abs_latitude[southern].max()) if southern.any() else float('nan'),
        'test_rows_with_unseen_level': int(unseen.sum()),
        'baseline_columns': int(baseline.shape[1]), 'baseline_rank': _rank(baseline),
        **_conditioning(baseline, 'baseline'),
        **{field: float('nan') for field in CANDIDATE_FIELDS},
        'latitude_state': '',
    }
    try:
        state = lb.fit_state(train.abs_latitude.to_numpy())
    except lb.DegenerateLatitudeBasis as exc:
        return {**record, 'structural_pass': False, 'failure': f'degenerate latitude state: {exc}'}
    added = lb.added_block(state, train.abs_latitude.to_numpy(), train.hemisphere.to_numpy())
    candidate = np.hstack([baseline, added])
    basis = np.linalg.svd(baseline, full_matrices=False)[0][:, :record['baseline_rank']]
    unit_added = _unit_columns(added)
    beyond = unit_added - basis @ (basis.T @ unit_added)
    held = (lb.extrapolation(state, test.abs_latitude.to_numpy()) if len(test)
            else {'below': 0, 'above': 0, 'min': float('nan'), 'max': float('nan')})
    record.update({
        **dict(zip(KNOT_FIELDS, state.knots)), 'latitude_state': state.to_json(),
        'test_below_lower_boundary': held['below'], 'test_above_upper_boundary': held['above'],
        'test_lat_min': held['min'], 'test_lat_max': held['max'],
        'candidate_columns': int(candidate.shape[1]), 'candidate_rank': _rank(candidate),
        **_conditioning(candidate, 'candidate'),
        'added_min_singular_value_unit_columns': float(np.linalg.svd(beyond, compute_uv=False).min()),
    })
    record['added_identifiable_dimensions'] = record['candidate_rank'] - record['baseline_rank']
    failures = []
    if record['baseline_rank'] != record['baseline_columns']:
        failures.append('M0* training design is rank deficient')
    if record['candidate_rank'] != record['candidate_columns']:
        failures.append('candidate design is rank deficient')
    if record['added_identifiable_dimensions'] != EXPECTED['added_dimensions']:
        failures.append(f"added identifiable dimensions {record['added_identifiable_dimensions']} != 3")
    record['structural_pass'] = not failures
    record['failure'] = '; '.join(failures)
    return record


def _column_groups(frame, names):
    owner = {f: g for g, features in _used_features(frame.columns, fs.SCHEMA_V1, fs.STATUS_AVAILABLE).items()
             for f in features}
    return ['intercept' if n == 'intercept' else owner[n.split('=')[0]] for n in names]


def coalition_records(frame: pd.DataFrame) -> list[dict]:
    """Full-sample rank of every group coalition; the added columns travel only with geography."""
    baseline, names = encode(frame, baseline_encoder(frame))
    groups = np.array(_column_groups(frame, names))
    try:
        state = lb.fit_state(frame.abs_latitude.to_numpy())
        added, failure = lb.added_block(state, frame.abs_latitude.to_numpy(), frame.hemisphere.to_numpy()), ''
    except lb.DegenerateLatitudeBasis as exc:
        added, failure = None, f'degenerate latitude state: {exc}'
    rows = []
    for size in range(len(GROUPS) + 1):
        for coalition in itertools.combinations(GROUPS, size):
            keep = (groups == 'intercept') | np.isin(groups, coalition)
            part = baseline[:, keep]
            row = {'groups': '+'.join(coalition), 'baseline_columns': int(part.shape[1]),
                   'baseline_rank': _rank(part), 'failure': failure}
            if added is None:
                rows.append({**row, 'candidate_columns': -1, 'candidate_rank': -1})
                continue
            with_added = np.hstack([part, added]) if 'geography' in coalition else part
            rows.append({**row, 'candidate_columns': int(with_added.shape[1]), 'candidate_rank': _rank(with_added)})
    return rows


def encoded_identity(frame, representation, encoded, table_numerics) -> dict:
    """The full-sample M0* encoding here against the frozen M1b encoded design, aligned by column name.

    Numeric values are taken from that design, so their identity holds by construction; the independent
    evidence is that the country-table labels rebuild every frozen dummy column exactly, that column
    names and groups agree, and that the table's rounded numeric copies align country by country.
    """
    baseline, names = encode(frame, baseline_encoder(frame))
    frozen = encoded[(encoded.representation == representation) & (encoded.model == 'M0star')]
    pivot = frozen.pivot(index='iso3', columns='column', values='value')
    group_of = frozen.drop_duplicates('column').set_index('column').group
    same_columns = set(pivot.columns) == set(names)
    values = pivot.loc[frame.iso3, names].to_numpy() if same_columns else None
    dummies = [j for j, n in enumerate(names) if '=' in n]
    rounded = transform_numerics(table_numerics)
    relative = {c: float(np.max(np.abs(rounded[c].to_numpy() - frame[c].to_numpy())
                                / np.maximum(np.abs(frame[c].to_numpy()), 1e-300)))
                for c in NUMERIC_COLUMNS[representation]}
    return {'comparison': 'value for value after aligning columns by name (the frozen artifact is in schema order)',
            'same_column_set': same_columns,
            'same_groups': same_columns and list(group_of.loc[names]) == _column_groups(frame, names),
            'dummy_columns_bit_identical_from_labels': bool(same_columns and np.array_equal(values[:, dummies],
                                                                                              baseline[:, dummies])),
            'bit_identical': bool(same_columns and np.array_equal(values, baseline)),
            'country_table_numeric_max_relative_difference': relative}


def fold_plan(folds: pd.DataFrame, iso3: list[str]) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    position = {code: i for i, code in enumerate(iso3)}
    everyone = np.arange(len(iso3))
    plan = [('full_sample', 'all', everyone, np.array([], int))]
    for row in folds.itertuples():
        train = np.array(sorted(position[c] for c in row.primary_training_iso3.split('|')))
        if len(train) != row.primary_n_train or position[row.iso3] in set(train):
            raise ValueError(f'primary training membership for {row.iso3} is inconsistent')
        plan.append(('primary_loco', row.iso3, train, np.array([position[row.iso3]])))
    for protocol, column in (('m49_subregion_lo', 'm49_fold'), ('random10', 'random10_fold')):
        labels = folds.set_index('iso3').loc[iso3, column].to_numpy()
        for k in sorted(set(labels)):
            plan.append((protocol, str(k), np.flatnonzero(labels != k), np.flatnonzero(labels == k)))
    return plan


def _ints(series) -> list[int]:
    return sorted(int(v) for v in series.dropna().unique())


def _maybe(series, how):
    values = series.dropna()
    return float(getattr(values, how)()) if len(values) else None


def _summary(records: pd.DataFrame) -> dict:
    out = {}
    for (measurement, representation, protocol), part in records.groupby(
            ['measurement', 'representation', 'protocol'], sort=True):
        passed = part.structural_pass.astype(bool)
        ok = part[passed]
        moved = part[(part.test_below_lower_boundary.fillna(0) + part.test_above_upper_boundary.fillna(0)) > 0]
        out[f'{measurement}/{representation}/{protocol}'] = {
            'fits': len(part), 'all_structural_pass': bool(passed.all()),
            'failing_fits': dict(zip(part.loc[~passed, 'fit'].astype(str), part.loc[~passed, 'failure'].astype(str))),
            'n_train_min': int(part.n_train.min()), 'n_train_max': int(part.n_train.max()),
            'baseline_columns': _ints(part.baseline_columns),
            'baseline_rank_equals_columns_in_all': bool((part.baseline_rank == part.baseline_columns).all()),
            'candidate_rank_equals_columns_in_all': bool(passed.all()
                                                         and (part.candidate_rank == part.candidate_columns).all()),
            'added_identifiable_dimensions': _ints(part.added_identifiable_dimensions),
            'n_train_sh_min': int(part.n_train_sh.min()), 'n_train_sh_max': int(part.n_train_sh.max()),
            'train_sh_lat_min_max': [_maybe(part.train_sh_lat_min, 'min'), _maybe(part.train_sh_lat_max, 'max')],
            'train_sh_lat_range_min_width': _maybe(part.train_sh_lat_max - part.train_sh_lat_min, 'min'),
            'knot_ranges': {k: [_maybe(part[k], 'min'), _maybe(part[k], 'max')] for k in KNOT_FIELDS},
            'min_knot_gap': float(np.diff(ok[list(KNOT_FIELDS)].to_numpy(), axis=1).min()) if len(ok) else None,
            'fits_with_held_out_extrapolation': {
                str(r.fit): {'below': int(r.test_below_lower_boundary), 'above': int(r.test_above_upper_boundary),
                             'train_lat_min': float(r.train_lat_min), 'train_lat_max': float(r.train_lat_max),
                             'test_lat_min': float(r.test_lat_min), 'test_lat_max': float(r.test_lat_max)}
                for r in moved.itertuples()},
            'test_rows_with_unseen_level': int(part.test_rows_with_unseen_level.sum()),
            'added_min_singular_value_unit_columns_min': _maybe(part.added_min_singular_value_unit_columns, 'min'),
            'baseline_condition_unit_columns_max': _maybe(part.baseline_condition_unit_columns, 'max'),
            'candidate_condition_unit_columns_max': _maybe(part.candidate_condition_unit_columns, 'max'),
            'baseline_condition_raw_max': _maybe(part.baseline_condition_raw, 'max'),
            'candidate_condition_raw_max': _maybe(part.candidate_condition_raw, 'max'),
        }
    return out


def _checks(records, coalitions, summary, identity, measurement) -> dict:
    part = records[records.measurement == measurement]
    full = part[part.protocol == 'full_sample']
    rows = [r for r in coalitions if r['measurement'] == measurement]
    compare = ['protocol', 'fit', 'baseline_rank', 'candidate_rank', *KNOT_FIELDS, 'n_train_sh']
    checks = {
        'full_sample_ranks_20_to_23': bool((full.baseline_rank == 20).all() and (full.baseline_columns == 20).all()
                                           and (full.candidate_rank == 23).all()),
        'fit_counts': all(summary[f'{measurement}/{r}/{p}']['fits'] == EXPECTED[f'{k}_fits'] for r in REPRESENTATIONS
                          for p, k in (('primary_loco', 'primary'), ('m49_subregion_lo', 'm49'),
                                       ('random10', 'random10'))),
        'every_fit_structural_pass': bool(part.structural_pass.astype(bool).all()),
        'coalition_gain_three_exactly_with_geography': bool(rows) and all(
            (r['candidate_rank'] - r['baseline_rank']) == (3 if 'geography' in r['groups'].split('+') else 0)
            and r['baseline_rank'] == r['baseline_columns'] and r['candidate_rank'] == r['candidate_columns']
            for r in rows),
        'representations_structurally_identical': bool(
            part[part.representation == 'primary_total_co2'][compare].reset_index(drop=True)
            .equals(part[part.representation == 'per_capita'][compare].reset_index(drop=True))),
    }
    if measurement == 'station':
        checks['encoded_m0star_identity'] = all(v['bit_identical'] and v['same_groups']
                                                 and v['dummy_columns_bit_identical_from_labels']
                                                 for v in identity.values())
        checks['min_training_sizes'] = all(
            summary[f'station/{r}/primary_loco']['n_train_min'] == EXPECTED['min_primary_n_train']
            and summary[f'station/{r}/m49_subregion_lo']['n_train_min'] == EXPECTED['min_m49_n_train']
            for r in REPRESENTATIONS)
    return checks


def _git(*args, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=check)


def code_state() -> dict:
    """Commit, cleanliness and push state of every file that shapes the record."""
    head = _git('rev-parse', 'HEAD').stdout.strip()
    tracked = all(_git('ls-files', '--error-unmatch', path, check=False).returncode == 0 for path in CODE)
    clean = tracked and _git('diff', '--quiet', 'HEAD', '--', *CODE, check=False).returncode == 0
    upstream = _git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}', check=False).stdout.strip()
    pushed = bool(upstream) and _git('merge-base', '--is-ancestor', head, upstream, check=False).returncode == 0
    return {'commit': head, 'upstream': upstream,
            'upstream_commit': _git('rev-parse', upstream, check=False).stdout.strip() if upstream else '',
            'tracked': tracked, 'clean': clean, 'pushed': pushed,
            'sha256': {path: sha256(ROOT / path) for path in CODE}}


def run(out_dir=OUT) -> dict:
    out_dir = Path(out_dir)
    canonical = out_dir.resolve() == OUT.resolve()
    if out_dir.exists():
        raise RuntimeError(f'{out_dir} already exists; the audit never overwrites a prior record')
    code = code_state()
    if canonical and not (code['tracked'] and code['clean'] and code['pushed']):
        raise RuntimeError('the canonical feasibility record is written only from committed, unmodified, '
                           f'pushed code: {code}')
    labels = read_table_columns(ROOT / COUNTRY_TABLE, PINS[COUNTRY_TABLE], LABEL_COLUMNS)
    table_numerics = read_table_columns(ROOT / COUNTRY_TABLE, PINS[COUNTRY_TABLE], ['iso3', *TABLE_NUMERIC_COLUMNS])
    folds = read_table_columns(ROOT / FOLDS, PINS[FOLDS], FOLD_COLUMNS)
    encoded = read_table_columns(ROOT / ENCODED_DESIGN, PINS[ENCODED_DESIGN], ENCODED_COLUMNS)
    m1a = read_table_columns(ROOT / M1A_FEATURES, PINS[M1A_FEATURES], M1A_COLUMNS)
    iso3 = folds.iso3.tolist()
    if len(iso3) != 151 or len(set(iso3)) != 151 or set(iso3) != set(labels.iso3) or set(iso3) != set(m1a.iso3):
        raise ValueError('the frozen fold artifact, country table and M1a file disagree on the 151 countries')
    labels = labels.set_index('iso3', drop=False).loc[iso3].reset_index(drop=True)
    table_numerics = table_numerics.set_index('iso3').loc[iso3].reset_index(drop=True)
    plan = fold_plan(folds, iso3)
    rows, identity, coalitions = [], {}, []
    for representation in REPRESENTATIONS:
        station = labels.merge(encoded_numerics(encoded, representation, iso3), on='iso3', how='left',
                               validate='one_to_one')
        if list(station.iso3) != iso3 or station[NUMERIC_COLUMNS[representation]].isna().any().any():
            raise ValueError(f'{representation}: the encoded design does not cover the 151 countries')
        identity[representation] = encoded_identity(station, representation, encoded, table_numerics)
        for measurement, frame in (('station', station), ('m1a_area', m1a_frame(station, m1a))):
            for protocol, fit, train, test in plan:
                rows.append({'measurement': measurement, 'representation': representation, 'protocol': protocol,
                             'fit': fit, **fit_record(frame, train, test)})
            coalitions += [{'measurement': measurement, 'representation': representation, **r}
                           for r in coalition_records(frame)]
    records = pd.DataFrame(rows)
    summary = _summary(records)
    checks = _checks(records, coalitions, summary, identity, 'station')
    arm_checks = _checks(records, coalitions, summary, identity, 'm1a_area')
    full = records[(records.protocol == 'full_sample')].set_index(['measurement', 'representation'])
    result = {
        'audit': 'M2 predictor-only structural feasibility (no outcome read, no estimation)',
        'design': 'research/model_v2/M2_DESIGN.md', 'basis_version': lb.BASIS_VERSION,
        'candidate': 'M0* + abs_latitude_ns1 + abs_latitude_ns2 + I(SH)*abs_latitude (all geography)',
        'canonical': canonical,
        'inputs': {p: {'sha256': sha256(ROOT / p), 'pinned': PINS[p]} for p in PINS},
        'columns_read': {COUNTRY_TABLE: {'labels': LABEL_COLUMNS,
                                         'rounded_numeric_comparison_only': TABLE_NUMERIC_COLUMNS},
                         FOLDS: FOLD_COLUMNS, ENCODED_DESIGN: ENCODED_COLUMNS, M1A_FEATURES: M1A_COLUMNS},
        'numeric_source': 'M0* rows of the frozen encoded design (exact scoring values, log10 applied); all '
                          'encoded-design rows are loaded and M0* rows selected in memory',
        'code': code, 'expected': EXPECTED, 'encoded_identity': identity,
        'full_sample': {f'{m}/{r}': {k: full.loc[(m, r), k] for k in ('baseline_columns', 'baseline_rank',
                                                                      'candidate_columns', 'candidate_rank',
                                                                      'latitude_state')}
                        for m in MEASUREMENTS for r in REPRESENTATIONS},
        'by_measurement_representation_protocol': summary,
        'checks': checks, 'structural_feasibility': 'pass' if all(checks.values()) else 'fail',
        'conditional_m1a_arm_checks': arm_checks,
        'conditional_m1a_arm_feasibility': 'pass' if all(arm_checks.values()) else 'fail',
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'platform': platform.platform(), 'machine': platform.machine()},
        'numerical_note': 'singular values and condition numbers are platform-scoped in their last bits; '
                          'ranks, knots and counts are the structural evidence',
    }
    out_dir.mkdir(parents=True)
    records.to_csv(out_dir / RECORD_FILES[0], index=False, lineterminator='\n')
    pd.DataFrame(coalitions).to_csv(out_dir / RECORD_FILES[1], index=False, lineterminator='\n')
    (out_dir / RECORD_FILES[2]).write_text(json.dumps(result, indent=2, default=_json_default) + '\n',
                                           encoding='utf-8')
    manifest = {name: sha256(out_dir / name) for name in RECORD_FILES}
    (out_dir / 'm2_feasibility_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return result


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description='M2 predictor-only structural feasibility audit.')
    parser.add_argument('--out', default=str(OUT), help='fresh output directory (never overwritten)')
    result = run(parser.parse_args(argv).out)
    print(json.dumps({'structural_feasibility': result['structural_feasibility'], 'checks': result['checks'],
                      'conditional_m1a_arm_feasibility': result['conditional_m1a_arm_feasibility'],
                      'canonical': result['canonical']}, indent=2))
    return result


if __name__ == '__main__':
    sys.exit(0 if main()['structural_feasibility'] == 'pass' else 5)
