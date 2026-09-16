"""M2 scoring under ``M2_EVALUATION_CONTRACT.md`` (with Amendment 1) and ``M2_EVALUATOR_SPEC.md``.

M2 = M0* + ``abs_latitude_ns1`` + ``abs_latitude_ns2`` + ``hemisphere=S:abs_latitude`` (all geography),
compared with M0* in the primary total-CO2 representation, with the per-capita representation reported
unconditionally. Only the primary representation determines the label.

* Numeric predictors are the exact frozen scoring values (``M0star`` rows of the M1b encoded design); the
  country table supplies identifiers, M49 labels, station coordinates and categorical labels only.
* The latitude state of every fit comes from that fit's training rows (``m2_latitude_basis``, frozen).
* Before any M2 fit, every fit's state, ranks and held-out extrapolation are checked against the committed
  predictor-only feasibility record, and both M0* representations must reproduce their frozen cards.
* Execution is enabled by provenance (``v2_provenance``): ``PYTHONHASHSEED=0``, a clean, pushed code
  closure and a fresh output root.

Run with ``PYTHONHASHSEED=0 uv run python -m research.model_v2.m2_evaluate [--out DIR]``.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from research.model_v2 import cv, territory
from research.model_v2 import m2_feasibility as feas
from research.model_v2 import m2_latitude_basis as lb
from research.model_v2 import v2_provenance as prov
from research.model_v2.m0 import load_inputs, m0_complete_design
from research.model_v2.regions import m49_table
from research.model_v2.run_m0_scorecard import MIN_REGION_N, PRIMARY_BUFFER_KM
from research.model_v2.run_territory_correction import paired_rmse_interval
from research.model_v2.spatial import haversine_matrix, knn_weights, morans_i
from src.decomposition import CATEGORICAL_FEATURES, OUTCOME_COL

ROOT = prov.ROOT
OUTPUTS = 'research/model_v2/outputs'
DEFAULT_OUT = ROOT / OUTPUTS / 'm2_primary'

CONTRACT = 'research/model_v2/M2_EVALUATION_CONTRACT.md'
SPEC = 'research/model_v2/M2_EVALUATOR_SPEC.md'
DOCUMENTS = (CONTRACT, SPEC, 'research/model_v2/M2_DESIGN.md', 'research/model_v2/M2_PROVENANCE_AUDIT.md',
             'research/model_v2/M2_FEASIBILITY_AUDIT.md', 'research/model_v2/M2_PRE_SCORE_RESOLUTIONS.md',
             'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md')
ENTRY_FILES = ('research/model_v2/m2_evaluate.py', 'research/model_v2/m2_conditional.py')

PRACTICAL = -0.002        # R1 and the static stopping indicator, degC/decade
VETO = 0.001              # R2 and R3, degC/decade
MORAN_CHANGE = 0.05
OVERFIT_GAP = 0.05
OVERFIT_DF_RISE = 5
MATERIAL_RESPONSIBILITY = 0.10
REFERENCE_ATOL = 1e-10
PREDICTION_ATOL = 1e-9
COEF_ATOL = 1e-8
SHARE_ATOL = 1e-9
IDENTITY_ATOL = 1e-12

GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')   # schema order (V1 Shapley order)
COMPARATOR, CANDIDATE = 'M0star', 'M2'
MODELS = (COMPARATOR, CANDIDATE)
GAMMA = 'hemisphere=S:abs_latitude'
LATITUDE_COEFFICIENTS = ('abs_latitude_ns1', 'abs_latitude_ns2', GAMMA, 'abs_latitude')
NOT_INTERPRETED = ('abs_latitude_ns1', 'abs_latitude_ns2', 'abs_latitude', 'hemisphere=S')
EXPECTED_FULL = {COMPARATOR: 20, CANDIDATE: 23}
PROTOCOLS = ('primary_loco', 'm49_subregion_lo', 'random10')
PRIMARY = 'primary_loco'
REPRESENTATIONS = {
    'primary_total_co2': {'drop': 'cum_co2_per_capita', 'reference_card': 'drop_per_capita',
                          'decides_verdict': True},
    'per_capita': {'drop': 'cum_co2_total', 'reference_card': 'drop_total', 'decides_verdict': False},
}
SUPPORT_LABELS = ('supported and promoted', 'predictive support, not promoted')

COUNTRY_TABLE = f'{OUTPUTS}/m0_countries.csv'
TABLE_COLUMNS = ['Country', 'iso3', 'm49_subregion', 'station_lon', 'station_lat', 'climate_zone',
                 'hemisphere', 'spatial_block', 'income_group']
ENCODED_DESIGN = f'{OUTPUTS}/m1b_primary/m1b_design_matrices.csv'
M1B_FOLDS = f'{OUTPUTS}/m1b_primary/m1b_cv_folds.csv'
M1B_MANIFEST = f'{OUTPUTS}/m1b_primary/m1b_result_manifest.json'
M1B_PROVENANCE = f'{OUTPUTS}/m1b_primary/m1b_provenance.json'
M1B_PREDICTIONS = f'{OUTPUTS}/m1b_primary/m1b_country_predictions.csv'
M1B_SCORECARD = f'{OUTPUTS}/m1b_primary/m1b_scorecard.json'
FEASIBILITY_SUMMARY = f'{OUTPUTS}/m2_feasibility/m2_feasibility_summary.json'
FEASIBILITY_FITS = f'{OUTPUTS}/m2_feasibility/m2_feasibility_fits.csv'
FEASIBILITY_MANIFEST = f'{OUTPUTS}/m2_feasibility/m2_feasibility_manifest.json'
RANK_AUDIT = f'{OUTPUTS}/rank_audit.json'
RANK_PREDICTIONS = f'{OUTPUTS}/rank_audit_country_predictions.csv'
M1A_SCORECARD = f'{OUTPUTS}/m1a_scorecard.json'
GEOMETRY = f'{OUTPUTS}/country_geometry.csv'
FOLD_MEMBERSHIP = f'{OUTPUTS}/cv_fold_membership.csv'
TERRITORY_MEMBERSHIP = f'{OUTPUTS}/territory_cv_membership_comparison.csv'
CORRECTED_ERRORS = f'{OUTPUTS}/m0_cv_errors_territory_corrected.csv'

# The M1b scoring-input pins (m1b_evaluate.FROZEN_SCORING_INPUTS, unchanged) plus the M1b result artifacts
# and the M2 feasibility record this evaluator binds to. A test asserts the M1b subset is identical.
FROZEN_SCORING_INPUTS = {
    f'{OUTPUTS}/m0_countries.csv': ('9e99b3798e91f2b7e40250921edd5e5a788e4191df3e9027dceb98fe76b43536', 'af97bd2'),
    f'{OUTPUTS}/country_geometry.csv': ('c8d72866d97d13288b3024781cd262f378ce08749d5166e0d4805bea01903e3a', 'af97bd2'),
    f'{OUTPUTS}/cv_fold_membership.csv': ('0100c5c7677ed0b4d3073b12b35300a3a973b3f126dff4cdcfacde4a86ac56f2', 'af97bd2'),
    f'{OUTPUTS}/territory_distance_lower_km.csv':
        ('79a20f007ebe08044833753956cdf859c27d505a2c10747b5e4df3469ee2c6e4', '4263429'),
    f'{OUTPUTS}/territory_cv_membership_comparison.csv':
        ('946be248edd0b7c3f49bd2bb34c94d3c921fd8cbf1b13e6be95e5ebf8b4706ef', '4263429'),
    f'{OUTPUTS}/m0_cv_errors_territory_corrected.csv':
        ('d9d800e322e27a7532d673a7613be5cccf4d694fc5914eb2b37f678ee3062d65', '4263429'),
    f'{OUTPUTS}/m0_scorecard_territory_corrected.json':
        ('396a6a6099db789c99a2e415c5bf5cae46d93e67b2f5a63d57c8f5efc89c7311', '4263429'),
    f'{OUTPUTS}/rank_audit.json': ('bb1a5ccd711a71969432ea6ce2e2a9c3b90ee0c45e3dc845bef25e5895bf362a', '9fb56e1'),
    f'{OUTPUTS}/rank_audit_country_predictions.csv':
        ('77ae9c6a40205354be7e98d9b41012f84e52df4c6e3f7b0e65518a42ff45217c', '9fb56e1'),
    f'{OUTPUTS}/product_stability_aligned_era5_trends.csv':
        ('23c96aeaf400433fea773903ede12fa593f549acf16a7286362b1f6078294922', '9fb56e1'),
    f'{OUTPUTS}/product_stability_aligned_summary.json':
        ('aaac5683728b22f23fb5e2392dd0f3a3f67d081db397b4a972051d75183f40f0', '9fb56e1'),
    f'{OUTPUTS}/m1a_geography_features.csv':
        ('a62d04d478b4500656c83e28d4fb4fb2d5307388b8ea5142a0f8de8dc059c4b3', 'a7dea34'),
    f'{OUTPUTS}/m1a_measurement_manifest.json':
        ('c9d9645203bb32a0698e07cc4cff4824bb5231194bdcc40ae2d91daf3d8b6947', 'a7dea34'),
    f'{OUTPUTS}/m1a_scorecard.json': ('9b6b0a778fd636a5373b33a611895c9310fbfd0c55088e3604e9712cbdd9ab24', '7003ff4'),
}
M2_BOUND_INPUTS = {
    ENCODED_DESIGN: ('bbdc8b1df1314b55c409f6ff59db03d39a9109d0a6827bef9ec58ce8a4caa982', '1680d85'),
    M1B_FOLDS: ('6a9e3976f51e60a66c6b70fd65cc56626df1eec28a41e774c39260d1a2a459f1', '1680d85'),
    M1B_MANIFEST: ('5998dd5cb7c675951c6275127338240e40e5755823312930db17fd67a2c4666a', '1680d85'),
    M1B_PROVENANCE: ('415a7c6437fa54f2013f894721c6c9dce4e255e173a67b4b3cd85e2650ea1b19', '1680d85'),
    M1B_PREDICTIONS: ('43f3f6fb61a067bcc1bca9017fcb60a08abfb8f437d6112ae769bd6a8faf2d56', '1680d85'),
    M1B_SCORECARD: ('c2a353fe37364711a905cb75380950ff7025d8dd3997276f994b2fd041c1f196', '1680d85'),
    FEASIBILITY_SUMMARY: ('9775714e57915336bd615e3907a89d2afec79a6c8a372dadfb62da3f8f5b6212', '23d31f4'),
    FEASIBILITY_FITS: ('810823b4c6e2ad4851499a6bfdd9b975beb0110604737ed09e2a4cbf1ffc252c', '23d31f4'),
    FEASIBILITY_MANIFEST: ('6a0c5e2eee534f1211624f0a3fce70fe7cc229920fac3dcc70f60bdc256f60f3', '23d31f4'),
}
ALL_PINS = {**FROZEN_SCORING_INPUTS, **M2_BOUND_INPUTS}

# Metrics on which M0* must reproduce its frozen reference card (the M1b list, unchanged).
REFERENCE_METRICS = (
    'n', 'in_sample_r2', 'in_sample_rmse', 'cv_r2', 'cv_rmse', 'cv_mae', 'fold_rmse_median',
    'fold_rmse_min', 'fold_rmse_max', 'fold_error_iqr', 'calibration_slope', 'calibration_intercept',
    'residual_morans_i_in_sample', 'residual_morans_i_cv', 'share_geography', 'share_emissions',
    'share_socioeconomic', 'share_population', 'residual_share', 'rows_with_unseen_level',
    'worst_region.rmse', 'secondary.cv_r2', 'secondary.cv_rmse', 'random_reference_only.cv_rmse',
)
EQUIVALENT_METRICS = (
    'in_sample_r2', 'in_sample_rmse', 'cv_r2', 'cv_rmse', 'cv_mae', 'residual_morans_i_in_sample',
    'residual_morans_i_cv', 'worst_region.rmse', 'secondary.cv_rmse', 'secondary.cv_r2',
    'random_reference_only.cv_rmse', 'recorded_area_centroid_moran_in_sample',
    'recorded_area_centroid_moran_cv',
)
DETERMINISTIC_ARTIFACTS = (
    'm2_scorecard.json', 'm2_country_predictions.csv', 'm2_primary_country_comparison.csv', 'm2_cv_folds.csv',
    'm2_fit_states.csv', 'm2_coefficients.csv', 'm2_coefficient_summary.json', 'm2_shapley_coalitions.csv',
    'm2_design_matrices.csv', 'm2_provenance.json',
)
RESULT_MANIFEST = 'm2_result_manifest.json'
RUN_METADATA = 'm2_run_metadata.json'
INTEGRITY_RECORD = 'm2_integrity_failure.json'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _refuse(condition, message):
    if not condition:
        raise prov.ExecutionRefused(message)


def code_files() -> list[str]:
    return sorted({*prov.import_closure(ENTRY_FILES), *DOCUMENTS, *prov.LOCK_FILES})


# ---------------------------------------------------------------------
# Scoring-input gate and the frozen frame
# ---------------------------------------------------------------------


def verify_pins() -> dict:
    digests = {}
    for rel, (pin, commit) in ALL_PINS.items():
        digest = prov.sha256(ROOT / rel)
        _refuse(digest == pin, f'{rel}: sha256 {digest} differs from the record frozen at {commit} ({pin})')
        digests[rel] = {'sha256': digest, 'frozen_at': commit}
    m1b_manifest = json.loads((ROOT / M1B_MANIFEST).read_text())
    for rel in (ENCODED_DESIGN, M1B_FOLDS, M1B_PROVENANCE, M1B_PREDICTIONS, M1B_SCORECARD):
        _refuse(m1b_manifest['artifact_sha256'][Path(rel).name] == ALL_PINS[rel][0],
                f'{rel} is not the artifact recorded by the committed M1b result manifest')
    feasibility_manifest = json.loads((ROOT / FEASIBILITY_MANIFEST).read_text())
    for rel in (FEASIBILITY_SUMMARY, FEASIBILITY_FITS):
        _refuse(feasibility_manifest[Path(rel).name] == ALL_PINS[rel][0],
                f'{rel} is not the artifact recorded by the committed feasibility manifest')
    return digests


@dataclasses.dataclass(frozen=True)
class Frozen:
    """Everything a scored arm needs, with its identity record."""

    table: pd.DataFrame
    frames: dict            # representation -> schema-named frame with the outcome column
    y: np.ndarray
    distance: np.ndarray
    station_w: np.ndarray
    area_w: np.ndarray
    folds: dict             # protocol -> (fold ids, buffer km or None)
    encoded: pd.DataFrame   # the frozen M1b encoded design
    identity: dict

    @property
    def n(self):
        return len(self.table)


def fold_plan(table, n):
    return {PRIMARY: (np.arange(n), PRIMARY_BUFFER_KM),
            'm49_subregion_lo': (cv.folds_from_labels(table.m49_subregion), None),
            'random10': (cv.folds_random(n, 10), None)}


def numeric_frame(encoded, representation, iso3):
    rows = encoded[(encoded.representation == representation) & (encoded.model == COMPARATOR)]
    wide = rows.pivot(index='iso3', columns='column', values='value')
    columns = feas.NUMERIC_COLUMNS[representation]
    _refuse(set(iso3) <= set(wide.index), f'{representation}: the frozen design does not cover the 151 countries')
    frame = wide.loc[list(iso3), columns].reset_index()
    _refuse(np.isfinite(frame[columns].to_numpy(float)).all(), f'{representation}: non-finite frozen numerics')
    return frame


def column_groups(frame, names):
    owner = feas._column_groups(frame, [n for n in names if n not in lb.ADDED_COLUMNS])
    groups = dict(zip([n for n in names if n not in lb.ADDED_COLUMNS], owner))
    return [lb.ADDED_GROUP if n in lb.ADDED_COLUMNS else groups[n] for n in names]


def encoded_identity(frame, representation, encoded):
    """Full-sample M0* encoding against the frozen design, aligned by column name (spec §2 item 3)."""
    matrix, names = feas.encode(frame, feas.baseline_encoder(frame))
    frozen = encoded[(encoded.representation == representation) & (encoded.model == COMPARATOR)]
    pivot = frozen.pivot(index='iso3', columns='column', values='value')
    group_of = frozen.drop_duplicates('column').set_index('column').group
    _refuse(set(pivot.columns) == set(names), f'{representation}: encoded column sets differ')
    values = pivot.loc[frame.iso3, names].to_numpy()
    dummies = [j for j, n in enumerate(names) if '=' in n]
    _refuse(np.array_equal(values[:, dummies], matrix[:, dummies]),
            f'{representation}: dummy columns rebuilt from the labels differ from the frozen design')
    _refuse(np.array_equal(values, matrix), f'{representation}: encoded design differs from the frozen design')
    _refuse(list(group_of.loc[names]) == column_groups(frame, names), f'{representation}: column groups differ')
    return {'columns': names, 'value_for_value_after_name_alignment': True,
            'dummy_columns_bit_identical_from_labels': True, 'groups_identical': True,
            'design_sha256': prov.array_digest(matrix)}


def outcome_vector(table):
    frozen = pd.read_csv(ROOT / RANK_PREDICTIONS, float_precision='round_trip')
    y = (frozen[frozen.variant == 'drop_per_capita'].set_index('iso3').loc[table.iso3, 'observed']
         .to_numpy(float))
    _refuse(np.isfinite(y).all(), 'non-finite frozen outcome')
    design = m0_complete_design(*load_inputs())
    _refuse(design.Country.tolist() == table.Country.tolist(), 'the pipeline country order differs')
    _refuse(np.array_equal(design[OUTCOME_COL].to_numpy(float), y),
            'the frozen ordered outcome differs from the pipeline outcome')
    m1b = pd.read_csv(ROOT / M1B_PREDICTIONS, float_precision='round_trip')
    m1b = m1b[(m1b.representation == 'primary_total_co2') & (m1b.model == COMPARATOR) & (m1b.protocol == 'in_sample')]
    _refuse(np.array_equal(m1b.set_index('iso3').loc[table.iso3, 'observed'].to_numpy(float), y),
            'the frozen ordered outcome differs from the committed M1b prediction record')
    return y


def fold_identities(table, distance, folds):
    derived = np.array([train for _f, _test, train in cv.iter_folds(folds[PRIMARY][0], distance, PRIMARY_BUFFER_KM)])
    membership = (pd.read_csv(ROOT / TERRITORY_MEMBERSHIP)
                  .pivot(index='held_out', columns='candidate', values='corrected_train')
                  .loc[table.iso3, table.iso3].to_numpy(bool))
    _refuse(np.array_equal(derived, membership), 'primary memberships differ from the corrected frozen memberships')
    n_train = pd.read_csv(ROOT / CORRECTED_ERRORS, usecols=['iso3', 'n_train'])
    _refuse(n_train.iso3.tolist() == table.iso3.tolist()
            and np.array_equal(derived.sum(axis=1), n_train.n_train.to_numpy()),
            'primary training counts differ from the frozen record')
    m1b_folds = pd.read_csv(ROOT / M1B_FOLDS)
    _refuse(m1b_folds.iso3.tolist() == table.iso3.tolist(), 'M1b fold artifact order')
    iso3 = table.iso3.to_numpy()
    for i, row in enumerate(m1b_folds.itertuples()):
        _refuse(sorted(row.primary_training_iso3.split('|')) == sorted(iso3[derived[i]].tolist()),
                f'primary membership of {row.iso3} differs from the committed M1b fold artifact')
    frozen_membership = pd.read_csv(ROOT / FOLD_MEMBERSHIP,
                                    usecols=['iso3', 'm49_subregion', 'm49_subregion_lo', 'random_10fold'])
    _refuse(frozen_membership.iso3.tolist() == table.iso3.tolist()
            and (frozen_membership.m49_subregion.to_numpy() == table.m49_subregion.to_numpy()).all(),
            'frozen fold-membership labels')
    for protocol, frozen_column, m1b_column in (('m49_subregion_lo', 'm49_subregion_lo', 'm49_fold'),
                                                ('random10', 'random_10fold', 'random10_fold')):
        ids = folds[protocol][0]
        _refuse(np.array_equal(ids, frozen_membership[frozen_column].to_numpy())
                and np.array_equal(ids, m1b_folds[m1b_column].to_numpy()),
                f'{protocol} fold ids differ from the frozen definitions')
    return {'primary_membership_sha256': prov.hashlib.sha256(derived.tobytes()).hexdigest(),
            'm49_fold_sha256': prov.hashlib.sha256(folds['m49_subregion_lo'][0].tobytes()).hexdigest(),
            'random10_fold_sha256': prov.hashlib.sha256(folds['random10'][0].tobytes()).hexdigest()}


def build_frozen() -> Frozen:
    """Load the frozen 151-country scoring frames and check every identity (spec §2)."""
    table = pd.read_csv(ROOT / COUNTRY_TABLE, usecols=TABLE_COLUMNS, float_precision='round_trip')[TABLE_COLUMNS]
    _refuse(len(table) == 151 and table.iso3.is_unique, 'the country table is not the frozen 151 countries')
    labels = table[['iso3']].merge(m49_table(), on='iso3', how='left', validate='one_to_one')
    _refuse(labels.m49_subregion.notna().all()
            and (labels.m49_subregion.to_numpy() == table.m49_subregion.to_numpy()).all(),
            'M49 labels differ from the frozen definition')
    encoded = pd.read_csv(ROOT / ENCODED_DESIGN, float_precision='round_trip')
    y = outcome_vector(table)
    frames, identity = {}, {'n': len(table), 'outcome_vector_sha256': prov.array_digest(y),
                            'outcome_binding': 'rank_audit_country_predictions.csv drop_per_capita observed; '
                                               'equal to the M0 pipeline and the committed M1b record',
                            'encoded_identity': {}}
    for representation in REPRESENTATIONS:
        frame = table.merge(numeric_frame(encoded, representation, table.iso3), on='iso3', how='left',
                            validate='one_to_one')
        _refuse(frame.iso3.tolist() == table.iso3.tolist(), 'frame order')
        frame[OUTCOME_COL] = y
        identity['encoded_identity'][representation] = encoded_identity(frame, representation, encoded)
        frames[representation] = frame
    total, per_capita = frames['primary_total_co2'], frames['per_capita']
    error = float(np.max(np.abs(total.cum_co2_total.to_numpy() - per_capita.cum_co2_per_capita.to_numpy()
                                - total.population.to_numpy() + 6.0)))
    _refuse(error <= IDENTITY_ATOL, f'the representation log identity fails ({error})')
    identity['representation_identity_max_abs_error'] = error
    distance = territory.load_bounds(table.iso3.tolist())
    folds = fold_plan(table, len(table))
    identity.update(fold_identities(table, distance, folds))
    station_w = cv.v1_weights(table.station_lon.to_numpy(), table.station_lat.to_numpy())
    geom = pd.read_csv(ROOT / GEOMETRY, usecols=['iso3', 'centroid_lon', 'centroid_lat']).set_index('iso3').loc[table.iso3]
    area_w = knn_weights(haversine_matrix(geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy()), 8)
    m1b_identity = json.loads((ROOT / M1B_PROVENANCE).read_text())['identity']
    _refuse(prov.array_digest(station_w) == m1b_identity['station_weights_sha256'], 'station weights digest')
    _refuse(prov.array_digest(area_w) == m1b_identity['area_weights_sha256'], 'area-centroid weights digest')
    identity.update({'station_weights_sha256': prov.array_digest(station_w),
                     'area_weights_sha256': prov.array_digest(area_w)})
    return Frozen(table=table, frames=frames, y=y, distance=distance, station_w=station_w, area_w=area_w,
                  folds=folds, encoded=encoded, identity=identity)


# ---------------------------------------------------------------------
# The estimator: M0* and M2 on exact transformed numerics
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Fitted:
    model: str
    encoder: feas.EncoderState
    state: lb.LatitudeState | None
    columns: list
    coef: np.ndarray
    matrix_columns: int
    matrix_rank: int
    baseline_columns: int
    baseline_rank: int


def design(model, rows, encoder, state):
    matrix, names = feas.encode(rows, encoder)
    if model == CANDIDATE:
        added = lb.added_block(state, rows.abs_latitude.to_numpy(float), rows.hemisphere.to_numpy())
        matrix, names = np.hstack([matrix, added]), [*names, *lb.ADDED_COLUMNS]
    return matrix, names


def fit_model(model, train, outcome=OUTCOME_COL) -> Fitted:
    encoder = feas.baseline_encoder(train)
    baseline, _ = feas.encode(train, encoder)
    state = lb.fit_state(train.abs_latitude.to_numpy(float)) if model == CANDIDATE else None
    matrix, names = design(model, train, encoder, state)
    coef, *_ = np.linalg.lstsq(matrix, train[outcome].to_numpy(float), rcond=None)
    return Fitted(model, encoder, state, names, coef, int(matrix.shape[1]), int(np.linalg.matrix_rank(matrix)),
                  int(baseline.shape[1]), int(np.linalg.matrix_rank(baseline)))


def predict(fitted: Fitted, rows) -> np.ndarray:
    """``cv.V1Design.predict`` semantics: X beta plus the mean-training-level-effect rule."""
    matrix, _ = design(fitted.model, rows, fitted.encoder, fitted.state)
    yhat = matrix @ fitted.coef
    for c in fitted.encoder.categorical:
        levels = fitted.encoder.levels[c]
        effects = np.array([0.0] + [fitted.coef[fitted.columns.index(f'{c}={lv}')] for lv in levels[1:]])
        unseen = ~rows[c].astype(str).isin(levels).to_numpy()
        yhat = yhat + unseen * effects.mean()
    return yhat


def structural_failures(fitted: Fitted) -> list[str]:
    failures = []
    if fitted.baseline_rank != fitted.baseline_columns:
        failures.append('M0* design is rank deficient')
    if fitted.model == CANDIDATE:
        if fitted.matrix_rank != fitted.matrix_columns:
            failures.append('M2 design is rank deficient')
        if fitted.matrix_rank - fitted.baseline_rank != 3:
            failures.append(f'M2 adds {fitted.matrix_rank - fitted.baseline_rank} identifiable dimensions, not 3')
    return failures


def fit_label(protocol, fold, table):
    return str(table.iso3.iloc[int(fold)]) if protocol == PRIMARY else str(int(fold))


def run_protocol(model, frame, frozen, protocol, outcome=OUTCOME_COL):
    """Out-of-fold predictions through the frozen ``cv.cross_validate`` with a record of every fit."""
    ids, buffer = frozen.folds[protocol]
    records = []

    def fit_predict(train, test):
        fitted = fit_model(model, train, outcome)
        fold = int(ids[test.index[0]])
        extrapolation = (lb.extrapolation(fitted.state, test.abs_latitude.to_numpy(float))
                         if fitted.state is not None else None)
        records.append({'protocol': protocol, 'fold': fold, 'fit': fit_label(protocol, fold, frozen.table),
                        'train_index': train.index.to_numpy(), 'test_index': test.index.to_numpy(),
                        'fitted': fitted, 'failures': structural_failures(fitted),
                        'extrapolation': extrapolation})
        return predict(fitted, test)

    frame = frame.copy()
    if outcome != OUTCOME_COL:
        frame[OUTCOME_COL] = frame[outcome]
    result = cv.cross_validate(frame, ids, fit_predict, dist=frozen.distance, buffer_km=buffer)
    return result, records


def predictor_only_states(frozen, frame, representation):
    """Every fit's latitude state, ranks and extrapolation, without any outcome (spec §3)."""
    rows = []
    fits = [('full_sample', 'all', np.arange(frozen.n), np.array([], int))]
    for protocol in PROTOCOLS:
        ids, buffer = frozen.folds[protocol]
        for fold, test, train in cv.iter_folds(ids, frozen.distance, buffer):
            fits.append((protocol, fit_label(protocol, fold, frozen.table), np.flatnonzero(train),
                         np.flatnonzero(test)))
    for protocol, label, train_idx, test_idx in fits:
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        encoder = feas.baseline_encoder(train)
        baseline, _ = feas.encode(train, encoder)
        state = lb.fit_state(train.abs_latitude.to_numpy(float))
        candidate, _ = design(CANDIDATE, train, encoder, state)
        held = (lb.extrapolation(state, test.abs_latitude.to_numpy(float)) if len(test)
                else {'below': 0, 'above': 0})
        rows.append({'representation': representation, 'protocol': protocol, 'fit': label,
                     'latitude_state': state.to_json(), 'baseline_columns': int(baseline.shape[1]),
                     'baseline_rank': int(np.linalg.matrix_rank(baseline)),
                     'candidate_columns': int(candidate.shape[1]),
                     'candidate_rank': int(np.linalg.matrix_rank(candidate)),
                     'test_below_lower_boundary': int(held['below']),
                     'test_above_upper_boundary': int(held['above'])})
    return rows


def check_against_feasibility(frozen) -> dict:
    """Pre-candidate refusal unless every fit reproduces the committed predictor-only record exactly."""
    record = pd.read_csv(ROOT / FEASIBILITY_FITS, dtype={'fit': str, 'latitude_state': str})
    record = record[record.measurement == 'station'].set_index(['representation', 'protocol', 'fit'])
    summary = json.loads((ROOT / FEASIBILITY_SUMMARY).read_text())
    compared = 0
    for representation, frame in frozen.frames.items():
        full = lb.fit_state(frame.abs_latitude.to_numpy(float))
        committed = lb.LatitudeState.from_json(summary['full_sample'][f'station/{representation}']['latitude_state'])
        _refuse(full == committed, f'{representation}: full-sample latitude state differs from the committed record')
        for row in predictor_only_states(frozen, frame, representation):
            key = (representation, row['protocol'], row['fit'])
            _refuse(key in record.index, f'{key} is absent from the feasibility record')
            saved = record.loc[key]
            _refuse(lb.LatitudeState.from_json(saved.latitude_state) == lb.LatitudeState.from_json(row['latitude_state']),
                    f'{key}: latitude state differs from the feasibility record')
            for field in ('baseline_columns', 'baseline_rank', 'candidate_columns', 'candidate_rank',
                          'test_below_lower_boundary', 'test_above_upper_boundary'):
                _refuse(int(saved[field]) == row[field], f'{key}: {field} differs from the feasibility record')
            _refuse(row['baseline_rank'] == row['baseline_columns'] and row['candidate_rank'] == row['candidate_columns']
                    and row['candidate_rank'] - row['baseline_rank'] == 3, f'{key}: structural requirement fails')
            compared += 1
    _refuse(compared == 2 * (1 + 151 + 20 + 10), f'compared {compared} fits, expected 364')
    return {'fits_compared': compared, 'states_ranks_and_extrapolation_identical': True,
            'full_sample_hex_states_identical': True}


# ---------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------


def lmg_shares(model, frame, outcome=OUTCOME_COL):
    """Group LMG/Shapley shares on the encoded blocks; every coalition uses the full-sample state."""
    state = lb.fit_state(frame.abs_latitude.to_numpy(float)) if model == CANDIDATE else None
    matrix, names = design(model, frame, feas.baseline_encoder(frame), state)
    groups = np.array(column_groups(frame, names))
    y = frame[outcome].to_numpy(float)
    tss = float(np.sum((y - y.mean()) ** 2))
    present = [g for g in GROUPS if (groups == g).any()]
    r2, rows = {}, []
    for size in range(len(present) + 1):
        for coalition in itertools.combinations(present, size):
            keep = (groups == 'intercept') | np.isin(groups, coalition)
            part = matrix[:, keep]
            coef, *_ = np.linalg.lstsq(part, y, rcond=None)
            value = float(1.0 - np.sum((y - part @ coef) ** 2) / tss)
            r2[frozenset(coalition)] = value
            rows.append({'coalition': '+'.join(coalition) or 'intercept_only', 'n_groups': size, 'r2': value})
    k = len(present)
    shares = {g: 0.0 for g in present}
    for g in present:
        others = [h for h in present if h != g]
        for size in range(k):
            weight = factorial(size) * factorial(k - size - 1) / factorial(k)
            for subset in itertools.combinations(others, size):
                s = frozenset(subset)
                shares[g] += weight * (r2[s | {g}] - r2[s])
    total = r2[frozenset(present)]
    return {'shares': shares, 'total_r2': total, 'residual_share': 1.0 - total, 'coalitions': rows,
            'state': state}


def protocol_card(frame, table, result, in_sample, lmg, rank, station_w, outcome=OUTCOME_COL):
    scored = frame if outcome == OUTCOME_COL else frame.assign(**{OUTCOME_COL: frame[outcome]})
    card = cv.scorecard(scored, in_sample, result, station_w, shares=False)
    card.update({f'share_{g}': float(v) for g, v in lmg['shares'].items()})
    card['residual_share'] = float(lmg['residual_share'])
    y = scored[OUTCOME_COL].to_numpy(float)
    errors = np.abs(y - result.yhat)
    card['fold_error_iqr'] = float(np.subtract(*np.quantile(errors, [0.75, 0.25])))
    card['effective_degrees_of_freedom'] = int(rank)
    card['residual_moran_in_sample'] = asdict(morans_i(y - in_sample, station_w))
    card['residual_moran_cv'] = asdict(morans_i(y - result.yhat, station_w))
    regions = cv.region_errors(result, y, table.m49_subregion)
    eligible = regions[regions.n >= MIN_REGION_N]
    card['worst_region'] = {'region': str(eligible.index[0]), 'rmse': float(eligible.iloc[0].rmse),
                            'n': int(eligible.iloc[0].n), 'min_region_n': MIN_REGION_N}
    card['region_rmse'] = {str(k): float(v) for k, v in regions.rmse.items()}
    card['region_bias'] = {str(k): float(v) for k, v in regions.bias.items()}
    card['region_mae'] = {str(k): float(v) for k, v in regions.mae.items()}
    worst = result.fold_id == card['worst_fold']
    card['worst_fold_units'] = sorted(table.iso3[worst].tolist())
    card['worst_fold_country'] = str(table.iloc[card['worst_fold']].Country) if card['n_folds'] == len(y) else None
    card['worst_countries_by_abs_error'] = [
        {'iso3': str(table.iso3.iloc[i]), 'Country': str(table.Country.iloc[i]), 'abs_error': float(errors[i])}
        for i in np.argsort(-errors, kind='stable')[:10]]
    return card


def coefficient_summary(full: Fitted, records):
    by_column = {c: [] for c in full.columns}
    for record in records:
        fitted = record['fitted']
        for column, value in zip(fitted.columns, fitted.coef):
            by_column.setdefault(column, []).append(float(value))
    summary = {}
    for column, value in zip(full.columns, full.coef):
        values = np.array(by_column.get(column, []), dtype=float)
        summary[column] = {
            'full': float(value), 'fold_values_n': int(len(values)),
            'fold_min': float(values.min()) if len(values) else None,
            'fold_median': float(np.median(values)) if len(values) else None,
            'fold_max': float(values.max()) if len(values) else None,
            'negative_n': int((values < 0).sum()), 'zero_n': int((values == 0).sum()),
            'interpretation': ('basis/parameterization coefficient: not interpreted and not stability evidence'
                               if column in NOT_INTERPRETED else
                               'gamma: southern-minus-northern latitude-slope difference' if column == GAMMA
                               else 'coefficient'),
        }
    return summary


def gamma_summary(full: Fitted, records, n_expected):
    values = np.array([r['fitted'].coef[r['fitted'].columns.index(GAMMA)] for r in records], dtype=float)
    full_gamma = float(full.coef[full.columns.index(GAMMA)])
    _require(len(values) == n_expected and np.isfinite(values).all() and np.isfinite(full_gamma),
             f'gamma must be finite in the full fit and in all {n_expected} primary training fits')
    return {'full': full_gamma, 'n_primary_training_fits': int(n_expected),
            'strictly_negative_n': int((values < 0).sum()), 'strictly_negative_fraction': float((values < 0).mean()),
            'zero_n': int((values == 0).sum()),
            'same_sign_as_full_fraction': float(np.mean(np.sign(values) == np.sign(full_gamma))),
            'median': float(np.median(values)), 'min': float(values.min()), 'max': float(values.max()),
            'cutoff': None,
            'interpretation': 'negative gamma means a lower southern latitude slope relative to the north; it does '
                              'not establish positive northern or non-positive southern absolute slopes and '
                              'identifies no mechanism; sign stability is descriptive and decides nothing'}


@dataclasses.dataclass(frozen=True)
class Scored:
    card: dict
    results: dict
    records: dict
    fitted: np.ndarray
    full: Fitted
    lmg: dict


def score_model(model, frame, frozen, outcome=OUTCOME_COL, expected_full=True) -> Scored:
    full = fit_model(model, frame, outcome)
    in_sample = predict(full, frame)
    results, records = {}, {}
    for protocol in PROTOCOLS:
        results[protocol], records[protocol] = run_protocol(model, frame, frozen, protocol, outcome)
    lmg = lmg_shares(model, frame, outcome)
    card = protocol_card(frame, frozen.table, results[PRIMARY], in_sample, lmg, full.matrix_rank, frozen.station_w, outcome)
    card['secondary'] = protocol_card(frame, frozen.table, results['m49_subregion_lo'], in_sample, lmg,
                                      full.matrix_rank, frozen.station_w, outcome)
    card['random_reference_only'] = protocol_card(frame, frozen.table, results['random10'], in_sample, lmg,
                                                  full.matrix_rank, frozen.station_w, outcome)
    y = frame[outcome].to_numpy(float)
    card['recorded_area_centroid_moran_in_sample'] = morans_i(y - in_sample, frozen.area_w).statistic
    card['recorded_area_centroid_moran_cv'] = morans_i(y - results[PRIMARY].yhat, frozen.area_w).statistic
    card['matrix_columns'], card['matrix_rank'] = full.matrix_columns, full.matrix_rank
    card['unseen_rows_by_categorical'] = {
        name: cv.design_descriptors(frame.assign(**{OUTCOME_COL: y}), ids, frozen.distance, buffer)[
            'test_rows_with_unseen_level_by_categorical']
        for name, (ids, buffer) in frozen.folds.items()}
    card['levels_present'] = {c: sorted(frame[c].astype(str).unique().tolist())
                              for c in sorted(CATEGORICAL_FEATURES) if c in frame}
    card['coefficients'] = coefficient_summary(full, records[PRIMARY])
    failures = structural_failures(full)
    if expected_full and not (full.matrix_columns == EXPECTED_FULL[model] == full.matrix_rank):
        failures.append(f'full-sample design is {full.matrix_columns} columns / rank {full.matrix_rank}, '
                        f'expected {EXPECTED_FULL[model]}')
    card['structural_integrity'] = {
        'full_sample_failures': failures,
        'fit_failures': [{'protocol': r['protocol'], 'fit': r['fit'], 'failures': r['failures']}
                         for p in PROTOCOLS for r in records[p] if r['failures']],
        'fits_checked': {p: len(records[p]) for p in PROTOCOLS},
    }
    card['structural_integrity']['passes'] = bool(not failures and not card['structural_integrity']['fit_failures']
                                                  and [len(records[p]) for p in PROTOCOLS]
                                                  == [len(np.unique(frozen.folds[p][0])) for p in PROTOCOLS])
    card['training_fits'] = {p: {'n': len(records[p]),
                                 'min_n_train': int(min(len(r['train_index']) for r in records[p])),
                                 'max_n_train': int(max(len(r['train_index']) for r in records[p])),
                                 'column_counts': sorted({r['fitted'].matrix_columns for r in records[p]})}
                             for p in PROTOCOLS}
    named = {g: card.get(f'share_{g}', 0.0) for g in GROUPS}
    card['share_accounting'] = {
        'named_share_sum': float(sum(named.values())),
        'named_share_sum_minus_in_sample_r2': float(sum(named.values()) - card['in_sample_r2']),
        'named_plus_residual_minus_one': float(sum(named.values()) + card['residual_share'] - 1.0),
        'tolerance': SHARE_ATOL,
        'passes': bool(abs(sum(named.values()) - card['in_sample_r2']) <= SHARE_ATOL
                       and abs(sum(named.values()) + card['residual_share'] - 1.0) <= SHARE_ATOL),
        'interpretation': 'fractions of explained outcome variance, not causal contributions',
    }
    if model == CANDIDATE:
        card['latitude_state_full_sample'] = json.loads(full.state.to_json())
        card['gamma'] = gamma_summary(full, records[PRIMARY], frozen.n)
    return Scored(card, results, records, in_sample, full, lmg)


def _metric(card, dotted):
    value = card
    for part in dotted.split('.'):
        value = value[part]
    return value


def reference_check(card, reference, label, metrics=REFERENCE_METRICS, atol=REFERENCE_ATOL):
    rows = {}
    for metric in metrics:
        try:
            got, want = float(_metric(card, metric)), float(_metric(reference, metric))
        except KeyError:
            raise prov.ExecutionRefused(f'{label}: reference card has no metric {metric!r}') from None
        _refuse(np.isfinite(got) and np.isfinite(want), f'{label}: {metric} is not finite')
        _refuse(abs(got - want) <= atol, f'{label} does not reproduce the frozen card: {metric} {got!r} vs {want!r}')
        rows[metric] = {'value': got, 'reference': want, 'abs_difference': abs(got - want)}
    return {'label': label, 'tolerance': atol, 'metrics': rows, 'reproduced': True}


# ---------------------------------------------------------------------
# Verdict, diagnostics, stopping indicator and equivalence
# ---------------------------------------------------------------------


def verdict(comparator, candidate, paired, integrity_ok):
    """Contract §5 (Amendment 1 §7), mechanically; non-finite inputs are refusals, never ``False``."""
    delta = float(paired['delta_rmse'])
    lo, hi = (float(v) for v in paired['country_bootstrap_95_interval'])
    m49 = float(candidate['secondary']['cv_rmse']) - float(comparator['secondary']['cv_rmse'])
    region = float(candidate['worst_region']['rmse']) - float(comparator['worst_region']['rmse'])
    for name, value in [('delta_rmse', delta), ('interval_low', lo), ('interval_high', hi),
                        ('m49_rmse_change', m49), ('worst_region_rmse_change', region)]:
        _require(np.isfinite(value), f'verdict input {name} is not finite')
    p1, p2 = bool(delta < 0), bool(hi < 0)
    r1, r2, r3, r4 = bool(delta <= PRACTICAL), bool(m49 <= VETO), bool(region <= VETO), bool(integrity_ok)
    if not r4:
        label = None
    elif p1 and p2 and r1 and r2 and r3:
        label = 'supported and promoted'
    elif p1 and p2:
        label = 'predictive support, not promoted'
    else:
        label = 'not supported'
    return {
        'verdict': label, 'predictive_support': bool(p1 and p2 and r4), 'promoted': label == 'supported and promoted',
        'P1_delta_below_zero': p1, 'P2_interval_upper_below_zero': p2,
        'R1_delta_le_minus_0.002': r1, 'R2_m49_veto_passed': r2, 'R3_worst_region_veto_passed': r3,
        'R4_structural_integrity': r4,
        'delta_rmse': delta, 'interval': [lo, hi], 'resamples': paired['resamples'], 'seed': paired['seed'],
        'interval_method': 'paired country resampling interval of the difference in fixed out-of-fold errors; '
                           'not a spatially corrected confidence interval',
        'practical_threshold': PRACTICAL, 'veto_threshold': VETO,
        'm49_rmse_change': m49, 'worst_region_rmse_change': region,
        'worst_region_comparator': comparator['worst_region'], 'worst_region_candidate': candidate['worst_region'],
        'worsened_generalization': bool(delta > 0 and lo > 0),
    }


def committed_stage_deltas():
    m1a = json.loads((ROOT / M1A_SCORECARD).read_text())['primary']['M1a']['paired_delta_vs_m0']['delta_rmse']
    m1b = json.loads((ROOT / M1B_SCORECARD).read_text())['verdict']['delta_rmse']
    return {'M1a': float(m1a), 'M1b': float(m1b)}


def stopping_indicator(delta_m2):
    deltas = {**committed_stage_deltas(), 'M2': float(delta_m2)}
    _require(all(np.isfinite(v) for v in deltas.values()), 'a stopping-rule input is not finite')
    best = min(deltas.values())
    return {'point_delta_rmse_vs_m0star': deltas, 'best': best, 'threshold': PRACTICAL,
            'fires': bool(best > PRACTICAL), 'm2_decides': bool(deltas['M1a'] > 0 and deltas['M1b'] > 0),
            'rule': 'fires iff the smallest primary total-CO2 Berkeley point delta over M1a, M1b and M2 is '
                    '> -0.002 degC/decade; equality does not fire; independent of promotion'}


def diagnostics(comparator, candidate, paired):
    delta = float(paired['delta_rmse'])
    lo, hi = (float(v) for v in paired['country_bootstrap_95_interval'])
    m49_ok = candidate['secondary']['cv_rmse'] - comparator['secondary']['cv_rmse'] <= VETO
    region_ok = candidate['worst_region']['rmse'] - comparator['worst_region']['rmse'] <= VETO
    transfer = ('improved generalization' if delta < 0 and hi < 0 and m49_ok and region_ok else
                'worsened generalization' if delta > 0 and lo > 0 else 'no detectable change in transfer')
    gain_in = candidate['in_sample_r2'] - comparator['in_sample_r2']
    gain_cv = candidate['cv_r2'] - comparator['cv_r2']
    d_in = candidate['residual_morans_i_in_sample'] - comparator['residual_morans_i_in_sample']
    d_cv = candidate['residual_morans_i_cv'] - comparator['residual_morans_i_cv']
    spatial = ('reduced' if d_in <= -MORAN_CHANGE and d_cv <= -MORAN_CHANGE else
               'increased' if d_in >= MORAN_CHANGE and d_cv >= MORAN_CHANGE else 'essentially unchanged')
    shares = {g: candidate.get(f'share_{g}', 0.0) for g in GROUPS}
    others = max(v for g, v in shares.items() if g != 'geography')
    material = bool(not shares['geography'] > others or shares['emissions'] > MATERIAL_RESPONSIBILITY)
    df_rise = candidate['effective_degrees_of_freedom'] - comparator['effective_degrees_of_freedom']
    clause_i = bool(gain_in - gain_cv > OVERFIT_GAP)
    clause_ii = bool(candidate['cv_rmse'] > comparator['cv_rmse'] and candidate['in_sample_rmse'] < comparator['in_sample_rmse'])
    clause_iv = bool(df_rise > OVERFIT_DF_RISE and delta > PRACTICAL)
    return {
        'three_dimension_summary': {
            'transfer': transfer, 'spatial_specification': f'residual spatial autocorrelation {spatial}',
            'stability': 'material change' if material else 'stable',
            'note': 'descriptive (M1A_EVALUATION_CONTRACT.md §4 dimensions); never a verdict condition'},
        'overfitting': {
            'clause_i_in_sample_gain_minus_cv_gain_gt_0.05': clause_i,
            'clause_ii_cv_worse_while_in_sample_better': clause_ii,
            'clause_iii_gamma_same_sign_fraction': candidate['gamma']['same_sign_as_full_fraction'],
            'clause_iii_cutoff': None,
            'clause_iv_df_rise_gt_5_with_gain_lt_0.002': clause_iv,
            'effective_df_rise': int(df_rise),
            'signal_from_clauses_i_ii_iv': bool(clause_i or clause_ii or clause_iv),
            'note': 'diagnostic only; clause (iii) is reported without a cutoff (M2_PRE_SCORE_RESOLUTIONS.md §5)'},
        'in_sample_r2_gain': gain_in, 'cv_r2_gain': gain_cv,
        'residual_spatial_structure': f'residual spatial autocorrelation {spatial}',
        'delta_moran_in_sample': d_in, 'delta_moran_cv': d_cv,
        'delta_area_centroid_moran_in_sample': candidate['recorded_area_centroid_moran_in_sample']
        - comparator['recorded_area_centroid_moran_in_sample'],
        'delta_area_centroid_moran_cv': candidate['recorded_area_centroid_moran_cv']
        - comparator['recorded_area_centroid_moran_cv'],
        'material_change_to_conclusion': material, 'scientific_stability': 'material change' if material else 'stable',
        'shares': shares, 'share_changes': {g: shares[g] - comparator.get(f'share_{g}', 0.0) for g in GROUPS},
        'delta_cv_rmse_m49': candidate['secondary']['cv_rmse'] - comparator['secondary']['cv_rmse'],
        'delta_cv_r2_m49': candidate['secondary']['cv_r2'] - comparator['secondary']['cv_r2'],
        'delta_cv_rmse_random_reference': candidate['random_reference_only']['cv_rmse']
        - comparator['random_reference_only']['cv_rmse'],
        'interpretation': 'Shares allocate shared explanatory variance among correlated groups; a change in the '
                          'geography share is an accounting consequence, not a physical source.',
    }


def extrapolation_disclosure(arm, frozen):
    """Held-out rows outside their fit's knot range, with both models' out-of-fold errors (descriptive)."""
    y = frozen.y if 'outcome' not in arm else arm['outcome']
    latitude = arm['frames'][CANDIDATE].abs_latitude
    rows = []
    for protocol in PROTOCOLS:
        for record in arm['scored'][CANDIDATE].records[protocol]:
            if not (record['extrapolation']['below'] or record['extrapolation']['above']):
                continue
            state = record['fitted'].state
            for i in record['test_index']:
                a = float(latitude.iloc[i])
                side = 'below' if a < state.knots[0] else 'above' if a > state.knots[-1] else None
                if side is None:
                    continue
                rows.append({'protocol': protocol, 'fit': record['fit'], 'iso3': str(frozen.table.iso3.iloc[i]),
                             'side': side, 'abs_latitude': a, 'knot_lower': state.knots[0], 'knot_upper': state.knots[-1],
                             'M0star_oof_error': float(y[i] - arm['scored'][COMPARATOR].results[protocol].yhat[i]),
                             'M2_oof_error': float(y[i] - arm['scored'][CANDIDATE].results[protocol].yhat[i])})
    counts = {p: {'below': sum(r['side'] == 'below' for r in rows if r['protocol'] == p),
                  'above': sum(r['side'] == 'above' for r in rows if r['protocol'] == p)} for p in PROTOCOLS}
    return {'rows': rows, 'counts': counts,
            'note': 'linear-tail extrapolation disclosed in advance (M2_FEASIBILITY_AUDIT.md §4); descriptive only'}


def equivalence(arms):
    a, b = arms['primary_total_co2'], arms['per_capita']
    report = {'prediction_tolerance': PREDICTION_ATOL, 'coefficient_tolerance': COEF_ATOL,
              'predictions': {}, 'scores': {}, 'coefficients': {}}
    passes = True
    for model in MODELS:
        diffs = {'in_sample': float(np.max(np.abs(a['scored'][model].fitted - b['scored'][model].fitted)))}
        for protocol in PROTOCOLS:
            diffs[protocol] = float(np.max(np.abs(a['scored'][model].results[protocol].yhat
                                                  - b['scored'][model].results[protocol].yhat)))
        report['predictions'][model] = diffs
        passes &= all(v <= PREDICTION_ATOL for v in diffs.values())
        scores = {m: abs(float(_metric(a['cards'][model], m)) - float(_metric(b['cards'][model], m)))
                  for m in EQUIVALENT_METRICS}
        report['scores'][model] = scores
        passes &= all(v <= PREDICTION_ATOL for v in scores.values())
    paired = {'delta_rmse': abs(a['paired']['delta_rmse'] - b['paired']['delta_rmse']),
              'interval': [abs(x - y) for x, y in zip(a['paired']['country_bootstrap_95_interval'],
                                                      b['paired']['country_bootstrap_95_interval'])]}
    report['scores']['paired'] = paired
    passes &= paired['delta_rmse'] <= PREDICTION_ATOL and all(v <= PREDICTION_ATOL for v in paired['interval'])
    for column in LATITUDE_COEFFICIENTS:
        fa, fb = a['scored'][CANDIDATE].full, b['scored'][CANDIDATE].full
        full_diff = abs(fa.coef[fa.columns.index(column)] - fb.coef[fb.columns.index(column)])
        fold = [abs(ra['fitted'].coef[ra['fitted'].columns.index(column)]
                    - rb['fitted'].coef[rb['fitted'].columns.index(column)])
                for ra, rb in zip(a['scored'][CANDIDATE].records[PRIMARY], b['scored'][CANDIDATE].records[PRIMARY])]
        report['coefficients'][column] = {'full': float(full_diff), 'primary_training_max': float(max(fold)),
                                          'training_fits_compared': len(fold)}
        passes &= full_diff <= COEF_ATOL and max(fold) <= COEF_ATOL and len(fold) == len(a['scored'][CANDIDATE].fitted)
    report['allocations_may_differ'] = {
        model: {g: {'primary_total_co2': a['cards'][model].get(f'share_{g}'),
                    'per_capita': b['cards'][model].get(f'share_{g}'),
                    'difference': a['cards'][model].get(f'share_{g}', 0.0) - b['cards'][model].get(f'share_{g}', 0.0)}
                for g in GROUPS} for model in MODELS}
    report['note'] = ('Shapley allocations and the population/intercept coefficients differ by construction; '
                      'they are reported, never required to match, and never select a representation.')
    report['passes'] = bool(passes)
    return report


# ---------------------------------------------------------------------
# Evidence tables
# ---------------------------------------------------------------------


def prediction_rows(arms, frozen):
    rows = []
    table = frozen.table
    for representation, arm in arms.items():
        y = arm.get('outcome', frozen.y)
        for model in MODELS:
            scored = arm['scored'][model]
            flags = {}
            if model == CANDIDATE:
                for protocol in PROTOCOLS:
                    for record in scored.records[protocol]:
                        state = record['fitted'].state
                        for i in record['test_index']:
                            a = float(arm['frames'][model].abs_latitude.iloc[i])
                            flags[(protocol, int(i))] = ('below' if a < state.knots[0] else
                                                         'above' if a > state.knots[-1] else '')
            base = {'representation': representation, 'model': model}
            for i in range(frozen.n):
                rows.append({**base, 'protocol': 'in_sample', 'iso3': table.iso3.iloc[i], 'Country': table.Country.iloc[i],
                             'm49_subregion': table.m49_subregion.iloc[i], 'observed': y[i],
                             'prediction': scored.fitted[i], 'error': y[i] - scored.fitted[i], 'fold_id': -1,
                             'n_train': frozen.n, 'nearest_train_km': np.nan, 'unseen_levels': 0, 'extrapolation': ''})
            for protocol in PROTOCOLS:
                result = scored.results[protocol]
                for i in range(frozen.n):
                    rows.append({**base, 'protocol': protocol, 'iso3': table.iso3.iloc[i],
                                 'Country': table.Country.iloc[i], 'm49_subregion': table.m49_subregion.iloc[i],
                                 'observed': y[i], 'prediction': result.yhat[i], 'error': y[i] - result.yhat[i],
                                 'fold_id': int(result.fold_id[i]), 'n_train': int(result.n_train[i]),
                                 'nearest_train_km': float(result.nearest_train_km[i]),
                                 'unseen_levels': int(result.unseen_levels[i]),
                                 'extrapolation': flags.get((protocol, i), '')})
    return pd.DataFrame(rows)


def comparison_rows(arms, frozen):
    parts = []
    for representation, arm in arms.items():
        y = arm.get('outcome', frozen.y)
        c0 = arm['scored'][COMPARATOR].results[PRIMARY].yhat
        m2 = arm['scored'][CANDIDATE].results[PRIMARY].yhat
        frame = pd.DataFrame({'representation': representation, 'iso3': frozen.table.iso3,
                              'Country': frozen.table.Country, 'm49_subregion': frozen.table.m49_subregion,
                              'observed': y, 'M0star_cv_prediction': c0, 'M2_cv_prediction': m2,
                              'M0star_abs_error': np.abs(y - c0), 'M2_abs_error': np.abs(y - m2)})
        frame['abs_error_change'] = frame.M2_abs_error - frame.M0star_abs_error
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def fold_rows(frozen):
    ids, _ = frozen.folds[PRIMARY]
    memberships, sizes = [], []
    for _fold, _test, train in cv.iter_folds(ids, frozen.distance, PRIMARY_BUFFER_KM):
        memberships.append('|'.join(frozen.table.iso3[train]))
        sizes.append(int(train.sum()))
    return pd.DataFrame({'iso3': frozen.table.iso3, 'Country': frozen.table.Country, 'primary_fold': ids,
                         'primary_n_train': sizes, 'primary_training_iso3': memberships,
                         'm49_subregion': frozen.table.m49_subregion, 'm49_fold': frozen.folds['m49_subregion_lo'][0],
                         'random10_fold': frozen.folds['random10'][0]})


def _state_row(fitted, frame, train_index, test_index):
    train = frame.iloc[train_index]
    row = {'n_train': len(train_index), 'n_test': len(test_index),
           'levels': json.dumps(fitted.encoder.levels, sort_keys=True), 'columns': '|'.join(fitted.columns),
           'matrix_columns': fitted.matrix_columns, 'matrix_rank': fitted.matrix_rank,
           'baseline_columns': fitted.baseline_columns, 'baseline_rank': fitted.baseline_rank,
           'n_train_sh': int((train.hemisphere.astype(str) == lb.SOUTHERN).sum()),
           'latitude_state': fitted.state.to_json() if fitted.state is not None else '',
           'test_below_lower_boundary': '', 'test_above_upper_boundary': ''}
    if fitted.state is not None and len(test_index):
        held = lb.extrapolation(fitted.state, frame.iloc[test_index].abs_latitude.to_numpy(float))
        row.update({'test_below_lower_boundary': held['below'], 'test_above_upper_boundary': held['above']})
    return row


def state_rows(arms):
    rows = []
    for representation, arm in arms.items():
        for model in MODELS:
            scored, frame = arm['scored'][model], arm['frames'][model]
            rows.append({'representation': representation, 'model': model, 'protocol': 'full_sample', 'fit': 'all',
                         'held_out_iso3': '', **_state_row(scored.full, frame, np.arange(len(frame)), [])})
            for protocol in PROTOCOLS:
                for record in scored.records[protocol]:
                    rows.append({'representation': representation, 'model': model, 'protocol': protocol,
                                 'fit': record['fit'],
                                 'held_out_iso3': '|'.join(frame.iso3.iloc[record['test_index']]),
                                 **_state_row(record['fitted'], frame, record['train_index'], record['test_index'])})
    return pd.DataFrame(rows)


def coefficient_rows(arms):
    rows = []
    for representation, arm in arms.items():
        for model in MODELS:
            scored = arm['scored'][model]
            fits = [('full_sample', 'all', scored.full)]
            fits += [(r['protocol'], r['fit'], r['fitted']) for p in PROTOCOLS for r in scored.records[p]]
            for protocol, fit, fitted in fits:
                for column, value in zip(fitted.columns, fitted.coef):
                    rows.append({'representation': representation, 'model': model, 'protocol': protocol,
                                 'fit': fit, 'column': column, 'coefficient': float(value)})
    return pd.DataFrame(rows)


def coalition_frame(arms):
    return pd.DataFrame([{'representation': representation, 'model': model, **row}
                         for representation, arm in arms.items() for model in MODELS
                         for row in arm['scored'][model].lmg['coalitions']])


def design_frame(arms):
    rows = []
    for representation, arm in arms.items():
        for model in MODELS:
            frame = arm['frames'][model]
            state = lb.fit_state(frame.abs_latitude.to_numpy(float)) if model == CANDIDATE else None
            matrix, names = design(model, frame, feas.baseline_encoder(frame), state)
            groups = column_groups(frame, names)
            for j, (name, group) in enumerate(zip(names, groups)):
                for i, iso3 in enumerate(frame.iso3):
                    rows.append({'representation': representation, 'model': model, 'iso3': iso3, 'column': name,
                                 'group': group, 'value': float(matrix[i, j])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


def representation_frames(frozen):
    return {r: {COMPARATOR: frozen.frames[r], CANDIDATE: frozen.frames[r]} for r in REPRESENTATIONS}


def score_comparators(frozen, frames, reference_cards, outcome=OUTCOME_COL):
    """Both M0* representations, checked against their frozen cards before any M2 fit (spec §3)."""
    arms = {}
    for representation in REPRESENTATIONS:
        scored = score_model(COMPARATOR, frames[representation][COMPARATOR], frozen, outcome)
        _refuse(scored.card['structural_integrity']['passes'], f'{representation} M0*: structural integrity fails')
        reference = (reference_check(scored.card, reference_cards[representation]['card'],
                                     f'{representation} M0*', reference_cards[representation]['metrics'])
                     if reference_cards else None)
        arms[representation] = {'frames': frames[representation], 'scored': {COMPARATOR: scored},
                                'cards': {COMPARATOR: scored.card}, 'reference': reference}
    return arms


def score_candidates(frozen, arms, outcome=OUTCOME_COL):
    integrity = []
    for representation, arm in arms.items():
        frame = arm['frames'][CANDIDATE]
        scored = score_model(CANDIDATE, frame, frozen, outcome)
        arm['scored'][CANDIDATE] = scored
        arm['cards'][CANDIDATE] = scored.card
        y = frame[outcome].to_numpy(float)
        arm['outcome'] = y
        arm['paired'] = paired_rmse_interval(y, scored.results[PRIMARY].yhat,
                                             arm['scored'][COMPARATOR].results[PRIMARY].yhat)
        for model in MODELS:
            card = arm['cards'][model]
            if not card['structural_integrity']['passes']:
                integrity.append(f'{representation} {model}: structural integrity fails')
            if not card['share_accounting']['passes']:
                integrity.append(f'{representation} {model}: share accounting outside {SHARE_ATOL}')
    for representation, arm in arms.items():
        ok = not any(msg.startswith(representation) for msg in integrity)
        try:
            arm['verdict'] = verdict(arm['cards'][COMPARATOR], arm['cards'][CANDIDATE], arm['paired'], ok)
        except ValueError as error:
            arm['verdict'] = {'verdict': None, 'refused': str(error)}
            integrity.append(f'{representation}: {error}')
        arm['diagnostics'] = diagnostics(arm['cards'][COMPARATOR], arm['cards'][CANDIDATE], arm['paired'])
    return integrity


def non_deciding_conditions(entry):
    conditions = dict(entry)
    label = conditions.pop('verdict', None)
    return {**conditions, 'label_descriptive_only': label,
            'note': 'the per-capita representation is reported unconditionally and never determines the label'}


def evaluate(frozen, out_dir, *, provenance, reference_cards):
    out_dir = prov.require_fresh(out_dir)
    feasibility_identity = check_against_feasibility(frozen)
    arms = score_comparators(frozen, representation_frames(frozen), reference_cards)
    # Every pre-candidate check has passed; M2 is fitted only from here on.
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        integrity = score_candidates(frozen, arms)
    except (lb.DegenerateLatitudeBasis, ValueError, np.linalg.LinAlgError) as error:
        prov.write_json({'integrity_passed': False, 'verdict': None,
                         'failure': f'{type(error).__name__}: {error}',
                         'rule': 'an integrity failure after any candidate fit writes no label and stops the line'},
                        out_dir / INTEGRITY_RECORD)
        return {'integrity_passed': False, 'verdict': None, 'failure': str(error)}
    equiv = equivalence(arms)
    if not equiv['passes']:
        integrity.append('representation equivalence outside the prespecified tolerances')
    primary = arms['primary_total_co2']
    extrapolation = extrapolation_disclosure(primary, frozen)
    expected_counts = {PRIMARY: {'below': 1, 'above': 1}, 'm49_subregion_lo': {'below': 1, 'above': 8},
                       'random10': {'below': 1, 'above': 4}}
    if extrapolation['counts'] != expected_counts:
        integrity.append(f"extrapolation counts {extrapolation['counts']} differ from the feasibility record")
    comparison = comparison_rows(arms, frozen)
    try:
        stopping = stopping_indicator(primary['paired']['delta_rmse'])
    except ValueError as error:
        stopping = {'refused': str(error)}
        integrity.append(str(error))
    if integrity:
        for arm in arms.values():
            if arm['verdict'].get('verdict') is not None:
                arm['verdict'] = {**arm['verdict'], 'verdict': None, 'predictive_support': None,
                                  'promoted': None, 'withheld': 'integrity failure'}
    result = {
        'contract': CONTRACT, 'amendment': 'research/model_v2/M2_PRE_SCORE_RESOLUTIONS.md', 'specification': SPEC,
        'primary_representation': 'primary_total_co2',
        'acceptance_rule': 'only the primary total-CO2 representation determines the label; the per-capita '
                           'representation is reported unconditionally',
        'pre_candidate_checks': {'feasibility_identity': feasibility_identity,
                                 'comparator_reference_reproduction': {r: arm['reference'] for r, arm in arms.items()}},
        'representations': {
            representation: {
                'decides_verdict': REPRESENTATIONS[representation]['decides_verdict'],
                'M0star': arm['cards'][COMPARATOR], 'M2': arm['cards'][CANDIDATE],
                'paired_primary_comparison': arm['paired'],
                **({'verdict': arm['verdict']} if REPRESENTATIONS[representation]['decides_verdict']
                   else {'representation_conditions': non_deciding_conditions(arm['verdict'])}),
                'diagnostics': arm['diagnostics'],
                'countries_improved': int((comparison[comparison.representation == representation].abs_error_change < 0).sum()),
                'countries_worsened': int((comparison[comparison.representation == representation].abs_error_change > 0).sum()),
                'countries_tied': int((comparison[comparison.representation == representation].abs_error_change == 0).sum()),
            } for representation, arm in arms.items()},
        'extrapolation_disclosure': extrapolation,
        'static_stopping_indicator': stopping,
        'representation_equivalence': equiv,
        'integrity_failures': integrity,
        'integrity_passed': not integrity,
    }
    result['verdict'] = result['representations']['primary_total_co2']['verdict']
    result['retained_static_specification'] = (
        None if integrity else 'M2' if result['verdict']['verdict'] == 'supported and promoted' else 'M0*')

    prov.write_csv(prediction_rows(arms, frozen), out_dir / 'm2_country_predictions.csv')
    prov.write_csv(comparison, out_dir / 'm2_primary_country_comparison.csv')
    prov.write_csv(fold_rows(frozen), out_dir / 'm2_cv_folds.csv')
    prov.write_csv(state_rows(arms), out_dir / 'm2_fit_states.csv')
    prov.write_csv(coefficient_rows(arms), out_dir / 'm2_coefficients.csv')
    prov.write_csv(coalition_frame(arms), out_dir / 'm2_shapley_coalitions.csv')
    prov.write_csv(design_frame(arms), out_dir / 'm2_design_matrices.csv')
    prov.write_json({r: {m: arm['cards'][m]['coefficients'] for m in MODELS} for r, arm in arms.items()},
                    out_dir / 'm2_coefficient_summary.json')
    prov.write_json(result, out_dir / 'm2_scorecard.json')
    prov.write_json({**provenance, 'identity': frozen.identity}, out_dir / 'm2_provenance.json')
    manifest = {
        'result': 'M2 primary and unconditional per-capita evaluation',
        'contract': CONTRACT, 'specification': SPEC,
        'evaluator_commit': provenance.get('code', {}).get('commit'),
        'verdict': result['verdict'].get('verdict'), 'predictive_support': result['verdict'].get('predictive_support'),
        'retained_static_specification': result['retained_static_specification'],
        'static_stopping_indicator_fires': stopping.get('fires'),
        'integrity_passed': result['integrity_passed'],
        'artifact_sha256': {name: prov.sha256(out_dir / name) for name in DETERMINISTIC_ARTIFACTS},
        'non_deterministic_artifacts': [RUN_METADATA],
    }
    prov.write_json(manifest, out_dir / RESULT_MANIFEST)
    result['result_manifest'] = manifest
    return result


def reproducibility_record(run_a, run_b, *, code=None):
    a, b = Path(run_a), Path(run_b)
    artifacts, identical = {}, True
    for name in (*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST):
        digest_a, digest_b = prov.sha256(a / name), prov.sha256(b / name)
        artifacts[name] = {'run_a_sha256': digest_a, 'run_b_sha256': digest_b, 'identical': digest_a == digest_b}
        identical &= digest_a == digest_b
    return {'comparison': 'unchanged-code rerun in a separate process and output root', 'run_a': str(a),
            'run_b': str(b), 'compared': [*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST],
            'excluded_as_run_metadata': [RUN_METADATA], 'artifacts': artifacts,
            'byte_identical': bool(identical), 'code': code}


def reference_cards():
    rank_audit = json.loads((ROOT / RANK_AUDIT).read_text())['cards']
    return {r: {'card': rank_audit[spec['reference_card']], 'metrics': REFERENCE_METRICS}
            for r, spec in REPRESENTATIONS.items()}


def software():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform(), 'machine': platform.machine()}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score M2 under the frozen contract.')
    parser.add_argument('--out', default=str(DEFAULT_OUT), help='fresh output root (never overwritten)')
    parser.add_argument('--compare', nargs=2, metavar=('RUN_A', 'RUN_B'),
                        help='byte-compare two completed runs instead of scoring')
    parser.add_argument('--record', help='where to write the --compare record')
    args = parser.parse_args(argv)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    if args.compare:
        record = reproducibility_record(*args.compare, code=prov.frozen_code_state(code_files()))
        if args.record:
            prov.write_json(record, args.record)
        print(prov.json_text({'byte_identical': record['byte_identical'],
                              'differing': [n for n, v in record['artifacts'].items() if not v['identical']]}))
        return record
    prov.require_hash_seed()
    code = prov.frozen_code_state(code_files())
    out_dir = prov.require_fresh(args.out)
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    digests = verify_pins()
    frozen = build_frozen()
    provenance = {
        'documents_sha256': {d: prov.sha256(ROOT / d) for d in DOCUMENTS},
        'code': {k: v for k, v in code.items() if k != 'live_remote_commit'},
        'inputs': digests,
        'reference_cards': {r: spec['reference_card'] for r, spec in REPRESENTATIONS.items()},
        'protocols': {'primary': 'corrected 500 km footprint-certified LOCO (4263429 bounds)',
                      'secondary': 'leave one M49 subregion out', 'reference_only': 'random 10-fold, seed 0'},
        'basis_version': lb.BASIS_VERSION,
        'software': software(),
    }
    result = evaluate(frozen, out_dir, provenance=provenance, reference_cards=reference_cards())
    prov.write_json({'started_utc': started_utc,
                     'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     'wall_seconds': time.perf_counter() - started, 'host': platform.node(),
                     'executable': sys.executable, 'argv': sys.argv, 'output_root': str(out_dir),
                     'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'upstream': code['upstream'],
                     'live_remote_commit': code['live_remote_commit'],
                     'note': 'run metadata is deliberately outside the byte-compared deterministic artifacts'},
                    out_dir / RUN_METADATA)
    print(prov.json_text({'verdict': result.get('verdict'), 'integrity_failures': result.get('integrity_failures'),
                          'retained_static_specification': result.get('retained_static_specification')}))
    return result


def cli_exit_code(outcome):
    """0 on success; 3 when evidence was written but integrity failed; 4 when --compare runs differ."""
    if not isinstance(outcome, dict) or ('integrity_passed' in outcome) == ('byte_identical' in outcome):
        raise RuntimeError('unrecognised CLI outcome: expected exactly one of integrity_passed or byte_identical')
    key, failure = ('integrity_passed', 3) if 'integrity_passed' in outcome else ('byte_identical', 4)
    if not isinstance(outcome[key], bool):
        raise RuntimeError(f'{key} must be a bool, found {type(outcome[key]).__name__}')
    return 0 if outcome[key] else failure


if __name__ == '__main__':
    sys.exit(cli_exit_code(main()))
