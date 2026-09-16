"""M1b scoring under ``M1B_EVALUATION_CONTRACT.md`` (``60111ae``) and ``M1B_EVALUATOR_SPEC.md``.

M1b = M0* + ``baseline_dryness`` in a research-only ``hydroclimate`` group, reported in both the
approved primary total-CO2 representation and, unconditionally (Amendment 3 A3.6), the per-capita
representation. Only the primary pair determines the contract verdict.

Scoring is enabled by provenance, not by a flag: :func:`frozen_code_state` refuses unless the whole
code path is committed, unmodified and pushed, ``PYTHONHASHSEED=0`` is set and the output root is
fresh. :func:`verify_scoring_inputs` then fails closed unless the exact accepted measurement package,
its redundancy record and every frozen scoring input match their recorded digests, before any target
column is read. Run with
``PYTHONHASHSEED=0 python -m research.model_v2.m1b_evaluate [--out DIR]``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from research.model_v2 import cv, territory
from research.model_v2 import m1b_hydroclimate as hydro
from research.model_v2 import m1b_redundancy as red
from research.model_v2.m0 import OUTPUT_DIR, ROOT, load_inputs, m0_complete_design
from research.model_v2.m0 import design_matrix as m0_design_matrix
from research.model_v2.regions import m49_table
from research.model_v2.run_m0_scorecard import MIN_REGION_N, PRIMARY_BUFFER_KM
from research.model_v2.run_territory_correction import paired_rmse_interval
from research.model_v2.spatial import haversine_matrix, knn_weights, morans_i
from src import feature_schema as fs
from src.decomposition import CATEGORICAL_FEATURES, OUTCOME_COL, _r2, feature_block, group_lmg_shares
from src.stability import _used_features

FEATURE = 'baseline_dryness'
HYDROCLIMATE = fs.FeatureGroup(
    key='hydroclimate', title='Baseline hydroclimate (research-only, M1b)',
    rationale='Pre-outcome-window (1920-1949) CRU TS v4.10 local log aridity over terrestrial area; '
              'a physical state concept, not geography and not attribution.',
    features=(fs.FeatureSpec(FEATURE, 'Area-weighted mean of cell log10(P/PET), 1920-1949 climatology.',
                             'log10(dimensionless)'),),
)
SCHEMA_M1B = dataclasses.replace(
    fs.SCHEMA_V1, version='v1+m1b-hydroclimate-research',
    groups=tuple(g for g in fs.SCHEMA_V1.groups if g.key != 'residual') + (HYDROCLIMATE,)
    + tuple(g for g in fs.SCHEMA_V1.groups if g.key == 'residual'))

# Contract thresholds (§4.4). None is recomputed, relaxed or made outcome-dependent here.
PRACTICAL = -0.002          # A2, degC/decade
VETO = 0.001                # A3 and A4, degC/decade
SIGN_STABILITY = Fraction(4, 5)   # S2/A5: >= 80% of the primary training fits, compared exactly
MORAN_CHANGE = 0.05
OVERFIT_GAP = 0.05
MATERIAL_RESPONSIBILITY = 0.10
NAMED_GROUPS = ('geography', 'emissions', 'socioeconomic', 'population', 'hydroclimate')
# Spec §3 and §6 tolerances, prespecified.
REFERENCE_ATOL = 1e-10      # rank-audit / frozen-card reproduction
PREDICTION_ATOL = 1e-9      # representation equivalence, predictions and non-allocation scores
COEF_ATOL = 1e-8            # representation equivalence, baseline_dryness coefficients
SHARE_ATOL = 1e-9           # named shares + residual accounting
IDENTITY_ATOL = 1e-12       # the log10 per-capita/total/population identity
EXPECTED_COLUMNS = {'comparator': 20, 'candidate': 21}

MEASUREMENT_BUILD_COMMIT = '58e64bfbe6a6b12503a8b08cea51fecd5415e1a5'
ACCEPTED_PACKAGE = {
    hydro.FEATURES: 'cf33ba7cfb87c2184a94df5e5686f0951adf692819af8de27329ea698d72e1f4',
    hydro.MANIFEST: '1974686199b9c7960ead2b105e6225f34b9b6fb33627858d2bc15239d7de4c9c',
    red.RECORD: '9acb680c09b1577592ca9c4d62f992f6dab9628d581bb58b4448ad36b80f7585',
}
ACCEPTED_REDUNDANCY = {
    'design_rank': 20, 'rank_with_c2': 21, 'near_redundant_flag': False,
    'r2_full_m0star': 0.6501911562624498, 'adjusted_r2_full_m0star': 0.5994555224379197,
    'vif': 2.8587041691555015,
}
# Frozen scoring inputs: SHA-256 and the commit that froze each file. Verified before any fit.
FROZEN_SCORING_INPUTS = {
    'research/model_v2/outputs/m0_countries.csv':
        ('9e99b3798e91f2b7e40250921edd5e5a788e4191df3e9027dceb98fe76b43536', 'af97bd2'),
    'research/model_v2/outputs/country_geometry.csv':
        ('c8d72866d97d13288b3024781cd262f378ce08749d5166e0d4805bea01903e3a', 'af97bd2'),
    'research/model_v2/outputs/cv_fold_membership.csv':
        ('0100c5c7677ed0b4d3073b12b35300a3a973b3f126dff4cdcfacde4a86ac56f2', 'af97bd2'),
    'research/model_v2/outputs/territory_distance_lower_km.csv':
        ('79a20f007ebe08044833753956cdf859c27d505a2c10747b5e4df3469ee2c6e4', '4263429'),
    'research/model_v2/outputs/territory_cv_membership_comparison.csv':
        ('946be248edd0b7c3f49bd2bb34c94d3c921fd8cbf1b13e6be95e5ebf8b4706ef', '4263429'),
    'research/model_v2/outputs/m0_cv_errors_territory_corrected.csv':
        ('d9d800e322e27a7532d673a7613be5cccf4d694fc5914eb2b37f678ee3062d65', '4263429'),
    'research/model_v2/outputs/m0_scorecard_territory_corrected.json':
        ('396a6a6099db789c99a2e415c5bf5cae46d93e67b2f5a63d57c8f5efc89c7311', '4263429'),
    'research/model_v2/outputs/rank_audit.json':
        ('bb1a5ccd711a71969432ea6ce2e2a9c3b90ee0c45e3dc845bef25e5895bf362a', '9fb56e1'),
    'research/model_v2/outputs/rank_audit_country_predictions.csv':
        ('77ae9c6a40205354be7e98d9b41012f84e52df4c6e3f7b0e65518a42ff45217c', '9fb56e1'),
    'research/model_v2/outputs/product_stability_aligned_era5_trends.csv':
        ('23c96aeaf400433fea773903ede12fa593f549acf16a7286362b1f6078294922', '9fb56e1'),
    'research/model_v2/outputs/product_stability_aligned_summary.json':
        ('aaac5683728b22f23fb5e2392dd0f3a3f67d081db397b4a972051d75183f40f0', '9fb56e1'),
    'research/model_v2/outputs/m1a_geography_features.csv':
        ('a62d04d478b4500656c83e28d4fb4fb2d5307388b8ea5142a0f8de8dc059c4b3', 'a7dea34'),
    'research/model_v2/outputs/m1a_measurement_manifest.json':
        ('c9d9645203bb32a0698e07cc4cff4824bb5231194bdcc40ae2d91daf3d8b6947', 'a7dea34'),
    'research/model_v2/outputs/m1a_scorecard.json':
        ('9b6b0a778fd636a5373b33a611895c9310fbfd0c55088e3604e9712cbdd9ab24', '7003ff4'),
}
# Local code whose bytes define this evaluation; all of it must be committed and pushed.
CODE_PATH = (
    'research/model_v2/m1b_evaluate.py', 'research/model_v2/m1b_sensitivity.py',
    'research/model_v2/M1B_EVALUATOR_SPEC.md', 'research/model_v2/M1B_EVALUATION_CONTRACT.md',
    'research/model_v2/m1b_hydroclimate.py', 'research/model_v2/m1b_redundancy.py',
    'research/model_v2/cv.py', 'research/model_v2/m0.py', 'research/model_v2/territory.py',
    'research/model_v2/spatial.py', 'research/model_v2/regions.py',
    'research/model_v2/run_m0_scorecard.py', 'research/model_v2/run_territory_correction.py',
    'src/decomposition.py', 'src/feature_schema.py', 'src/stability.py', 'src/explain.py',
    'src/emissions.py',
)
CONTRACT_PATH = 'research/model_v2/M1B_EVALUATION_CONTRACT.md'
SPEC_PATH = 'research/model_v2/M1B_EVALUATOR_SPEC.md'
COUNTRY_TABLE = 'research/model_v2/outputs/m0_countries.csv'
RANK_AUDIT = 'research/model_v2/outputs/rank_audit.json'
RANK_PREDICTIONS = 'research/model_v2/outputs/rank_audit_country_predictions.csv'
FOLD_MEMBERSHIP = 'research/model_v2/outputs/cv_fold_membership.csv'
TERRITORY_MEMBERSHIP = 'research/model_v2/outputs/territory_cv_membership_comparison.csv'
CORRECTED_ERRORS = 'research/model_v2/outputs/m0_cv_errors_territory_corrected.csv'
GEOMETRY = 'research/model_v2/outputs/country_geometry.csv'
# The country table also holds outcome, fitted and residual columns; only these are ever read.
TABLE_COLUMNS = ['Country', 'iso3', 'm49_subregion', 'station_lon', 'station_lat']
OUTCOME_SOURCE_COLUMN = 'trend_c_per_decade_area_weighted'

REPRESENTATIONS = {
    'primary_total_co2': {'drop': 'cum_co2_per_capita', 'responsibility': 'cum_co2_total',
                          'reference_card': 'drop_per_capita', 'decides_verdict': True},
    'per_capita': {'drop': 'cum_co2_total', 'responsibility': 'cum_co2_per_capita',
                   'reference_card': 'drop_total', 'decides_verdict': False},
}
PROTOCOLS = ('primary_loco', 'm49_subregion_lo', 'random10')
PRIMARY = 'primary_loco'
COMPARATOR, CANDIDATE = 'M0star', 'M1b'
DEFAULT_OUT = OUTPUT_DIR / 'm1b_primary'
# Metrics on which a comparator must reproduce its frozen reference card (spec §3).
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
    'm1b_scorecard.json', 'm1b_country_predictions.csv', 'm1b_primary_country_comparison.csv',
    'm1b_cv_folds.csv', 'm1b_coefficients.csv', 'm1b_coefficient_summary.json',
    'm1b_shapley_coalitions.csv', 'm1b_design_matrices.csv', 'm1b_provenance.json',
)
RESULT_MANIFEST = 'm1b_result_manifest.json'
RUN_METADATA = 'm1b_run_metadata.json'


class M1bDesign(cv.V1Design):
    """The frozen V1 estimator under the research schema extension (M0* when C2 is absent)."""

    def _features(self, columns):
        used = _used_features(columns, SCHEMA_M1B, fs.STATUS_AVAILABLE)
        return [name for names in used.values() for name in names]


def fit_predict(train, test):
    return M1bDesign().fit(train).predict(test)


def design_matrix(frame):
    used = _used_features(frame.columns, SCHEMA_M1B, fs.STATUS_AVAILABLE)
    blocks = [feature_block(frame, f)[0] for names in used.values() for f in names]
    return np.hstack([np.ones((len(frame), 1)), *blocks])


def design_columns(frame):
    """Encoded column names of :func:`design_matrix`, with their declared group."""
    used = _used_features(frame.columns, SCHEMA_M1B, fs.STATUS_AVAILABLE)
    names, groups = ['intercept'], ['intercept']
    for group, features in used.items():
        for f in features:
            cols = feature_block(frame, f)[1]
            names.extend(cols)
            groups.extend([group] * len(cols))
    return names, groups


def training_matrix(model, frame):
    """The design matrix a fitted encoder builds for ``frame`` (train-only levels)."""
    return model._matrix(frame)


# ---------------------------------------------------------------------
# Execution boundary and the scoring-input gate
# ---------------------------------------------------------------------


def _git(*args, check=True):
    done = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done


def frozen_code_state(code_path=CODE_PATH):
    """Fail closed unless every code-path file is committed, unmodified and pushed."""
    head = _git('rev-parse', 'HEAD').stdout.strip()
    missing = [p for p in code_path if not (ROOT / p).exists()]
    if missing:
        raise RuntimeError(f'code path files are missing: {missing}')
    untracked = _git('ls-files', '--others', '--exclude-standard', '--', *code_path).stdout.split()
    if untracked:
        raise RuntimeError(f'untracked files on the code path: {untracked}')
    if _git('diff', '--quiet', 'HEAD', '--', *code_path, check=False).returncode != 0:
        changed = _git('diff', '--name-only', 'HEAD', '--', *code_path).stdout.split()
        raise RuntimeError(f'the code path differs from HEAD; commit before scoring: {changed}')
    upstream = _git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}').stdout.strip()
    if _git('merge-base', '--is-ancestor', head, upstream, check=False).returncode != 0:
        raise RuntimeError(f'HEAD {head} is not contained in {upstream}; push before scoring')
    return {
        'commit': head, 'upstream': upstream,
        'upstream_commit': _git('rev-parse', upstream).stdout.strip(),
        'code_path_clean_and_pushed': True,
        'code_path_sha256': {p: hydro.sha256(ROOT / p) for p in code_path},
    }


def require_hash_seed():
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('contract 4.1 requires PYTHONHASHSEED=0 for byte-reproducible scoring')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_manifest_fields(manifest):
    """Explicit required-field and type validation of the measurement manifest (spec 2.3)."""
    types = {'gate_version': str, 'all_hard_stops_pass': bool, 'hard_stops': list, 'git': dict,
             'code_sha256': str, 'contract_freeze_commit': str, 'support_inputs': dict,
             'artifact_sha256': dict, 'window': list, 'n_months': int, 'coverage_min': float,
             'amendments': list, 'formula': str}
    for key, kind in types.items():
        _require(key in manifest, f'measurement manifest is missing {key!r}')
        _require(isinstance(manifest[key], kind) and not (kind is int and isinstance(manifest[key], bool)),
                 f'measurement manifest field {key!r} has the wrong type')
    _require(manifest['gate_version'] == hydro.GATE_VERSION, 'measurement manifest gate version')
    _require(manifest['all_hard_stops_pass'] is True and manifest['hard_stops'] == [],
             'measurement manifest does not record a passing gate')
    _require(manifest['contract_freeze_commit'] == hydro.CONTRACT_COMMIT, 'contract freeze commit')
    _require(manifest['git'].get('commit') == MEASUREMENT_BUILD_COMMIT,
             'measurement package was not built at the accepted build commit')
    _require(manifest['git'].get('code_path_matches_commit') is True,
             'measurement package was not built from committed construction code')
    _require(sorted(manifest['artifact_sha256']) == sorted(hydro.PACKAGE_ARTIFACTS),
             'measurement manifest does not record exactly the package artifact digests')
    _require(isinstance(manifest['support_inputs'].get('country_list_sha256'), str),
             'measurement manifest is missing the country list digest')
    _require(manifest['window'] == list(hydro.WINDOW) and manifest['n_months'] == hydro.N_MONTHS,
             'measurement manifest window')
    _require(manifest['coverage_min'] == hydro.COVERAGE_MIN, 'measurement manifest coverage gate')


def _require_record_fields(record, manifest, manifest_sha256):
    """Explicit required-field validation of the redundancy record (spec 2.3)."""
    for key, value in ACCEPTED_REDUNDANCY.items():
        _require(record.get(key) == value,
                 f'redundancy record {key!r} is {record.get(key)!r}, not the accepted {value!r}')
    provenance = record['provenance']
    expected = {
        'measurement_manifest_sha256': manifest_sha256,
        'c2_features_sha256': manifest['artifact_sha256'][hydro.FEATURES],
        'measurement_git_commit': manifest['git']['commit'],
        'measurement_code_sha256': manifest['code_sha256'],
        'measurement_gate_version': manifest['gate_version'],
        'country_list_sha256': manifest['support_inputs']['country_list_sha256'],
        'predictor_columns': red.PREDICTOR_COLUMNS,
        'outcome_column_present': False,
    }
    for key, value in expected.items():
        _require(provenance.get(key) == value, f'redundancy provenance {key!r} does not match the package')
    for key in ('predictor_frame_sha256', 'design_matrix_sha256', 'c2_vector_sha256',
                'redundancy_code_sha256'):
        _require(isinstance(provenance.get(key), str) and len(provenance[key]) == 64,
                 f'redundancy provenance {key!r} is not a SHA-256 digest')
    _require(isinstance(provenance.get('design_columns'), list)
             and len(provenance['design_columns']) == record['design_column_count']
             and provenance['design_columns'][0] == 'intercept',
             'redundancy provenance design columns')
    _require(provenance.get('predictor_input_sha256') == {
        str(Path(p).resolve().relative_to(ROOT)): d for p, d in red.PREDICTOR_INPUT_SHA256.items()},
        'redundancy record used different predictor inputs')


@dataclasses.dataclass(frozen=True)
class ScoringInputs:
    manifest: dict
    manifest_sha256: str
    record: dict
    features_bytes: bytes
    digests: dict


def verify_scoring_inputs(out_dir=OUTPUT_DIR):
    """Fail closed before any target column is loaded (spec 2)."""
    out_dir = Path(out_dir)
    for name, pin in ACCEPTED_PACKAGE.items():
        digest = hydro.sha256(out_dir / name)
        _require(digest == pin, f'{name}: sha256 {digest} is not the accepted package digest {pin}')
    manifest, manifest_sha256 = hydro.verify_package(out_dir)
    _require(manifest_sha256 == ACCEPTED_PACKAGE[hydro.MANIFEST], 'manifest digest')
    _require_manifest_fields(manifest)
    features_bytes = hydro.read_verified(out_dir, hydro.FEATURES, manifest)
    record = red.verify_record(out_dir, manifest_sha256, manifest['artifact_sha256'][hydro.FEATURES])
    _require_record_fields(record, manifest, manifest_sha256)
    hydro.check_frozen_inputs()
    digests = {'measurement_package': dict(ACCEPTED_PACKAGE),
               'c2_construction_inputs': {str(Path(p).resolve().relative_to(ROOT)): pin
                                          for p, pin in hydro.FROZEN_INPUT_SHA256.items()},
               'predictor_inputs': red.check_predictor_inputs(),
               'frozen_scoring_inputs': {}}
    for rel, (pin, commit) in FROZEN_SCORING_INPUTS.items():
        digest = hydro.sha256(ROOT / rel)
        _require(digest == pin,
                 f'{rel}: sha256 {digest} differs from the record frozen at {commit} ({pin})')
        digests['frozen_scoring_inputs'][rel] = {'sha256': digest, 'frozen_at': commit}
    return ScoringInputs(manifest, manifest_sha256, record, features_bytes, digests)


# ---------------------------------------------------------------------
# Frozen scoring frame
# ---------------------------------------------------------------------


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Frozen:
    """Everything a scored arm needs, with its identity record."""

    design: pd.DataFrame
    table: pd.DataFrame
    c2: np.ndarray
    distance: np.ndarray
    station_w: np.ndarray
    area_w: np.ndarray
    area_centroids: np.ndarray   # carried so an arm rebuilds weights without re-reading a file
    folds: dict
    identity: dict

    @property
    def n(self):
        return len(self.design)


def fold_plan(table, distance, n):
    return {PRIMARY: (np.arange(n), PRIMARY_BUFFER_KM),
            'm49_subregion_lo': (cv.folds_from_labels(table.m49_subregion), None),
            'random10': (cv.folds_random(n, 10), None)}


def build_frozen(inputs):
    """Load the frozen 151-country scoring frame and check every identity (spec 2.5)."""
    features = pd.read_csv(io.BytesIO(inputs.features_bytes))
    _require(features.columns.tolist() == ['iso3', 'Country', FEATURE], 'C2 columns')
    support = pd.read_csv(ROOT / 'research/model_v2/outputs/m1a_geography_qa.csv',
                          usecols=['iso3', 'Country'])
    _require(features.iso3.tolist() == support.iso3.tolist()
             and features.Country.tolist() == support.Country.tolist(),
             'C2 row order differs from the frozen M1a support record')
    country_list = hashlib.sha256(('\n'.join(features.iso3) + '\n').encode()).hexdigest()
    _require(country_list == inputs.manifest['support_inputs']['country_list_sha256'],
             'C2 country list differs from the measurement manifest')
    c2 = features[FEATURE].to_numpy(float)
    _require(np.isfinite(c2).all(), 'non-finite C2')
    _require(array_digest(c2) == inputs.record['provenance']['c2_vector_sha256'],
             'C2 vector digest differs from the redundancy record')

    design = m0_complete_design(*load_inputs())
    table = pd.read_csv(ROOT / COUNTRY_TABLE, usecols=TABLE_COLUMNS)[TABLE_COLUMNS]
    _require(len(design) == len(features) == len(table), 'complete-case population size')
    _require(design.Country.tolist() == features.Country.tolist() == table.Country.tolist(),
             'the scored design is not the frozen country population in the frozen order')
    predictors = design.drop(columns=[OUTCOME_COL, 'cum_co2_per_capita']).reset_index(drop=True)
    _require(predictors.columns.tolist() == red.PREDICTOR_COLUMNS, 'predictor frame columns')
    provenance = inputs.record['provenance']
    _require(red.frame_sha256(predictors) == provenance['predictor_frame_sha256'],
             'the scored predictor frame differs from the redundancy record')
    x, columns, _groups = m0_design_matrix(predictors)
    _require(columns == provenance['design_columns'], 'scored design columns')
    _require(array_digest(x) == provenance['design_matrix_sha256'], 'scored design matrix digest')

    y = design[OUTCOME_COL].to_numpy(float)
    _require(np.isfinite(y).all(), 'non-finite outcome')
    frozen_predictions = pd.read_csv(ROOT / RANK_PREDICTIONS, float_precision='round_trip')
    observed = (frozen_predictions[frozen_predictions.variant == 'drop_per_capita']
                .set_index('iso3').loc[table.iso3, 'observed'].to_numpy(float))
    _require(np.array_equal(y, observed),
             'the Berkeley outcome differs from the frozen ordered outcome record')
    identity_error = float(np.max(np.abs(
        np.log10(design.cum_co2_per_capita.to_numpy(float))
        - np.log10(design.cum_co2_total.to_numpy(float))
        + np.log10(design.population.to_numpy(float)) - 6)))
    _require(identity_error <= IDENTITY_ATOL, 'the representation log identity does not hold')

    labels = table[['iso3']].merge(m49_table(), on='iso3', how='left', validate='one_to_one')
    _require(labels.m49_subregion.notna().all()
             and (labels.m49_subregion.to_numpy() == table.m49_subregion.to_numpy()).all(),
             'M49 labels differ from the frozen definition')
    membership = pd.read_csv(ROOT / FOLD_MEMBERSHIP,
                             usecols=['iso3', 'm49_subregion', 'm49_subregion_lo', 'random_10fold'])
    _require(membership.iso3.tolist() == table.iso3.tolist()
             and (membership.m49_subregion.to_numpy() == table.m49_subregion.to_numpy()).all(),
             'frozen fold-membership labels')
    distance = territory.load_bounds(table.iso3.tolist())
    folds = fold_plan(table, distance, len(table))
    _require(np.array_equal(folds['m49_subregion_lo'][0], membership.m49_subregion_lo.to_numpy()),
             'M49 holdout construction differs from the frozen definition')
    _require(np.array_equal(folds['random10'][0], membership.random_10fold.to_numpy()),
             'the random 10-fold seed-0 reference differs from the frozen definition')
    derived = np.array([train for _f, _test, train in
                        cv.iter_folds(folds[PRIMARY][0], distance, PRIMARY_BUFFER_KM)])
    frozen_membership = (pd.read_csv(ROOT / TERRITORY_MEMBERSHIP)
                         .pivot(index='held_out', columns='candidate', values='corrected_train')
                         .loc[table.iso3, table.iso3].to_numpy(bool))
    _require(np.array_equal(derived, frozen_membership),
             'primary fold memberships differ from the corrected frozen memberships')
    n_train = pd.read_csv(ROOT / CORRECTED_ERRORS, usecols=['iso3', 'n_train'])
    _require(n_train.iso3.tolist() == table.iso3.tolist()
             and np.array_equal(derived.sum(axis=1), n_train.n_train.to_numpy()),
             'primary training counts differ from the frozen record')

    station_w = cv.v1_weights(table.station_lon.to_numpy(), table.station_lat.to_numpy())
    geom = (pd.read_csv(ROOT / GEOMETRY, usecols=['iso3', 'centroid_lon', 'centroid_lat'])
            .set_index('iso3').loc[table.iso3])
    area_centroids = np.column_stack([geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy()])
    area_w = knn_weights(haversine_matrix(area_centroids[:, 0], area_centroids[:, 1]), 8)
    identity = {
        'n': len(design), 'country_list_sha256': country_list,
        'outcome_source_column': OUTCOME_SOURCE_COLUMN,
        'outcome_vector_sha256': array_digest(y), 'c2_vector_sha256': array_digest(c2),
        'predictor_frame_sha256': provenance['predictor_frame_sha256'],
        'design_matrix_sha256': provenance['design_matrix_sha256'],
        'representation_identity_max_abs_error': identity_error,
        'primary_membership_sha256': hashlib.sha256(derived.tobytes()).hexdigest(),
        'm49_fold_sha256': hashlib.sha256(folds['m49_subregion_lo'][0].tobytes()).hexdigest(),
        'random10_fold_sha256': hashlib.sha256(folds['random10'][0].tobytes()).hexdigest(),
        'station_weights_sha256': array_digest(station_w),
        'area_weights_sha256': array_digest(area_w),
        'frozen_outcome_record': RANK_PREDICTIONS,
    }
    return Frozen(design=design.reset_index(drop=True), table=table, c2=c2, distance=distance,
                  station_w=station_w, area_w=area_w, area_centroids=area_centroids, folds=folds,
                  identity=identity)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


def protocol_card(frame, table, result, station_w):
    """The frozen metric set for one protocol. No paired field lives in a card (spec 4)."""
    fitted = M1bDesign().fit(frame).predict(frame)
    card = cv.scorecard(frame, fitted, result, station_w, shares=False)
    lmg = group_lmg_shares(frame, schema=SCHEMA_M1B)
    card.update({f'share_{k}': float(v) for k, v in lmg.shares.items()})
    card['residual_share'] = float(lmg.residual_share)
    y = frame[OUTCOME_COL].to_numpy(float)
    errors = np.abs(y - result.yhat)
    card['fold_error_iqr'] = float(np.subtract(*np.quantile(errors, [0.75, 0.25])))
    card['effective_degrees_of_freedom'] = int(np.linalg.matrix_rank(design_matrix(frame)))
    card['residual_moran_in_sample'] = asdict(morans_i(y - fitted, station_w))
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
    card['worst_fold_country'] = (str(table.iloc[card['worst_fold']].Country)
                                  if card['n_folds'] == len(y) else None)
    card['worst_countries_by_abs_error'] = [
        {'iso3': str(table.iso3.iloc[i]), 'Country': str(table.Country.iloc[i]),
         'abs_error': float(errors[i])} for i in np.argsort(-errors)[:10]]
    return card


def training_fits(frame, distance, folds, table):
    """Every primary training fit: coefficients, rank and whether C2 is identified."""
    rows = []
    for fold, _test, train in cv.iter_folds(folds, distance, PRIMARY_BUFFER_KM):
        model = M1bDesign().fit(frame[train])
        matrix, columns = training_matrix(model, frame[train])
        rank = int(np.linalg.matrix_rank(matrix))
        identified = None
        if FEATURE in columns:
            j = columns.index(FEATURE)
            without = int(np.linalg.matrix_rank(np.delete(matrix, j, axis=1)))
            identified = bool(rank == without + 1)
        rows.append({'fold': int(fold), 'held_out_iso3': str(table.iso3.iloc[int(fold)]),
                     'held_out_country': str(table.Country.iloc[int(fold)]),
                     'n_train': int(train.sum()), 'matrix_columns': int(matrix.shape[1]),
                     'matrix_rank': rank, 'feature_identified': identified,
                     'columns': list(columns), 'coef': [float(v) for v in model.coef]})
    return rows


def coefficient_summary(full, fits):
    by_column = {c: [] for c in full.columns}
    for fit in fits:
        for column, value in zip(fit['columns'], fit['coef']):
            by_column.setdefault(column, []).append(float(value))
    summary = {}
    for column, value in zip(full.columns, full.coef):
        values = np.array(by_column.get(column, []), dtype=float)
        finite = values[np.isfinite(values)]
        summary[column] = {
            'full': float(value), 'fold_values_n': int(len(values)),
            'fold_values_finite_n': int(len(finite)),
            'fold_min': float(finite.min()) if len(finite) else None,
            'fold_median': float(np.median(finite)) if len(finite) else None,
            'fold_max': float(finite.max()) if len(finite) else None,
            'same_sign_fraction': float(np.mean(np.sign(finite) == np.sign(value))) if len(finite) else None,
            'negative_n': int((finite < 0).sum()),
            'negative_fraction': float((finite < 0).mean()) if len(finite) else None,
            'zero_n': int((finite == 0).sum()),
        }
    return summary


def coalition_rows(frame):
    """Every group-coalition R^2 behind the LMG/Shapley shares (Shapley aggregation evidence)."""
    from itertools import combinations
    lmg = group_lmg_shares(frame, schema=SCHEMA_M1B)
    y = frame[OUTCOME_COL].to_numpy(float)
    groups = list(lmg.group_features)
    rows = []
    for size in range(len(groups) + 1):
        for subset in combinations(groups, size):
            blocks = [feature_block(frame, f)[0] for g in subset for f in lmg.group_features[g]]
            rows.append({'coalition': '+'.join(subset) or 'intercept_only', 'n_groups': size,
                         'r2': float(_r2(blocks, y))})
    return rows, lmg


@dataclasses.dataclass(frozen=True)
class Scored:
    card: dict
    results: dict
    fitted: np.ndarray
    full: object
    fits: list
    coalitions: list


def score_model(frame, frozen, *, expected_columns=None):
    results = {name: cv.cross_validate(frame, ids, fit_predict, dist=frozen.distance, buffer_km=buffer)
               for name, (ids, buffer) in frozen.folds.items()}
    card = protocol_card(frame, frozen.table, results[PRIMARY], frozen.station_w)
    card['secondary'] = protocol_card(frame, frozen.table, results['m49_subregion_lo'], frozen.station_w)
    card['random_reference_only'] = protocol_card(frame, frozen.table, results['random10'], frozen.station_w)
    fitted = M1bDesign().fit(frame).predict(frame)
    y = frame[OUTCOME_COL].to_numpy(float)
    card['recorded_area_centroid_moran_in_sample'] = morans_i(y - fitted, frozen.area_w).statistic
    card['recorded_area_centroid_moran_cv'] = morans_i(y - results[PRIMARY].yhat, frozen.area_w).statistic
    x = design_matrix(frame)
    card['matrix_columns'], card['matrix_rank'] = int(x.shape[1]), int(np.linalg.matrix_rank(x))
    if expected_columns is not None:
        _require(card['matrix_columns'] == expected_columns and card['matrix_rank'] == expected_columns,
                 f"full-sample design is {card['matrix_columns']} columns / rank {card['matrix_rank']}, "
                 f'expected {expected_columns} / {expected_columns}')
    card['unseen_rows_by_categorical'] = {
        name: cv.design_descriptors(frame, ids, frozen.distance, buffer)['test_rows_with_unseen_level_by_categorical']
        for name, (ids, buffer) in frozen.folds.items()}
    card['levels_present'] = {c: sorted(frame[c].astype(str).unique().tolist())
                              for c in sorted(CATEGORICAL_FEATURES) if c in frame}
    full = M1bDesign().fit(frame)
    fits = training_fits(frame, frozen.distance, frozen.folds[PRIMARY][0], frozen.table)
    card['coefficients'] = coefficient_summary(full, fits)
    card['training_fits'] = {
        'n': len(fits), 'min_n_train': int(min(f['n_train'] for f in fits)),
        'max_n_train': int(max(f['n_train'] for f in fits)),
        'all_full_rank': bool(all(f['matrix_rank'] == f['matrix_columns'] for f in fits)),
        'feature_identified_in_all': bool(all(f['feature_identified'] for f in fits))
        if FEATURE in full.columns else None,
        'column_counts': sorted({f['matrix_columns'] for f in fits}),
    }
    coalitions, lmg = coalition_rows(frame)
    named = {g: card.get(f'share_{g}', 0.0) for g in NAMED_GROUPS}
    card['share_accounting'] = {
        'named_share_sum': float(sum(named.values())),
        'named_share_sum_minus_in_sample_r2': float(sum(named.values()) - card['in_sample_r2']),
        'named_plus_residual_minus_one': float(sum(named.values()) + card['residual_share'] - 1.0),
        'tolerance': SHARE_ATOL,
        'passes': bool(abs(sum(named.values()) - card['in_sample_r2']) <= SHARE_ATOL
                       and abs(sum(named.values()) + card['residual_share'] - 1.0) <= SHARE_ATOL),
        'interpretation': 'fractions of explained outcome variance, not causal contributions',
    }
    card['groups_present'] = list(lmg.group_features)
    return Scored(card, results, fitted, full, fits, coalitions)


def _metric(card, dotted):
    value = card
    for part in dotted.split('.'):
        value = value[part]
    return value


def reference_check(card, reference, label, metrics=REFERENCE_METRICS, atol=REFERENCE_ATOL):
    """Reproduce a frozen reference card to ``atol``; every compared value must be finite."""
    rows = {}
    for metric in metrics:
        try:
            got, want = float(_metric(card, metric)), float(_metric(reference, metric))
        except KeyError:
            raise ValueError(f'{label}: reference card has no metric {metric!r}') from None
        _require(np.isfinite(got) and np.isfinite(want), f'{label}: {metric} is not finite')
        _require(abs(got - want) <= atol,
                 f'{label} does not reproduce the frozen card: {metric} {got!r} vs {want!r}')
        rows[metric] = {'value': got, 'reference': want, 'abs_difference': abs(got - want)}
    return {'label': label, 'tolerance': atol, 'metrics': rows, 'reproduced': True}


# ---------------------------------------------------------------------
# Verdict, diagnostics and equivalence
# ---------------------------------------------------------------------


def sign_condition(summary, n_expected):
    """S2/A5 inputs: the strictly-negative fraction over exactly ``n_expected`` finite fits."""
    stats = summary[FEATURE]
    _require(stats['fold_values_n'] == n_expected and stats['fold_values_finite_n'] == n_expected,
             f"expected exactly {n_expected} finite {FEATURE} training-fit coefficients, "
             f"got {stats['fold_values_finite_n']} finite of {stats['fold_values_n']}")
    _require(np.isfinite(stats['full']), f'{FEATURE} full-fit coefficient is not finite')
    fraction = Fraction(stats['negative_n'], n_expected)
    return {'full': float(stats['full']), 'negative_n': int(stats['negative_n']),
            'n_fits': int(n_expected), 'negative_fraction': float(fraction),
            'zero_n': int(stats['zero_n']),
            'negative_and_stable': bool(stats['full'] < 0 and fraction >= SIGN_STABILITY)}


def verdict(comparator, candidate, paired, sign):
    """Contract 4.4, mechanically. Non-finite inputs are a refusal, never a ``False``."""
    delta = float(paired['delta_rmse'])
    lo, hi = (float(v) for v in paired['country_bootstrap_95_interval'])
    m49 = float(candidate['secondary']['cv_rmse']) - float(comparator['secondary']['cv_rmse'])
    region = float(candidate['worst_region']['rmse']) - float(comparator['worst_region']['rmse'])
    for name, value in [('delta_rmse', delta), ('interval_low', lo), ('interval_high', hi),
                        ('m49_rmse_change', m49), ('worst_region_rmse_change', region),
                        ('coefficient_full', sign['full'])]:
        _require(np.isfinite(value), f'verdict input {name} is not finite')
    s1 = bool(delta < 0 and hi < 0)
    s2 = bool(sign['negative_and_stable'])
    a2 = bool(delta <= PRACTICAL)
    a3 = bool(m49 <= VETO)
    a4 = bool(region <= VETO)
    if s1 and s2 and a2 and a3 and a4:
        label = 'supported and promoted'
    elif s1 and s2:
        label = 'supported but sub-material / not promoted'
    else:
        label = 'not supported'
    return {
        'verdict': label, 'association_supported': bool(s1 and s2), 'promoted': label == 'supported and promoted',
        'S1_interval_below_zero': s1, 'S2_sign_negative_and_stable': s2,
        'A1': s1, 'A2_practical_le_minus_0.002': a2, 'A3_m49_veto_passed': a3,
        'A4_worst_region_veto_passed': a4, 'A5': s2,
        'delta_rmse': delta, 'interval': [lo, hi], 'resamples': paired['resamples'],
        'seed': paired['seed'], 'interval_method': 'paired country resampling interval of the '
                                                   'difference in fixed out-of-fold errors; not a '
                                                   'spatially corrected confidence interval',
        'coefficient_full': sign['full'], 'coefficient_negative_n': sign['negative_n'],
        'coefficient_n_fits': sign['n_fits'],
        'coefficient_negative_fraction': sign['negative_fraction'],
        'coefficient_zero_n': sign['zero_n'], 'sign_stability_threshold': float(SIGN_STABILITY),
        'practical_threshold': PRACTICAL, 'veto_threshold': VETO,
        'm49_rmse_change': m49, 'worst_region_rmse_change': region,
        'worst_region_comparator': comparator['worst_region'], 'worst_region_candidate': candidate['worst_region'],
        'improvement_without_prestated_sign': bool(s1 and not s2),
        'worsened_generalization': bool(delta > 0 and lo > 0),
    }


def diagnostics(comparator, candidate, paired):
    """Contract 4.5 diagnostics and the three-dimension summary (never verdict conditions)."""
    delta = float(paired['delta_rmse'])
    lo, hi = (float(v) for v in paired['country_bootstrap_95_interval'])
    m49_ok = candidate['secondary']['cv_rmse'] - comparator['secondary']['cv_rmse'] <= VETO
    region_ok = candidate['worst_region']['rmse'] - comparator['worst_region']['rmse'] <= VETO
    transfer = ('improved generalization' if delta < 0 and hi < 0 and m49_ok and region_ok else
                'worsened generalization' if delta > 0 and lo > 0 else
                'no detectable change in transfer')
    gain_in = candidate['in_sample_r2'] - comparator['in_sample_r2']
    gain_cv = candidate['cv_r2'] - comparator['cv_r2']
    d_in = candidate['residual_morans_i_in_sample'] - comparator['residual_morans_i_in_sample']
    d_cv = candidate['residual_morans_i_cv'] - comparator['residual_morans_i_cv']
    spatial = ('reduced' if d_in <= -MORAN_CHANGE and d_cv <= -MORAN_CHANGE else
               'increased' if d_in >= MORAN_CHANGE and d_cv >= MORAN_CHANGE else 'essentially unchanged')
    shares = {g: candidate.get(f'share_{g}', 0.0) for g in NAMED_GROUPS}
    largest = max(shares, key=shares.get)
    material = bool(largest != 'geography' or shares['emissions'] > MATERIAL_RESPONSIBILITY)
    return {
        'three_dimension_summary': {
            'transfer': transfer,
            'spatial_specification': f'residual spatial autocorrelation {spatial}',
            'stability': 'material change' if material else 'stable',
            'note': 'the same three dimensions M1A_EVALUATION_CONTRACT.md §4 reports; descriptive, '
                    'never a verdict condition'},
        'overfitting_signal': bool(gain_in - gain_cv > OVERFIT_GAP
                                   or (candidate['cv_rmse'] > comparator['cv_rmse']
                                       and candidate['in_sample_rmse'] < comparator['in_sample_rmse'])),
        'in_sample_r2_gain': gain_in, 'cv_r2_gain': gain_cv,
        'residual_spatial_structure': f'residual spatial autocorrelation {spatial}',
        'delta_moran_in_sample': d_in, 'delta_moran_cv': d_cv,
        'delta_area_centroid_moran_in_sample': candidate['recorded_area_centroid_moran_in_sample']
        - comparator['recorded_area_centroid_moran_in_sample'],
        'delta_area_centroid_moran_cv': candidate['recorded_area_centroid_moran_cv']
        - comparator['recorded_area_centroid_moran_cv'],
        'largest_named_group': largest, 'material_change_to_conclusion': material,
        'scientific_stability': 'material change' if material else 'stable',
        'shares': shares,
        'share_changes': {g: shares[g] - comparator.get(f'share_{g}', 0.0) for g in NAMED_GROUPS},
        'delta_cv_rmse_m49': candidate['secondary']['cv_rmse'] - comparator['secondary']['cv_rmse'],
        'delta_cv_r2_m49': candidate['secondary']['cv_r2'] - comparator['secondary']['cv_r2'],
        'delta_cv_rmse_random_reference': (candidate['random_reference_only']['cv_rmse']
                                           - comparator['random_reference_only']['cv_rmse']),
        'interpretation': 'Shares allocate shared explanatory variance; a nonzero hydroclimate share '
                          'is not a fraction of warming physically caused by hydroclimate.',
    }


def equivalence(arms):
    """Spec 6: the two representations span the same column space, so predictions must agree."""
    a, b = arms['primary_total_co2'], arms['per_capita']
    report = {'prediction_tolerance': PREDICTION_ATOL, 'coefficient_tolerance': COEF_ATOL,
              'predictions': {}, 'scores': {}, 'coefficients': {}, 'allocations_may_differ': {}}
    passes = True
    for model in (COMPARATOR, CANDIDATE):
        diffs = {'in_sample': float(np.max(np.abs(a['scored'][model].fitted - b['scored'][model].fitted)))}
        for protocol in PROTOCOLS:
            diffs[protocol] = float(np.max(np.abs(a['scored'][model].results[protocol].yhat
                                                  - b['scored'][model].results[protocol].yhat)))
        report['predictions'][model] = diffs
        passes &= all(v <= PREDICTION_ATOL for v in diffs.values())
        scores = {}
        for metric in EQUIVALENT_METRICS:
            got, want = float(_metric(a['cards'][model], metric)), float(_metric(b['cards'][model], metric))
            scores[metric] = abs(got - want)
        report['scores'][model] = scores
        passes &= all(v <= PREDICTION_ATOL for v in scores.values())
    for key in ('delta_rmse',):
        diff = abs(float(a['paired'][key]) - float(b['paired'][key]))
        report['scores'].setdefault('paired', {})[key] = diff
        passes &= diff <= PREDICTION_ATOL
    interval = [abs(float(x) - float(y)) for x, y in
                zip(a['paired']['country_bootstrap_95_interval'], b['paired']['country_bootstrap_95_interval'])]
    report['scores']['paired']['country_bootstrap_95_interval'] = interval
    passes &= all(v <= PREDICTION_ATOL for v in interval)
    full_diff = abs(a['cards'][CANDIDATE]['coefficients'][FEATURE]['full']
                    - b['cards'][CANDIDATE]['coefficients'][FEATURE]['full'])
    fold_diffs = []
    for fit_a, fit_b in zip(a['scored'][CANDIDATE].fits, b['scored'][CANDIDATE].fits):
        if fit_a['feature_identified'] and fit_b['feature_identified']:
            fold_diffs.append(abs(fit_a['coef'][fit_a['columns'].index(FEATURE)]
                                  - fit_b['coef'][fit_b['columns'].index(FEATURE)]))
    report['coefficients'] = {'baseline_dryness_full': full_diff,
                              'baseline_dryness_training_max': max(fold_diffs) if fold_diffs else None,
                              'training_fits_compared': len(fold_diffs)}
    passes &= full_diff <= COEF_ATOL and (not fold_diffs or max(fold_diffs) <= COEF_ATOL)
    report['allocations_may_differ'] = {
        model: {g: {'primary_total_co2': a['cards'][model].get(f'share_{g}'),
                    'per_capita': b['cards'][model].get(f'share_{g}'),
                    'difference': (a['cards'][model].get(f'share_{g}', 0.0)
                                   - b['cards'][model].get(f'share_{g}', 0.0))}
                for g in NAMED_GROUPS if f'share_{g}' in a['cards'][model]}
        for model in (COMPARATOR, CANDIDATE)}
    report['note'] = ('Shapley allocations and the population/intercept coefficients differ by '
                      'construction; they are reported, never required to match, and never select a '
                      'representation.')
    report['passes'] = bool(passes)
    return report


# ---------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------


def prediction_rows(arms, frozen):
    rows = []
    table, y = frozen.table, frozen.design[OUTCOME_COL].to_numpy(float)
    for representation, arm in arms.items():
        for model in (COMPARATOR, CANDIDATE):
            scored = arm['scored'][model]
            base = {'representation': representation, 'model': model}
            for i in range(len(table)):
                rows.append({**base, 'protocol': 'in_sample', 'iso3': table.iso3.iloc[i],
                             'Country': table.Country.iloc[i], 'm49_subregion': table.m49_subregion.iloc[i],
                             'observed': y[i], 'prediction': scored.fitted[i],
                             'error': y[i] - scored.fitted[i], 'abs_error': abs(y[i] - scored.fitted[i]),
                             'fold_id': -1, 'n_train': len(table), 'nearest_train_km': np.nan,
                             'unseen_levels': 0})
            for protocol in PROTOCOLS:
                result = scored.results[protocol]
                for i in range(len(table)):
                    rows.append({**base, 'protocol': protocol, 'iso3': table.iso3.iloc[i],
                                 'Country': table.Country.iloc[i],
                                 'm49_subregion': table.m49_subregion.iloc[i], 'observed': y[i],
                                 'prediction': result.yhat[i], 'error': y[i] - result.yhat[i],
                                 'abs_error': abs(y[i] - result.yhat[i]),
                                 'fold_id': int(result.fold_id[i]), 'n_train': int(result.n_train[i]),
                                 'nearest_train_km': float(result.nearest_train_km[i]),
                                 'unseen_levels': int(result.unseen_levels[i])})
    return pd.DataFrame(rows)


def comparison_rows(arms, frozen):
    rows = []
    table, y = frozen.table, frozen.design[OUTCOME_COL].to_numpy(float)
    for representation, arm in arms.items():
        c0 = arm['scored'][COMPARATOR].results[PRIMARY]
        m1 = arm['scored'][CANDIDATE].results[PRIMARY]
        frame = pd.DataFrame({
            'representation': representation, 'iso3': table.iso3, 'Country': table.Country,
            'm49_subregion': table.m49_subregion, 'observed': y,
            'M0star_cv_prediction': c0.yhat, 'M1b_cv_prediction': m1.yhat,
            'M0star_abs_error': np.abs(y - c0.yhat), 'M1b_abs_error': np.abs(y - m1.yhat),
            'M0star_unseen_levels': c0.unseen_levels, 'M1b_unseen_levels': m1.unseen_levels})
        frame['abs_error_change'] = frame.M1b_abs_error - frame.M0star_abs_error
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def fold_rows(frozen):
    ids, _buffer = frozen.folds[PRIMARY]
    memberships, sizes = [], []
    for _fold, _test, train in cv.iter_folds(ids, frozen.distance, PRIMARY_BUFFER_KM):
        memberships.append('|'.join(frozen.table.iso3[train]))
        sizes.append(int(train.sum()))
    return pd.DataFrame({
        'iso3': frozen.table.iso3, 'Country': frozen.table.Country,
        'primary_fold': ids, 'primary_n_train': sizes,
        'primary_training_iso3': memberships,
        'm49_subregion': frozen.table.m49_subregion,
        'm49_fold': frozen.folds['m49_subregion_lo'][0], 'random10_fold': frozen.folds['random10'][0]})


def coefficient_rows(arms):
    rows = []
    for representation, arm in arms.items():
        for model in (COMPARATOR, CANDIDATE):
            scored = arm['scored'][model]
            for column, value in zip(scored.full.columns, scored.full.coef):
                rows.append({'representation': representation, 'model': model, 'fit': 'full_sample',
                             'held_out_iso3': '', 'fold': -1, 'n_train': len(scored.fitted),
                             'matrix_columns': len(scored.full.columns), 'matrix_rank': '',
                             'feature_identified': '', 'column': column, 'coefficient': float(value)})
            for fit in scored.fits:
                for column, value in zip(fit['columns'], fit['coef']):
                    rows.append({'representation': representation, 'model': model,
                                 'fit': 'primary_training', 'held_out_iso3': fit['held_out_iso3'],
                                 'fold': fit['fold'], 'n_train': fit['n_train'],
                                 'matrix_columns': fit['matrix_columns'], 'matrix_rank': fit['matrix_rank'],
                                 'feature_identified': fit['feature_identified'], 'column': column,
                                 'coefficient': float(value)})
    return pd.DataFrame(rows)


def coalition_frame(arms):
    rows = []
    for representation, arm in arms.items():
        for model in (COMPARATOR, CANDIDATE):
            for row in arm['scored'][model].coalitions:
                rows.append({'representation': representation, 'model': model, **row})
    return pd.DataFrame(rows)


def design_frame(arms, frozen):
    rows = []
    for representation, arm in arms.items():
        for model in (COMPARATOR, CANDIDATE):
            frame = arm['frames'][model]
            matrix = design_matrix(frame)
            names, groups = design_columns(frame)
            for j, (name, group) in enumerate(zip(names, groups)):
                for i, iso3 in enumerate(frozen.table.iso3):
                    rows.append({'representation': representation, 'model': model, 'iso3': iso3,
                                 'column': name, 'group': group, 'value': float(matrix[i, j])})
    return pd.DataFrame(rows)


def json_text(payload):
    return json.dumps(payload, indent=2, allow_nan=False, default=float) + '\n'


def write_csv(frame, path):
    path.write_text(frame.to_csv(index=False, lineterminator='\n'), encoding='utf-8')


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


def build_arms(frozen):
    """Comparator and candidate frames per representation; identical sample, outcome and C2."""
    arms = {}
    for representation, spec in REPRESENTATIONS.items():
        comparator = frozen.design.drop(columns=spec['drop']).reset_index(drop=True)
        candidate = comparator.copy()
        candidate[FEATURE] = frozen.c2
        _require(FEATURE not in comparator.columns, 'comparator must not carry C2')
        _require(spec['responsibility'] in comparator.columns, 'responsibility column')
        arms[representation] = {'frames': {COMPARATOR: comparator, CANDIDATE: candidate}}
    return arms


def score_representations(frozen, *, reference_cards, expected_columns=EXPECTED_COLUMNS):
    """Score both representations under one set of inputs; returns ``(arms, integrity_failures)``.

    Both comparators are scored and checked against their frozen reference cards **before** either
    candidate is fitted (spec 3). ``expected_columns`` is ``None`` for an arm whose categorical
    levels may legitimately differ; the design must still be full column rank.
    """
    arms = build_arms(frozen)
    integrity = []
    for representation, arm in arms.items():
        scored = score_model(arm['frames'][COMPARATOR], frozen,
                             expected_columns=None if expected_columns is None
                             else expected_columns['comparator'])
        arm['scored'] = {COMPARATOR: scored}
        arm['cards'] = {COMPARATOR: scored.card}
        arm['reference'] = (reference_check(scored.card, reference_cards[representation]['card'],
                                            f'{representation} comparator',
                                            metrics=reference_cards[representation]['metrics'])
                            if reference_cards else None)
    for representation, arm in arms.items():
        scored = score_model(arm['frames'][CANDIDATE], frozen,
                             expected_columns=None if expected_columns is None
                             else expected_columns['candidate'])
        arm['scored'][CANDIDATE] = scored
        arm['cards'][CANDIDATE] = scored.card
        y = frozen.design[OUTCOME_COL].to_numpy(float)
        arm['paired'] = paired_rmse_interval(y, scored.results[PRIMARY].yhat,
                                             arm['scored'][COMPARATOR].results[PRIMARY].yhat)
        try:
            sign = sign_condition(scored.card['coefficients'], frozen.n)
            arm['verdict'] = verdict(arm['cards'][COMPARATOR], arm['cards'][CANDIDATE],
                                     arm['paired'], sign)
        except ValueError as error:
            arm['verdict'] = {'verdict': None, 'refused': str(error)}
            integrity.append(f'{representation}: {error}')
        arm['diagnostics'] = diagnostics(arm['cards'][COMPARATOR], arm['cards'][CANDIDATE], arm['paired'])
        for model in (COMPARATOR, CANDIDATE):
            card = arm['cards'][model]
            if not card['share_accounting']['passes']:
                integrity.append(f'{representation} {model}: share accounting outside {SHARE_ATOL}')
            if card['training_fits']['all_full_rank'] is False:
                integrity.append(f'{representation} {model}: a primary training fit is rank deficient')
            if card['matrix_rank'] != card['matrix_columns']:
                integrity.append(f'{representation} {model}: the full-sample design is rank deficient')
        if arm['cards'][CANDIDATE]['training_fits']['feature_identified_in_all'] is False:
            integrity.append(f'{representation}: {FEATURE} is not identified in every training fit')
    return arms, integrity


def support_context(package_dir=OUTPUT_DIR):
    """Contract §2.5 provenance QA, reported alongside the result (context, never a condition)."""
    checkpoint = json.loads((Path(package_dir) / 'm1b_support_checkpoint.json').read_text())
    extremes = checkpoint['extremes_top10']
    return {'pre_station_support_distribution': checkpoint['distributions'],
            'lowest_pre_station_supported_share': extremes['lowest_station_supported_share'][:5],
            'highest_pure_climatology_share': extremes['highest_pure_climatology_share'][:5],
            'pet_limitation': checkpoint['pet_limitation'],
            'no_exclusion_threshold': 'no station-count threshold excludes, reweights or adjusts any '
                                      'country; these quantities are reported only'}


def non_deciding_conditions(entry):
    """A non-deciding representation reports the same Booleans without a §4.4 verdict label.

    Contract §4.4 reports exactly one verdict, and A3.6 gives acceptance to the primary total-CO2
    representation alone, so the per-capita arm carries a descriptive label, exactly as a conditional
    robustness arm does.
    """
    conditions = dict(entry)
    label = conditions.pop('verdict', None)
    return {**conditions, 'label_descriptive_only': label,
            'note': 'the per-capita representation is reported unconditionally (Amendment 3 A3.6) '
                    'and never determines acceptance or promotion'}


def redundancy_context_from(record):
    """§3 context, taken from the verified redundancy record rather than transcribed by hand."""
    return {'r2_full_m0star': record['r2_full_m0star'],
            'adjusted_r2_full_m0star': record['adjusted_r2_full_m0star'],
            'vif': record['vif'],
            'r2_by_group_alone': record['r2_by_group_alone'],
            'rank_without_c2': record['design_rank'], 'rank_with_c2': record['rank_with_c2'],
            'near_redundant_flag': record['near_redundant_flag'],
            'note': 'C2 is not independent of geography: most of its cross-country variance that the '
                    'M0* design spans is geography. The diagnostic says nothing about warming.'}


def evaluate(frozen, out_dir, *, provenance, reference_cards, support_qa=None,
             redundancy_context=None):
    """Score both representations, apply the frozen verdict and write the evidence.

    The output root is created only once the comparators have reproduced their frozen cards and both
    candidates have been scored, so a pre-scoring refusal leaves no directory behind.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f'{out_dir} already exists; a run never overwrites a prior run')
    arms, integrity = score_representations(frozen, reference_cards=reference_cards)
    out_dir.mkdir(parents=True, exist_ok=False)
    equiv = equivalence(arms)
    if not equiv['passes']:
        integrity.append('representation equivalence outside the prespecified tolerance')
    comparison = comparison_rows(arms, frozen)
    result = {
        'contract': CONTRACT_PATH, 'specification': SPEC_PATH,
        'primary_representation': 'primary_total_co2',
        'acceptance_rule': 'only the primary total-CO2 representation determines the verdict '
                           '(Amendment 3 A3.6); the per-capita representation is reported unconditionally',
        'representations': {
            representation: {
                'decides_verdict': REPRESENTATIONS[representation]['decides_verdict'],
                'comparator_reference': arm['reference'],
                'M0star': arm['cards'][COMPARATOR], 'M1b': arm['cards'][CANDIDATE],
                'paired_primary_comparison': arm['paired'],
                # Exactly one §4.4 verdict is reported: the primary representation's. The per-capita
                # arm carries the same Booleans under a descriptive label (A3.6).
                **({'verdict': arm['verdict']} if REPRESENTATIONS[representation]['decides_verdict']
                   else {'representation_conditions': non_deciding_conditions(arm['verdict'])}),
                'diagnostics': arm['diagnostics'],
                'countries_improved': int((comparison[comparison.representation == representation]
                                           .abs_error_change < 0).sum()),
                'countries_worsened': int((comparison[comparison.representation == representation]
                                           .abs_error_change > 0).sum()),
            } for representation, arm in arms.items()},
        'representation_equivalence': equiv,
        'redundancy_context': redundancy_context,
        'provenance_limitation': 'C2 is CRU-reconstructed 1920-1949 baseline hydroclimatic dryness over '
                                 'resolved support, not a predictor built only from information '
                                 'available before 1950.',
        'support_quality_context': support_qa,
        'integrity_failures': integrity,
        'integrity_passed': not integrity,
    }
    result['verdict'] = result['representations']['primary_total_co2']['verdict']

    write_csv(prediction_rows(arms, frozen), out_dir / 'm1b_country_predictions.csv')
    write_csv(comparison, out_dir / 'm1b_primary_country_comparison.csv')
    write_csv(fold_rows(frozen), out_dir / 'm1b_cv_folds.csv')
    write_csv(coefficient_rows(arms), out_dir / 'm1b_coefficients.csv')
    write_csv(coalition_frame(arms), out_dir / 'm1b_shapley_coalitions.csv')
    write_csv(design_frame(arms, frozen), out_dir / 'm1b_design_matrices.csv')
    (out_dir / 'm1b_coefficient_summary.json').write_text(json_text({
        representation: {model: arm['cards'][model]['coefficients'] for model in (COMPARATOR, CANDIDATE)}
        for representation, arm in arms.items()}), encoding='utf-8')
    (out_dir / 'm1b_scorecard.json').write_text(json_text(result), encoding='utf-8')
    (out_dir / 'm1b_provenance.json').write_text(json_text(
        {**provenance, 'identity': frozen.identity}), encoding='utf-8')
    manifest = {
        'result': 'M1b primary and unconditional per-capita evaluation',
        'contract': CONTRACT_PATH, 'specification': SPEC_PATH,
        'evaluator_commit': provenance.get('code', {}).get('commit'),
        'verdict': result['verdict'].get('verdict'),
        'association_supported': result['verdict'].get('association_supported'),
        'integrity_passed': result['integrity_passed'],
        'artifact_sha256': {name: hydro.sha256(out_dir / name) for name in DETERMINISTIC_ARTIFACTS},
        'non_deterministic_artifacts': [RUN_METADATA],
    }
    (out_dir / RESULT_MANIFEST).write_text(json_text(manifest), encoding='utf-8')
    result['result_manifest'] = manifest
    return result


def reproducibility_record(run_a, run_b, *, code=None):
    """Byte-compare two unchanged-code runs' deterministic artifacts (spec §8).

    Run metadata is deliberately excluded: it is the one artifact allowed to differ.
    """
    a, b = Path(run_a), Path(run_b)
    artifacts, identical = {}, True
    for name in (*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST):
        digest_a, digest_b = hydro.sha256(a / name), hydro.sha256(b / name)
        artifacts[name] = {'run_a_sha256': digest_a, 'run_b_sha256': digest_b,
                           'identical': digest_a == digest_b}
        identical &= digest_a == digest_b
    return {
        'comparison': 'unchanged-code rerun in a separate process and output root',
        'run_a': str(a), 'run_b': str(b),
        'compared': [*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST],
        'excluded_as_run_metadata': [RUN_METADATA],
        'artifacts': artifacts, 'byte_identical': bool(identical),
        'code': code,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score M1b under the frozen contract.')
    parser.add_argument('--out', default=str(DEFAULT_OUT), help='fresh output root (never overwritten)')
    parser.add_argument('--compare', nargs=2, metavar=('RUN_A', 'RUN_B'),
                        help='byte-compare two completed runs instead of scoring')
    parser.add_argument('--record', help='where to write the --compare record')
    args = parser.parse_args(argv)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    if args.compare:
        record = reproducibility_record(*args.compare, code=frozen_code_state())
        if args.record:
            Path(args.record).write_text(json_text(record), encoding='utf-8')
        print(json_text({'byte_identical': record['byte_identical'],
                         'differing': [n for n, v in record['artifacts'].items() if not v['identical']]}))
        return record
    require_hash_seed()
    code = frozen_code_state()
    out_dir = Path(args.out)
    if out_dir.exists():
        raise RuntimeError(f'{out_dir} already exists; a run never overwrites a prior run')
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    inputs = verify_scoring_inputs()
    frozen = build_frozen(inputs)
    rank_audit = json.loads((ROOT / RANK_AUDIT).read_text())['cards']
    reference_cards = {r: {'card': rank_audit[spec['reference_card']], 'metrics': REFERENCE_METRICS}
                       for r, spec in REPRESENTATIONS.items()}
    provenance = {
        'contract_sha256': hydro.sha256(ROOT / CONTRACT_PATH),
        'specification_sha256': hydro.sha256(ROOT / SPEC_PATH),
        # The upstream head moves independently of this run, so it belongs to run metadata, not to
        # the deterministic provenance of the scored result.
        'code': {k: v for k, v in code.items() if k != 'upstream_commit'},
        'inputs': inputs.digests,
        'measurement_manifest_sha256': inputs.manifest_sha256,
        'measurement_build_commit': inputs.manifest['git']['commit'],
        'redundancy_record_sha256': ACCEPTED_PACKAGE[red.RECORD],
        'reference_cards': {r: spec['reference_card'] for r, spec in REPRESENTATIONS.items()},
        'protocols': {'primary': 'corrected 500 km footprint-certified LOCO (4263429 bounds)',
                      'secondary': 'leave one M49 subregion out', 'reference_only': 'random 10-fold, seed 0'},
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'platform': platform.platform(), 'machine': platform.machine()},
    }
    result = evaluate(frozen, out_dir, provenance=provenance, reference_cards=reference_cards,
                      support_qa=support_context(),
                      redundancy_context=redundancy_context_from(inputs.record))
    (out_dir / RUN_METADATA).write_text(json_text({
        'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'wall_seconds': time.perf_counter() - started, 'host': platform.node(),
        'executable': sys.executable, 'argv': sys.argv, 'output_root': str(out_dir),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'upstream': code['upstream'], 'upstream_commit': code['upstream_commit'],
        'note': 'run metadata is deliberately outside the byte-compared deterministic artifacts',
    }), encoding='utf-8')
    print(json_text({'verdict': result['verdict'], 'integrity_failures': result['integrity_failures']}))
    return result


def cli_exit_code(outcome):
    """Process exit status for a :func:`main` outcome (post-M1b auxiliary CLI maintenance).

    Scoring returns a result carrying ``integrity_passed`` (0 when it holds, else 3, as frozen at
    ``618c8b8``); ``--compare`` returns a reproducibility record carrying ``byte_identical`` (0 when
    the runs are byte-identical, else 4). The shapes are disjoint, so an outcome with neither key,
    both keys, or a non-boolean flag is malformed and raises instead of exiting 0.
    """
    if not isinstance(outcome, dict) or ('integrity_passed' in outcome) == ('byte_identical' in outcome):
        keys = sorted(outcome) if isinstance(outcome, dict) else type(outcome).__name__
        raise RuntimeError(f'unrecognised CLI outcome: expected exactly one of integrity_passed '
                           f'(scoring) or byte_identical (--compare), found {keys}')
    key, failure = ('integrity_passed', 3) if 'integrity_passed' in outcome else ('byte_identical', 4)
    if not isinstance(outcome[key], bool):
        raise RuntimeError(f'{key} must be a bool, found {type(outcome[key]).__name__}')
    return 0 if outcome[key] else failure


if __name__ == '__main__':
    sys.exit(cli_exit_code(main()))
