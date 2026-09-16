"""The strict M1b scoring-input gate: exact accepted artifacts, field validation and frame identity.

Packages here are synthetic unless a test explicitly checks the accepted package's digests. No model
is fitted: these tests exercise refusals that must fire before any target column is used.
"""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m1b_evaluate as ev
from research.model_v2 import m1b_hydroclimate as hydro
from research.model_v2 import m1b_redundancy as red
from research.model_v2.m0 import DEFAULT_FEATURES_PATH, DEFAULT_INEQUALITY_PATH, ROOT

FROZEN = pd.read_csv(red.SUPPORT_RECORD, usecols=['iso3', 'Country'])
REAL_INPUTS = DEFAULT_INEQUALITY_PATH.exists() and DEFAULT_FEATURES_PATH.exists()
needs_inputs = pytest.mark.skipif(not REAL_INPUTS, reason='requires the local frozen inputs')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_package(out, c2=None, manifest_overrides=None, record_overrides=None,
                  provenance_overrides=None):
    """A synthetic package whose manifest, record and artifacts are mutually consistent."""
    out.mkdir(parents=True, exist_ok=True)
    values = np.linspace(-1.0, 0.5, len(FROZEN)) if c2 is None else np.asarray(c2, dtype=float)
    features = FROZEN.assign(baseline_dryness=values)
    texts = {hydro.FEATURES: features.to_csv(index=False, lineterminator='\n'),
             'm1b_hydroclimate_qa.csv': 'iso3,coverage\nAAA,1.0\n',
             'm1b_harmonization_cells.csv': 'target_lat_index\n1\n',
             'm1b_unresolved_support.csv': 'target_lat_index\n2\n',
             'm1b_support_checkpoint.json': json.dumps({'extremes_top10': {
                 'lowest_station_supported_share': [[code, 0.1 * i] for i, code in
                                                    enumerate(['SAU', 'YEM', 'HTI', 'OMN', 'TCD'])]}}) + '\n'}
    for name, text in texts.items():
        (out / name).write_text(text, encoding='utf-8')
    country_list = sha(('\n'.join(features.iso3) + '\n').encode())
    manifest = {
        'contract': ev.CONTRACT_PATH, 'contract_freeze_commit': hydro.CONTRACT_COMMIT,
        'amendments': list(hydro.AMENDMENTS), 'gate_version': hydro.GATE_VERSION,
        'git': {'commit': ev.MEASUREMENT_BUILD_COMMIT, 'code_path_matches_commit': True},
        'formula': 'x_j = log10(P/E)', 'window': list(hydro.WINDOW), 'n_months': hydro.N_MONTHS,
        'coverage_min': hydro.COVERAGE_MIN, 'code_sha256': 'e' * 64,
        'support_inputs': {'country_list_sha256': country_list},
        'hard_stops': [], 'all_hard_stops_pass': True,
        'artifact_sha256': {n: sha(t.encode('utf-8')) for n, t in texts.items()},
    }
    manifest.update(manifest_overrides or {})
    (out / hydro.MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    manifest_sha = sha((out / hydro.MANIFEST).read_bytes())
    provenance = {
        'measurement_manifest_sha256': manifest_sha,
        'c2_features_sha256': manifest['artifact_sha256'][hydro.FEATURES],
        'measurement_git_commit': manifest['git']['commit'],
        'measurement_code_sha256': manifest['code_sha256'],
        'measurement_gate_version': manifest['gate_version'],
        'country_list_sha256': manifest['support_inputs']['country_list_sha256'],
        'predictor_columns': red.PREDICTOR_COLUMNS, 'outcome_column_present': False,
        'predictor_input_sha256': {str(p.resolve().relative_to(ROOT)): d
                                   for p, d in red.PREDICTOR_INPUT_SHA256.items()},
        'predictor_frame_sha256': 'a' * 64, 'design_matrix_sha256': 'b' * 64,
        'c2_vector_sha256': ev.array_digest(values), 'redundancy_code_sha256': 'd' * 64,
        'design_columns': ['intercept'] + [f'x{i}' for i in range(19)],
    }
    provenance.update(provenance_overrides or {})
    record = {'status': 'pass', 'all_hard_stops_pass': True, 'hard_stops': [],
              'design_column_count': 20, **ev.ACCEPTED_REDUNDANCY, 'provenance': provenance}
    record.update(record_overrides or {})
    (out / red.RECORD).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return manifest, record


def pins(out):
    return {name: hydro.sha256(out / name) for name in ev.ACCEPTED_PACKAGE}


def accept(monkeypatch, out):
    monkeypatch.setattr(ev, 'ACCEPTED_PACKAGE', pins(out))


# --- the accepted package ---------------------------------------------------------------------

def test_the_accepted_package_digests_are_the_committed_ones():
    accepted = {hydro.FEATURES: 'cf33ba7cfb87c2184a94df5e5686f0951adf692819af8de27329ea698d72e1f4',
                hydro.MANIFEST: '1974686199b9c7960ead2b105e6225f34b9b6fb33627858d2bc15239d7de4c9c',
                red.RECORD: '9acb680c09b1577592ca9c4d62f992f6dab9628d581bb58b4448ad36b80f7585'}
    assert ev.ACCEPTED_PACKAGE == accepted
    for name, digest in accepted.items():
        assert hydro.sha256(hydro.OUT / name) == digest


def test_an_internally_consistent_but_unapproved_package_is_rejected(tmp_path):
    write_package(tmp_path)
    hydro.verify_package(tmp_path)          # self-consistent...
    with pytest.raises(ValueError, match='not the accepted package digest'):
        ev.verify_scoring_inputs(tmp_path)  # ...but not the accepted measurement package


@needs_inputs
def test_a_consistent_accepted_package_passes_the_gate(tmp_path, monkeypatch):
    manifest, _record = write_package(tmp_path)
    accept(monkeypatch, tmp_path)
    inputs = ev.verify_scoring_inputs(tmp_path)
    assert inputs.manifest['gate_version'] == hydro.GATE_VERSION
    assert inputs.features_bytes == (tmp_path / hydro.FEATURES).read_bytes()
    assert inputs.record['status'] == 'pass'
    assert inputs.digests['frozen_scoring_inputs'][ev.COUNTRY_TABLE]['frozen_at'] == 'af97bd2'


@needs_inputs
@pytest.mark.parametrize('tamper', ['c2', 'manifest', 'redundancy'])
def test_altered_c2_stale_manifest_or_stale_redundancy_are_rejected(tmp_path, monkeypatch, tamper):
    write_package(tmp_path)
    accept(monkeypatch, tmp_path)
    if tamper == 'c2':
        frame = pd.read_csv(tmp_path / hydro.FEATURES)
        frame.loc[0, 'baseline_dryness'] += 1e-12
        frame.to_csv(tmp_path / hydro.FEATURES, index=False, lineterminator='\n')
    elif tamper == 'manifest':
        manifest = json.loads((tmp_path / hydro.MANIFEST).read_text())
        manifest['code_sha256'] = 'f' * 64
        (tmp_path / hydro.MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        record = json.loads((tmp_path / red.RECORD).read_text())
        record['provenance']['measurement_manifest_sha256'] = '0' * 64
        (tmp_path / red.RECORD).write_text(json.dumps(record, indent=2) + '\n')
    with pytest.raises(ValueError):
        ev.verify_scoring_inputs(tmp_path)


@pytest.mark.parametrize('mutate', [
    lambda m: m.pop('gate_version'),
    lambda m: m.pop('hard_stops'),
    lambda m: m.update(hard_stops=[{'iso3': 'PER'}]),
    lambda m: m.update(all_hard_stops_pass='true'),
    lambda m: m.update(n_months='360'),
    lambda m: m.update(n_months=True),
    lambda m: m.update(coverage_min=0.95),
    lambda m: m.update(window=['1920-01', '1950-12']),
    lambda m: m.update(contract_freeze_commit='0' * 40),
    lambda m: m['git'].update(commit='0' * 40),
    lambda m: m['git'].update(code_path_matches_commit=False),
    lambda m: m.pop('support_inputs'),
    lambda m: m['support_inputs'].pop('country_list_sha256'),
    lambda m: m['artifact_sha256'].pop('m1b_unresolved_support.csv'),
    lambda m: m.pop('amendments'),
])
def test_missing_malformed_or_wrong_identity_manifest_fields_are_rejected(mutate):
    manifest = {
        'contract_freeze_commit': hydro.CONTRACT_COMMIT, 'amendments': list(hydro.AMENDMENTS),
        'gate_version': hydro.GATE_VERSION, 'formula': 'x', 'window': list(hydro.WINDOW),
        'n_months': hydro.N_MONTHS, 'coverage_min': hydro.COVERAGE_MIN, 'code_sha256': 'e' * 64,
        'git': {'commit': ev.MEASUREMENT_BUILD_COMMIT, 'code_path_matches_commit': True},
        'support_inputs': {'country_list_sha256': 'c' * 64}, 'hard_stops': [],
        'all_hard_stops_pass': True,
        'artifact_sha256': {name: '0' * 64 for name in hydro.PACKAGE_ARTIFACTS},
    }
    ev._require_manifest_fields(manifest)
    mutate(manifest)
    with pytest.raises(ValueError):
        ev._require_manifest_fields(manifest)


@needs_inputs
@pytest.mark.parametrize('override', [
    {'status': 'hard_stop'}, {'all_hard_stops_pass': False}, {'hard_stops': ['x']},
    {'design_rank': 19}, {'rank_with_c2': 20}, {'near_redundant_flag': True},
    {'r2_full_m0star': 0.65}, {'vif': 2.85}, {'adjusted_r2_full_m0star': 0.6},
])
def test_a_redundancy_record_with_different_recorded_diagnostics_is_rejected(tmp_path, monkeypatch, override):
    write_package(tmp_path, record_overrides=override)
    accept(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        ev.verify_scoring_inputs(tmp_path)


@needs_inputs
@pytest.mark.parametrize('override', [
    {'outcome_column_present': True}, {'predictor_columns': ['Country']},
    {'measurement_git_commit': '0' * 40}, {'measurement_gate_version': 'Amendment 1'},
    {'country_list_sha256': '0' * 64}, {'predictor_frame_sha256': 'short'},
    {'predictor_input_sha256': {}}, {'design_columns': ['not_intercept']},
])
def test_a_redundancy_record_with_wrong_source_identity_is_rejected(tmp_path, monkeypatch, override):
    write_package(tmp_path, provenance_overrides=override)
    accept(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        ev.verify_scoring_inputs(tmp_path)


# --- frame identity ----------------------------------------------------------------------------

@needs_inputs
def test_the_real_package_and_frozen_inputs_build_the_frozen_frame():
    inputs = ev.verify_scoring_inputs()
    frozen = ev.build_frozen(inputs)
    assert frozen.n == 151 and frozen.design.Country.tolist() == frozen.table.Country.tolist()
    assert frozen.identity['outcome_source_column'] == 'trend_c_per_decade_area_weighted'
    assert frozen.identity['representation_identity_max_abs_error'] <= ev.IDENTITY_ATOL
    assert frozen.identity['c2_vector_sha256'] == inputs.record['provenance']['c2_vector_sha256']
    assert set(frozen.folds) == {ev.PRIMARY, 'm49_subregion_lo', 'random10'}


@needs_inputs
@pytest.mark.parametrize('tamper', ['reordered', 'duplicated', 'missing', 'value'])
def test_reordered_duplicated_or_missing_countries_are_rejected(tamper):
    inputs = ev.verify_scoring_inputs()
    features = pd.read_csv(io.BytesIO(inputs.features_bytes))
    if tamper == 'reordered':
        features = pd.concat([features.iloc[1:2], features.iloc[0:1], features.iloc[2:]])
    elif tamper == 'duplicated':
        features = pd.concat([features.iloc[0:1], features.iloc[1:]])
        features.iloc[0, features.columns.get_loc('iso3')] = features.iso3.iloc[1]
    elif tamper == 'missing':
        features = features.iloc[1:]
    else:
        features = features.copy()
        features.loc[0, 'baseline_dryness'] += 1e-9
    payload = features.to_csv(index=False, lineterminator='\n').encode('utf-8')
    with pytest.raises(ValueError):
        ev.build_frozen(ev.ScoringInputs(inputs.manifest, inputs.manifest_sha256, inputs.record,
                                         payload, inputs.digests))


@needs_inputs
def test_altered_territorial_bounds_are_rejected(monkeypatch):
    inputs = ev.verify_scoring_inputs()
    real = ev.territory.load_bounds

    def perturbed(codes, *args, **kwargs):
        bounds = real(codes, *args, **kwargs).copy()
        bounds[0, 1] = bounds[1, 0] = 499.0
        return bounds
    monkeypatch.setattr(ev.territory, 'load_bounds', perturbed)
    with pytest.raises(ValueError, match='primary fold memberships'):
        ev.build_frozen(inputs)


@needs_inputs
def test_altered_m49_labels_are_rejected(monkeypatch):
    inputs = ev.verify_scoring_inputs()
    real = ev.m49_table

    def relabelled():
        table = real().copy()
        table.loc[table.iso3 == 'DZA', 'm49_subregion'] = 'Elsewhere'
        return table
    monkeypatch.setattr(ev, 'm49_table', relabelled)
    with pytest.raises(ValueError, match='M49 labels'):
        ev.build_frozen(inputs)


@needs_inputs
def test_a_changed_outcome_vector_is_rejected(monkeypatch):
    inputs = ev.verify_scoring_inputs()
    real = ev.m0_complete_design

    def shifted(*args, **kwargs):
        design = real(*args, **kwargs).copy()
        design.loc[0, 'warming_trend'] += 1e-12
        return design
    monkeypatch.setattr(ev, 'm0_complete_design', shifted)
    with pytest.raises(ValueError, match='frozen ordered outcome record'):
        ev.build_frozen(inputs)


@needs_inputs
def test_a_changed_frozen_scoring_input_is_rejected(monkeypatch):
    pinned = dict(ev.FROZEN_SCORING_INPUTS)
    pinned[ev.RANK_AUDIT] = ('0' * 64, '9fb56e1')
    monkeypatch.setattr(ev, 'FROZEN_SCORING_INPUTS', pinned)
    with pytest.raises(ValueError, match='differs from the record frozen at'):
        ev.verify_scoring_inputs()


# --- the frozen baseline anchor ------------------------------------------------------------------

@needs_inputs
def test_the_comparators_reproduce_the_frozen_rank_audit_cards():
    """M0* must reproduce its frozen card (contract §4.1). No C2 is added, so nothing is fitted to it."""
    inputs = ev.verify_scoring_inputs()
    frozen = ev.build_frozen(inputs)
    cards = json.loads((ROOT / ev.RANK_AUDIT).read_text())['cards']
    for representation, spec in ev.REPRESENTATIONS.items():
        comparator = frozen.design.drop(columns=spec['drop']).reset_index(drop=True)
        assert ev.FEATURE not in comparator.columns
        scored = ev.score_model(comparator, frozen,
                                expected_columns=ev.EXPECTED_COLUMNS['comparator'])
        check = ev.reference_check(scored.card, cards[spec['reference_card']], representation)
        assert check['reproduced']
        assert max(m['abs_difference'] for m in check['metrics'].values()) <= ev.REFERENCE_ATOL
        fits = scored.card['training_fits']
        assert fits['n'] == 151 and fits['all_full_rank'] and fits['min_n_train'] >= 2
        # The full-sample expectation is 20 columns, but training folds legitimately encode fewer.
        assert min(fits['column_counts']) < ev.EXPECTED_COLUMNS['comparator']
        assert scored.card['rows_with_unseen_level'] == cards[spec['reference_card']]['rows_with_unseen_level']
    with pytest.raises(ValueError, match='does not reproduce the frozen card'):
        ev.reference_check(scored.card, {**cards['drop_per_capita'], 'cv_rmse': 0.05}, 'tampered')


@needs_inputs
def test_a_candidate_design_on_real_predictors_is_full_rank_at_twenty_one_columns():
    """Structural only: C2 is replaced by a deterministic placeholder, never the real values."""
    inputs = ev.verify_scoring_inputs()
    frozen = ev.build_frozen(inputs)
    placeholder = np.linspace(-1.0, 1.0, frozen.n)
    frame = frozen.design.drop(columns='cum_co2_per_capita').reset_index(drop=True)
    frame[ev.FEATURE] = placeholder
    matrix = ev.design_matrix(frame)
    assert matrix.shape[1] == ev.EXPECTED_COLUMNS['candidate']
    assert np.linalg.matrix_rank(matrix) == ev.EXPECTED_COLUMNS['candidate']
    names, groups = ev.design_columns(frame)
    assert names[-1] == ev.FEATURE and groups[-1] == 'hydroclimate'
