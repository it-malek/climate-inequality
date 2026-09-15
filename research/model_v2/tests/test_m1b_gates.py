"""M1b Amendment 3 gates: package integrity, strict gate records and outcome-free redundancy loading.

Every package here is synthetic. The instrumented redundancy run reads the frozen predictor inputs with
synthetic C2 values; no warming outcome, no real C2 and no evaluator scoring is involved.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m0
from research.model_v2 import m1b_hydroclimate as h
from research.model_v2 import m1b_redundancy as r

FROZEN = pd.read_csv(r.SUPPORT_RECORD, usecols=['iso3', 'Country'])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_package(out, c2=None, **manifest_overrides):
    """A synthetic measurement package in ``out`` whose manifest digests match its artifacts."""
    out.mkdir(parents=True, exist_ok=True)
    values = np.linspace(-1.0, 0.5, len(FROZEN)) if c2 is None else c2
    texts = {h.FEATURES: FROZEN.assign(baseline_dryness=values).to_csv(index=False, lineterminator='\n'),
             'm1b_hydroclimate_qa.csv': 'iso3,coverage\nAAA,1.0\n',
             'm1b_harmonization_cells.csv': 'target_lat_index\n1\n',
             'm1b_unresolved_support.csv': 'target_lat_index\n2\n',
             'm1b_support_checkpoint.json': '{}\n'}
    for name, text in texts.items():
        (out / name).write_text(text, encoding='utf-8')
    manifest = {'gate_version': h.GATE_VERSION, 'all_hard_stops_pass': True, 'hard_stops': [],
                'git': {'commit': 'f' * 40, 'code_path_matches_commit': True}, 'code_sha256': 'c' * 64,
                'support_inputs': {'country_list_sha256': sha(('\n'.join(FROZEN.iso3) + '\n').encode())},
                'artifact_sha256': {n: sha(t.encode('utf-8')) for n, t in texts.items()}}
    manifest.update(manifest_overrides)
    (out / h.MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def rewrite_manifest(out, mutate):
    manifest = json.loads((out / h.MANIFEST).read_text())
    mutate(manifest)
    (out / h.MANIFEST).write_text(json.dumps(manifest), encoding='utf-8')


# --- measurement package ------------------------------------------------------------------------

def test_a_consistent_passing_package_verifies(tmp_path):
    write_package(tmp_path)
    manifest, digest = h.verify_package(tmp_path)
    assert digest == sha((tmp_path / h.MANIFEST).read_bytes()) and manifest['hard_stops'] == []


def test_changed_c2_with_a_stale_manifest_fails(tmp_path):
    write_package(tmp_path)
    frame = pd.read_csv(tmp_path / h.FEATURES)
    frame.loc[0, 'baseline_dryness'] += 1e-12
    frame.to_csv(tmp_path / h.FEATURES, index=False, lineterminator='\n')
    with pytest.raises(ValueError, match='digest'):
        h.verify_package(tmp_path)


@pytest.mark.parametrize('mutate', [
    lambda m: m.pop('hard_stops'),
    lambda m: m.pop('all_hard_stops_pass'),
    lambda m: m.update(all_hard_stops_pass='true'),
    lambda m: m.update(all_hard_stops_pass=1),
    lambda m: m.update(hard_stops=[{'iso3': 'PER', 'rule': 'x'}]),
    lambda m: m.update(hard_stops=None),
    lambda m: m.update(gate_version='M1B_EVALUATION_CONTRACT.md Amendment 1'),
    lambda m: m['git'].update(code_path_matches_commit=False),
    lambda m: m.pop('git'),
    lambda m: m['artifact_sha256'].pop('m1b_unresolved_support.csv'),
    lambda m: m['artifact_sha256'].update(extra='0' * 64),
    lambda m: m.pop('artifact_sha256'),
])
def test_missing_malformed_or_failing_manifest_fields_fail_closed(tmp_path, mutate):
    write_package(tmp_path)
    rewrite_manifest(tmp_path, mutate)
    with pytest.raises(ValueError):
        h.verify_package(tmp_path)


def test_non_object_or_missing_manifest_and_missing_artifact_fail(tmp_path):
    write_package(tmp_path)
    (tmp_path / 'm1b_support_checkpoint.json').unlink()
    with pytest.raises(FileNotFoundError):
        h.verify_package(tmp_path)
    (tmp_path / h.MANIFEST).write_text('[]')
    with pytest.raises(ValueError):
        h.verify_package(tmp_path)
    (tmp_path / h.MANIFEST).unlink()
    with pytest.raises(FileNotFoundError):
        h.verify_package(tmp_path)


def test_the_stopped_amendment1_build_cannot_pass_the_gate():
    with pytest.raises(ValueError):
        h.verify_package(h.OUT / 'm1b_amendment1_stopped_build')


# --- redundancy record and the evaluator gate ----------------------------------------------------

def passing_record(out, manifest_sha, features_sha, **overrides):
    record = {'status': 'pass', 'all_hard_stops_pass': True, 'hard_stops': [],
              'provenance': {'measurement_manifest_sha256': manifest_sha, 'c2_features_sha256': features_sha}}
    record.update(overrides)
    (out / r.RECORD).write_text(json.dumps(record), encoding='utf-8')


def test_scoring_gate_accepts_only_a_redundancy_record_from_the_current_package(tmp_path):
    from research.model_v2.m1b_evaluate import scoring_gate
    manifest = write_package(tmp_path)
    manifest_sha = sha((tmp_path / h.MANIFEST).read_bytes())
    features_sha = manifest['artifact_sha256'][h.FEATURES]
    with pytest.raises(FileNotFoundError):
        scoring_gate(tmp_path)
    passing_record(tmp_path, manifest_sha, features_sha)
    assert scoring_gate(tmp_path) == (tmp_path / h.FEATURES).read_bytes()
    for overrides in [{'provenance': {'measurement_manifest_sha256': '0' * 64, 'c2_features_sha256': features_sha}},
                      {'provenance': {'measurement_manifest_sha256': manifest_sha, 'c2_features_sha256': '0' * 64}},
                      {'provenance': None}, {'hard_stops': None}, {'hard_stops': ['exact rank redundancy']},
                      {'status': 'hard_stop'}, {'all_hard_stops_pass': 'true'}]:
        passing_record(tmp_path, manifest_sha, features_sha, **overrides)
        with pytest.raises(ValueError):
            scoring_gate(tmp_path)
    passing_record(tmp_path, manifest_sha, features_sha)
    write_package(tmp_path, c2=np.linspace(-2.0, 1.0, len(FROZEN)))   # a rebuilt package makes the record stale
    with pytest.raises(ValueError):
        scoring_gate(tmp_path)


@pytest.mark.parametrize('tamper', ['features', 'unresolved', 'failing_manifest'])
def test_scoring_gate_verifies_the_package_itself_not_only_the_record(tmp_path, tamper):
    from research.model_v2.m1b_evaluate import scoring_gate
    manifest = write_package(tmp_path)
    if tamper == 'failing_manifest':
        manifest = write_package(tmp_path, all_hard_stops_pass=False, hard_stops=[{'iso3': 'PER', 'rule': 'x'}])
    passing_record(tmp_path, sha((tmp_path / h.MANIFEST).read_bytes()), manifest['artifact_sha256'][h.FEATURES])
    if tamper == 'features':
        frame = pd.read_csv(tmp_path / h.FEATURES)
        frame.loc[3, 'baseline_dryness'] += 1e-12
        frame.to_csv(tmp_path / h.FEATURES, index=False, lineterminator='\n')
    elif tamper == 'unresolved':
        (tmp_path / 'm1b_unresolved_support.csv').write_text('target_lat_index\n3\n', encoding='utf-8')
    with pytest.raises(ValueError):
        scoring_gate(tmp_path)


def test_json_safe_keeps_non_finite_statistics_explicit():
    safe = r.json_safe({'a': np.inf, 'b': -np.inf, 'c': np.nan, 'd': np.float64(1.5), 'e': [np.int64(3), np.bool_(True)]})
    assert safe == {'a': 'inf', 'b': '-inf', 'c': 'nan', 'd': 1.5, 'e': [3, True]}
    json.dumps(safe, allow_nan=False)


def test_predictor_input_pins_fail_closed(tmp_path):
    path = tmp_path / 'x.parquet'
    path.write_bytes(b'abc')
    assert list(r.check_predictor_inputs({path: sha(b'abc')}).values()) == [sha(b'abc')]
    with pytest.raises(ValueError):
        r.check_predictor_inputs({path: '0' * 64})


# --- the actual redundancy entry point, instrumented, on synthetic C2 ------------------------------

OPENS = {'active': False, 'paths': []}


def _audit(event, args):
    if OPENS['active'] and event == 'open' and isinstance(args[0], (str, bytes)) or (OPENS['active'] and event == 'open'
                                                                                  and hasattr(args[0], '__fspath__')):
        OPENS['paths'].append(str(args[0]))


sys.addaudithook(_audit)


@pytest.fixture
def instrumented(monkeypatch):
    if not (r.DEFAULT_INEQUALITY_PATH.exists() and r.DEFAULT_FEATURES_PATH.exists()):
        pytest.skip('requires local frozen predictor inputs')
    calls = {'parquet': [], 'csv': [], 'opens': OPENS['paths']}
    OPENS['paths'].clear()
    OPENS['active'] = True
    real_parquet, real_csv = pd.read_parquet, pd.read_csv

    def parquet(path, columns=None, **kwargs):
        calls['parquet'].append((str(path), columns))
        if columns is None:
            raise AssertionError('parquet read without a column allow-list')
        return real_parquet(path, columns=columns, **kwargs)

    def csv(path, *args, **kwargs):
        calls['csv'].append((str(path) if not hasattr(path, 'read') else '<verified bytes>', kwargs.get('usecols')))
        return real_csv(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('outcome-bearing design loader called')
    monkeypatch.setattr(pd, 'read_parquet', parquet)
    monkeypatch.setattr(pd, 'read_csv', csv)
    monkeypatch.setattr(m0, 'load_inputs', forbidden)
    monkeypatch.setattr(m0, 'm0_complete_design', forbidden)
    import src.decomposition
    monkeypatch.setattr(src.decomposition, 'build_country_design', forbidden)
    yield calls
    OPENS['active'] = False


def data_opens(calls):
    """Opened non-code files under the repository or the synthetic package, in call order."""
    root = str(m0.ROOT)
    return [p for p in calls['opens'] if not p.endswith(('.py', '.pyc')) and '__pycache__' not in p
            and (p.startswith(root) or 'pytest' in p)]


def test_redundancy_entry_point_reads_only_approved_predictor_columns(tmp_path, instrumented):
    manifest = write_package(tmp_path)
    result = r.main(tmp_path)
    assert instrumented['parquet'] == [(str(r.DEFAULT_INEQUALITY_PATH), r.INEQUALITY_COLUMNS),
                                       (str(r.DEFAULT_FEATURES_PATH), r.CITY_COLUMNS)]
    forbidden = ('trend', 'slope', 'warming', 'fitted', 'resid', 'era5', 'score', 'coverage')
    assert not any(word in col.lower() for _, cols in instrumented['parquet'] for col in cols for word in forbidden)
    assert instrumented['csv'] == [('<verified bytes>', None), (str(r.SUPPORT_RECORD), ['iso3', 'Country']),
                                   (str(r.INCOME_PATH), None)]
    allowed = {str(tmp_path / n) for n in (*h.DETERMINISTIC_OUTPUTS, r.RECORD)} | {str(p) for p in r.PREDICTOR_INPUT_SHA256}
    assert data_opens(instrumented) and set(data_opens(instrumented)) <= allowed, data_opens(instrumented)
    record = json.loads((tmp_path / r.RECORD).read_text())
    prov = record['provenance']
    assert record['status'] == 'pass' and record['hard_stops'] == [] and result['n'] == len(FROZEN)
    assert prov['measurement_manifest_sha256'] == sha((tmp_path / h.MANIFEST).read_bytes())
    assert prov['c2_features_sha256'] == manifest['artifact_sha256'][h.FEATURES]
    assert prov['predictor_columns'] == r.PREDICTOR_COLUMNS and prov['outcome_column_present'] is False
    assert prov['design_columns'][0] == 'intercept' and len(prov['design_columns']) == record['design_column_count']
    assert set(prov['predictor_input_sha256'].values()) == set(r.PREDICTOR_INPUT_SHA256.values())
    assert r.verify_record(tmp_path, prov['measurement_manifest_sha256'], prov['c2_features_sha256']) == record


@pytest.mark.parametrize('case', ['failing_manifest', 'stale_c2', 'predictor_pin', 'stale_unresolved', 'old_gate_version'])
def test_redundancy_refuses_before_reading_c2_or_predictors(tmp_path, instrumented, monkeypatch, case):
    overrides = {'failing_manifest': {'all_hard_stops_pass': False},
                 'old_gate_version': {'gate_version': 'M1B_EVALUATION_CONTRACT.md Amendment 1'}}.get(case, {})
    write_package(tmp_path, **overrides)
    if case == 'stale_unresolved':
        (tmp_path / 'm1b_unresolved_support.csv').write_text('target_lat_index\n9\n', encoding='utf-8')
    if case == 'stale_c2':
        text = (tmp_path / h.FEATURES).read_text().replace(',-1.0\n', ',-1.0000000001\n', 1)
        (tmp_path / h.FEATURES).write_text(text, encoding='utf-8')
    if case == 'predictor_pin':
        pins = dict(r.PREDICTOR_INPUT_SHA256)
        pins[r.DEFAULT_FEATURES_PATH] = '0' * 64
        monkeypatch.setattr(r, 'PREDICTOR_INPUT_SHA256', pins)
    with pytest.raises(ValueError):
        r.main(tmp_path)
    assert instrumented['parquet'] == [] and instrumented['csv'] == [] and not (tmp_path / r.RECORD).exists()


def test_redundancy_hard_stop_is_recorded_as_valid_json(tmp_path, instrumented):
    frozen_lat = r.m0star_predictors(FROZEN.Country).abs_latitude.to_numpy()
    write_package(tmp_path, c2=2.0 * frozen_lat)                                     # exactly redundant with M0*
    result = r.main(tmp_path)
    record = json.loads((tmp_path / r.RECORD).read_text())
    assert result['all_hard_stops_pass'] is False and record['status'] == 'hard_stop'
    assert any('exact rank redundancy' in s for s in record['hard_stops'])
    with pytest.raises(ValueError):
        r.verify_record(tmp_path, record['provenance']['measurement_manifest_sha256'],
                        record['provenance']['c2_features_sha256'])
    write_package(tmp_path, c2=np.ones(len(FROZEN)))                                  # zero variance
    r.main(tmp_path)
    assert json.loads((tmp_path / r.RECORD).read_text())['hard_stops'] == ['zero-variance C2']


def test_evaluator_main_refuses_before_any_gate_or_input_read(monkeypatch):
    from research.model_v2 import m1b_evaluate as e

    def forbidden(*args, **kwargs):
        raise AssertionError('scoring touched a gate or an input before the disabled-scoring check')
    for name in ('scoring_gate', 'load_inputs', 'm0_complete_design'):
        monkeypatch.setattr(e, name, forbidden)
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    assert e.SCORING_ENABLED is False
    with pytest.raises(RuntimeError, match='scoring is disabled'):
        e.main()
