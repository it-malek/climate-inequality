"""Disposable evidence-stage and portable initialization contracts."""
import importlib.util
import json

import pytest

from scripts.init_current_validation import initialize
from src.correction_identity import CORRECTION_SET_ID
from src.validation_receipts import ValidationReceipts, dump, sha


@pytest.fixture
def configured(tmp_path):
    root = tmp_path / 'repository'
    root.mkdir()
    for name in ['uv.lock', 'pyproject.toml', 'driver.py']:
        (root / name).write_text('# synthetic source\n')
    raw = tmp_path / 'raw'
    raw.mkdir()
    source = raw / 'input.txt'
    source.write_text('before\n')
    run = root / 'outputs/prospective' / CORRECTION_SET_ID / 'synthetic-run'
    run.mkdir(parents=True)
    dump(run / 'VALIDATION_SPEC.json', {'synthetic': True})
    return root, run, raw, source, ValidationReceipts(root, run, raw, sha(run / 'VALIDATION_SPEC.json'))


def test_source_archive_before_action_and_no_overwrite(configured):
    root, run, raw, source, receipts = configured
    def action(directory):
        started = json.loads((directory / 'started.json').read_text())
        assert started['input_sha256'][str(source)] == sha(source)
        assert (directory / 'execution-source/driver.py').read_bytes() == (root / 'driver.py').read_bytes()
        (directory / 'answer.txt').write_text('synthetic answer')
        return {'done': True}
    receipts.stage('first', [source], action, runner=root / 'driver.py')
    assert receipts.verified_stage('first') == run / 'first'
    with pytest.raises(FileExistsError):
        receipts.stage('first', [source], action, runner=root / 'driver.py')


def test_input_drift_records_failure_and_preserves_stage(configured):
    root, run, raw, source, receipts = configured
    def drift(directory):
        source.write_text('changed')
    with pytest.raises(ValueError, match='input changed'):
        receipts.stage('drift', [source], drift, runner=root / 'driver.py')
    assert (run / 'drift/failed.json').is_file()
    assert not (run / 'drift/completed.json').exists()
    with pytest.raises(FileExistsError):
        receipts.stage('drift', [source], drift, runner=root / 'driver.py')


def test_output_and_spec_drift_block_dependencies(configured):
    root, run, raw, source, receipts = configured
    receipts.stage('first', [source], lambda d: (d / 'answer').write_text('ok'), runner=root / 'driver.py')
    (run / 'first/answer').write_text('drift')
    with pytest.raises(ValueError, match='Checkpoint drift'):
        receipts.stage('second', [source], lambda d: None, runner=root / 'driver.py', dependencies=['first'])
    assert not (run / 'second').exists()
    (run / 'VALIDATION_SPEC.json').write_text('{}')
    with pytest.raises(ValueError, match='specification drift'):
        receipts.stage('third', [source], lambda d: None, runner=root / 'driver.py')


def test_confinement_rejects_escape_and_links(configured, tmp_path):
    root, run, raw, source, receipts = configured
    with pytest.raises(ValueError, match='prospective namespace'):
        ValidationReceipts(root, tmp_path / 'outside', raw, 'unused')
    for name in ['../outside', '/absolute', '.']:
        with pytest.raises(ValueError):
            receipts.stage(name, [source], lambda d: None, runner=root / 'driver.py')
    (run / 'alias').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        receipts.stage('alias/child', [source], lambda d: None, runner=root / 'driver.py')


def test_initializer_is_portable_exclusive_and_preserves_original(configured):
    root, existing, raw, source, receipts = configured
    docs = root / 'docs'
    docs.mkdir()
    template = {'inputs': {'raw_root': '${RAW_ROOT}', 'previous_run': '${PREVIOUS_RUN}'}, 'provenance': {'original_frozen_spec_sha256': 'original-digest'}, 'scientific_settings': {'k': 30}}
    dump(docs / 'current_validation_spec.json', template)
    new = existing.parent / 'fresh-run'
    initialize(new, raw, repository_root=root)
    written = json.loads((new / 'VALIDATION_SPEC.json').read_text())
    assert written['scientific_settings'] == template['scientific_settings']
    assert written['inputs']['raw_root'] == str(raw)
    assert json.loads((new / 'SPEC_FREEZE.json').read_text())['original_frozen_spec_sha256'] == 'original-digest'
    assert json.loads((docs / 'current_validation_spec.json').read_text()) == template
    adapter = new / 'control/runlib.py'
    spec = importlib.util.spec_from_file_location('synthetic_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RUN == new
    assert module.RAW == raw
    assert module.SPEC_HASH == sha(new / 'VALIDATION_SPEC.json')
    with pytest.raises(FileExistsError):
        initialize(new, raw, repository_root=root)
    with pytest.raises(ValueError, match='prospective namespace'):
        initialize(root / 'outside', raw, repository_root=root)


def test_initializer_rejects_nested_run_before_creating_anything(configured):
    root, existing, raw, source, receipts = configured
    docs = root / 'docs'
    docs.mkdir()
    dump(docs / 'current_validation_spec.json', {'inputs': {'raw_root': '${RAW_ROOT}', 'previous_run': '${PREVIOUS_RUN}'}, 'provenance': {'original_frozen_spec_sha256': 'original-digest'}})
    nested = existing.parent / 'outer' / 'inner'
    with pytest.raises(ValueError, match='simple direct child'):
        initialize(nested, raw, repository_root=root)
    assert not nested.parent.exists()
