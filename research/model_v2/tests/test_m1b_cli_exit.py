"""Post-M1b auxiliary CLI maintenance: mode-specific process exit status.

``main()`` returns a scoring result (``integrity_passed``) or, under ``--compare``, a reproducibility
record (``byte_identical``). The frozen ``__main__`` line read ``integrity_passed`` unconditionally,
so a completed comparison raised ``KeyError`` (M1B_REPORT.md §8). Every fixture here is synthetic:
no test loads a scoring input, fits a model or reads a warming outcome, and the comparison runs over
placeholder bytes in temporary directories.
"""
import ast
import json
import os
import subprocess
import sys

import pytest

from research.model_v2 import m1b_evaluate as ev
from research.model_v2.m0 import ROOT

ARTIFACT_NAMES = (*ev.DETERMINISTIC_ARTIFACTS, ev.RESULT_MANIFEST)
FROZEN_EVALUATOR = '618c8b83b3630e06c9d33a295825389ed79f1cc4'

# The driver forbids every scoring-side entry point before calling main(), so a comparison that
# reached scoring-input loading, arm construction or fitting would fail the process rather than pass.
DRIVER = '''
import sys
from research.model_v2 import m1b_evaluate as ev


def forbidden(name):
    def refuse(*args, **kwargs):
        raise AssertionError(f'--compare reached {name}')
    return refuse


for name in ('require_hash_seed', 'verify_scoring_inputs', 'build_frozen', 'evaluate', 'build_arms',
             'score_model', 'support_context', 'redundancy_context_from'):
    setattr(ev, name, forbidden(name))
ev.frozen_code_state = lambda *args, **kwargs: {'commit': 'synthetic'}
sys.exit(ev.cli_exit_code(ev.main(sys.argv[1:])))
'''


def synthetic_runs(root, differ=()):
    """Two completed-run directories of placeholder bytes; names in ``differ`` get distinct content."""
    runs = root / 'run_a', root / 'run_b'
    for run in runs:
        run.mkdir()
        (run / ev.RUN_METADATA).write_text(f'{{"host": "{run.name}"}}\n', encoding='utf-8')
    for name in ARTIFACT_NAMES:
        runs[0].joinpath(name).write_bytes(f'synthetic {name}\n'.encode())
        runs[1].joinpath(name).write_bytes(f'synthetic {name}{" changed" if name in differ else ""}\n'.encode())
    return runs


def run_cli(tmp_path, *args):
    driver = tmp_path / 'driver.py'
    driver.write_text(DRIVER, encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    return subprocess.run([sys.executable, str(driver), *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


# --- the defect, characterized ------------------------------------------------------------------

def test_a_comparison_record_carries_byte_identical_and_no_integrity_flag(tmp_path):
    record = ev.reproducibility_record(*synthetic_runs(tmp_path))
    assert record['byte_identical'] is True
    assert 'integrity_passed' not in record


# --- exit status mapping ------------------------------------------------------------------------

@pytest.mark.parametrize('passed, status', [(True, 0), (False, 3)])
def test_scoring_exit_semantics_are_unchanged(passed, status):
    assert ev.cli_exit_code({'verdict': {}, 'integrity_passed': passed}) == status


@pytest.mark.parametrize('identical, succeeds', [(True, True), (False, False)])
def test_a_comparison_exits_zero_only_when_byte_identical(identical, succeeds):
    status = ev.cli_exit_code({'artifacts': {}, 'byte_identical': identical})
    assert (status == 0) is succeeds
    assert status != 3 or succeeds, 'a differing comparison must not reuse the integrity-failure status'


@pytest.mark.parametrize('outcome', [
    {},
    {'verdict': 'not supported'},
    {'integrity_passed': True, 'byte_identical': True},
    {'integrity_passed': False, 'byte_identical': True},
])
def test_an_outcome_that_is_neither_or_both_shapes_raises(outcome):
    with pytest.raises(RuntimeError, match='unrecognised CLI outcome'):
        ev.cli_exit_code(outcome)


@pytest.mark.parametrize('key', ['integrity_passed', 'byte_identical'])
@pytest.mark.parametrize('flag', [1, 0, 'true', 'false', None, 1.0, [True]])
def test_a_non_boolean_flag_raises_rather_than_being_read_as_truthy(key, flag):
    with pytest.raises(RuntimeError, match='must be a bool'):
        ev.cli_exit_code({key: flag})


@pytest.mark.parametrize('outcome', [None, [], 'byte_identical', ('integrity_passed', True)])
def test_a_non_mapping_outcome_raises(outcome):
    with pytest.raises(RuntimeError, match='unrecognised CLI outcome'):
        ev.cli_exit_code(outcome)


def test_the_main_guard_delegates_its_exit_status_to_cli_exit_code():
    tree = ast.parse((ROOT / 'research/model_v2/m1b_evaluate.py').read_text(encoding='utf-8'))
    guards = [node for node in tree.body
              if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"]
    assert len(guards) == 1
    assert [ast.unparse(statement) for statement in guards[0].body] == ['sys.exit(cli_exit_code(main()))']


def _top_level_nodes(source):
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            key = '__main__ guard'
        else:
            key = ast.unparse(node)[:120]
        assert key not in nodes, key
        nodes[key] = ast.dump(node)
    return nodes


def test_the_maintenance_leaves_every_scoring_definition_identical_to_the_frozen_evaluator():
    """Only the exit mapping may differ from 618c8b8; every other function, class, constant, pin and
    the rest of the code path must be unchanged, so the historical result stays pinned to that commit."""
    frozen = subprocess.run(['git', 'show', f'{FROZEN_EVALUATOR}:research/model_v2/m1b_evaluate.py'],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    before = _top_level_nodes(frozen)
    after = _top_level_nodes((ROOT / 'research/model_v2/m1b_evaluate.py').read_text(encoding='utf-8'))
    assert set(after) - set(before) == {'cli_exit_code'}
    assert set(before) <= set(after)
    assert [key for key in before if before[key] != after[key]] == ['__main__ guard']
    for path in ev.CODE_PATH:
        if path != 'research/model_v2/m1b_evaluate.py':
            blob = subprocess.run(['git', 'show', f'{FROZEN_EVALUATOR}:{path}'], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            assert (ROOT / path).read_bytes() == blob, path


# --- the process boundary -----------------------------------------------------------------------

def test_an_identical_comparison_exits_zero_in_a_subprocess(tmp_path):
    run_a, run_b = synthetic_runs(tmp_path)
    record = tmp_path / 'record.json'
    completed = run_cli(tmp_path, '--compare', run_a, run_b, '--record', record)
    assert completed.returncode == 0, completed.stderr
    assert 'KeyError' not in completed.stderr
    assert json.loads(completed.stdout) == {'byte_identical': True, 'differing': []}
    saved = json.loads(record.read_text(encoding='utf-8'))
    assert saved['byte_identical'] is True and saved['compared'] == list(ARTIFACT_NAMES)


def test_a_differing_comparison_exits_nonzero_in_a_subprocess(tmp_path):
    run_a, run_b = synthetic_runs(tmp_path, differ={'m1b_cv_folds.csv'})
    completed = run_cli(tmp_path, '--compare', run_a, run_b)
    assert completed.returncode not in (0, 3), completed.stderr
    assert 'KeyError' not in completed.stderr and 'AssertionError' not in completed.stderr
    assert json.loads(completed.stdout) == {'byte_identical': False, 'differing': ['m1b_cv_folds.csv']}


def test_run_metadata_alone_differing_still_exits_zero_in_a_subprocess(tmp_path):
    run_a, run_b = synthetic_runs(tmp_path)
    assert (run_a / ev.RUN_METADATA).read_bytes() != (run_b / ev.RUN_METADATA).read_bytes()
    assert run_cli(tmp_path, '--compare', run_a, run_b).returncode == 0


def test_the_driver_forbids_scoring_entry_points_for_real(tmp_path):
    """Negative control: without --compare the driver must reach a forbidden stub and fail."""
    completed = run_cli(tmp_path, '--out', tmp_path / 'never_created')
    assert completed.returncode != 0
    assert 'AssertionError: --compare reached require_hash_seed' in completed.stderr
    assert not (tmp_path / 'never_created').exists()
