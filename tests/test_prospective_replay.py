"""Contract tests for prospective replay and run namespace isolation."""
import json

import numpy as np
import pandas as pd
import pytest

from research.model_v2.prospective_diagnostics import (
    ordered_group, prediction_diagnostics, sha256, verify_artifacts,
)
from src import prospective


def test_replay_preserves_country_order_and_unavailable_evidence():
    frame = pd.DataFrame({'iso3': ['C', 'A', 'B'], 'protocol': ['in_sample'] * 3,
                          'observed': [2., 1., 3.], 'prediction': [1., 0., 2.],
                          'trend': [0., 0., 0.]})
    weights = (np.ones((3, 3)) - np.eye(3)) / 2
    result = prediction_diagnostics(frame, ['A', 'B', 'C'], {'station': weights},
                                    ['protocol'], permutations=19)
    assert result[0]['status'] == 'unavailable'
    assert result[0]['reason'] == 'constant_residuals'
    assert result[0]['p_value'] is None
    assert result[1]['status'] == 'available'
    assert ordered_group(frame, ['A', 'B', 'C']).observed.tolist() == [1., 3., 2.]
    with pytest.raises(ValueError, match='countries differ'):
        ordered_group(frame, ['A', 'B', 'D'])
    with pytest.raises(ValueError, match='Duplicate'):
        ordered_group(pd.concat([frame, frame.iloc[:1]]), ['A', 'B', 'C'])


def test_artifact_verification_refuses_drift_and_traversal(tmp_path):
    artifact = tmp_path / 'predictions.csv'
    artifact.write_text('immutable')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'artifact_sha256': {'predictions.csv': sha256(artifact)}}))
    assert verify_artifacts(tmp_path, 'manifest.json')['predictions.csv'] == sha256(artifact)
    artifact.write_text('changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_artifacts(tmp_path, 'manifest.json')
    manifest.write_text(json.dumps({'artifact_sha256': {'../outside.csv': 'fake'}}))
    with pytest.raises(ValueError, match='escapes'):
        verify_artifacts(tmp_path, 'manifest.json')


def test_run_names_confine_outputs_without_implicit_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(prospective, 'PROSPECTIVE_OUTPUT_ROOT', tmp_path / 'prospective')
    with prospective.prospective_execution(run_id='run-1') as run:
        assert run.root == tmp_path / 'prospective/run-1'
        assert not run.root.exists()
        with pytest.raises(prospective.ProspectiveExecutionRequired):
            run.destination(tmp_path / 'prospective/run-2/result.json')
    for name in ('../escape', '/absolute', '.', 'x/y', ''):
        with pytest.raises(prospective.ProspectiveExecutionRequired):
            with prospective.prospective_execution(run_id=name):
                pass
    with pytest.raises(prospective.ProspectiveExecutionRequired):
        with prospective.prospective_execution(run_id='run-1', scratch=tmp_path):
            pass


def test_injected_pipeline_keeps_exact_outcome_identity_gate(monkeypatch):
    from research.model_v2 import m2_evaluate as ev
    from research.model_v2 import v2_provenance as prov
    table = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'Country': ['A', 'B']})
    rank = table.assign(variant='drop_per_capita', observed=[1., 2.])
    m1b = table.assign(representation='primary_total_co2', model=ev.COMPARATOR,
                      protocol='in_sample', observed=[1., 2.])
    pipeline = table.assign(warming_trend=[1., 2.])
    monkeypatch.setattr(pd, 'read_csv', lambda path, **kwargs:
                        rank if str(path).endswith(ev.RANK_PREDICTIONS) else m1b)
    monkeypatch.setattr(ev, 'load_inputs', lambda: pytest.fail('implicit input read'))
    monkeypatch.setattr(ev, 'm0_complete_design', lambda frame: frame)
    assert ev.outcome_vector(table, pipeline_inputs=(pipeline,)).tolist() == [1., 2.]
    changed = pipeline.assign(warming_trend=[1., 2.000000000001])
    with pytest.raises(prov.ExecutionRefused, match='differs from the pipeline'):
        ev.outcome_vector(table, pipeline_inputs=(changed,))


def test_replay_refuses_to_overwrite_completed_output(tmp_path):
    from research.model_v2.prospective_diagnostics import write_replay
    with prospective.prospective_execution(scratch=tmp_path):
        path = tmp_path / 'replay.json'
        write_replay([], {}, out_path=path, limitations=[])
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            write_replay([], {}, out_path=path, limitations=['changed'])
        assert path.read_bytes() == before
