"""Temporary/synthetic checks of opt-in, confinement, and historical compatibility."""
from contextvars import Context
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import prospective as p
from src.correction_identity import (
    CORRECTION_SET_ID, EVIDENCE_SCHEMA_ID, FORECAST_METHOD_ID, summary_origin,
)
from src.data_io import OUTPUTS_DIR


@pytest.mark.parametrize('module,function', [
    ('src.vulnerability', 'build_vulnerability'),
    ('src.physical_model', 'build_physical_model'),
    ('src.stability', 'main'), ('src.explain', 'main'),
    ('src.app_assets', 'build_app_assets'), ('src.app_assets', 'main'),
    ('src.app_assets', 'build_vulnerability_asset'),
    ('src.app_assets', 'build_physical_summary_asset'),
    ('src.app_assets', 'build_stability_summary_asset'),
    ('src.interpolate', 'build_interpolated_surface'),
    ('research.model_v2.diagnostics', 'run'),
    ('research.model_v2.borders', 'main'),
    ('research.model_v2.product_stability', 'run'),
    ('research.model_v2.product_stability', 'build_aligned_outcome'),
    ('research.model_v2.run_m0_scorecard', 'main'),
    ('research.model_v2.run_m0_scorecard', 'legacy_main'),
    ('research.model_v2.run_cv_design', 'main'),
    ('research.model_v2.run_territory_correction', 'main'),
    ('research.model_v2.rank_audit', 'main'),
    ('research.model_v2.m1a_evaluate', 'main'),
])
def test_writer_refuses_without_context_before_inputs(module, function, monkeypatch):
    target = getattr(importlib.import_module(module), function)
    def forbidden(*a, **k):
        raise AssertionError('refused writer reached an input read')
    monkeypatch.setattr(pd, 'read_parquet', forbidden)
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    with pytest.raises(p.ProspectiveExecutionRequired):
        Context().run(target)


def test_canonical_output_path_and_distinct_identifiers():
    assert p.PROSPECTIVE_OUTPUT_ROOT == OUTPUTS_DIR / 'prospective' / CORRECTION_SET_ID
    assert CORRECTION_SET_ID == 'prospective-corrections-2026-10-22'
    assert len({CORRECTION_SET_ID, EVIDENCE_SCHEMA_ID, FORECAST_METHOD_ID}) == 3
    assert EVIDENCE_SCHEMA_ID == 'finite-evidence-v1'
    assert FORECAST_METHOD_ID == 'fixed-rho-conditional-t-v1'


def test_omitted_legacy_destinations_are_replaced_not_overwritten(tmp_path):
    legacy = tmp_path / 'published'
    legacy.mkdir()
    sentinel = legacy / 'summary.json'
    sentinel.write_text('frozen')
    @p.prospective_outputs(out_dir='app_bundle', summary='processed/summary.json')
    def writer(out_dir=legacy, summary=sentinel):
        return out_dir, summary
    scratch = tmp_path / 'scratch'
    with p.prospective_execution(scratch=scratch):
        assert writer() == (scratch / 'app_bundle', scratch / 'processed/summary.json')
        with pytest.raises(p.ProspectiveExecutionRequired):
            writer(out_dir=legacy)
    assert sentinel.read_text() == 'frozen'
    assert not scratch.exists()  # selecting a scope alone performs no writes


@pytest.mark.parametrize('suffix', ['app/data', 'research/model_v2/outputs',
                                     'data/processed', 'outputs/figures', '../escape'])
def test_archived_or_outside_destinations_rejected(tmp_path, suffix):
    scratch = tmp_path / 'scratch'
    archived = tmp_path / suffix
    with p.prospective_execution(scratch=scratch) as execution:
        with pytest.raises(p.ProspectiveExecutionRequired):
            execution.destination(archived)


@pytest.mark.parametrize('link_kind', ['symlink', 'hardlink'])
def test_all_destinations_checked_before_first_write(tmp_path, link_kind):
    from src.explain import write_explain_summary
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    saved = tmp_path / 'saved.parquet'
    saved.write_bytes(b'unchanged')
    link = scratch / 'explain_features.parquet'
    first = scratch / 'summary.json'
    first.write_text('also unchanged')
    with p.prospective_execution(scratch=scratch):
        if link_kind == 'symlink':
            link.symlink_to(saved)
        else:
            link.hardlink_to(saved)
        with pytest.raises(p.ProspectiveExecutionRequired):
            write_explain_summary([], [], pd.DataFrame(), summary_path=first, bundle_path=link)
    assert first.read_text() == 'also unchanged'
    assert saved.read_bytes() == b'unchanged'


def test_directory_child_symlink_and_root_alias_are_rejected(tmp_path, monkeypatch):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    saved = tmp_path / 'published'
    saved.mkdir()
    with p.prospective_execution(scratch=scratch) as execution:
        (scratch / 'summary.json').symlink_to(saved / 'summary.json')
        with pytest.raises(p.ProspectiveExecutionRequired):
            execution.destination(scratch)
    alias = tmp_path / 'prospective-alias'
    alias.symlink_to(saved, target_is_directory=True)
    monkeypatch.setattr(p, 'PROSPECTIVE_OUTPUT_ROOT', alias)
    with pytest.raises(p.ProspectiveExecutionRequired):
        with p.prospective_execution():
            pass


def test_scratch_cannot_target_worktree_or_published_bundle():
    from src.bundle import APP_DATA_DIR
    with pytest.raises(p.ProspectiveExecutionRequired):
        with p.prospective_execution(scratch=APP_DATA_DIR):
            pass


@pytest.mark.parametrize('module', ['m1b_evaluate', 'm2_evaluate', 'm3_evaluate'])
def test_unadapted_evaluate_refuses_real_prospective_scope(module, tmp_path):
    evaluate = getattr(importlib.import_module('research.model_v2.' + module), 'evaluate')
    with p.prospective_execution(), pytest.raises(p.ProspectiveExecutionRequired, match='historical evaluator'):
        evaluate(None, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('module,args', [
    ('m1b_evaluate', []), ('m2_evaluate', []),
    ('m3_evaluate', ['--weights', 'station']), ('m4_evaluate', []),
    ('m1b_sensitivity', []), ('m2_conditional', []),
])
def test_unadapted_main_refuses_before_scoring(module, args):
    main = getattr(importlib.import_module('research.model_v2.' + module), 'main')
    with p.prospective_execution(), pytest.raises(p.ProspectiveExecutionRequired):
        main(args)


def test_context_does_not_leak_or_respond_to_environment(tmp_path, monkeypatch):
    def isolated():
        monkeypatch.setenv('PROSPECTIVE', '1')
        with pytest.raises(p.ProspectiveExecutionRequired):
            p.require_prospective()
        with p.prospective_execution(scratch=tmp_path):
            assert p.require_prospective().synthetic
        with pytest.raises(p.ProspectiveExecutionRequired):
            p.require_prospective()
    Context().run(isolated)


def test_synthetic_physical_builder_writes_only_scoped_outputs(tmp_path):
    from src.physical_model import build_physical_model
    from tests.test_physical_model import make_synthetic_forcings
    inputs = tmp_path / 'synthetic.parquet'
    make_synthetic_forcings().to_parquet(inputs)
    scratch = tmp_path / 'scratch'
    with p.prospective_execution(scratch=scratch):
        result = build_physical_model(forcings_path=inputs)
    assert result['summary_path'] == scratch / 'processed/physical_summary.json'
    assert result['trajectory_path'] == scratch / 'processed/physical_trajectory.parquet'
    summary = json.loads(result['summary_path'].read_text())
    assert summary['correction_set_id'] == CORRECTION_SET_ID
    assert summary['evidence_schema_id'] == EVIDENCE_SCHEMA_ID
    assert summary['forecast_method_id'] == FORECAST_METHOD_ID


def test_missing_income_summary_is_scoped_and_versioned(tmp_path):
    from src.vulnerability import build_vulnerability
    scratch = tmp_path / 'scratch'
    with p.prospective_execution(scratch=scratch):
        result = build_vulnerability(income_path=tmp_path / 'absent.csv')
    summary = json.loads(result['summary_path'].read_text())
    assert result['summary_path'].is_relative_to(scratch)
    assert summary['available'] is False
    assert summary['correction_set_id'] == CORRECTION_SET_ID


def test_statistic_and_p_remain_available_when_z_score_is_undefined():
    from research.model_v2.spatial import morans_i
    values = np.array([2., -1., 4., 0., -5.])
    result = morans_i(values, np.ones((5, 5)) - np.eye(5), 19, 0)
    assert result.statistic == -.25 and result.p_value == 1.
    assert result.status == 'available' and result.reason is None
    assert result.z_score is None and result.z_score_reason


def test_historical_reading_does_not_stamp_or_reclassify_record():
    historical = {'triple_inequality': {'holds': True}, 'forecast_uncertainty_method': 'archived'}
    before = json.dumps(historical, sort_keys=True)
    assert 'Historical' in summary_origin(historical)
    assert json.dumps(historical, sort_keys=True) == before
    assert 'Not a validated scientific release' in summary_origin({'correction_set_id': CORRECTION_SET_ID})


def test_unavailable_local_evidence_is_not_absent_evidence():
    from research.model_v2.product_stability import evidence_class, local_level
    row = {'w_b_residual_label': 'unavailable', 'w_b_residual_fdr_label': 'unavailable'}
    assert local_level(row, 'w', 'b', 'HH') == 'unavailable'
    assert evidence_class('strong', 'unavailable') == 'unresolved'


def test_cli_refuses_without_prospective_opt_in():
    done = subprocess.run([sys.executable, '-m', 'src.physical_model'],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert done.returncode == 2
    assert '--prospective' in done.stderr


@pytest.mark.parametrize('function', ['add_elevation', 'add_koppen'])
def test_feature_sampling_never_acquires_missing_grid(tmp_path, monkeypatch, function):
    from src import explain
    def forbidden(*args, **kwargs):
        raise AssertionError('input acquisition was reached')
    monkeypatch.setattr(explain, 'download_file', forbidden)
    monkeypatch.setattr(explain, 'prepare_koppen_grid', forbidden)
    with pytest.raises(FileNotFoundError, match='read-only inputs'):
        getattr(explain, function)(pd.DataFrame(), nc_path=tmp_path / 'absent.nc')


def test_explain_main_requires_prepared_inputs_without_download(tmp_path, monkeypatch):
    from src import explain
    monkeypatch.setattr(explain, 'ETOPO_PATH', tmp_path / 'absent.nc')
    def forbidden(*args, **kwargs):
        raise AssertionError('input acquisition/preflight was reached')
    for name in ('download_file', 'prepare_koppen_grid', 'run_geo_preflight'):
        monkeypatch.setattr(explain, name, forbidden)
    with p.prospective_execution(scratch=tmp_path), pytest.raises(FileNotFoundError):
        explain.main()


def test_surface_builder_never_downloads_land_or_scores_before_geometry(tmp_path, monkeypatch):
    from src import interpolate
    def missing():
        raise FileNotFoundError('prepared land missing')
    def forbidden(*args, **kwargs):
        raise AssertionError('download or scoring was reached')
    monkeypatch.setattr(interpolate, 'load_land_geometry', missing)
    monkeypatch.setattr(interpolate, 'download_land_polygons', forbidden)
    monkeypatch.setattr(interpolate, 'leave_one_out_cv', forbidden)
    with p.prospective_execution(scratch=tmp_path), pytest.raises(FileNotFoundError):
        interpolate.build_interpolated_surface()



def test_direct_optional_asset_copy_requires_scope_and_confined_destination(tmp_path):
    from src.app_assets import _copy_findings_parquet
    source = tmp_path / 'source.parquet'
    pd.DataFrame({'value': [1]}).to_parquet(source)
    dest = tmp_path / 'no-write.parquet'
    with pytest.raises(p.ProspectiveExecutionRequired):
        Context().run(_copy_findings_parquet, source, dest, ('value',))
    assert not dest.exists()
    with p.prospective_execution(scratch=tmp_path / 'scratch'), pytest.raises(p.ProspectiveExecutionRequired):
        _copy_findings_parquet(source, dest, ('value',))
    assert not dest.exists()
