"""The committed M3 structural feasibility record: manifest, canonical provenance and its pass flags.

The suite never re-runs the audit; it checks the committed record only.
"""
import json

import pytest

from research.model_v2 import m3_feasibility as mf
from research.model_v2 import v2_provenance as prov

RECORD = mf.OUT


@pytest.fixture(scope='module')
def summary():
    if not (RECORD / 'm3_feasibility_manifest.json').exists():
        pytest.skip('the feasibility record is not committed')
    manifest = json.loads((RECORD / 'm3_feasibility_manifest.json').read_text())
    assert sorted(manifest) == sorted(mf.RECORD_FILES)
    for name, digest in manifest.items():
        assert prov.sha256(RECORD / name) == digest
    return json.loads((RECORD / 'm3_feasibility_summary.json').read_text())


def test_the_record_is_canonical_and_bound_to_pushed_code(summary):
    assert summary['canonical'] is True
    assert summary['code']['code_closure_clean_and_pushed'] is True
    assert summary['inputs'] == mf.PINS


def test_every_graph_and_synthetic_fit_passes(summary):
    assert summary['graphs']['fits'] == 968 and summary['graphs']['all_structural_pass']
    synthetic = summary['synthetic_estimator']
    assert synthetic['fits'] == 728 and synthetic['non_computable'] == 0 and synthetic['held_out_invariance_all']


def test_the_null_boundary_frequencies_of_the_record(summary):
    rows = {(r['design'], r['family'], r['true_theta']): r for r in summary['synthetic_shrinkage_monte_carlo']['rows']}
    assert rows[('M0star', 'sem', 0.0)]['at_domain_bound'] == 14
    assert rows[('M2', 'sem', 0.0)]['at_domain_bound'] == 13
