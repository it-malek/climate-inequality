"""The committed final Model V2 record agrees with the committed evidence it cites."""
import json

import pytest

from research.model_v2 import v2_final_record as vf
from research.model_v2 import v2_provenance as prov


@pytest.fixture(scope='module')
def record():
    path = vf.ROOT / vf.OUT
    if not path.exists():
        pytest.skip('the final record is committed at closure')
    return json.loads(path.read_text())


def test_every_cited_source_matches_its_recorded_digest(record):
    for name, entry in record['sources'].items():
        assert prov.sha256(vf.ROOT / entry['path']) == entry['sha256'], name
    for name, entry in record['verification'].items():
        assert prov.sha256(vf.ROOT / entry['path']) == entry['sha256'], name
        assert entry['passes'] is True, name


def test_the_final_identities_agree_with_the_m3_and_m4_manifests(record):
    station = json.loads((vf.ROOT / vf.SOURCES['m3_station_manifest']).read_text())['final_naming']
    m4 = json.loads((vf.ROOT / vf.SOURCES['m4_result_manifest']).read_text())['final']
    for key in ('retained_static_specification', 'qualifying_spatial_extensions', 'final_primary_predictive_model'):
        assert record[key] == station[key] == m4[key]
