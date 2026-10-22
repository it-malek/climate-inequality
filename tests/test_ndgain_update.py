import hashlib
import zipfile
import numpy as np
import pandas as pd
import pytest
from src.ndgain_update import archive_snapshot, revision_table


def test_explicit_year_missingness_and_units(tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for metric, old, new in [('vulnerability', .4, .9), ('readiness', .6, .1), ('gain', 60., 10.)]:
            z.writestr(f'resources/{metric}/{metric}.csv',
                       pd.DataFrame({'ISO3': ['AAA', 'BBB'], '2023': [old, np.nan], '2024': [new, new]}).to_csv(index=False))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame, meta = archive_snapshot(path, expected_sha256=digest, year=2023, prefix='resources')
    assert frame.loc[0, 'vulnerability'] == .4
    assert frame.loc[1, ['vulnerability', 'ndgain_year']].isna().all()
    assert meta['observation_year'] == 2023
    with pytest.raises(ValueError, match='checksum'):
        archive_snapshot(path, expected_sha256='incorrect', year=2023, prefix='resources')
    with pytest.raises(ValueError, match='year unavailable'):
        archive_snapshot(path, expected_sha256=digest, year=2022, prefix='resources')


def test_revision_and_coverage_are_distinct():
    old = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'vulnerability': [.4, np.nan], 'readiness': [.6, .5], 'gain': [60., np.nan]})
    new = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'vulnerability': [.45, .5], 'readiness': [.6, .5], 'gain': [57.5, 50.]})
    table = revision_table(old, new)
    assert table.vulnerability_status.tolist() == ['overlapping_values', 'newly_available']
    assert table.loc[0, 'vulnerability_change'] == pytest.approx(.05)
