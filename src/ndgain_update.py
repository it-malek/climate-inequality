"""Explicit-year ND-GAIN snapshots and revision accounting from verified archives."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

METRICS = ('vulnerability', 'readiness', 'gain')


def archive_snapshot(path, *, expected_sha256, year, prefix):
    """Read one named vintage/year; never silently choose latest or fill missing years."""
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError('ND-GAIN archive checksum differs from the selected release')
    components, metadata = {}, {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for metric in METRICS:
            member = f'{prefix}/{metric}/{metric}.csv'
            raw = archive.read(member)
            frame = pd.read_csv(io.BytesIO(raw))
            if frame.ISO3.isna().any() or not frame.ISO3.is_unique:
                raise ValueError(f'{metric}: missing or duplicate ISO3')
            if str(year) not in frame.columns:
                raise ValueError(f'{metric}: requested observation year unavailable')
            values = pd.to_numeric(frame[str(year)], errors='raise')
            bound = 100 if metric == 'gain' else 1
            finite = values.dropna()
            if not np.isfinite(finite).all() or not finite.between(0, bound).all():
                raise ValueError(f'{metric}: invalid score range')
            components[metric] = pd.DataFrame({'iso3': frame.ISO3, metric: values})
            metadata[metric] = {'member': member, 'sha256': hashlib.sha256(raw).hexdigest(),
                                'year': year, 'rows': len(frame), 'finite': int(values.notna().sum()),
                                'missing_iso3': frame.loc[values.isna(), 'ISO3'].tolist()}
    result = components['vulnerability']
    for metric in METRICS[1:]:
        result = result.merge(components[metric], on='iso3', how='outer', validate='one_to_one')
    result['ndgain_year'] = np.where(result.vulnerability.notna(), float(year), np.nan)
    paired = result.dropna(subset=list(METRICS))
    error = abs(paired.gain - (paired.readiness - paired.vulnerability + 1) * 50)
    if not np.all(error <= 1e-8):
        raise ValueError('ND-GAIN aggregate formula does not match documented units')
    result = result[['iso3', 'ndgain_year', *METRICS]].sort_values('iso3').reset_index(drop=True)
    return result, {'archive_sha256': actual, 'observation_year': year, 'components': metadata,
                    'gain_formula_max_abs_error': float(error.max()) if len(error) else None}


def revision_table(reference, updated):
    """Separate existing-value revisions from additions/removals in the selected year."""
    table = reference.merge(updated, on='iso3', how='outer', suffixes=('_reference', '_updated'),
                            validate='one_to_one', indicator=True)
    for metric in METRICS:
        before, after = table[f'{metric}_reference'], table[f'{metric}_updated']
        table[f'{metric}_change'] = after - before
        table[f'{metric}_status'] = np.select(
            [before.notna() & after.notna(), before.isna() & after.notna(),
             before.notna() & after.isna()], ['overlapping_values', 'newly_available', 'now_missing'],
            default='missing_both')
    return table.sort_values('iso3').reset_index(drop=True)
