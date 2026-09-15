"""M1b feasibility audit under the ORIGINAL coverage rule (builder ``m1b_hydroclimate.py`` sha256 57cf5b64…, commit c6ef919).

Read-only characterisation of the original C2 coverage failure, plus a byte comparison of two
independent rebuilds against the preserved outputs. Inputs are only country identifiers, the frozen
M1a land support, the GPW country grid and the CRU PRE/PET files; the warming outcome is never read.
No harmonization, fill or neighbour value is computed here.

Run after two rebuilds with the unmodified builder::

    python -m research.model_v2.m1b_hydroclimate BUILD_A
    python -m research.model_v2.m1b_hydroclimate BUILD_B
    python -m research.model_v2.feasibility.m1b_original_coverage BUILD_A BUILD_B

Writes three files into ``outputs/m1b_feasibility_original_coverage/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from research.model_v2 import m1b_hydroclimate as h
from research.model_v2.geometry import GPW_ENGINE
from research.model_v2.m1a_geography import gpw_country_grid, row_band_area_km2
from src.population import GPW_PATH

AUDIT = h.OUT / 'm1b_feasibility_original_coverage'
BUILDER_SHA256 = '57cf5b64ad87fab6f6e7554ce97b163567c4b32372b513212d21f94f1f70bfee'
OUTPUT_FILES = ['m1b_hydroclimate_features.csv', 'm1b_hydroclimate_qa.csv', 'm1b_measurement_manifest.json',
                'm1b_support_checkpoint.json']
NEAR_THRESHOLD = 0.99
CHUNK_MONTHS = 120
CLASSES = ['both_pre_and_pet_masked', 'pet_masked_pre_valid', 'pre_masked_pet_valid', 'temporal_partial',
           'nonpositive_mean']


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def determinism(build_a, build_b):
    files = {}
    for name in OUTPUT_FILES:
        digests = {'preserved': file_sha256(AUDIT / name), 'build_a': file_sha256(build_a / name),
                   'build_b': file_sha256(build_b / name)}
        files[name] = {**digests, 'byte_identical': len(set(digests.values())) == 1}
    return {'builder_sha256': BUILDER_SHA256, 'files': files,
            'all_byte_identical': all(f['byte_identical'] for f in files.values())}


def grid_alignment():
    """GPW 0.25° cells nest exactly 2 × 2 inside CRU 0.5° cells after the ascending-latitude flip."""
    with xr.open_dataset(GPW_PATH, engine=GPW_ENGINE) as ds:
        lats, lons = ds['latitude'].to_numpy().astype(float), ds['longitude'].to_numpy().astype(float)
    ascending = lats[::-1]
    r, c = np.arange(720), np.arange(1440)
    cru_lat, cru_lon = -89.75 + 0.5 * (r // 2), -179.75 + 0.5 * (c // 2)
    checks = {
        'gpw_native_latitude_descending': bool(np.all(np.diff(lats) < 0)),
        'gpw_ascending_lat_centres': bool(np.allclose(ascending, -89.875 + 0.25 * r, rtol=0, atol=1e-9)),
        'gpw_lon_centres': bool(np.allclose(lons, -179.875 + 0.25 * c, rtol=0, atol=1e-9)),
        'gpw_lat_edges_inside_cru_cell': bool(np.all((ascending - 0.125 >= cru_lat - 0.25 - 1e-9)
                                                     & (ascending + 0.125 <= cru_lat + 0.25 + 1e-9))),
        'gpw_lon_edges_inside_cru_cell': bool(np.all((lons - 0.125 >= cru_lon - 0.25 - 1e-9)
                                                     & (lons + 0.125 <= cru_lon + 0.25 + 1e-9))),
    }
    if not all(checks.values()):
        raise AssertionError(f'GPW/CRU grid alignment failed: {checks}')
    return checks


def full_record_valid_counts(path, var):
    """Per-cell count of non-fill finite values over every month of the file (raw, unmasked reads)."""
    with netCDF4.Dataset(path) as nc:
        v = nc.variables[var]
        v.set_auto_maskandscale(False)
        fills = {float(v.getncattr(a)) for a in ('_FillValue', 'missing_value') if a in v.ncattrs()}
        count = np.zeros(v.shape[1:], dtype=np.int32)
        for t0 in range(0, v.shape[0], CHUNK_MONTHS):
            block = v[t0:t0 + CHUNK_MONTHS]
            ok = np.isfinite(block)
            for fill in fills:
                ok &= block != np.float32(fill)
            count += ok.sum(axis=0, dtype=np.int32)
        return count, int(v.shape[0])


def window_arrays(path, var):
    """The 360-month window exactly as the builder loads it (xarray decoding, float64)."""
    with xr.open_dataset(path) as ds:
        months, days = h.window_months(ds['time'].to_numpy())
        return ds[var].isel(time=months).to_numpy().astype(np.float64), days


def classify(pre_n, pet_n, valid, n):
    """Per-cell class for invalid cells; '' for valid cells."""
    both = (pre_n == 0) & (pet_n == 0)
    pet_only = (pre_n == n) & (pet_n == 0)
    pre_only = (pre_n == 0) & (pet_n == n)
    partial = ((pre_n > 0) & (pre_n < n)) | ((pet_n > 0) & (pet_n < n))
    nonpositive = (pre_n == n) & (pet_n == n) & ~valid
    out = np.full(valid.shape, '', dtype=object)
    for name, mask in zip(CLASSES, [both, pet_only, pre_only, partial, nonpositive]):
        out[mask & ~valid] = name
    if np.any((out == '') & ~valid):
        raise AssertionError('an invalid cell falls in no class')
    return out


def main(build_a, build_b):
    det = determinism(build_a, build_b)
    if not det['all_byte_identical']:
        raise AssertionError(f'rebuilds do not reproduce the preserved outputs: {det}')
    alignment = grid_alignment()
    pre_path = h.CRU_DIR / 'cru_ts4.10.1901.2025.pre.dat.nc'
    pet_path = h.CRU_DIR / 'cru_ts4.10.1901.2025.pet.dat.nc'
    pre, days = window_arrays(pre_path, 'pre')
    pet, _ = window_arrays(pet_path, 'pet')
    pre_n, pet_n = np.isfinite(pre).sum(axis=0), np.isfinite(pet).sum(axis=0)
    p_bar, e_bar, x, valid, _ = h.cell_climatology(pre, pet, days)
    del pre, pet
    pre_full, n_full = full_record_valid_counts(pre_path, 'pre')
    pet_full, _ = full_record_valid_counts(pet_path, 'pet')
    klass = classify(pre_n, pet_n, valid, h.N_MONTHS)

    table = pd.read_csv(h.COUNTRY_TABLE, usecols=['Country', 'iso3'])
    codes = table.iso3.tolist()
    land = np.load(h.LAND_AREA_NPZ)['cell_land_area_km2']
    country_of_cell, _, _ = gpw_country_grid(codes)
    cells = h.country_cells(codes, land, country_of_cell)
    i, k, a, idx = cells.i.to_numpy(), cells.k.to_numpy(), cells.area.to_numpy(), cells.idx.to_numpy()
    cru_cell_area = row_band_area_km2(-90 + 0.5 * i, -90 + 0.5 * (i + 1), 0.5)
    land_151 = np.bincount(i * 720 + k, weights=a, minlength=360 * 720).reshape(360, 720)
    fraction = land_151[i, k] / cru_cell_area
    ok = valid[i, k]
    cell_class = klass[i, k]
    fully_masked_record = (pre_full[i, k] == 0) | (pet_full[i, k] == 0)

    n = len(codes)
    total = np.bincount(idx, weights=a, minlength=n)
    missing = np.bincount(idx[~ok], weights=a[~ok], minlength=n)
    rows = pd.DataFrame({'iso3': codes, 'Country': table.Country, 'terrestrial_area_km2': total,
                         'valid_area_km2': total - missing, 'missing_area_km2': missing,
                         'coverage': (total - missing) / total})
    rows['passes_0_98'] = rows.coverage >= h.COVERAGE_MIN
    rows['n_support_cells'] = np.bincount(idx, minlength=n)
    rows['n_invalid_support_cells'] = np.bincount(idx[~ok], minlength=n)
    for name in CLASSES:
        m = cell_class == name
        rows[f'missing_area_{name}_km2'] = np.bincount(idx[m], weights=a[m], minlength=n)
    m = ~ok & fully_masked_record
    rows['missing_area_masked_all_1500_months_km2'] = np.bincount(idx[m], weights=a[m], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        rows['missing_cells_area_weighted_mean_151_country_land_fraction'] = (
            np.bincount(idx[~ok], weights=a[~ok] * fraction[~ok], minlength=n) / missing)
    rows['missing_cells_max_151_country_land_fraction'] = (
        pd.Series(fraction[~ok]).groupby(idx[~ok]).max().reindex(range(n)).to_numpy())

    preserved = pd.read_csv(AUDIT / 'm1b_hydroclimate_qa.csv')
    np.testing.assert_allclose(rows.coverage, preserved.coverage, rtol=1e-12, atol=0)
    rows.to_csv(AUDIT / 'm1b_original_coverage_by_country.csv', index=False)

    support = ~ok
    cells_unique = pd.DataFrame({'cell': i * 720 + k, 'cls': cell_class, 'invalid': ~ok}).drop_duplicates('cell')
    summary = {
        'builder_sha256': BUILDER_SHA256,
        'grid_alignment': alignment,
        'cru_grid_cells': int(valid.size),
        'window_months': h.N_MONTHS, 'full_record_months': n_full,
        'pre_window': {'all_valid': int((pre_n == h.N_MONTHS).sum()), 'all_missing': int((pre_n == 0).sum()),
                       'partial': int(((pre_n > 0) & (pre_n < h.N_MONTHS)).sum())},
        'pet_window': {'all_valid': int((pet_n == h.N_MONTHS).sum()), 'all_missing': int((pet_n == 0).sum()),
                       'partial': int(((pet_n > 0) & (pet_n < h.N_MONTHS)).sum())},
        'pre_full_record': {'all_valid': int((pre_full == n_full).sum()), 'all_missing': int((pre_full == 0).sum()),
                            'partial': int(((pre_full > 0) & (pre_full < n_full)).sum())},
        'pet_full_record': {'all_valid': int((pet_full == n_full).sum()), 'all_missing': int((pet_full == 0).sum()),
                            'partial': int(((pet_full > 0) & (pet_full < n_full)).sum())},
        'land_masks_window': {'pre_and_pet_land': int(((pre_n == h.N_MONTHS) & (pet_n == h.N_MONTHS)).sum()),
                              'pre_land_pet_masked': int(((pre_n == h.N_MONTHS) & (pet_n == 0)).sum()),
                              'pet_land_pre_masked': int(((pet_n == h.N_MONTHS) & (pre_n == 0)).sum())},
        'valid_c2_cells_global': int(valid.sum()),
        'valid_cells_with_nonpositive_mean': int((((pre_n == h.N_MONTHS) & (pet_n == h.N_MONTHS)) & ~valid).sum()),
        'support': {
            'distinct_cru_cells': int(len(cells_unique)),
            'distinct_invalid_cru_cells': int(cells_unique.invalid.sum()),
            'invalid_distinct_cells_by_class': {c: int((cells_unique.cls == c).sum()) for c in CLASSES},
            'invalid_country_cell_area_km2_by_class': {c: float(a[cell_class == c].sum()) for c in CLASSES},
            'invalid_area_km2_total': float(a[support].sum()),
            'invalid_area_masked_all_1500_months_km2': float(a[support & fully_masked_record].sum()),
            'invalid_area_fraction_of_151_country_support': float(a[support].sum() / a.sum()),
        },
        'coverage_failures': rows.loc[~rows.passes_0_98, ['iso3', 'coverage', 'missing_area_km2']].to_dict('records'),
        'near_threshold_passes_below_0_99': rows.loc[rows.passes_0_98 & (rows.coverage < NEAR_THRESHOLD),
                                                     ['iso3', 'coverage', 'missing_area_km2']].to_dict('records'),
        'warming_outcome_read': False,
    }
    (AUDIT / 'm1b_original_coverage_structural_mask.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (AUDIT / 'm1b_original_coverage_determinism.json').write_text(json.dumps(det, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ['pre_window', 'pet_window', 'pre_full_record', 'pet_full_record',
                                              'land_masks_window', 'support', 'coverage_failures',
                                              'near_threshold_passes_below_0_99']}, indent=2))
    low = rows[rows.coverage < NEAR_THRESHOLD].sort_values('coverage')
    cols = ['iso3', 'Country', 'terrestrial_area_km2', 'valid_area_km2', 'missing_area_km2', 'coverage',
            'n_support_cells', 'n_invalid_support_cells'] + [f'missing_area_{c}_km2' for c in CLASSES] + [
            'missing_area_masked_all_1500_months_km2', 'missing_cells_area_weighted_mean_151_country_land_fraction',
            'missing_cells_max_151_country_land_fraction']
    print(low[cols].to_string(index=False))
    return summary


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
