"""M1b C2: baseline hydroclimatic dryness, built under ``M1B_EVALUATION_CONTRACT.md`` (pushed ``60111ae``).

For each CRU TS v4.10 0.5° cell j, 1920-01..1949-12::

    P̄_j = (1/30) Σ_t PRE_{j,t}          (mm/month)
    Ē_j = (1/30) Σ_t PET_{j,t} · N_t    (mm/day × calendar days)
    x_j = log10(P̄_j / Ē_j)

A_{c,j} is the frozen M1a terrestrial area. Under the contract's Amendment 1, terrestrial area in a
cell that CRU's fixed land mask excludes (a structural-mask target) takes the already-computed x of
the nearest native-valid cell among its eight neighbours (WGS84 geodesic); native values never
change, harmonized values never act as donors, and a target without such a neighbour stays
unresolved::

    C2_c = Σ_{native ∪ harmonized} A_{c,j} x_j / Σ_{native ∪ harmonized} A_{c,j}

Inputs are only the frozen M1a support record (country identifiers, terrestrial area, per-cell land
area), the GPW country grid and the pinned CRU PRE/PET files; no file holding a warming outcome is
opened. Writes only ``outputs/m1b_*``.
Run with ``python -m research.model_v2.m1b_hydroclimate [out_dir]``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from research.model_v2 import m0
from research.model_v2.m1a_geography import gpw_country_grid
from research.model_v2.territory import GEOD
from src.area_weighting import GPW_NATID_LOOKUP_PATH
from src.population import GPW_ENGINE, GPW_PATH

OUT = m0.OUTPUT_DIR
CRU_DIR = m0.ROOT / 'data' / 'raw' / 'cru_ts_4.10'
BASE_URL = 'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/cruts.2604091129.v4.10/'
# Contract §1.2 pins: sizes from the pre-acquisition review, SHA-256 committed with the feasibility audit (c5ff9b5).
FILES = {
    'pre': {'path': 'pre/cru_ts4.10.1901.2025.pre.dat.nc.gz', 'bytes': 698_006_393, 'units': 'mm/month',
            'gz_sha256': 'b7e7a3b74e9887db74d8db62227800708ff89ebbabeed5b11f302c99a551edaa',
            'nc_bytes': 6_220_812_084, 'nc_sha256': 'efe27c453101fd8b98b0544a198f457b872cf983a9cadb791ecd8e9cddbb3eae'},
    'pet': {'path': 'pet/cru_ts4.10.1901.2025.pet.dat.nc.gz', 'bytes': 72_951_284, 'units': 'mm/day',
            'gz_sha256': 'fcc792b090a651d9a19187acf105cd726d680171d01ee48630f0a1a10816f013',
            'nc_bytes': 1_555_211_692, 'nc_sha256': '0bc9a1319f76d93a9f8fb8517ce82399b8c5f6e59dfcf26d802f0dbbd1e00715'},
}
NETCDF_ENGINE = 'netcdf4'
WINDOW = ('1920-01', '1949-12')
N_MONTHS = 360
COVERAGE_MIN = 0.98
STN_MAX = 8
LAND_AREA_NPZ = OUT / 'm1a_cell_land_area_km2.npz'
M1A_QA = OUT / 'm1a_geography_qa.csv'  # frozen at a7dea34; the outcome-free source of identifiers and row order
SUPPORT_COLUMNS = ['iso3', 'Country', 'terrestrial_area_km2']
GRID_ATOL_DEG = 1e-6  # ~0.1 m; GPW and CRU store exact multiples of 0.125°
CONTRACT_COMMIT = '60111ae1d44bdf9da6c613852a32de541f9b2d7a'  # the original freeze; amendments are recorded separately
CONTRACT_PATH = Path(__file__).with_name('M1B_EVALUATION_CONTRACT.md')
LAND_SUPPORT_COMMIT = 'a7dea346b0d808aab8d9cf84058cfac25c5796b0'
LAND_SUPPORT_SHA256 = {LAND_AREA_NPZ: '642cbe5abe48457682f312884c03510c13b133a74bc083039b537f0b738466cf',
                       M1A_QA: 'bdc4f1caff740fea7949b636c367afb6c68b423eccebb07c9391b357358425c8'}
CRU_VERSION, CRU_RUN_ID = '4.10', '2604091129'
# Local code on the construction path, checked against the build commit.
CODE_PATH = ('research/model_v2/m1b_hydroclimate.py', 'research/model_v2/M1B_EVALUATION_CONTRACT.md',
             'research/model_v2/m0.py', 'research/model_v2/m1a_geography.py', 'research/model_v2/geometry.py',
             'research/model_v2/territory.py', 'src/population.py', 'src/area_weighting.py')

# Amendment 1: one-ring structural land-mask harmonization.
AMENDMENT = 'M1B_EVALUATION_CONTRACT.md Amendment 1 (2026-09-15)'
N_LAT, N_LON, CELL_DEG = 360, 720, 0.5
NEIGHBOUR_OFFSETS = tuple((di, dk) for di in (-1, 0, 1) for dk in (-1, 0, 1) if (di, dk) != (0, 0))
NATIVE, HARMONIZED, UNRESOLVED = 'native', 'harmonized', 'unresolved'
STRUCTURAL, NON_STRUCTURAL = 'structural_cru_mask', 'non_structural_invalid'
MASKED, COMPLETE = 'masked', 'complete'
AREA_RTOL = 1e-10  # float64 summation error over <= ~1e5 cell addends stays below this relative bound
FILL_CHUNK_MONTHS = 60
DETERMINISTIC_OUTPUTS = ('m1b_hydroclimate_features.csv', 'm1b_hydroclimate_qa.csv', 'm1b_harmonization_cells.csv',
                         'm1b_support_checkpoint.json', 'm1b_measurement_manifest.json')
RUN_METADATA = 'm1b_build_run.json'  # wall-clock metadata, deliberately outside the byte-compared outputs


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def decompressed(gz_path):
    """Deterministically gunzip next to the archive (once); returns the .nc path."""
    nc = gz_path.with_suffix('')
    if not nc.exists():
        tmp = nc.with_suffix('.nc.partial')
        with gzip.open(gz_path, 'rb') as src, tmp.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        tmp.rename(nc)
    return nc


def pinned_source(spec):
    """The decompressed file and its hash record, after size and SHA-256 checks of both files; fails closed."""
    gz = CRU_DIR / Path(spec['path']).name
    if gz.stat().st_size != spec['bytes']:
        raise ValueError(f'{gz.name}: {gz.stat().st_size} bytes, contract pins {spec["bytes"]}')
    gz_hash = sha256(gz)
    if gz_hash != spec['gz_sha256']:
        raise ValueError(f'{gz.name}: sha256 {gz_hash} differs from the pin')
    nc = decompressed(gz)
    nc_hash = sha256(nc)
    if nc.stat().st_size != spec['nc_bytes'] or nc_hash != spec['nc_sha256']:
        raise ValueError(f'{nc.name}: size or sha256 differs from the pin')
    return nc, {'gz_bytes': gz.stat().st_size, 'gz_sha256': gz_hash, 'nc_bytes': nc.stat().st_size, 'nc_sha256': nc_hash}


def check_support_nesting(gpw_lat, gpw_lon):
    """Assert each GPW 0.25° cell lies inside CRU 0.5° cell (row // 2, col // 2) (contract §2.3).

    ``gpw_lat`` is in GPW storage order (north to south), which ``gpw_country_grid`` flips to ascending.
    Centres at -89.875 + 0.25 r and -179.875 + 0.25 c put both edges of every GPW cell on the CRU cell's
    edges or inside it.
    """
    lat, lon = np.asarray(gpw_lat, dtype=float)[::-1], np.asarray(gpw_lon, dtype=float)
    nested = (lat.shape == (720,) and lon.shape == (1440,)
              and np.allclose(lat, -89.875 + 0.25 * np.arange(720), rtol=0, atol=GRID_ATOL_DEG)
              and np.allclose(lon, -179.875 + 0.25 * np.arange(1440), rtol=0, atol=GRID_ATOL_DEG))
    if not nested:
        raise ValueError('GPW 0.25° grid does not nest 2 × 2 inside the CRU 0.5° grid')


def window_months(times):
    """Indices of exactly the 360 contract months, and their calendar day counts."""
    index = pd.DatetimeIndex(times)
    period = index.to_period('M')
    keep = (period >= pd.Period(WINDOW[0], 'M')) & (period <= pd.Period(WINDOW[1], 'M'))
    selected = period[keep]
    expected = pd.period_range(WINDOW[0], WINDOW[1], freq='M')
    if len(selected) != N_MONTHS or not (selected == expected).all():
        raise ValueError('time axis does not contain exactly the 360 contract months')
    return np.flatnonzero(keep), np.asarray(expected.days_in_month, dtype=float)


def check_grid(ds):
    lat, lon = ds['lat'].to_numpy(), ds['lon'].to_numpy()
    ok_lat = len(lat) == 360 and np.allclose(lat, -89.75 + 0.5 * np.arange(360))
    ok_lon = len(lon) == 720 and np.allclose(lon, -179.75 + 0.5 * np.arange(720))
    if not (ok_lat and ok_lon):
        raise ValueError('CRU grid is not the documented ascending 0.5° grid')


def version_attrs(ds):
    attrs = {k: str(v) for k, v in ds.attrs.items()}
    if not any('4.10' in v for v in attrs.values()):
        raise ValueError('CRU global metadata does not identify version 4.10')
    return attrs


def cell_climatology(pre, pet, days):
    """Per-cell P̄, Ē, x and validity for arrays shaped (360 months, lat, lon)."""
    finite = np.isfinite(pre).all(axis=0) & np.isfinite(pet).all(axis=0)
    p_bar = np.nansum(pre, axis=0) / 30.0
    e_bar = np.nansum(pet * days[:, None, None], axis=0) / 30.0
    valid = finite & (p_bar > 0) & (e_bar > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where(valid, np.log10(p_bar / e_bar), np.nan)
    reason = np.where(~finite, 'non-finite month', np.where(p_bar <= 0, 'P_bar <= 0',
                      np.where(e_bar <= 0, 'E_bar <= 0', '')))
    return p_bar, e_bar, x, valid, reason


def country_cells(codes, land_area, country_of_cell):
    """Long table of (country index, CRU row, CRU col, terrestrial km²) from GPW 0.25° cells."""
    r, c = np.nonzero((country_of_cell >= 0) & (land_area > 0))
    frame = pd.DataFrame({'idx': country_of_cell[r, c].astype(int), 'i': r // 2, 'k': c // 2,
                          'area': land_area[r, c]})
    return frame.groupby(['idx', 'i', 'k'], as_index=False, sort=True).area.sum()


def support_quantities(stn, valid):
    """Per-cell PRE station support over the window, audited over valid cells (contract §2.5).

    CRU stores fill values for ``stn`` in non-land cells, where PRE is also missing. Those cells
    are already invalid for C2 and are not part of the audit. Any missing or out-of-range count in
    a valid cell raises.
    """
    audited = stn[:, valid]
    if not np.isfinite(audited).all():
        raise ValueError('PRE stn has non-finite values in valid cells in the window')
    if (audited < 0).any() or (audited > STN_MAX).any() or not np.array_equal(audited, np.round(audited)):
        raise ValueError('PRE stn outside the documented integer 0-8 range in valid cells')
    with np.errstate(invalid='ignore'):
        supported = np.where(valid, (stn >= 1).mean(axis=0), np.nan)
        mean_stn = np.where(valid, np.nanmean(np.where(valid, stn, 0), axis=0), np.nan)
    pure = valid & (np.where(valid, stn, 1) == 0).all(axis=0)
    return supported, mean_stn, pure


def lat_centre(lat_index):
    return -89.75 + CELL_DEG * np.asarray(lat_index)


def lon_centre(lon_index):
    return -179.75 + CELL_DEG * np.asarray(lon_index)


def _neighbour_distance_table():
    """WGS84 geodesic distance (km) from a row-i CRU centre to its neighbour at (Δi, |Δk|); shape (360, 3, 2).

    Karney's inverse geodesic (pyproj ``Geod``) on the WGS84 ellipsoid of the frozen territorial CV,
    evaluated once from whole-cell offsets with a non-negative longitude step. East/west mirror-image
    neighbours therefore share one stored value and tie exactly; rows beyond a pole are +inf.
    """
    table = np.full((N_LAT, 3, 2), np.inf)
    i, d_lat, d_lon = np.meshgrid(np.arange(N_LAT), np.array([-1, 0, 1]), np.array([0, 1]), indexing='ij')
    lat1, lat2 = lat_centre(i), lat_centre(i + d_lat)
    ok = (np.abs(lat2) < 90) & ((d_lat != 0) | (d_lon != 0))
    _, _, metres = GEOD.inv(np.zeros(int(ok.sum())), lat1[ok], CELL_DEG * d_lon[ok], lat2[ok])
    table[ok] = np.asarray(metres, dtype=float) / 1000
    return table


NEIGHBOUR_DISTANCE_KM = _neighbour_distance_table()


def neighbour_distance_km(lat_index, d_lat, d_lon):
    """Distance between the CRU centre in row ``lat_index`` and its neighbour offset by (d_lat, d_lon) cells."""
    return NEIGHBOUR_DISTANCE_KM[lat_index, np.asarray(d_lat) + 1, np.abs(d_lon)]


def fill_count(nc_path, var, months):
    """Per-cell count of raw values equal to the variable's declared fill or missing value over ``months``.

    Values are read undecoded in chunks, so a fill value is never confused with a malformed NaN or
    infinity. ``months`` must be a contiguous index range.
    """
    months = np.asarray(months)
    if months.size == 0 or np.any(np.diff(months) != 1):
        raise ValueError('months must be a contiguous, non-empty index range')
    with netCDF4.Dataset(nc_path) as nc:
        v = nc.variables[var]
        v.set_auto_maskandscale(False)
        fills = [v.getncattr(name) for name in ('_FillValue', 'missing_value') if name in v.ncattrs()]
        if not fills:
            raise ValueError(f'{var} declares no fill or missing value')
        count = np.zeros(v.shape[1:], dtype=np.int32)
        for start in range(int(months[0]), int(months[-1]) + 1, FILL_CHUNK_MONTHS):
            block = v[start:min(start + FILL_CHUNK_MONTHS, int(months[-1]) + 1)]
            is_fill = np.zeros(block.shape, dtype=bool)
            for fill in fills:
                is_fill |= block == fill
            count += is_fill.sum(axis=0, dtype=np.int32)
    return count


def variable_state(record_fill, window_fill, window_finite, window_mean, n_record_months, n_window_months=N_MONTHS):
    """Per-cell state of one variable (Amendment 1 A1.2), checked in this order.

    * 'masked': the declared fill value in every month of the pinned record (CRU's fixed land mask);
    * 'complete': every window month finite and a positive climatological mean;
    * 'nonpositive_mean': every window month finite, mean not positive;
    * 'partial_fill': fill in some window months but not in the whole record;
    * 'malformed': a non-finite window value that is not the fill value.
    """
    return np.select([record_fill == n_record_months, (window_finite == n_window_months) & (window_mean > 0),
                      window_finite == n_window_months, window_fill > 0],
                     [MASKED, COMPLETE, 'nonpositive_mean', 'partial_fill'], 'malformed').astype(object)


def structural_mask_cells(pre_state, pet_state):
    """Cells lacking C2 only because CRU's fixed mask excludes PRE, PET or both: at least one variable
    masked and each variable masked or complete."""
    usable = [MASKED, COMPLETE]
    return (((pre_state == MASKED) | (pet_state == MASKED))
            & np.isin(pre_state, usable) & np.isin(pet_state, usable))


def one_ring_donors(targets, native, x):
    """Nearest native-valid donor among the eight cells adjacent to each target (Amendment 1 A1.3-A1.5).

    Longitude wraps; latitude does not. Eligibility reads only ``native``, so a harmonized target is
    never a candidate. Ties go to the smaller latitude index, then the smaller (wrapped) longitude
    index. Returns one row per target in (latitude index, longitude index) order; unresolved targets
    have donor indices -1 and NaN distance, centres and value.
    """
    targets, native = np.asarray(targets, dtype=bool), np.asarray(native, dtype=bool)
    if not (targets.shape == native.shape == np.shape(x) == (N_LAT, N_LON)):
        raise ValueError('targets, native and x must be CRU (360, 720) grids')
    if (targets & native).any():
        raise ValueError('a structural-mask target is also a native-valid cell')
    ti, tk = np.nonzero(targets)
    offsets = np.asarray(NEIGHBOUR_OFFSETS)
    d_lat, d_lon = offsets[:, 0][None, :], offsets[:, 1][None, :]
    ci = ti[:, None] + d_lat
    ck = (tk[:, None] + d_lon) % N_LON
    inside = (ci >= 0) & (ci < N_LAT)
    ci = np.clip(ci, 0, N_LAT - 1)
    eligible = inside & native[ci, ck]
    distance = np.where(eligible, NEIGHBOUR_DISTANCE_KM[ti[:, None], d_lat + 1, np.abs(d_lon)], np.inf)
    best = np.lexsort((ck, ci, distance), axis=1)[:, 0]
    rows = np.arange(len(ti))
    found = eligible[rows, best] & np.isfinite(distance[rows, best])  # beyond-pole table entries are +inf
    donor_i = np.where(found, ci[rows, best], -1)
    donor_k = np.where(found, ck[rows, best], -1)
    return pd.DataFrame({
        'target_lat_index': ti, 'target_lon_index': tk, 'target_lat': lat_centre(ti), 'target_lon': lon_centre(tk),
        'donor_lat_index': donor_i, 'donor_lon_index': donor_k,
        'donor_lat': np.where(found, lat_centre(donor_i), np.nan), 'donor_lon': np.where(found, lon_centre(donor_k), np.nan),
        'donor_distance_km': np.where(found, distance[rows, best], np.nan),
        'donor_c2': np.where(found, np.asarray(x)[np.maximum(donor_i, 0), np.maximum(donor_k, 0)], np.nan),
    })


def resolve_support(cells, x, native, structural, donors):
    """Status, reason and C2 value of every (country, CRU cell) support row, in ``cells`` order.

    * native: the cell's own x;
    * harmonized: a structural-mask target with a one-ring donor, carrying the donor's x;
    * unresolved: a structural target without a donor, or any other invalid cell
      (reason ``non_structural_invalid``), which never receives a value.
    """
    i, k, idx = cells.i.to_numpy(), cells.k.to_numpy(), cells.idx.to_numpy()
    is_native, is_structural = np.asarray(native)[i, k], np.asarray(structural)[i, k]
    if (is_native & is_structural).any():
        raise ValueError('a support cell is both native-valid and a structural-mask target')
    frame = cells.merge(donors.drop(columns=['target_lat', 'target_lon']), how='left', validate='many_to_one',
                        left_on=['i', 'k'], right_on=['target_lat_index', 'target_lon_index'])
    if frame.target_lat_index.isna().to_numpy()[is_structural].any():
        raise ValueError('a structural-mask support cell has no donor-search record')
    harmonized = is_structural & (frame.donor_lat_index.fillna(-1).to_numpy() >= 0)
    donor_key = np.where(harmonized, frame.donor_lat_index.fillna(0).to_numpy() * N_LON
                         + frame.donor_lon_index.fillna(0).to_numpy(), -1).astype(np.int64)
    support_pairs = (i.astype(np.int64) * N_LON + k) * 1000 + idx
    cross_border = harmonized & ~np.isin(donor_key * 1000 + idx, support_pairs)
    out = cells.copy()
    out['target_lat'], out['target_lon'] = lat_centre(i), lon_centre(k)
    out['status'] = np.select([is_native, harmonized], [NATIVE, HARMONIZED], UNRESOLVED)
    out['reason'] = np.select([is_native, is_structural], ['', STRUCTURAL], NON_STRUCTURAL)
    out['c2_value'] = np.where(is_native, np.asarray(x)[i, k], np.where(harmonized, frame.donor_c2.to_numpy(), np.nan))
    for col in ['donor_lat_index', 'donor_lon_index']:
        out[col] = frame[col].where(harmonized).astype('Int64').array
    for col in ['donor_lat', 'donor_lon', 'donor_distance_km', 'donor_c2']:
        out[col] = frame[col].where(harmonized).to_numpy(dtype=float)
    out['cross_border'] = pd.array(np.where(harmonized, cross_border, None).tolist(), dtype='boolean')
    return out


def aggregate(resolved, p_bar, e_bar, supported, mean_stn, pure, n):
    """Per-country C2, area partition, donor-distance QA and the native-cell §2.5 support audit."""
    idx, i, k = resolved.idx.to_numpy(), resolved.i.to_numpy(), resolved.k.to_numpy()
    a, status = resolved.area.to_numpy(dtype=float), resolved.status.to_numpy()
    value, dist = resolved.c2_value.to_numpy(dtype=float), resolved.donor_distance_km.to_numpy(dtype=float)
    native, harmonized, unresolved = status == NATIVE, status == HARMONIZED, status == UNRESOLVED
    non_structural = resolved.reason.to_numpy() == NON_STRUCTURAL
    covered = native | harmonized
    cross = harmonized & resolved.cross_border.fillna(False).to_numpy(dtype=bool)

    def area(mask):
        return np.bincount(idx[mask], weights=a[mask], minlength=n)

    def count(mask):
        return np.bincount(idx[mask], minlength=n)

    def weighted(values, mask):
        return np.bincount(idx[mask], weights=a[mask] * values[mask], minlength=n)

    total = np.bincount(idx, weights=a, minlength=n)
    native_area, harmonized_area, unresolved_area, covered_area = area(native), area(harmonized), area(unresolved), area(covered)
    with np.errstate(invalid='ignore', divide='ignore'):
        c2_native = weighted(value, native) / native_area
        c2_harmonized = weighted(value, covered) / covered_area
        qa = pd.DataFrame({
            'total_land_area_km2': total,
            'native_valid_area_km2': native_area,
            'harmonized_area_km2': harmonized_area,
            'unresolved_area_km2': unresolved_area,
            'unresolved_structural_area_km2': area(unresolved & ~non_structural),
            'non_structural_invalid_area_km2': area(non_structural),
            'native_fraction': native_area / total,
            'harmonized_fraction': harmonized_area / total,
            'unresolved_fraction': unresolved_area / total,
            'coverage': covered_area / total,
            'n_cru_cells': count(np.ones_like(native)),
            'n_native_cells': count(native),
            'n_harmonized_cells': count(harmonized),
            'n_unresolved_cells': count(unresolved),
            'n_non_structural_invalid_cells': count(non_structural),
            'n_cross_border_cells': count(cross),
            'cross_border_area_km2': area(cross),
            'mean_donor_distance_km': np.bincount(idx[harmonized], weights=dist[harmonized], minlength=n) / count(harmonized),
            'area_weighted_mean_donor_distance_km': weighted(dist, harmonized) / harmonized_area,
            'max_donor_distance_km': pd.Series(dist[harmonized]).groupby(idx[harmonized]).max().reindex(range(n)).to_numpy(),
            'c2_native_support_only': c2_native,
            'c2_harmonized': c2_harmonized,
            'delta_c2_harmonization': c2_harmonized - c2_native,
            'p_bar_native_area_mean_mm_yr': weighted(p_bar[i, k], native) / native_area,
            'e_bar_native_area_mean_mm_yr': weighted(e_bar[i, k], native) / native_area,
            'pre_station_supported_share': weighted(supported[i, k], native) / native_area,
            'pre_mean_station_count': weighted(mean_stn[i, k], native) / native_area,
            'pre_pure_climatology_share': weighted(pure[i, k].astype(float), native) / native_area,
        })
    return qa


def cell_audit(resolved, codes, pre_state, pet_state):
    """One row per (non-native support cell, country): variable states, donor, distance, value, cross-border flag."""
    rows = resolved[resolved.status != NATIVE]
    harmonized = (rows.status == HARMONIZED).to_numpy()
    donor_i = rows.donor_lat_index.fillna(-1).to_numpy(dtype=np.int64)
    donor_k = rows.donor_lon_index.fillna(-1).to_numpy(dtype=np.int64)
    donor_cells = set(zip(donor_i[harmonized].tolist(), donor_k[harmonized].tolist()))
    in_donor = [(a, b) in donor_cells for a, b in zip(resolved.i.tolist(), resolved.k.tolist())]
    names = (resolved[in_donor].assign(iso3=lambda f: [codes[j] for j in f.idx])
             .groupby(['i', 'k']).iso3.agg(lambda s: ';'.join(sorted(set(s)))))
    donor_countries = [names.get((a, b), 'none') if h else None for a, b, h in zip(donor_i, donor_k, harmonized)]
    ri, rk = rows.i.to_numpy(), rows.k.to_numpy()
    audit = pd.DataFrame({
        'target_lat_index': ri, 'target_lon_index': rk,
        'target_lat': rows.target_lat.to_numpy(), 'target_lon': rows.target_lon.to_numpy(),
        'iso3': [codes[j] for j in rows.idx], 'target_land_area_km2': rows.area.to_numpy(),
        'pre_state': np.asarray(pre_state)[ri, rk], 'pet_state': np.asarray(pet_state)[ri, rk],
        'status': rows.status.to_numpy(), 'reason': rows.reason.to_numpy(),
        'donor_lat_index': rows.donor_lat_index.array, 'donor_lon_index': rows.donor_lon_index.array,
        'donor_lat': rows.donor_lat.to_numpy(), 'donor_lon': rows.donor_lon.to_numpy(),
        'donor_distance_km': rows.donor_distance_km.to_numpy(), 'donor_c2': rows.donor_c2.to_numpy(),
        'donor_countries': pd.array(donor_countries, dtype=object),
        'cross_border': rows.cross_border.array,
    })
    return audit.sort_values(['target_lat_index', 'target_lon_index', 'iso3'], kind='stable').reset_index(drop=True)


def harmonization_summary(resolved, support):
    """Global Amendment 1 counts, areas and donor distances (distinct cells unless stated)."""
    targets = resolved[resolved.reason == STRUCTURAL]
    cells = targets.drop_duplicates(['i', 'k'])
    done = cells[cells.status == HARMONIZED]
    d = done.donor_distance_km.to_numpy(dtype=float)
    cross = targets[targets.cross_border.fillna(False).to_numpy(dtype=bool)]
    donors = done[['donor_lat_index', 'donor_lon_index']].astype(int).drop_duplicates()
    return {
        'n_structural_target_cells': int(len(cells)),
        'n_harmonized_cells': int(len(done)),
        'n_unresolved_structural_cells': int((cells.status == UNRESOLVED).sum()),
        'n_non_structural_invalid_cells': int(resolved[resolved.reason == NON_STRUCTURAL].drop_duplicates(['i', 'k']).shape[0]),
        'harmonized_area_km2': float(resolved.area[resolved.status == HARMONIZED].sum()),
        'unresolved_area_km2': float(resolved.area[resolved.status == UNRESOLVED].sum()),
        'unresolved_structural_area_km2': float(targets.area[targets.status == UNRESOLVED].sum()),
        'non_structural_invalid_area_km2': float(resolved.area[resolved.reason == NON_STRUCTURAL].sum()),
        'donor_distance_km': {'min': float(d.min()), 'median': float(np.median(d)), 'mean': float(d.mean()),
                              'p95': float(np.quantile(d, 0.95)), 'max': float(d.max())} if d.size else None,
        'donor_distance_statistics_over': 'distinct harmonized target cells; quantiles by numpy linear interpolation',
        'n_cross_border_country_cells': int(len(cross)),
        'cross_border_area_km2': float(cross.area.sum()),
        'n_distinct_donor_cells': int(len(donors)),
        'n_donor_cells_outside_all_support': int((~support[donors.donor_lat_index.to_numpy(), donors.donor_lon_index.to_numpy()]).sum()),
    }


def hard_stops(qa, codes):
    """Contract §2.4-§2.5 and Amendment 1 A1.7 stops, one record per country and rule (NaN fails closed)."""
    stops = []
    for j, code in enumerate(codes):
        r = qa.iloc[j]
        if not r.coverage >= COVERAGE_MIN:
            stops.append({'iso3': code, 'rule': 'coverage < 0.98', 'coverage': float(r.coverage),
                          'unresolved_structural_area_km2': float(r.unresolved_structural_area_km2),
                          'non_structural_invalid_area_km2': float(r.non_structural_invalid_area_km2)})
        if not r.n_non_structural_invalid_cells == 0:
            stops.append({'iso3': code, 'rule': 'non-structural invalid support cells (Amendment 1 A1.2): review',
                          'cells': int(r.n_non_structural_invalid_cells)})
        if not r.native_valid_area_km2 > 0:
            stops.append({'iso3': code, 'rule': 'no native-valid terrestrial area: station audit undefined'})
        if not r.pre_pure_climatology_share < 1:
            stops.append({'iso3': code, 'rule': 'pathological: entire terrestrial area pure climatology for all 360 months'})
        if not np.isfinite(r.c2_harmonized):
            stops.append({'iso3': code, 'rule': 'non-finite C2'})
    return stops


def distribution(series):
    q = [0, .1, .25, .5, .75, .9, 1]
    return dict(zip(['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'], map(float, series.quantile(q))))


def csv_text(frame):
    return frame.to_csv(index=False, na_rep='NA', lineterminator='\n')


def write_csv(frame, path):
    Path(path).write_text(csv_text(frame))


def git_state():
    def run(*args):
        return subprocess.run(['git', *args], cwd=m0.ROOT, capture_output=True, text=True)
    head = run('rev-parse', 'HEAD')
    if head.returncode != 0:
        return {'commit': None, 'code_path_matches_commit': None, 'contract_amendment_commit': None}
    return {'commit': head.stdout.strip(),
            'code_path': list(CODE_PATH),
            'code_path_matches_commit': run('diff', '--quiet', 'HEAD', '--', *CODE_PATH).returncode == 0,
            'contract_amendment_commit': run('log', '-1', '--format=%H', '--', CODE_PATH[1]).stdout.strip()}


def build(out=OUT):
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    code_hash = sha256(Path(__file__))  # pinned before any work; never edit the file mid-run
    for path, pin in LAND_SUPPORT_SHA256.items():
        if sha256(path) != pin:
            raise ValueError(f'{path.name} differs from the frozen M1a support at {LAND_SUPPORT_COMMIT[:7]}')
    table = pd.read_csv(M1A_QA, usecols=SUPPORT_COLUMNS)[SUPPORT_COLUMNS]
    if table.columns.tolist() != SUPPORT_COLUMNS or len(table) != 151:
        raise ValueError('M1a support record does not have the expected identifier and area columns')
    codes = table.iso3.tolist()
    with xr.open_dataset(GPW_PATH, engine=GPW_ENGINE) as ds:
        check_support_nesting(ds['latitude'].to_numpy(), ds['longitude'].to_numpy())
    sources, arrays, record_fill, window_fill, window_finite = {}, {}, {}, {}, {}
    for var, spec in FILES.items():
        nc, hashes = pinned_source(spec)
        with xr.open_dataset(nc, engine=NETCDF_ENGINE) as ds:
            check_grid(ds)
            if var not in ds or ds[var].attrs.get('units') != spec['units']:
                raise ValueError(f'{nc.name}: variable {var} with units {spec["units"]} not found')
            if var == 'pre' and 'stn' not in ds:
                raise ValueError('PRE file lacks the stn station-count variable')
            months, days = window_months(ds['time'].to_numpy())
            n_record = int(ds.sizes['time'])
            arrays[var] = ds[var].isel(time=months).to_numpy().astype(np.float64)
            if var == 'pre':
                arrays['stn'] = ds['stn'].isel(time=months).to_numpy().astype(np.float64)
            sources[var] = {'url': BASE_URL + spec['path'], **hashes,
                            'units': ds[var].attrs.get('units'), 'global_attrs': version_attrs(ds),
                            'window_first_last': [str(pd.DatetimeIndex(ds['time'].to_numpy()[months])[i].date())
                                                  for i in (0, -1)], 'record_months': n_record}
        record_fill[var] = fill_count(nc, var, np.arange(n_record))
        window_fill[var] = fill_count(nc, var, months)
        window_finite[var] = np.isfinite(arrays[var]).sum(axis=0)
    p_bar, e_bar, x, native, _ = cell_climatology(arrays['pre'], arrays['pet'], days)
    supported, mean_stn, pure = support_quantities(arrays['stn'], native)
    del arrays
    pre_state = variable_state(record_fill['pre'], window_fill['pre'], window_finite['pre'], p_bar, sources['pre']['record_months'])
    pet_state = variable_state(record_fill['pet'], window_fill['pet'], window_finite['pet'], e_bar, sources['pet']['record_months'])
    if not np.array_equal(native, (pre_state == COMPLETE) & (pet_state == COMPLETE)):
        raise AssertionError('variable states do not reproduce the §2.2 native validity')
    structural = structural_mask_cells(pre_state, pet_state)
    if (structural & native).any():
        raise AssertionError('a native-valid cell is classified as a structural-mask cell')

    land = np.load(LAND_AREA_NPZ)['cell_land_area_km2']
    country_of_cell, _, _ = gpw_country_grid(codes)
    cells = country_cells(codes, land, country_of_cell)
    support = np.zeros((N_LAT, N_LON), dtype=bool)
    support[cells.i.to_numpy(), cells.k.to_numpy()] = True
    donors = one_ring_donors(structural & support, native, x)
    resolved = resolve_support(cells, x, native, structural, donors)
    qa = aggregate(resolved, p_bar, e_bar, supported, mean_stn, pure, len(codes))
    qa.insert(0, 'iso3', codes)
    qa.insert(1, 'Country', table.Country)

    rel = np.abs(qa.total_land_area_km2.to_numpy() / table.terrestrial_area_km2.to_numpy() - 1)
    if not np.all(rel <= 1e-9):
        raise ValueError(f'CRU-cell land area does not reproduce M1a terrestrial area (max rel {np.nanmax(rel):.2e})')
    parts = qa.native_valid_area_km2 + qa.harmonized_area_km2 + qa.unresolved_area_km2
    partition = np.abs(parts.to_numpy() / qa.total_land_area_km2.to_numpy() - 1)
    if not np.all(partition <= AREA_RTOL):
        raise AssertionError(f'native + harmonized + unresolved area differs from terrestrial area ({np.nanmax(partition):.2e})')
    stops = hard_stops(qa, codes)

    checkpoint = {
        'distributions': {col: distribution(qa[col]) for col in
                          ['pre_station_supported_share', 'pre_mean_station_count', 'pre_pure_climatology_share']},
        'extremes_top10': {
            'lowest_station_supported_share': qa.nsmallest(10, 'pre_station_supported_share')[['iso3', 'pre_station_supported_share']].values.tolist(),
            'lowest_mean_station_count': qa.nsmallest(10, 'pre_mean_station_count')[['iso3', 'pre_mean_station_count']].values.tolist(),
            'highest_pure_climatology_share': qa.nlargest(10, 'pre_pure_climatology_share')[['iso3', 'pre_pure_climatology_share']].values.tolist(),
        },
        'pet_limitation': 'PET has no meaningful station-count field (Harris et al. 2020); its support is not quantified.',
    }
    audit = cell_audit(resolved, codes, pre_state, pet_state)
    manifest = {
        'contract': 'research/model_v2/M1B_EVALUATION_CONTRACT.md', 'contract_freeze_commit': CONTRACT_COMMIT,
        'contract_sha256': sha256(CONTRACT_PATH), 'amendment': AMENDMENT, 'git': git_state(),
        'description': 'CRU-reconstructed 1920-1949 baseline hydroclimatic dryness (not purely observed pre-1950 dryness)',
        'sources': sources, 'cru_version': CRU_VERSION, 'cru_run_id': CRU_RUN_ID,
        'support_inputs': {'m1a_cell_land_area_sha256': sha256(LAND_AREA_NPZ), 'm1a_geography_qa_sha256': sha256(M1A_QA),
                           'land_support_commit': LAND_SUPPORT_COMMIT, 'land_support_pins_asserted': True,
                           'gpw_path': str(GPW_PATH.relative_to(m0.ROOT)), 'gpw_sha256': sha256(GPW_PATH),
                           'gpw_national_identifier_lookup_sha256': sha256(GPW_NATID_LOOKUP_PATH),
                           'country_list_sha256': hashlib.sha256(('\n'.join(codes) + '\n').encode()).hexdigest(),
                           'grid_nesting': 'GPW 0.25 deg nests 2x2 inside CRU 0.5 deg (asserted)'},
        'window': list(WINDOW), 'n_months': N_MONTHS,
        'formula': 'x_j = log10((1/30)*sum PRE / ((1/30)*sum PET*days)); C2 = terrestrial-area-weighted mean of x_j '
                   'over native-valid cells and Amendment 1 harmonized structural-mask cells',
        'formula_version': 'contract section 2 + Amendment 1',
        'harmonization_rule': {
            'targets': 'M1a support cell, not native-valid, at least one of PRE/PET fill in every record month, '
                       'each variable masked or complete (all 360 window months finite with a positive mean)',
            'candidates': 'the 8 adjacent CRU cells; longitude index wraps modulo 720; latitude does not wrap',
            'donor_eligibility': 'native-valid CRU cell, any country or none; harmonized cells never donate',
            'distance': 'WGS84 ellipsoidal geodesic between cell centres (Karney inverse, pyproj Geod), '
                        'tabulated from whole-cell offsets',
            'tie_break': ['minimum distance', 'minimum latitude index', 'minimum wrapped longitude index'],
            'value': "donor's native x_j as one derived value",
        },
        'coverage_min': COVERAGE_MIN, 'cells_native_valid_global': int(native.sum()),
        'cells_fixed_mask_any_variable_global': int(((pre_state == MASKED) | (pet_state == MASKED)).sum()),
        'variable_states_in_support': {v: {s: int(n) for s, n in zip(*np.unique(state[support], return_counts=True))}
                                       for v, state in (('pre', pre_state), ('pet', pet_state))},
        'harmonization': harmonization_summary(resolved, support),
        'country_area_reproduces_m1a_max_relative_error': float(rel.max()),
        'area_partition_max_relative_error': float(partition.max()),
        'hard_stops': stops, 'all_hard_stops_pass': not stops,
        'code_sha256': code_hash,
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'xarray': xr.__version__, 'netcdf4_python': netCDF4.__version__,
                     'netcdf_c': netCDF4.__netcdf4libversion__, 'hdf5': netCDF4.__hdf5libversion__,
                     'pyproj': pyproj.__version__, 'proj': pyproj.proj_version_str,
                     'xarray_engine': NETCDF_ENGINE, 'platform': platform.platform(), 'machine': platform.machine()},
    }
    texts = {  # serialize everything before writing, so a failure never leaves a partial output set
        'm1b_hydroclimate_features.csv': csv_text(qa[['iso3', 'Country']].assign(baseline_dryness=qa.c2_harmonized)),
        'm1b_hydroclimate_qa.csv': csv_text(qa),
        'm1b_harmonization_cells.csv': csv_text(audit),
        'm1b_support_checkpoint.json': json.dumps(checkpoint, indent=2, allow_nan=False) + '\n',
        'm1b_measurement_manifest.json': json.dumps(manifest, indent=2, allow_nan=False) + '\n',
    }
    if tuple(texts) != DETERMINISTIC_OUTPUTS:
        raise AssertionError('output set differs from DETERMINISTIC_OUTPUTS')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / name).write_text(text)
    run = {'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'wall_seconds': round(time.perf_counter() - started, 1), 'argv': sys.argv,
           'deterministic_output_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}}
    (out / RUN_METADATA).write_text(json.dumps(run, indent=2) + '\n')
    return qa, manifest


if __name__ == '__main__':
    qa, manifest = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(json.dumps({'all_hard_stops_pass': manifest['all_hard_stops_pass'], 'hard_stops': manifest['hard_stops']}, indent=2))
