"""M1b C2: baseline hydroclimatic dryness, built under ``M1B_EVALUATION_CONTRACT.md`` (pushed ``60111ae``).

For each CRU TS v4.10 0.5° cell j, 1920-01..1949-12::

    P̄_j = (1/30) Σ_t PRE_{j,t}          (mm/month)
    Ē_j = (1/30) Σ_t PET_{j,t} · N_t    (mm/day × calendar days)
    x_j = log10(P̄_j / Ē_j)

The country value is C2_c = Σ_valid A_{c,j} x_j / Σ_valid A_{c,j}, with A_{c,j} the frozen M1a
terrestrial area. Inputs are only the frozen M1a support record (country identifiers, terrestrial
area, per-cell land area), the GPW country grid and the pinned CRU PRE/PET files; no file holding a
warming outcome is opened. Writes only ``outputs/m1b_*``.
Run with ``python -m research.model_v2.m1b_hydroclimate [out_dir]``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from research.model_v2 import m0
from research.model_v2.m1a_geography import gpw_country_grid
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
CONTRACT_COMMIT = '60111ae1d44bdf9da6c613852a32de541f9b2d7a'


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


def aggregate(cells, x, valid, reason, p_bar, e_bar, supported, mean_stn, pure, n):
    i, k, a = cells.i.to_numpy(), cells.k.to_numpy(), cells.area.to_numpy()
    idx = cells.idx.to_numpy()
    ok = valid[i, k]

    def wsum(values, mask):
        return np.bincount(idx[mask], weights=a[mask] * values[mask], minlength=n)
    total = np.bincount(idx, weights=a, minlength=n)
    valid_area = np.bincount(idx[ok], weights=a[ok], minlength=n)
    out = pd.DataFrame({
        'terrestrial_area_km2': total,
        'valid_area_km2': valid_area,
        'coverage': valid_area / total,
        'baseline_dryness': wsum(x[i, k], ok) / valid_area,
        'p_bar_area_mean_mm_yr': wsum(p_bar[i, k], ok) / valid_area,
        'e_bar_area_mean_mm_yr': wsum(e_bar[i, k], ok) / valid_area,
        'pre_station_supported_share': wsum(supported[i, k], ok) / valid_area,
        'pre_mean_station_count': wsum(mean_stn[i, k], ok) / valid_area,
        'pre_pure_climatology_share': wsum(pure[i, k].astype(float), ok) / valid_area,
        'n_cru_cells': np.bincount(idx, minlength=n),
        'n_valid_cru_cells': np.bincount(idx[ok], minlength=n),
    })
    reasons = pd.DataFrame({'idx': idx[~ok], 'reason': reason[i, k][~ok], 'area': a[~ok]})
    out['invalid_reasons'] = [json.dumps(reasons[reasons.idx == j].groupby('reason').area.sum().round(6).to_dict())
                              for j in range(n)]
    return out


def distribution(series):
    q = [0, .1, .25, .5, .75, .9, 1]
    return dict(zip(['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'], map(float, series.quantile(q))))


def build(out=OUT):
    code_hash = sha256(Path(__file__))  # pinned before any work; never edit the file mid-run
    table = pd.read_csv(M1A_QA, usecols=SUPPORT_COLUMNS)[SUPPORT_COLUMNS]
    if table.columns.tolist() != SUPPORT_COLUMNS or len(table) != 151:
        raise ValueError('M1a support record does not have the expected identifier and area columns')
    codes = table.iso3.tolist()
    with xr.open_dataset(GPW_PATH, engine=GPW_ENGINE) as ds:
        check_support_nesting(ds['latitude'].to_numpy(), ds['longitude'].to_numpy())
    sources, arrays = {}, {}
    for var, spec in FILES.items():
        nc, hashes = pinned_source(spec)
        with xr.open_dataset(nc, engine=NETCDF_ENGINE) as ds:
            check_grid(ds)
            if var not in ds or ds[var].attrs.get('units') != spec['units']:
                raise ValueError(f'{nc.name}: variable {var} with units {spec["units"]} not found')
            if var == 'pre' and 'stn' not in ds:
                raise ValueError('PRE file lacks the stn station-count variable')
            months, days = window_months(ds['time'].to_numpy())
            arrays[var] = ds[var].isel(time=months).to_numpy().astype(np.float64)
            if var == 'pre':
                arrays['stn'] = ds['stn'].isel(time=months).to_numpy().astype(np.float64)
            sources[var] = {'url': BASE_URL + spec['path'], **hashes,
                            'units': ds[var].attrs.get('units'), 'global_attrs': version_attrs(ds),
                            'window_first_last': [str(pd.DatetimeIndex(ds['time'].to_numpy()[months])[i].date())
                                                  for i in (0, -1)]}
    p_bar, e_bar, x, valid, reason = cell_climatology(arrays['pre'], arrays['pet'], days)
    supported, mean_stn, pure = support_quantities(arrays['stn'], valid)

    land = np.load(LAND_AREA_NPZ)['cell_land_area_km2']
    country_of_cell, _, _ = gpw_country_grid(codes)
    cells = country_cells(codes, land, country_of_cell)
    qa = aggregate(cells, x, valid, reason, p_bar, e_bar, supported, mean_stn, pure, len(codes))
    qa.insert(0, 'iso3', codes)
    qa.insert(1, 'Country', table.Country)

    rel = np.abs(qa.terrestrial_area_km2.to_numpy() / table.terrestrial_area_km2.to_numpy() - 1)
    if rel.max() > 1e-9:
        raise ValueError(f'CRU-cell land area does not reproduce M1a terrestrial area (max rel {rel.max():.2e})')

    hard_stops = []
    failures = qa[~(qa.coverage >= COVERAGE_MIN)]
    for r in failures.itertuples():
        hard_stops.append({'iso3': r.iso3, 'rule': 'coverage < 0.98', 'coverage': float(r.coverage),
                           'missing_area_km2': float(r.terrestrial_area_km2 - r.valid_area_km2),
                           'reasons': json.loads(r.invalid_reasons)})
    for r in qa[qa.pre_pure_climatology_share >= 1].itertuples():
        hard_stops.append({'iso3': r.iso3, 'rule': 'pathological: entire terrestrial area pure climatology for all 360 months'})
    if not np.isfinite(qa.baseline_dryness).all():
        hard_stops.append({'rule': 'non-finite C2', 'iso3': qa.iso3[~np.isfinite(qa.baseline_dryness)].tolist()})

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
    manifest = {
        'contract': 'research/model_v2/M1B_EVALUATION_CONTRACT.md', 'contract_commit': CONTRACT_COMMIT,
        'sources': sources,
        'support_inputs': {'m1a_cell_land_area_sha256': sha256(LAND_AREA_NPZ), 'm1a_geography_qa_sha256': sha256(M1A_QA)},
        'window': list(WINDOW), 'n_months': N_MONTHS,
        'formula': 'x_j = log10((1/30)*sum PRE / ((1/30)*sum PET*days)); C2 = terrestrial-area-weighted mean of x_j over valid cells',
        'coverage_min': COVERAGE_MIN, 'cells_valid_global': int(valid.sum()),
        'country_area_reproduces_m1a_max_relative_error': float(rel.max()),
        'hard_stops': hard_stops, 'all_hard_stops_pass': not hard_stops,
        'code_sha256': code_hash,
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'xarray': xr.__version__},
    }
    out.mkdir(exist_ok=True)
    qa[['iso3', 'Country', 'baseline_dryness']].to_csv(out / 'm1b_hydroclimate_features.csv', index=False)
    qa.to_csv(out / 'm1b_hydroclimate_qa.csv', index=False)
    (out / 'm1b_support_checkpoint.json').write_text(json.dumps(checkpoint, indent=2, allow_nan=False) + '\n')
    (out / 'm1b_measurement_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return qa, manifest


if __name__ == '__main__':
    qa, manifest = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(json.dumps({'all_hard_stops_pass': manifest['all_hard_stops_pass'], 'hard_stops': manifest['hard_stops']}, indent=2))
