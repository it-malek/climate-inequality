"""Fixed-M0 paired product audit; run with python -m research.model_v2.product_stability.

ERA5 is an external outcome robustness source, never a predictor or tuning target.
Only product_stability-prefixed research artifacts are written.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr

from research.model_v2 import m0, spatial
from research.model_v2.regions import m49_table
from src import area_weighting as aw
from src import era5_weighting as ew
from src.cleaning import DEFAULT_END, DEFAULT_START
from src.grids import decode_fractional_years
from src.population import GPW_PATH
from src.stability import country_centroids

OUT = m0.OUTPUT_DIR


def bh_adjust(p_values):
    """Benjamini-Hochberg adjusted p-values within the supplied finite family."""
    p = np.asarray(p_values, dtype=float)
    result = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    q = p[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return result


def paired_refit(x, berkeley, era5):
    """Use precisely the same rows and fixed design columns in both least squares."""
    keep = np.isfinite(berkeley) & np.isfinite(era5) & np.isfinite(x).all(axis=1)
    xc = x[keep]
    residuals = [y[keep] - xc @ np.linalg.lstsq(xc, y[keep], rcond=None)[0]
                 for y in (berkeley, era5)]
    return keep, *residuals


def classify_region(row, b_threshold, e_threshold):
    """Descriptive rule: >=3 countries, material means, >=75% country signs agree."""
    if row['n'] < 3:
        return 'unresolved'
    b, e = row['b_mean'], row['e_mean']
    if abs(b) >= b_threshold and abs(e) >= e_threshold:
        if b * e < 0:
            return 'product-sensitive'
        if row['sign_agreement'] >= .75:
            return 'product-robust'
    if (abs(b) >= b_threshold) != (abs(e) >= e_threshold):
        return 'product-sensitive'
    return 'unresolved'


def correlations(a, b):
    return {'pearson': float(pearsonr(a, b).statistic),
            'spearman': float(spearmanr(a, b).statistic)}


def lisa_overlap(table):
    """Report cluster and spatial-outlier membership, including adjusted results."""
    result = {}
    for weight in ['station_knn8', 'area_knn8']:
        for inference in ['label', 'fdr_label']:
            for quadrant in ['HH', 'LL', 'HL', 'LH']:
                sets = [set(table.loc[
                    table[f'{weight}_{product}_residual_{inference}'] == quadrant,
                    'iso3']) for product in ['b', 'e']]
                a, b = sets
                result[f'{weight}_{inference}_{quadrant}'] = {
                    'berkeley': sorted(a), 'era5': sorted(b),
                    'intersection': sorted(a & b), 'n_overlap': len(a & b),
                    'jaccard': len(a & b) / len(a | b) if a | b else None,
                }
    return result


CLASSES = ('product-robust', 'product-sensitive', 'unresolved')
TAIL_K, TAIL_NEAR_K = 15, 30


def combine_arms(absolute, aligned):
    """Robust only when both ERA5 constructions agree; any disagreement is unresolved."""
    return absolute if absolute == aligned else 'unresolved'


def evidence_class(b_level, e_level):
    """Levels are 'strong', 'weak' or 'none' for the same pattern in each product."""
    if b_level == e_level == 'strong':
        return 'product-robust'
    if {b_level, e_level} == {'strong', 'none'}:
        return 'product-sensitive'
    return 'unresolved'


def local_level(row, weight, product, quadrant):
    if row[f'{weight}_{product}_residual_fdr_label'] == quadrant:
        return 'strong'
    return 'weak' if row[f'{weight}_{product}_residual_label'] == quadrant else 'none'


def tail_level(scores, position):
    """Top-15 same-direction residual is strong; ranks 16-30 are weak (run() tie order)."""
    rank = int(np.flatnonzero(np.argsort(-scores, kind='stable') == position)[0]) + 1
    return 'strong' if rank <= TAIL_K else 'weak' if rank <= TAIL_NEAR_K else 'none'


def classify_patterns(arms):
    """Classify M0 residual patterns from {'absolute': (table, summary, region), 'aligned': ...}.

    Declared rules, applied identically to both ERA5 constructions and combined with
    ``combine_arms``: global presence needs positive Moran I with p<0.05 in both
    products; global magnitude is robust when ERA5 I >= 0.5 Berkeley I; regions use
    ``classify_region``; local and tail entries use ``evidence_class``.
    """
    rows = []

    def add(kind, weight, pattern, per_arm, evidence, note=''):
        rows.append({'pattern_type': kind, 'weights': weight, 'pattern': pattern,
                     **{f'evidence_{arm}': evidence[arm] for arm in arms},
                     **{f'class_{arm}_arm': per_arm[arm] for arm in arms},
                     'classification': combine_arms(per_arm['absolute'], per_arm['aligned']),
                     'note': note})

    for weight in ['station_knn8', 'area_knn8']:
        stats = {arm: arms[arm][1]['spatial'][weight] for arm in arms}
        significant = {arm: {p: s[f'{p}_residual']['statistic'] > 0 and s[f'{p}_residual']['p_value'] < .05
                             for p in 'be'} for arm, s in stats.items()}
        ratio = {arm: s['e_residual']['statistic'] / s['b_residual']['statistic'] for arm, s in stats.items()}
        evidence = {arm: f"I_B={s['b_residual']['statistic']:.4f} (p={s['b_residual']['p_value']:.4f}); "
                         f"I_E={s['e_residual']['statistic']:.4f} (p={s['e_residual']['p_value']:.4f})"
                    for arm, s in stats.items()}
        add('global', weight, 'positive residual spatial autocorrelation (presence)',
            {arm: evidence_class(*['strong' if significant[arm][p] else 'none' for p in 'be'])
             for arm in arms}, evidence)
        add('global', weight, 'residual spatial autocorrelation magnitude',
            {arm: 'product-robust' if ratio[arm] >= .5 else 'product-sensitive' for arm in arms},
            {arm: f'{evidence[arm]}; ratio={ratio[arm]:.3f}' for arm in arms})
    regions = {arm: arms[arm][2] for arm in arms}
    for name in regions['aligned'].index:
        add('regional', 'm49_subregion', name,
            {arm: regions[arm].loc[name, 'classification'] for arm in arms},
            {arm: f"n={regions[arm].loc[name, 'n']}; B={regions[arm].loc[name, 'b_mean']:+.4f}; "
                  f"E={regions[arm].loc[name, 'e_mean']:+.4f}; "
                  f"signs={regions[arm].loc[name, 'sign_agreement']:.0%}" for arm in arms})
    tables = {arm: arms[arm][0].set_index('iso3') for arm in arms}
    for weight in ['station_knn8', 'area_knn8']:
        candidates = set()
        for arm, t in tables.items():
            for quadrant in ['HH', 'LL', 'HL', 'LH']:
                for iso3, row in t.iterrows():
                    levels = [local_level(row, weight, p, quadrant) for p in 'be']
                    if 'strong' in levels or levels == ['weak', 'weak']:
                        candidates.add((iso3, quadrant))
        for iso3, quadrant in sorted(candidates):
            levels = {arm: [local_level(tables[arm].loc[iso3], weight, p, quadrant) for p in 'be']
                      for arm in arms}
            add('local', weight, f'{iso3} {quadrant}',
                {arm: evidence_class(*levels[arm]) for arm in arms},
                {arm: f'B={levels[arm][0]}; E={levels[arm][1]}' for arm in arms},
                'strong=BH q<=0.05; weak=nominal p<0.05 only')
    for kind in ['positive', 'negative', 'absolute']:
        overlap = {arm: arms[arm][1]['tail_overlap'][kind]['n_overlap'] for arm in arms}
        add('tail_list', f'top{TAIL_K}', f'{kind} residual top-{TAIL_K} list',
            {arm: 'product-robust' if 2 * overlap[arm] > TAIL_K else 'product-sensitive' for arm in arms},
            {arm: f'overlap={overlap[arm]}/{TAIL_K}' for arm in arms})
    for kind, sign in [('positive', 1), ('negative', -1)]:
        scores = {arm: {p: sign * t[f'{p}_residual'].to_numpy() for p in 'be'}
                  for arm, t in tables.items()}
        members = set()
        for arm in arms:
            for p in 'be':
                members |= set(tables[arm].index[np.argsort(-scores[arm][p], kind='stable')[:TAIL_K]])
        for iso3 in sorted(members):
            levels = {arm: [tail_level(scores[arm][p], tables[arm].index.get_loc(iso3)) for p in 'be']
                      for arm in arms}
            add('tail_country', f'top{TAIL_K}', f'{iso3} {kind}',
                {arm: evidence_class(*levels[arm]) for arm in arms},
                {arm: f'B={levels[arm][0]}; E={levels[arm][1]}' for arm in arms},
                f'strong=top {TAIL_K}; weak=rank {TAIL_K + 1}-{TAIL_NEAR_K}')
    return pd.DataFrame(rows)


def across_arm_summary(table, region):
    """Compare aligned outcome arm with the preserved absolute-temperature arm."""
    previous = pd.read_csv(OUT / 'product_stability_countries.csv').set_index('iso3').loc[table.iso3]
    old_regions = pd.read_csv(OUT / 'product_stability_regions.csv').set_index('m49_subregion')
    change = table.era5.to_numpy() - previous.era5.to_numpy()
    return {
        'aligned_minus_absolute_era5_mean': float(change.mean()),
        'aligned_minus_absolute_era5_sample_sd': float(change.std(ddof=1)),
        'aligned_minus_absolute_era5_rmse': float(np.sqrt(np.mean(change**2))),
        'era5_outcome_correlations_across_preprocessing': correlations(table.era5, previous.era5),
        'era5_residual_correlations_across_preprocessing': correlations(table.e_residual, previous.e_residual),
        'era5_residual_sign_agreement_across_preprocessing': int(np.sum(table.e_sign.to_numpy() == previous.e_sign.to_numpy())),
        'regional_classifications': {
            name: {'absolute': old_regions.loc[name, 'classification'],
                   'aligned': row['classification'],
                   'robust_in_both': old_regions.loc[name, 'classification'] == row['classification'] == 'product-robust'}
            for name, row in region.iterrows()},
    }


def provenance():
    """Inspect raw coordinates/time/masks, and hash actual local inputs and code."""
    metadata = {}
    for name, path, decode, converter in [
        ('berkeley', aw.BERKELEY_GRID_PATH, False, decode_fractional_years),
        ('era5', ew.ERA5_GRID_PATH, True, ew.era5_time_to_months),
    ]:
        with xr.open_dataset(path, decode_times=decode) as ds:
            months = converter(ds.time.to_numpy())
            lat, lon = ds.latitude.to_numpy(), ds.longitude.to_numpy()
            window = months[(months >= DEFAULT_START) & (months <= DEFAULT_END)]
            var = 'temperature' if name == 'berkeley' else 't2m'
            mask = (aw.assign_cell_iso3(lat, lon) if name == 'berkeley'
                    else ew.era5_cell_iso3(lat, lon))
            metadata[name] = {
                'path': str(path.relative_to(m0.ROOT)), 'variable': var,
                'units': str(ds[var].attrs.get('units')),
                'raw_months': len(months), 'raw_start': str(months.min()),
                'raw_end': str(months.max()), 'window_months': len(window),
                'window_start': str(window.min()), 'window_end': str(window.max()),
                'latitude_count': len(lat), 'longitude_count': len(lon),
                'latitude_first_last': [float(lat[0]), float(lat[-1])],
                'longitude_first_last': [float(lon[0]), float(lon[-1])],
                'latitude_step': float(lat[1] - lat[0]),
                'longitude_step': float(lon[1] - lon[0]),
                'assigned_cells': int(np.sum(mask != None)),  # noqa: E711
                'assigned_country_counts': pd.Series(mask.ravel()).dropna().value_counts().to_dict(),
                'variable_attrs': {k: str(v) for k, v in ds[var].attrs.items()},
            }
    paths = [aw.BERKELEY_GRID_PATH, ew.ERA5_GRID_PATH, GPW_PATH,
             aw.GPW_NATID_LOOKUP_PATH, m0.DEFAULT_INEQUALITY_PATH,
             m0.DEFAULT_FEATURES_PATH, m0.INCOME_PATH, m0.OWID_CO2_PATH,
             m0.BUNDLE_DIR / 'era5_area_trends.parquet',
             m0.BUNDLE_DIR / 'era5_validation_summary.json',
             OUT / 'm0_countries.csv', OUT / 'country_geometry.csv']
    paths += [m0.ROOT / p for p in ['src/area_weighting.py', 'src/era5_weighting.py',
              'src/cleaning.py', 'src/grids.py', 'scripts/fetch_era5.py',
              'src/decomposition.py', 'src/feature_schema.py',
              'research/model_v2/m0.py', 'research/model_v2/spatial.py',
              'research/model_v2/product_stability.py']]
    hashes = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        hashes[str(path.relative_to(m0.ROOT))] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return {'raw_grids': metadata, 'files': hashes,
            'limitation': 'Raw coordinate/time/mask inspection and code tracing; full expensive per-cell Theil-Sen rebuild not rerun.'}


def run(n_permutations=9999, seed=0, aligned=False):
    inequality, features, income = m0.load_inputs()
    complete = m0.m0_complete_design(inequality, features, income)
    x, names, groups = m0.design_matrix(complete)
    ids = complete.Country.map(m0.iso3_bridge(inequality))
    era_path = (OUT / 'product_stability_aligned_era5_trends.csv' if aligned
                else m0.BUNDLE_DIR / 'era5_area_trends.parquet')
    era = pd.read_csv(era_path) if aligned else pd.read_parquet(era_path)
    if era.iso3.duplicated().any() or ids.duplicated().any():
        raise ValueError('Country/ISO3 keys must be unique')
    b = complete[m0.OUTCOME_COL].to_numpy(float)
    e = ids.map(era.set_index('iso3')[ew.ERA5_AREA_COL]).to_numpy(float)
    keep, rb, re = paired_refit(x, b, e)
    table = pd.DataFrame({'Country': complete.Country[keep], 'iso3': ids[keep],
                          'berkeley': b[keep], 'era5': e[keep],
                          'b_residual': rb, 'e_residual': re}).reset_index(drop=True)
    table = table.merge(m49_table(), on='iso3', validate='one_to_one', how='left')
    if table.m49_subregion.isna().any():
        raise ValueError('Missing region assignment')
    table['b_minus_e'] = table.berkeley - table.era5
    # Round only for sign comparisons, to avoid effectively exact dummy fits.
    table['b_sign'] = np.sign(np.where(abs(rb) < 1e-10, 0, rb)).astype(int)
    table['e_sign'] = np.sign(np.where(abs(re) < 1e-10, 0, re)).astype(int)
    table['sign_agreement'] = table.b_sign == table.e_sign
    gap = table.b_minus_e.to_numpy()
    table['squared_gap_share'] = gap**2 / np.sum(gap**2)
    cen = country_centroids(features).set_index('Country').loc[table.Country]
    geom = pd.read_csv(OUT / 'country_geometry.csv').set_index('iso3').loc[table.iso3]
    weights = {
        'station_knn8': spatial.knn_weights(spatial.haversine_matrix(cen.Longitude.to_numpy(), cen.Latitude.to_numpy()), 8),
        'area_knn8': spatial.knn_weights(spatial.haversine_matrix(geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy()), 8),
    }
    spatial_results = {}
    for weight_name, w in weights.items():
        spatial_results[weight_name] = {}
        for col in ['b_minus_e', 'b_residual', 'e_residual']:
            spatial_results[weight_name][col] = asdict(spatial.morans_i(table[col].to_numpy(), w, n_permutations, seed))
            local = spatial.local_morans(table[col].to_numpy(), w, n_permutations, seed)
            local['q_value'] = bh_adjust(local.p_value)
            local['fdr_label'] = np.where(local.q_value <= .05, local.quadrant, 'ns')
            for field in ['local_i', 'p_value', 'q_value', 'quadrant', 'label', 'fdr_label']:
                table[f'{weight_name}_{col}_{field}'] = local[field].to_numpy()
    region = table.groupby('m49_subregion', sort=True).agg(
        n=('iso3', 'size'), b_mean=('b_residual', 'mean'), e_mean=('e_residual', 'mean'),
        difference_mean=('b_minus_e', 'mean'), difference_median=('b_minus_e', 'median'),
        difference_sd=('b_minus_e', 'std'), squared_gap_share=('squared_gap_share', 'sum'),
        sign_agreement=('sign_agreement', 'mean'))
    bt, et = .5 * rb.std(ddof=1), .5 * re.std(ddof=1)
    region['classification'] = region.apply(classify_region, axis=1, args=(bt, et))
    tails = {}
    for kind in ['positive', 'negative', 'absolute']:
        scores = [v if kind == 'positive' else -v if kind == 'negative' else abs(v) for v in [rb, re]]
        a, c = [set(table.iloc[np.argsort(-v, kind='stable')[:15]].iso3) for v in scores]
        tails[kind] = {'k': 15, 'berkeley': sorted(a), 'era5': sorted(c),
                       'intersection': sorted(a & c), 'n_overlap': len(a & c), 'jaccard': len(a & c) / len(a | c)}
    summary = {
        'common_n': int(keep.sum()), 'm0_n': len(complete), 'era5_bundle_n': len(era),
        'excluded_m0_countries': complete.Country[~keep].tolist(),
        'design_columns': names, 'groups': groups, 'design_rank': int(np.linalg.matrix_rank(x[keep])),
        'outcomes': {key: {'mean': float(v.mean()), 'sample_sd': float(v.std(ddof=1))}
                     for key, v in [('berkeley', b[keep]), ('era5', e[keep])]},
        'b_minus_e': {'mean': float(gap.mean()), 'median': float(np.median(gap)),
                      'sample_sd': float(gap.std(ddof=1)), 'rmse': float(np.sqrt(np.mean(gap**2)))},
        'systematic_bias': {
            'era5_greater_n': int(np.sum(gap < 0)), 'berkeley_greater_n': int(np.sum(gap > 0)),
            'era5_on_berkeley_ols': dict(zip(['slope', 'intercept'], map(float, np.polyfit(b[keep], e[keep], 1)))),
            'era5_to_berkeley_sd_ratio': float(e[keep].std(ddof=1) / b[keep].std(ddof=1)),
        },
        'outcome_correlations': correlations(b[keep], e[keep]),
        'residual_correlations': correlations(rb, re),
        'berkeley_residual_vs_era5_minus_berkeley': correlations(rb, -gap),
        'residual_sample_sd': {'berkeley': float(rb.std(ddof=1)), 'era5': float(re.std(ddof=1))},
        'r2': {key: float(1 - np.sum(r**2)/np.sum((v-v.mean())**2))
               for key, r, v in [('berkeley', rb, b[keep]), ('era5', re, e[keep])]},
        'sign_agreement_n': int(table.sign_agreement.sum()), 'tail_overlap': tails,
        'region_material_threshold': {'berkeley': bt, 'era5': et},
        'spatial': spatial_results, 'permutations': n_permutations, 'seed': seed,
        'lisa_overlap': lisa_overlap(table),
        'lisa_fdr_counts': {col: table[col].value_counts().to_dict() for col in table if col.endswith('fdr_label')},
        'provenance': provenance(),
    }
    OUT.mkdir(exist_ok=True)
    prefix = 'product_stability_aligned' if aligned else 'product_stability'
    if aligned:
        summary['across_preprocessing_arms'] = across_arm_summary(table, region)
        summary['alignment'] = json.loads((OUT / 'product_stability_aligned_preprocessing.json').read_text())
        summary['aligned_outcome_sha256'] = hashlib.sha256(era_path.read_bytes()).hexdigest()
        absolute = (pd.read_csv(OUT / 'product_stability_countries.csv'),
                    json.loads((OUT / 'product_stability_summary.json').read_text()),
                    pd.read_csv(OUT / 'product_stability_regions.csv').set_index('m49_subregion'))
        patterns = classify_patterns({'absolute': absolute, 'aligned': (table, summary, region)})
        patterns.to_csv(OUT / 'product_stability_pattern_classification.csv', index=False)
        summary['pattern_classification_counts'] = {
            kind: group.classification.value_counts().reindex(CLASSES, fill_value=0).to_dict()
            for kind, group in patterns.groupby('pattern_type')}
    table.to_csv(OUT / f'{prefix}_countries.csv', index=False)
    region.to_csv(OUT / f'{prefix}_regions.csv')
    (OUT / f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['provenance', 'groups', 'design_columns']}, indent=2))
    return summary


def monthly_anomalies(temperature):
    """Subtract each cell's 1951-1980 calendar-month mean (never a fitted trend)."""
    baseline = temperature.sel(time=slice('1951-01-01', '1980-12-31'))
    counts = baseline.time.groupby('time.month').count()
    if len(counts) != 12 or not np.all(counts.to_numpy() == 30):
        raise ValueError('Expected exactly 30 baseline timestamps per calendar month')
    climatology = baseline.groupby('time.month').mean('time')
    return temperature.groupby('time.month') - climatology


def build_aligned_outcome():
    """Run shared estimator on research-only deseasonalized ERA5; delete temp NC."""
    with xr.open_dataset(aw.BERKELEY_GRID_PATH, decode_times=False) as ds:
        baseline_metadata = str(ds.climatology.attrs['long_name'])
        if 'Jan 1951 - Dec 1980' not in baseline_metadata:
            raise ValueError('Berkeley climatology baseline changed')
    with xr.open_dataset(ew.ERA5_GRID_PATH) as ds:
        anomalies = monthly_anomalies(ds.t2m).load()
        anomalies.attrs = {'units': 'K', 'long_name': 'ERA5 anomalies relative to 1951\u20131980 calendar-month means'}
    # tempfile is confined to research; remove after success or exception.
    with tempfile.TemporaryDirectory(prefix='product_stability_', dir=OUT) as scratch:
        path = Path(scratch) / 'era5_monthly_anomalies.nc'
        anomalies.to_dataset(name='t2m').to_netcdf(path)
        mask, lats, slopes = ew.era5_cell_slopes(path)
    np.savez_compressed(OUT / 'product_stability_aligned_cell_slopes.npz',
                        iso3=np.where(mask == None, '', mask).astype('U3'),  # noqa: E711
                        latitude=lats, longitude=anomalies.longitude.to_numpy(),
                        slope=slopes)
    ew.reduce_era5_slopes(mask, lats, slopes).to_csv(
        OUT / 'product_stability_aligned_era5_trends.csv', index=False)
    metadata = {'berkeley_climatology_long_name': baseline_metadata,
                'era5_normalization': 'Subtract own 1951\u20131980 cell/calendar-month mean before shared Theil\u2013Sen.',
                'analysis_window': [DEFAULT_START, DEFAULT_END],
                'mask_and_grid': 'Native ERA5, unchanged GPW operator; half-degree offset persists.',
                'selected_before_aligned_results': True,
                'source_sha256': hashlib.sha256(ew.ERA5_GRID_PATH.read_bytes()).hexdigest(),
                'baseline_months': 360, 'baseline_years_per_calendar_month': 30}
    (OUT / 'product_stability_aligned_preprocessing.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--build-aligned', action='store_true')
    args = parser.parse_args()
    if args.build_aligned:
        build_aligned_outcome()
    run(aligned=args.aligned or args.build_aligned)
