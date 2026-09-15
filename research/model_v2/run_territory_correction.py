"""Reproduce corrected M0 and the complete legacy-to-corrected fold audit.

Never overwrites any Session 1 artifact. No choices depend on corrected scores.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from research.model_v2 import cv, territory
from research.model_v2.m0 import (
    OUTPUT_DIR, design_matrix, load_inputs, m0_complete_design, verify_against_bundle,
)
from research.model_v2.run_m0_scorecard import MIN_REGION_N, PRIMARY_BUFFER_KM
from research.model_v2.spatial import morans_i
from src.area_weighting import GPW_NATID_LOOKUP_PATH
from src.population import GPW_PATH

CORRECTED_PATH = OUTPUT_DIR / "m0_scorecard_territory_corrected.json"


def paired_rmse_interval(y, new, reference):
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(y), size=(2000, len(y)))
    delta = (np.sqrt(np.mean((y - new)[indices]**2, axis=1))
             - np.sqrt(np.mean((y - reference)[indices]**2, axis=1)))
    return {
        "delta_rmse": float(np.sqrt(np.mean((y-new)**2)) - np.sqrt(np.mean((y-reference)**2))),
        "country_bootstrap_95_interval": np.quantile(delta, [0.025, 0.975]).tolist(),
        "resamples": 2000, "seed": 0,
    }


def complete_score(design, table, result, *, reference=None):
    fitted = cv.V1Design().fit(design).predict(design)
    w = cv.v1_weights(table.station_lon.to_numpy(), table.station_lat.to_numpy())
    card = cv.scorecard(design, fitted, result, w)
    y = design.warming_trend.to_numpy()
    errors = np.abs(y - result.yhat)
    card["fold_error_iqr"] = float(np.subtract(*np.quantile(errors, [0.75, 0.25])))
    card["effective_degrees_of_freedom"] = int(np.linalg.matrix_rank(design_matrix(design)[0]))
    card["residual_moran_in_sample"] = asdict(morans_i(y - fitted, w))
    card["residual_moran_cv"] = asdict(morans_i(y - result.yhat, w))
    regions = cv.region_errors(result, y, table.m49_subregion)
    eligible = regions[regions.n >= MIN_REGION_N]
    card["worst_region"] = {
        "region": str(eligible.index[0]), "rmse": float(eligible.iloc[0].rmse),
        "n": int(eligible.iloc[0].n), "min_region_n": MIN_REGION_N,
    }
    card["region_rmse"] = regions.rmse.to_dict()
    card["region_bias"] = regions.bias.to_dict()
    card["region_mae"] = regions.mae.to_dict()
    card["worst_fold_country"] = str(table.iloc[card["worst_fold"]].Country) if card['n_folds'] == len(y) else None
    card["paired_delta_vs_m0"] = paired_rmse_interval(y, result.yhat, result.yhat if reference is None else reference)
    return card


def main():
    codes_table = pd.read_csv(OUTPUT_DIR / 'm0_countries.csv')
    codes = codes_table.iso3.tolist()
    units = territory.gpw_units(codes)
    lower, upper = territory.distance_bounds(units)
    lower2, upper2 = territory.distance_bounds(units)
    assert np.array_equal(lower, lower2) and np.array_equal(upper, upper2)
    np.savez_compressed(territory.SNAPSHOT_PATH, **units)
    lower.to_csv(territory.LOWER_PATH)
    upper.to_csv(territory.UPPER_PATH)
    distance = territory.load_bounds(codes)
    assert np.array_equal(distance, lower.to_numpy())
    old = pd.read_csv(OUTPUT_DIR / 'country_border_distance_km.csv', index_col=0).loc[codes, codes].to_numpy()
    design = m0_complete_design(*load_inputs())
    assert design.Country.tolist() == codes_table.Country.tolist()
    folds = np.arange(len(codes))
    old_result = cv.cross_validate(design, folds, dist=old, buffer_km=PRIMARY_BUFFER_KM)
    new_result = cv.cross_validate(design, folds, dist=distance, buffer_km=PRIMARY_BUFFER_KM)
    repeat = cv.cross_validate(design, folds, dist=distance, buffer_km=PRIMARY_BUFFER_KM)
    assert np.array_equal(new_result.yhat, repeat.yhat)
    assert np.all(new_result.nearest_train_km > PRIMARY_BUFFER_KM)
    memberships, changes, pairs = [], [], []
    for i, code in enumerate(codes):
        old_train = (old[i] > PRIMARY_BUFFER_KM)
        new_train = (distance[i] > PRIMARY_BUFFER_KM)
        newly_excluded = old_train & ~new_train
        newly_retained = new_train & ~old_train
        changes.append(dict(
            iso3=code, Country=codes_table.iloc[i].Country,
            changed=bool(np.any(new_train != old_train)),
            old_n_train=int(old_train.sum()), corrected_n_train=int(new_train.sum()),
            newly_excluded=int(newly_excluded.sum()), newly_retained=int(newly_retained.sum()),
            excluded_codes='|'.join(np.array(codes)[newly_excluded]),
            retained_codes='|'.join(np.array(codes)[newly_retained]),
            minimum_retained_clearance_lower_km=float(distance[i, new_train].min()),
        ))
        for j, other in enumerate(codes):
            memberships.append(dict(held_out=code, candidate=other,
                                    legacy_train=bool(old_train[j]), corrected_train=bool(new_train[j])))
            if i < j and old_train[j] != new_train[j]:
                pairs.append(dict(a=code, b=other, legacy_km=old[i, j],
                                  corrected_lower_km=distance[i, j], corrected_upper_km=upper.iloc[i, j],
                                  change='newly_excluded' if newly_excluded[j] else 'newly_retained'))
    pd.DataFrame(memberships).to_csv(OUTPUT_DIR / 'territory_cv_membership_comparison.csv', index=False)
    pd.DataFrame(changes).to_csv(OUTPUT_DIR / 'territory_cv_fold_changes.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUTPUT_DIR / 'territory_cv_changed_pairs.csv', index=False)
    y = design.warming_trend.to_numpy()
    pd.DataFrame(dict(Country=design.Country, iso3=codes, m49_subregion=codes_table.m49_subregion,
                      observed=y, cv_prediction=new_result.yhat, cv_error=y-new_result.yhat,
                      n_train=new_result.n_train, nearest_train_lower_km=new_result.nearest_train_km,
                      unseen_levels=new_result.unseen_levels)).to_csv(
                          OUTPUT_DIR / 'm0_cv_errors_territory_corrected.csv', index=False)
    legacy = json.loads((OUTPUT_DIR / 'm0_scorecard.json').read_text())
    old_card = complete_score(design, codes_table, old_result)
    new_card = complete_score(design, codes_table, new_result)
    # Legacy scorecard precision check, independent recomputation on unchanged folds.
    legacy_primary = legacy['protocols']['primary_border_buffered_loo_500km']
    for k, value in legacy_primary.items():
        if isinstance(value, (float, int)):
            assert np.isclose(old_card[k], value, rtol=0, atol=1e-10), k
    secondary = cv.cross_validate(design, cv.folds_from_labels(codes_table.m49_subregion), dist=old)
    secondary_card = complete_score(design, codes_table, secondary)
    legacy_secondary = legacy['protocols']['secondary_m49_subregion_lo']
    for k, value in legacy_secondary.items():
        if isinstance(value, (float, int)):
            assert np.isclose(secondary_card[k], value, rtol=0, atol=1e-10), k
    reference = cv.cross_validate(design, cv.folds_random(len(codes), 10), dist=old)
    sensitivity = cv.cross_validate(design, folds, dist=distance, buffer_km=1000)
    from research.model_v2.spatial import haversine_matrix
    centroid = haversine_matrix(codes_table.centroid_lon.to_numpy(), codes_table.centroid_lat.to_numpy())
    centroid_result = cv.cross_validate(design, folds, dist=centroid, buffer_km=1500)
    # Coefficient signs use precisely the training encoders, including absent levels.
    full = cv.V1Design().fit(design)
    coefs = []
    for _, _, train in cv.iter_folds(folds, distance, PRIMARY_BUFFER_KM):
        enc = cv.V1Design().fit(design[train])
        coefs.append(pd.Series(enc.coef, index=enc.columns))
    coef_table = pd.DataFrame(coefs).reindex(columns=full.columns)
    new_card['coefficient_sign_agreement'] = {
        name: float((np.sign(coef_table[name].dropna()) == np.sign(full.coef[j])).mean())
        for j, name in enumerate(full.columns)
    }
    off = ~np.eye(len(codes), dtype=bool)
    uncertain = (distance <= 500) & (upper.to_numpy() > 500) & off
    out = {
        'baseline_verification': verify_against_bundle(),
        'geometry': {
            'source': 'GPW v4 rev11 adjusted population file, national identifier band 11',
            'source_sha256': territory.sha256(GPW_PATH),
            'lookup_sha256': territory.sha256(GPW_NATID_LOOKUP_PATH),
            'snapshot_sha256': territory.sha256(territory.SNAPSHOT_PATH),
            'crs': 'WGS84 (EPSG:4326); ECEF on same ellipsoid, km',
            'cell_degrees': 0.25, 'cell_radius_upper_km': territory.radius_bound_km(),
            'pair_radius_margin_km': 2*territory.radius_bound_km(),
            'units': len(codes), 'assigned_cells': sum(map(len, units.values())),
            'cell_counts_by_unit': {c: len(p) for c, p in units.items()},
            'retention_rule': 'ECEF minimum centre chord - 2*cell radius bound - 1 mm > 500 km',
            'distance_fields_are_bounds_not_exact_border_distances': True,
            'conservatively_excluded_pairs_with_upper_above_500': int(uncertain.sum() // 2),
        },
        'determinism': {'geometry_recomputed_identically': True, 'predictions_recomputed_identically': True},
        'clearance': {'all_retained_lower_bounds_above_500': True,
                      'minimum_retained_lower_km': float(new_result.nearest_train_km.min())},
        'fold_changes': {'changed_folds': sum(r['changed'] for r in changes),
                         'newly_excluded_directed': sum(r['newly_excluded'] for r in changes),
                         'newly_retained_directed': sum(r['newly_retained'] for r in changes)},
        'legacy_primary_recomputed': old_card,
        'corrected_primary': new_card,
        'secondary_m49_unchanged': secondary_card,
        'reference_random_10fold': complete_score(design, codes_table, reference),
        'sensitivity_territory_1000km': complete_score(design, codes_table, sensitivity),
        'sensitivity_centroid_1500km': complete_score(design, codes_table, centroid_result),
        'corrected_minus_legacy_rmse': paired_rmse_interval(y, new_result.yhat, old_result.yhat),
    }
    CORRECTED_PATH.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    return out


if __name__ == '__main__':
    result = main()
    print(json.dumps({k: result[k] for k in ['geometry', 'clearance', 'fold_changes']}, indent=2))
    for name in ['legacy_primary_recomputed', 'corrected_primary', 'secondary_m49_unchanged']:
        print(name, {k: result[name][k] for k in ['cv_r2', 'cv_rmse', 'fold_error_iqr', 'worst_region', 'rows_with_unseen_level']})
