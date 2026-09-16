"""M4 final assessment under ``DOWNSTREAM_COMPLETION_SPEC.md`` §3 and ``M4_EVIDENCE_INVENTORY.md``.

M4 changes no model. It computes only the evidence the frozen inventory marks as missing, for the retained
static specification S (branch A: M0*; branch B: M2), its qualifying station-weight spatial extensions and, in
branch B, M0* as S's comparator:

* **static share uncertainty** for S: the V1 country bootstrap (2,000 draws, seed 0) and continent block
  bootstrap (seed 1), both representations, with a new latitude state per draw in branch B;
* **buffer sensitivities**: territorial 1000 km and land-centroid 1500 km leave-one-country-out;
* **product check of the eventual model**: aligned ERA5 (preferred) and legacy ERA5 (preprocessing
  sensitivity);

and consolidates committed evidence into one table, the inventory status and the final specification record.

Run with ``PYTHONHASHSEED=0 uv run python -m research.model_v2.m4_evaluate``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.model_v2 import m2_evaluate as ev
from research.model_v2 import m2_feasibility as feas
from research.model_v2 import m2_latitude_basis as lb
from research.model_v2 import m3_evaluate as m3
from research.model_v2 import v2_provenance as prov
from research.model_v2.run_m0_scorecard import SENSITIVITY_BORDER_BUFFER_KM, SENSITIVITY_CENTROID_BUFFER_KM
from research.model_v2.run_territory_correction import paired_rmse_interval
from research.model_v2.spatial import haversine_matrix
from src.decomposition import OUTCOME_COL

ROOT = prov.ROOT
OUTPUTS = ev.OUTPUTS
SPEC = 'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md'
INVENTORY = 'research/model_v2/M4_EVIDENCE_INVENTORY.md'
EVALUATOR_SPEC = 'research/model_v2/M4_EVALUATOR_SPEC.md'
DOCUMENTS = (SPEC, INVENTORY, EVALUATOR_SPEC)
ENTRY_FILES = ('research/model_v2/m4_evaluate.py',)
OUT = ROOT / OUTPUTS / 'm4_final'
M3_STATION, M3_LAND = m3.OUT['station'], m3.OUT['land']
LEGACY_ERA5 = 'app/data/era5_area_trends.parquet'
LEGACY_ERA5_SHA256 = '5a18ccbaea44bfef09788b7261ea081a132ff794295e382fc504d459b982a43c'
ALIGNED_ERA5 = f'{OUTPUTS}/product_stability_aligned_era5_trends.csv'
ALIGNED_SUMMARY = f'{OUTPUTS}/product_stability_aligned_summary.json'
LEGACY_SUMMARY = f'{OUTPUTS}/product_stability_summary.json'
LEGACY_SUMMARY_SHA256 = None   # pinned at freeze time below through PINS
ERA5_COLUMN = 'trend_c_per_decade_era5_area'
M0_CORRECTED = f'{OUTPUTS}/m0_scorecard_territory_corrected.json'
M0_LEGACY = f'{OUTPUTS}/m0_scorecard.json'
N_BOOT = 2000
COUNTRY_SEED, BLOCK_SEED = 0, 1
SHARE_ORDER = ('emissions', 'geography', 'socioeconomic', 'population')   # V1 schema order; ties to the earlier
MATERIAL_RESPONSIBILITY = 0.10
REFERENCE_ATOL = 1e-10
SENSITIVITY_ANCHOR_ATOL = 1e-9
SENSITIVITY_PROTOCOLS = {'buffer_territory_1000km': SENSITIVITY_BORDER_BUFFER_KM,
                         'buffer_centroid_1500km': SENSITIVITY_CENTROID_BUFFER_KM}
PRODUCTS = ('aligned_era5', 'legacy_era5')
DETERMINISTIC_ARTIFACTS = ('m4_bootstrap_draws.csv', 'm4_bootstrap_summary.json', 'm4_sensitivities.json',
                           'm4_sensitivity_predictions.csv', 'm4_products.json', 'm4_product_predictions.csv',
                           'm4_consolidated_table.json', 'm4_consolidated_table.csv', 'm4_inventory_status.json',
                           'm4_final_specification.json', 'm4_provenance.json')
RESULT_MANIFEST = 'm4_result_manifest.json'
RUN_METADATA = 'm4_run_metadata.json'


def code_files() -> list[str]:
    return sorted({*prov.import_closure(ENTRY_FILES), *DOCUMENTS, *prov.LOCK_FILES})


# ---------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------


def verified_prerequisites() -> dict:
    m2 = m3.verified_m2()
    station = m3.verified_station(M3_STATION)
    land = prov.verify_manifest(M3_LAND, m3.RESULT_MANIFEST)
    if land.get('integrity_passed') is not True or land.get('weights') != 'land':
        raise prov.ExecutionRefused('the land-centroid M3 package is missing or not integrity-valid')
    prov.tracked_and_pushed(m3._repo_paths(M3_LAND, [*m3.DETERMINISTIC_ARTIFACTS, m3.RESULT_MANIFEST, m3.RUN_METADATA]))
    naming = station['manifest'].get('final_naming')
    if not naming or naming.get('retained_static_specification') != m2['retained_static_specification']:
        raise prov.ExecutionRefused('the M3 station package naming is missing or inconsistent with the M2 branch')
    for rel, pin in ((LEGACY_ERA5, LEGACY_ERA5_SHA256),):
        if prov.sha256(ROOT / rel) != pin:
            raise prov.ExecutionRefused(f'{rel} does not match its pinned digest')
    return {'m2': m2, 'station': station, 'land_manifest': land, 'naming': naming,
            'land_manifest_sha256': prov.sha256(M3_LAND / m3.RESULT_MANIFEST)}


def introducing_commit(path) -> str | None:
    done = subprocess.run(['git', 'log', '--format=%H', '--diff-filter=A', '--', str(path)], cwd=ROOT,
                          capture_output=True, text=True)
    commits = done.stdout.split()
    return commits[-1] if commits else None


# ---------------------------------------------------------------------
# A. Static share uncertainty (spec §3.2)
# ---------------------------------------------------------------------


def draw_shares(model, draw):
    """One bootstrap draw: V1 per-draw encoding, a new latitude state in branch B, projection-R2 LMG."""
    try:
        result = ev.lmg_shares(model, draw)
    except (lb.DegenerateLatitudeBasis, ValueError, np.linalg.LinAlgError) as error:
        return None, f'{type(error).__name__}: {error}', None
    # A group whose features take a single level in a draw has no columns; as in src.decomposition it keeps a
    # zero share, and a Shapley null player leaves the other groups' shares unchanged.
    shares = {g: result['shares'].get(g, 0.0) for g in SHARE_ORDER}
    values = [shares[g] for g in SHARE_ORDER] + [result['residual_share']]
    if not all(np.isfinite(values)):
        return None, 'non-finite share', None
    state = lb.fit_state(draw.abs_latitude.to_numpy(float)) if model == ev.CANDIDATE else None
    matrix, _ = ev.design(model, draw, feas.baseline_encoder(draw), state)
    rank_deficient = bool(np.linalg.matrix_rank(matrix) < matrix.shape[1])
    return {**shares, 'residual': result['residual_share']}, None, rank_deficient


def bootstrap(model, frame, kind):
    rng = np.random.default_rng(COUNTRY_SEED if kind == 'country' else BLOCK_SEED)
    blocks = pd.unique(frame['spatial_block'])
    rows = []
    for b in range(N_BOOT):
        if kind == 'country':
            idx = rng.integers(0, len(frame), len(frame))
            draw = frame.iloc[idx].reset_index(drop=True)
        else:
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            draw = pd.concat([frame[frame['spatial_block'] == c] for c in chosen], ignore_index=True)
        shares, failure, rank_deficient = draw_shares(model, draw)
        row = {'bootstrap': kind, 'draw': b, 'n_rows': len(draw), 'usable': shares is not None,
               'failure': failure or '', 'rank_deficient': rank_deficient if shares is not None else ''}
        row.update({g: (shares[g] if shares is not None else np.nan) for g in (*SHARE_ORDER, 'residual')})
        rows.append(row)
    return pd.DataFrame(rows)


def largest_named(row):
    return max(SHARE_ORDER, key=lambda g: row[g])        # ties go to the earlier group, as src.stability


def summarize_bootstrap(point, draws):
    out = {}
    for kind, frame in draws.groupby('bootstrap', sort=False):
        usable = frame[frame.usable]
        groups = {}
        for g in (*SHARE_ORDER, 'residual'):
            values = usable[g].to_numpy(float)
            lo, hi = np.percentile(values, [2.5, 97.5])
            groups[g] = {'point': point[g], 'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
                         'ci_low': float(lo), 'ci_high': float(hi)}
        margin = usable['geography'].to_numpy(float) - usable[[g for g in SHARE_ORDER if g != 'geography']].max(axis=1).to_numpy(float)
        out[kind] = {'n_draws': int(len(frame)), 'usable': int(len(usable)), 'degenerate': int((~frame.usable).sum()),
                     'rank_deficient_usable': int(usable.rank_deficient.astype(bool).sum()),
                     'groups': groups,
                     'p_geography_largest': float(np.mean([largest_named(r) == 'geography' for _, r in usable.iterrows()])),
                     'geography_margin_percentile_2_5': float(np.percentile(margin, 2.5)),
                     'seed': COUNTRY_SEED if kind == 'country' else BLOCK_SEED}
    others = max(point[g] for g in SHARE_ORDER if g != 'geography')
    material = {
        'point_geography_not_largest': bool(not point['geography'] > others),
        'interval_geography_not_established_country': bool(out['country']['geography_margin_percentile_2_5'] <= 0),
        'interval_geography_not_established_block': bool(out['block']['geography_margin_percentile_2_5'] <= 0),
        'responsibility_exceeds_0_10': bool(point['emissions'] > MATERIAL_RESPONSIBILITY),
        'responsibility_upper_country_ci_exceeds_0_10': bool(out['country']['groups']['emissions']['ci_high']
                                                             > MATERIAL_RESPONSIBILITY),
    }
    material['material_change_to_v1_conclusion'] = bool(material['point_geography_not_largest']
                                                         or material['interval_geography_not_established_country']
                                                         or material['responsibility_exceeds_0_10'])
    return {**out, 'material_change': material,
            'rule': 'plan "Material change" with the country bootstrap as the headline interval; the block '
                    'bootstrap version is reported alongside; descriptive allocation, never causal'}


def static_uncertainty(frozen, static_model):
    summaries, frames = {}, []
    for representation in ev.REPRESENTATIONS:
        frame = frozen.frames[representation]
        full = ev.lmg_shares(static_model, frame)
        point = {**full['shares'], 'residual': full['residual_share']}
        draws = pd.concat([bootstrap(static_model, frame, 'country'), bootstrap(static_model, frame, 'block')],
                          ignore_index=True)
        summaries[representation] = summarize_bootstrap(point, draws)
        frames.append(draws.assign(representation=representation, model=static_model))
    return summaries, pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# B. Buffer sensitivities (spec §3.4)
# ---------------------------------------------------------------------


def sensitivity_frozen(frozen, protocol):
    n = frozen.n
    if protocol == 'buffer_territory_1000km':
        distance = frozen.distance
    else:
        table = pd.read_csv(ROOT / ev.COUNTRY_TABLE, usecols=['iso3', 'centroid_lon', 'centroid_lat'],
                            float_precision='round_trip').set_index('iso3').loc[frozen.table.iso3]
        distance = haversine_matrix(table.centroid_lon.to_numpy(), table.centroid_lat.to_numpy())
    return dataclasses.replace(frozen, distance=distance, folds={protocol: (np.arange(n), SENSITIVITY_PROTOCOLS[protocol])})


def cv_scores(y, yhat):
    e = y - yhat
    return {'cv_r2': float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), 'cv_rmse': float(np.sqrt(np.mean(e ** 2))),
            'cv_mae': float(np.mean(np.abs(e)))}


def sensitivities(frozen, static, qualifying, station_dist):
    static_model = m3.STATIC_MODEL[static]
    committed = json.loads((ROOT / M0_CORRECTED).read_text())
    anchors = {'buffer_territory_1000km': committed['sensitivity_territory_1000km'],
               'buffer_centroid_1500km': committed['sensitivity_centroid_1500km']}
    models = [static_model] + ([ev.COMPARATOR] if static_model == ev.CANDIDATE else [])
    report, rows = {}, []
    for protocol in SENSITIVITY_PROTOCOLS:
        sf = sensitivity_frozen(frozen, protocol)
        for representation in ev.REPRESENTATIONS:
            frame = frozen.frames[representation]
            y = frame[OUTCOME_COL].to_numpy(float)
            entry, predictions = {}, {}
            for model in models:
                try:
                    result, records = ev.run_protocol(model, frame, sf, protocol)
                    failures = [r['failures'] for r in records if r['failures']]
                    if failures:
                        entry[model] = {'status': 'non_computable', 'reason': failures[0]}
                        continue
                except (lb.DegenerateLatitudeBasis, ValueError) as error:
                    entry[model] = {'status': 'non_computable', 'reason': str(error)}
                    continue
                predictions[model] = result.yhat
                entry[model] = {'status': 'computed', **cv_scores(y, result.yhat)}
                if model == ev.COMPARATOR:
                    for metric in ('cv_r2', 'cv_rmse', 'cv_mae'):
                        difference = abs(entry[model][metric] - float(anchors[protocol][metric]))
                        if not difference <= SENSITIVITY_ANCHOR_ATOL:
                            raise prov.ExecutionRefused(f'{protocol} {representation} M0*: {metric} differs from the '
                                                        f'committed M0 sensitivity card by {difference}')
                    entry[model]['anchor_m0_committed_card'] = 'reproduced to 1e-9'
            for name in qualifying:
                family = 'sem' if name.startswith('M3-SEM') else 'sar'
                run = m3.run_family(family, static_model, frame, sf, station_dist, representation, protocols=(protocol,))
                if run.status != 'computable':
                    entry[name] = {'status': 'non_computable', 'failure': run.failure}
                    continue
                predictions[name] = run.results[protocol].yhat
                entry[name] = {'status': 'computed', **cv_scores(y, run.results[protocol].yhat),
                               'at_domain_bound_fits': int(sum(r['fit'].at_domain_bound for r in run.records[protocol]))}
            if static_model == ev.CANDIDATE and all(m in predictions for m in (ev.CANDIDATE, ev.COMPARATOR)):
                entry['paired_S_minus_M0star'] = paired_rmse_interval(y, predictions[ev.CANDIDATE], predictions[ev.COMPARATOR])
            for name in qualifying:
                if name in predictions and static_model in predictions:
                    entry[f'paired_{name}_minus_S'] = paired_rmse_interval(y, predictions[name], predictions[static_model])
            report[f'{protocol}/{representation}'] = entry
            for model, values in predictions.items():
                rows += [{'protocol': protocol, 'representation': representation, 'model': model,
                          'iso3': frozen.table.iso3.iloc[i], 'observed': y[i], 'prediction': values[i]}
                         for i in range(frozen.n)]
    report['scope'] = ('retained specifications only: S, each qualifying spatial extension and, in branch B, M0* as '
                       "S's comparator; rejected candidates and non-qualifying families are not executed")
    return report, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# C. Product check of the eventual model (spec §3.5)
# ---------------------------------------------------------------------


def product_outcomes(frozen):
    aligned_summary = json.loads((ROOT / ALIGNED_SUMMARY).read_text())
    if prov.sha256(ROOT / ALIGNED_ERA5) != aligned_summary['aligned_outcome_sha256']:
        raise prov.ExecutionRefused('the aligned ERA5 file differs from its summary digest')
    aligned = pd.read_csv(ROOT / ALIGNED_ERA5, float_precision='round_trip').set_index('iso3')
    legacy = pd.read_parquet(ROOT / LEGACY_ERA5).set_index('iso3')
    legacy_summary = json.loads((ROOT / LEGACY_SUMMARY).read_text())
    if legacy_summary['provenance']['files'][LEGACY_ERA5]['sha256'] != LEGACY_ERA5_SHA256:
        raise prov.ExecutionRefused('the legacy ERA5 digest differs from the M0.5 audit record')
    out = {}
    for name, table, summary in (('aligned_era5', aligned, aligned_summary), ('legacy_era5', legacy, legacy_summary)):
        y = table.loc[frozen.table.iso3, ERA5_COLUMN].to_numpy(float)
        out[name] = {'y': y, 'finite': bool(np.isfinite(y).all()), 'r2_anchor': float(summary['r2']['era5'])}
    return out


def residual_agreement(a, b):
    return {'pearson': float(np.corrcoef(a, b)[0, 1]), 'spearman': float(stats.spearmanr(a, b).statistic),
            'sign_agreement_n': int(np.sum(np.sign(a) == np.sign(b)))}


def products(frozen, static, qualifying, station_dist, m2_conditional=ev.ROOT / OUTPUTS / 'm2_conditional'):
    static_model = m3.STATIC_MODEL[static]
    outcomes = product_outcomes(frozen)
    report, rows = {}, []
    for product, spec in outcomes.items():
        if not spec['finite']:
            report[product] = {'status': 'non_computable', 'reason': 'a country has a non-finite outcome'}
            continue
        for representation in ev.REPRESENTATIONS:
            frame = frozen.frames[representation].assign(**{OUTCOME_COL: spec['y']})
            berkeley = frozen.frames[representation]
            pf = dataclasses.replace(frozen, y=spec['y'], frames={**frozen.frames, representation: frame})
            scored = ev.score_model(static_model, frame, pf)
            berkeley_scored = ev.score_model(static_model, berkeley, frozen)
            card = scored.card
            shares = {g: card[f'share_{g}'] for g in SHARE_ORDER}
            others = max(v for g, v in shares.items() if g != 'geography')
            entry = {'S': {'model': static_model, 'card': card,
                           'material_change_point': {'point_geography_not_largest': bool(not shares['geography'] > others),
                                                     'responsibility_exceeds_0_10': bool(shares['emissions'] > MATERIAL_RESPONSIBILITY)},
                           'residual_agreement_with_berkeley': residual_agreement(
                               spec['y'] - scored.fitted, berkeley[OUTCOME_COL].to_numpy(float) - berkeley_scored.fitted)}}
            if static == 'M0*':
                difference = abs(card['in_sample_r2'] - spec['r2_anchor'])
                if not difference <= REFERENCE_ATOL:
                    raise prov.ExecutionRefused(f'{product} {representation}: S in-sample R2 differs from the M0.5 anchor '
                                                f'by {difference}')
                entry['S']['anchor'] = f'in-sample R2 reproduces the M0.5 {product} summary to 1e-10'
            elif product == 'aligned_era5':
                entry['S']['anchor'] = aligned_arm_anchor(card, representation, m2_conditional)
            for name in qualifying:
                family = 'sem' if name.startswith('M3-SEM') else 'sar'
                run = m3.run_family(family, static_model, frame, pf, station_dist, representation)
                if run.status != 'computable':
                    entry[name] = {'status': 'non_computable', 'failure': run.failure}
                    continue
                ext_card = m3.family_card(run, frame, pf, static, frozen.station_w, frozen.area_w, station_dist)
                paired = m3.paired_block(spec['y'], run, scored)
                conditions = m3.qualification(ext_card, card, paired['vs_static'], True)
                entry[name] = {'status': 'computed', 'theta': run.full.theta, 'at_domain_bound': run.full.at_domain_bound,
                               'card': ext_card, 'paired': paired, 'accounting': m3.accounting_block(run, frame, scored),
                               'arm_conditions': {**conditions, 'descriptive_only': True},
                               'residual_agreement_with_berkeley': None}
                for protocol in ev.PROTOCOLS:
                    rows += [{'product': product, 'representation': representation, 'model': name, 'protocol': protocol,
                              'iso3': frozen.table.iso3.iloc[i], 'observed': spec['y'][i],
                              'prediction': run.results[protocol].yhat[i]} for i in range(frozen.n)]
            for protocol in ev.PROTOCOLS:
                rows += [{'product': product, 'representation': representation, 'model': static_model, 'protocol': protocol,
                          'iso3': frozen.table.iso3.iloc[i], 'observed': spec['y'][i],
                          'prediction': scored.results[protocol].yhat[i]} for i in range(frozen.n)]
            report[f'{product}/{representation}'] = entry
    for name in qualifying:
        aligned = report.get('aligned_era5/primary_total_co2', {}).get(name, {})
        report[f'product_sensitive/{name}'] = (None if aligned.get('status') != 'computed'
                                               else not aligned['arm_conditions']['qualifies'])
    report['interpretation'] = ('aligned ERA5 is the preferred construction; legacy ERA5 is the preprocessing '
                                'sensitivity. Product disagreement is not an identified amount or cause of '
                                'observational error, and ERA5 selects nothing.')
    return report, pd.DataFrame(rows)


def aligned_arm_anchor(card, representation, conditional_dir):
    path = Path(conditional_dir) / 'm2_conditional_scorecard.json'
    if not path.exists():
        return 'not applicable: no computed M2 conditional aligned-ERA5 arm'
    arm = json.loads(path.read_text())['arms']['aligned_era5_outcome']
    if arm['status'] != 'computed':
        return f"not applicable: the M2 conditional aligned-ERA5 arm is {arm['status']}"
    committed = arm['representations'][representation]['M2']
    check = ev.reference_check(card, committed, f'aligned ERA5 {representation} M2', ev.REFERENCE_METRICS, REFERENCE_ATOL)
    return {'reproduces_committed_m2_conditional_arm': check['reproduced'], 'tolerance': REFERENCE_ATOL}


# ---------------------------------------------------------------------
# D. Consolidation, inventory status and the final record
# ---------------------------------------------------------------------


def _load(rel):
    return json.loads((ROOT / rel).read_text())


def consolidated(static, naming, m2_manifest, station_manifest, land_manifest):
    committed_m0 = _load(M0_CORRECTED)
    corrected = {**committed_m0['corrected_primary'], 'secondary': committed_m0['secondary_m49_unchanged'],
                 'random_reference_only': committed_m0['reference_random_10fold']}
    rank = _load(ev.RANK_AUDIT)['cards']
    m1a = _load(ev.M1A_SCORECARD)
    m1b = _load(ev.M1B_SCORECARD)
    m2 = _load(f'{OUTPUTS}/m2_primary/m2_scorecard.json')
    m3s = _load(f'{OUTPUTS}/m3_station/m3_scorecard.json')
    m3l = _load(f'{OUTPUTS}/m3_land_centroid/m3_scorecard.json')

    def row(stage, status, card, comparator, paired, allocation, source):
        return {'stage': stage, 'status': status, 'comparator': comparator,
                'primary_cv_r2': card.get('cv_r2'), 'primary_cv_rmse': card.get('cv_rmse'), 'primary_cv_mae': card.get('cv_mae'),
                'm49_cv_rmse': card.get('secondary', {}).get('cv_rmse'),
                'random_cv_rmse': card.get('random_reference_only', {}).get('cv_rmse'),
                'worst_region': card.get('worst_region'), 'oof_morans_i': card.get('residual_morans_i_cv'),
                'paired_delta_rmse': None if paired is None else paired.get('delta_rmse'),
                'paired_interval': None if paired is None else paired.get('country_bootstrap_95_interval'),
                'allocation': allocation, 'source': source}

    shares = lambda c: {g: c.get(f'share_{g}') for g in SHARE_ORDER} | {'residual': c.get('residual_share')}  # noqa: E731
    rows = [
        row('M0 (V1 frozen, corrected 500 km primary)', 'frozen baseline', corrected, None, None,
            {'V1_rank_deficient_shares': shares(corrected), 'note': 'V1 allocation; not restated under V2'}, M0_CORRECTED),
        row('M0* primary total CO2', 'approved full-rank development baseline', rank['drop_per_capita'], None, None,
            shares(rank['drop_per_capita']), ev.RANK_AUDIT),
        row('M0* per-capita', 'registered representation sensitivity', rank['drop_total'], None, None,
            shares(rank['drop_total']), ev.RANK_AUDIT),
        row('M1a (area geography)', 'not promoted; measurement sensitivity', m1a['primary']['M1a'], 'M0* (C0)',
            m1a['primary']['M1a']['paired_delta_vs_m0'], shares(m1a['primary']['M1a']), ev.M1A_SCORECARD),
        row('M1b (M0* + C2 dryness)', 'not supported', m1b['representations']['primary_total_co2']['M1b'], 'M0*',
            m1b['representations']['primary_total_co2']['paired_primary_comparison'],
            shares(m1b['representations']['primary_total_co2']['M1b']), ev.M1B_SCORECARD),
        row('M2 (M0* + latitude functional form)', m2['verdict']['verdict'], m2['representations']['primary_total_co2']['M2'],
            'M0*', m2['representations']['primary_total_co2']['paired_primary_comparison'],
            shares(m2['representations']['primary_total_co2']['M2']), f'{OUTPUTS}/m2_primary/m2_scorecard.json'),
    ]
    for scorecard, arm in ((m3s, 'station'), (m3l, 'land')):
        for family in m3.FAMILIES:
            entry = scorecard['families'][f'primary_total_co2/{family}']
            name = m3.extension_name(family, static) + ('' if arm == 'station' else ' [land-centroid kNN8 sensitivity]')
            if entry['status'] != 'computable':
                status = 'non-computable'
            elif arm == 'station':
                status = 'qualifying' if entry['qualification']['qualifies'] else 'non-qualifying'
            else:
                status = 'weight sensitivity (descriptive)'
            rows.append(row(name, status, entry.get('card', {}), static, entry.get('paired', {}).get('vs_static'),
                            {'accounting': entry.get('accounting', {}).get('estimands'),
                             'filtered_trend_shares': entry.get('accounting', {}).get('filtered_trend_shares', {}).get('shares')},
                            f'{OUTPUTS}/m3_{"station" if arm == "station" else "land_centroid"}/m3_scorecard.json'))
    legacy = {'M0 legacy 1° primary (defective approximation; superseded)': _load(M0_LEGACY)}
    return {'rows': rows, 'legacy_block': {k: {'primary_cv_rmse': v['protocols']['primary_border_buffered_loo_500km']['cv_rmse'],
                                               'primary_cv_r2': v['protocols']['primary_border_buffered_loo_500km']['cv_r2']}
                                           for k, v in legacy.items()},
            'separations': ['legacy 1° CV rows are kept outside the corrected table',
                            'V1 allocations (rank-deficient M0) are kept separate from V2 full-rank allocations']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='M4 final assessment under the frozen specification.')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--compare', nargs=2, metavar=('RUN_A', 'RUN_B'))
    parser.add_argument('--record')
    args = parser.parse_args(argv)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    if args.compare:
        record = m3.reproducibility_record(*args.compare, code=prov.frozen_code_state(code_files()))
        record['artifacts'] = {n: {'run_a_sha256': prov.sha256(Path(args.compare[0]) / n),
                                   'run_b_sha256': prov.sha256(Path(args.compare[1]) / n)}
                               for n in (*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST)}
        for value in record['artifacts'].values():
            value['identical'] = value['run_a_sha256'] == value['run_b_sha256']
        record['byte_identical'] = all(v['identical'] for v in record['artifacts'].values())
        if args.record:
            prov.write_json(record, args.record)
        print(prov.json_text({'byte_identical': record['byte_identical']}))
        return record
    prov.require_hash_seed()
    code = prov.frozen_code_state(code_files())
    out_dir = prov.require_fresh(args.out)
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    pre = verified_prerequisites()
    digests = ev.verify_pins()
    frozen = ev.build_frozen()
    static = pre['m2']['retained_static_specification']
    naming = pre['naming']
    qualifying = naming['qualifying_spatial_extensions']
    station_dist = m3.arm_distance(frozen, 'station')

    uncertainty, draws = static_uncertainty(frozen, m3.STATIC_MODEL[static])
    sensitivity, sensitivity_rows = sensitivities(frozen, static, qualifying, station_dist)
    product, product_rows = products(frozen, static, qualifying, station_dist)
    table = consolidated(static, naming, pre['m2']['manifest'], pre['station']['manifest'], pre['land_manifest'])
    m2_scorecard = _load(f'{OUTPUTS}/m2_primary/m2_scorecard.json')
    station_scorecard = _load(f'{OUTPUTS}/m3_station/m3_scorecard.json')
    retained = [static] + qualifying
    three_numbers = {}
    for name in retained:
        if name == static:
            card = (_load(ev.RANK_AUDIT)['cards']['drop_per_capita'] if static == 'M0*'
                    else m2_scorecard['representations']['primary_total_co2']['M2'])
            geography = uncertainty['primary_total_co2']['country']['groups']['geography']
            three_numbers[name] = {'primary_rmse_minus_m0star': card['cv_rmse'] - _load(ev.RANK_AUDIT)['cards']['drop_per_capita']['cv_rmse'],
                                   'oof_morans_i': card['residual_morans_i_cv'],
                                   'geography_share': geography['point'],
                                   'geography_share_country_bootstrap_95': [geography['ci_low'], geography['ci_high']]}
        else:
            family = 'sem' if name.startswith('M3-SEM') else 'sar'
            entry = station_scorecard['families'][f'primary_total_co2/{family}']
            three_numbers[name] = {'primary_rmse_minus_m0star': entry['card']['cv_rmse'] - _load(ev.RANK_AUDIT)['cards']['drop_per_capita']['cv_rmse'],
                                   'oof_morans_i': entry['card']['residual_morans_i_cv'],
                                   'geography_share_filtered_trend': entry['accounting']['filtered_trend_shares']['shares']['geography'],
                                   'geography_share_interval': 'not applicable (spec §3.3)'}
    stopping = m2_scorecard['static_stopping_indicator']
    commits = {
        'downstream_specification': introducing_commit('research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md'),
        'downstream_specification_amendment_A1': 'ef04958',
        'm2_evaluator': pre['m2']['manifest']['evaluator_commit'],
        'm2_result': introducing_commit(f'{OUTPUTS}/m2_primary/m2_result_manifest.json'),
        'm2_conditional_disposition': introducing_commit(f'{OUTPUTS}/m2_conditional/m2_conditional_run_metadata.json'),
        'm3_evaluator': pre['station']['manifest']['evaluator_commit'],
        'm3_station_result': introducing_commit(f'{OUTPUTS}/m3_station/m3_result_manifest.json'),
        'm3_land_result': introducing_commit(f'{OUTPUTS}/m3_land_centroid/m3_result_manifest.json'),
        'm4_evaluator': code['commit'],
    }
    final = {
        'retained_static_specification': static,
        'qualifying_spatial_extensions': qualifying,
        'final_primary_predictive_model': naming['final_primary_predictive_model'],
        'final_naming_reason': naming['reason'],
        'm2_label': m2_scorecard['verdict']['verdict'],
        'static_stopping_indicator': stopping,
        'm3_family_dispositions': {f'{arm}/{family}': (card['families'][f'primary_total_co2/{family}']['status'],
                                                       card['families'][f'primary_total_co2/{family}'].get(
                                                           'qualification', {}).get('qualifies'))
                                   for arm, card in (('station', station_scorecard),
                                                     ('land', _load(f'{OUTPUTS}/m3_land_centroid/m3_scorecard.json')))
                                   for family in m3.FAMILIES},
        'three_numbers_per_retained_stage': three_numbers,
        'material_change_to_v1_conclusion': {r: uncertainty[r]['material_change'] for r in ev.REPRESENTATIONS},
        'commits': commits,
        'residual_description': 'variance in cross-country warming differences not captured at country resolution by '
                                'the tested models with these observations; not an unknown climate cause',
    }
    inventory = inventory_status(static, qualifying, pre, uncertainty, sensitivity, product)
    out_dir.mkdir(parents=True)
    prov.write_csv(draws, out_dir / 'm4_bootstrap_draws.csv')
    prov.write_json(uncertainty, out_dir / 'm4_bootstrap_summary.json')
    prov.write_json(sensitivity, out_dir / 'm4_sensitivities.json')
    prov.write_csv(sensitivity_rows, out_dir / 'm4_sensitivity_predictions.csv')
    prov.write_json(product, out_dir / 'm4_products.json')
    prov.write_csv(product_rows, out_dir / 'm4_product_predictions.csv')
    prov.write_json(table, out_dir / 'm4_consolidated_table.json')
    prov.write_csv(pd.json_normalize(table['rows']), out_dir / 'm4_consolidated_table.csv')
    prov.write_json(inventory, out_dir / 'm4_inventory_status.json')
    prov.write_json(final, out_dir / 'm4_final_specification.json')
    prov.write_json({'documents_sha256': {d: prov.sha256(ROOT / d) for d in DOCUMENTS},
                     'code': {k: v for k, v in code.items() if k != 'live_remote_commit'}, 'inputs': digests,
                     'legacy_era5_sha256': LEGACY_ERA5_SHA256, 'identity': frozen.identity,
                     'prerequisite_manifests': {'m2': pre['m2']['manifest_sha256'],
                                                'm3_station': pre['station']['manifest_sha256'],
                                                'm3_land': pre['land_manifest_sha256']},
                     'software': ev.software()}, out_dir / 'm4_provenance.json')
    complete = all(item['status'] in ('verified evidence', 'permitted disposition') for item in inventory['items'])
    manifest = {'result': 'M4 final assessment', 'evaluator_commit': code['commit'], 'inventory_complete': complete,
                'final': {k: final[k] for k in ('retained_static_specification', 'qualifying_spatial_extensions',
                                                'final_primary_predictive_model')},
                'integrity_passed': complete,
                'artifact_sha256': {n: prov.sha256(out_dir / n) for n in DETERMINISTIC_ARTIFACTS},
                'non_deterministic_artifacts': [RUN_METADATA]}
    prov.write_json(manifest, out_dir / RESULT_MANIFEST)
    prov.write_json({'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     'wall_seconds': time.perf_counter() - started, 'host': platform.node(), 'executable': sys.executable,
                     'argv': sys.argv, 'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
                     'live_remote_commit': code['live_remote_commit']}, out_dir / RUN_METADATA)
    print(prov.json_text({'final': manifest['final'], 'inventory_complete': complete}))
    return manifest


def inventory_status(static, qualifying, pre, uncertainty, sensitivity, product):
    def item(identifier, status, evidence):
        return {'id': identifier, 'status': status, 'evidence': evidence}

    usable = all(uncertainty[r][k]['usable'] > 0 for r in ev.REPRESENTATIONS for k in ('country', 'block'))
    items = [
        item('I1', 'verified evidence', 'm4_consolidated_table.json (committed stage scorecards)'),
        item('I2', 'verified evidence', f'{M0_LEGACY} kept in the legacy block'),
        item('I3', 'verified evidence', 'V1 allocations labelled separately in m4_consolidated_table.json'),
        item('I4', 'verified evidence', 'paired intervals in the consolidated rows (M1a, M1b, M2, M3)'),
        item('I5', 'verified evidence', 'region tables in the committed stage cards and the M3 scorecards'),
        item('I6', 'verified evidence', 'OOF Moran in the consolidated rows'),
        item('I7', 'verified evidence', 'effective degrees of freedom in the stage cards'),
        item('I8', 'verified evidence', 'random-reference rows in the stage cards'),
        item('I9', 'verified evidence', 'both representations in every new computation'),
        item('I10', 'verified evidence' if usable else 'missing', 'm4_bootstrap_summary.json (S only; others not applicable)'),
        item('I11', 'verified evidence', 'm4_bootstrap_summary.json material_change'),
        item('I12', 'verified evidence', 'm2_scorecard.json static_stopping_indicator'),
        item('I13', 'verified evidence', f'{OUTPUTS}/m3_station/'),
        item('I14', 'permitted disposition', 'spatial range not applicable; graph descriptors in m3_scorecard.json'),
        item('I15', 'verified evidence', 'dependence-parameter se/Wald or not-applicable reason in m3_scorecard.json; '
                                         'bootstrap of accounting parts not applicable'),
        item('I16', 'verified evidence', f'{OUTPUTS}/m3_land_centroid/'),
        item('I17', 'verified evidence', 'm4_sensitivities.json buffer_territory_1000km'),
        item('I18', 'verified evidence', 'm4_sensitivities.json buffer_centroid_1500km'),
        item('I19', 'verified evidence' if 'aligned_era5/primary_total_co2' in product else 'permitted disposition',
             'm4_products.json aligned_era5'),
        item('I20', 'verified evidence' if 'legacy_era5/primary_total_co2' in product else 'permitted disposition',
             'm4_products.json legacy_era5; legacy M2-vs-M0* replication not executed (contract §10)'),
        item('I21', 'permitted disposition', 'two-product mean not executed: never authorized (owner 2026-09-16)'),
        item('I22', 'verified evidence', f'{ev.M1A_SCORECARD}' if static == 'M0*' else f'{OUTPUTS}/m2_conditional/'),
        item('I23', 'verified evidence', 'M1b C2 and M2 gamma summaries; M3 theta same-sign fraction'),
        item('I24', 'permitted disposition', 'written in the closure report'),
        item('I25', 'verified evidence', 'm4_final_specification.json'),
        item('I26', 'permitted disposition', 'not pursued (register)'),
        item('I27', 'permitted disposition', 'not pursued, not experimentally rejected'),
    ]
    return {'items': items, 'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying}


if __name__ == '__main__':
    outcome = main()
    sys.exit(0 if outcome.get('integrity_passed', outcome.get('byte_identical')) else 3)
