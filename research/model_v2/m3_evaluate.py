"""M3 scoring: spatial error and spatial lag extensions of the retained static specification.

Two spatial families, a spatial error model (SEM) and a spatial lag model (SAR), are fitted by Gaussian ML
on the retained static specification S (M0* or M2, read from the committed M2 result) with kNN8 graphs:
station centroids (``--weights station``, the arm that decides qualification and final naming) or land
centroids (``--weights land``, the historically required weight sensitivity, run after the station package is
committed). Held-out nodes attach as sinks to their eight nearest training nodes; only training outcomes enter
any estimate, lag or prediction. Accounting is the fixed multi-part identity with filtered-trend group shares.

Run with ``PYTHONHASHSEED=0 uv run python -m research.model_v2.m3_evaluate --weights station|land``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v2 import cv
from research.model_v2 import m2_evaluate as ev
from research.model_v2 import m2_feasibility as feas
from research.model_v2 import m2_latitude_basis as lb
from research.model_v2 import m3_spatial as ms
from research.model_v2 import v2_provenance as prov
from research.model_v2.run_m0_scorecard import MIN_REGION_N
from research.model_v2.run_territory_correction import paired_rmse_interval
from research.model_v2.spatial import haversine_matrix, knn_weights, morans_i
from src.decomposition import OUTCOME_COL

ROOT = prov.ROOT
OUTPUTS = ev.OUTPUTS
SPEC = 'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md'
EVALUATOR_SPEC = 'research/model_v2/M3_EVALUATOR_SPEC.md'
DOCUMENTS = (SPEC, EVALUATOR_SPEC, 'research/model_v2/M3_FEASIBILITY_AUDIT.md')
ENTRY_FILES = ('research/model_v2/m3_evaluate.py',)
M2_PRIMARY = ROOT / OUTPUTS / 'm2_primary'
M2_CONDITIONAL = ROOT / OUTPUTS / 'm2_conditional'
OUT = {'station': ROOT / OUTPUTS / 'm3_station', 'land': ROOT / OUTPUTS / 'm3_land_centroid'}
FAMILIES = ms.FAMILIES
FAMILY_NAMES = {'sem': 'SEM', 'sar': 'SAR'}
PROTOCOLS = ev.PROTOCOLS
PRIMARY = ev.PRIMARY
VETO = 0.001
OVERFIT_GAP = 0.05
OVERFIT_DF_RISE = 5
PRACTICAL = -0.002
REFERENCE_ATOL = 1e-10
PREDICTION_ATOL = 1e-6    # spatial representation identity tolerance
THETA_ATOL = 1e-5
MORAN_PERMUTATIONS = 999
STATIC_MODEL = {'M0*': ev.COMPARATOR, 'M2': ev.CANDIDATE}
DETERMINISTIC_ARTIFACTS = ('m3_scorecard.json', 'm3_country_predictions.csv', 'm3_fits.csv', 'm3_graphs.csv',
                           'm3_full_sample_grid.csv', 'm3_provenance.json')
RESULT_MANIFEST = 'm3_result_manifest.json'
RUN_METADATA = 'm3_run_metadata.json'
EQUIVALENT_SCORES = ('in_sample_one_step_r2', 'trend_r2', 'cv_r2', 'cv_rmse', 'cv_mae', 'residual_morans_i_in_sample',
                     'residual_morans_i_cv', 'worst_region.rmse', 'secondary.cv_rmse', 'random_reference_only.cv_rmse')
ACCOUNTING_PARTS = ('a_static', 'a_dependence', 'a_innovation', 'absorbed_fraction_of_static_residual')


def code_files() -> list[str]:
    return sorted({*prov.import_closure(ENTRY_FILES), *DOCUMENTS, *prov.LOCK_FILES})


def extension_name(family, static):
    return f'M3-{FAMILY_NAMES[family]}({static})'


# ---------------------------------------------------------------------
# Prerequisites: committed, pushed, verified M2 (and station) packages
# ---------------------------------------------------------------------


def _repo_paths(directory, names):
    return [str((Path(directory) / n).resolve().relative_to(ROOT)) for n in names]


def verified_m2(primary=M2_PRIMARY, conditional=M2_CONDITIONAL) -> dict:
    manifest = prov.verify_manifest(primary, ev.RESULT_MANIFEST)
    if manifest.get('integrity_passed') is not True or manifest.get('verdict') is None:
        raise prov.ExecutionRefused('the M2 primary result has no integrity-valid label')
    static = manifest.get('retained_static_specification')
    if static not in STATIC_MODEL:
        raise prov.ExecutionRefused(f'the M2 result does not record a retained static specification ({static!r})')
    names = [*ev.DETERMINISTIC_ARTIFACTS, ev.RESULT_MANIFEST, ev.RUN_METADATA]
    conditional = Path(conditional)
    if (conditional / 'm2_conditional_not_run.json').exists():
        conditional_names = ['m2_conditional_not_run.json', 'm2_conditional_run_metadata.json']
        disposition = 'not executed'
    else:
        conditional_manifest = prov.verify_manifest(conditional, 'm2_conditional_manifest.json')
        conditional_names = [*conditional_manifest['artifact_sha256'], 'm2_conditional_manifest.json',
                             'm2_conditional_run_metadata.json']
        disposition = conditional_manifest['arm_status']
    prov.tracked_and_pushed(_repo_paths(primary, names) + _repo_paths(conditional, conditional_names))
    expected = 'M2' if manifest['verdict'] == 'supported and promoted' else 'M0*'
    if static != expected:
        raise prov.ExecutionRefused(f'retained static {static!r} is inconsistent with the label {manifest["verdict"]!r}')
    return {'manifest': manifest, 'manifest_sha256': prov.sha256(Path(primary) / ev.RESULT_MANIFEST),
            'retained_static_specification': static, 'conditional_disposition': disposition}


def verified_station(directory=None) -> dict:
    directory = Path(directory or OUT['station'])
    manifest = prov.verify_manifest(directory, RESULT_MANIFEST)
    if manifest.get('integrity_passed') is not True or manifest.get('weights') != 'station':
        raise prov.ExecutionRefused('the station M3 package is missing, not integrity-valid or not the station arm')
    prov.tracked_and_pushed(_repo_paths(directory, [*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST, RUN_METADATA]))
    return {'manifest': manifest, 'manifest_sha256': prov.sha256(directory / RESULT_MANIFEST)}


# ---------------------------------------------------------------------
# Static reproduction
# ---------------------------------------------------------------------


def reproduce_static(frozen, models, primary=M2_PRIMARY) -> dict:
    """Refit the static models with the M2 evaluator's functions; predictions and cards must match to 1e-10."""
    saved = pd.read_csv(Path(primary) / 'm2_country_predictions.csv', float_precision='round_trip')
    scorecard = json.loads((Path(primary) / 'm2_scorecard.json').read_text())
    record, scored = {}, {}
    for representation in ev.REPRESENTATIONS:
        frame = frozen.frames[representation]
        for model in models:
            result = ev.score_model(model, frame, frozen)
            rows = saved[(saved.representation == representation) & (saved.model == model)]
            worst = 0.0
            for protocol in ('in_sample', *PROTOCOLS):
                values = rows[rows.protocol == protocol].set_index('iso3').loc[frozen.table.iso3, 'prediction'].to_numpy(float)
                ours = result.fitted if protocol == 'in_sample' else result.results[protocol].yhat
                worst = max(worst, float(np.max(np.abs(values - ours))))
            if not worst <= REFERENCE_ATOL:
                raise prov.ExecutionRefused(f'{representation} {model}: static predictions differ from the M2 result by {worst}')
            check = ev.reference_check(result.card, scorecard['representations'][representation][model],
                                       f'{representation} {model}', ev.REFERENCE_METRICS, REFERENCE_ATOL)
            record[f'{representation}/{model}'] = {'max_prediction_difference': worst, 'card': check}
            scored[(representation, model)] = result
    return {'record': record, 'scored': scored}


# ---------------------------------------------------------------------
# One spatial family on one representation and weights arm
# ---------------------------------------------------------------------


@dataclasses.dataclass
class FamilyRun:
    family: str
    representation: str
    status: str = 'computable'
    failure: dict | None = None
    full: ms.SpatialFit | None = None
    full_design: np.ndarray | None = None
    full_columns: list | None = None
    full_graph: np.ndarray | None = None
    quantities: ms.FittedQuantities | None = None
    interval: ms.DependenceInterval | None = None
    results: dict = dataclasses.field(default_factory=dict)
    records: dict = dataclasses.field(default_factory=dict)
    adjustments: dict = dataclasses.field(default_factory=dict)


def static_design(static_model, rows, encoder, state):
    return ev.design(static_model, rows, encoder, state)


def fit_one(family, static_model, frame, dist, train, test):
    """One spatial fit on training rows ``train`` with held-out rows ``test`` (canonical indices)."""
    tr = frame.iloc[train]
    encoder = feas.baseline_encoder(tr)
    state = lb.fit_state(tr.abs_latitude.to_numpy(float)) if static_model == ev.CANDIDATE else None
    x_train, names = static_design(static_model, tr, encoder, state)
    if np.linalg.matrix_rank(x_train) != x_train.shape[1]:
        raise ms.SpatialFitFailure('rank-deficient', 'the static training design is rank deficient')
    w = ms.training_graph(dist, train)
    y_train = tr[OUTCOME_COL].to_numpy(float)
    fit = ms.fit_spatial(family, y_train, x_train, w, names)
    record = {'train': train, 'test': test, 'fit': fit, 'encoder': encoder, 'state': state,
              'neighbours': ms.training_neighbours(dist, train)}
    if len(test):
        te = frame.iloc[test]
        neighbours, weights = ms.attachment(dist, train, test)
        x_test, _ = static_design(static_model, te, encoder, state)
        base = ms.predict_held_out(fit, y_train, x_train, x_test, ms.local_positions(train, neighbours), weights)
        levels = {c: encoder.levels[c] for c in encoder.categorical}
        labels = {c: te[c].astype(str).tolist() for c in encoder.categorical}
        prediction = ms.apply_unseen_level_adjustment(base, fit, levels, labels)
        record.update({'attachments': neighbours, 'prediction': prediction, 'adjustment': prediction - base,
                       'unseen': np.array([sum(lab not in levels[c] for c, labs in labels.items() for lab in [labs[k]])
                                           for k in range(len(test))])})
    return record


def run_family(family, static_model, frame, frozen, dist, representation, protocols=PROTOCOLS) -> FamilyRun:
    """Full-sample fit and every fit of ``protocols`` (fold ids and buffers from ``frozen``)."""
    run = FamilyRun(family, representation)
    n = frozen.n
    everyone = np.arange(n)
    try:
        encoder = feas.baseline_encoder(frame)
        state = lb.fit_state(frame.abs_latitude.to_numpy(float)) if static_model == ev.CANDIDATE else None
        x, names = static_design(static_model, frame, encoder, state)
        w = ms.training_graph(dist, everyone)
        y = frame[OUTCOME_COL].to_numpy(float)
        run.full = ms.fit_spatial(family, y, x, w, names)
        run.full_design, run.full_columns, run.full_graph = x, names, w
        run.quantities = ms.fitted_quantities(run.full, y, x, w)
        run.interval = ms.dependence_interval(run.full, x, w)
    except ms.SpatialFitFailure as failure:
        run.status, run.failure = 'non_computable', {'protocol': 'full_sample', 'fit': 'all', 'reason': failure.reason,
                                                     'detail': failure.detail}
        return run
    for protocol in protocols:
        ids, buffer = frozen.folds[protocol]
        yhat, n_train = np.full(n, np.nan), np.zeros(n, int)
        nearest, unseen, adjust = np.full(n, np.nan), np.zeros(n, int), np.zeros(n)
        records = []
        for fold, test_mask, train_mask in cv.iter_folds(ids, frozen.distance, buffer):
            train, test = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
            label = ev.fit_label(protocol, fold, frozen.table)
            try:
                record = fit_one(family, static_model, frame, dist, train, test)
            except (ms.SpatialFitFailure, lb.DegenerateLatitudeBasis) as failure:
                run.status = 'non_computable'
                run.failure = {'protocol': protocol, 'fit': label, 'reason': getattr(failure, 'reason', 'degenerate-state'),
                               'detail': str(failure)}
                return run
            record.update({'protocol': protocol, 'fold': int(fold), 'label': label})
            records.append(record)
            yhat[test] = record['prediction']
            adjust[test] = record['adjustment']
            unseen[test] = record['unseen']
            n_train[test] = len(train)
            nearest[test] = frozen.distance[np.ix_(test, train)].min(axis=1)
        run.results[protocol] = cv.CVResult(yhat, ids.copy(), n_train, nearest, unseen)
        run.records[protocol] = records
        run.adjustments[protocol] = adjust
    return run


# ---------------------------------------------------------------------
# Cards, qualification, accounting
# ---------------------------------------------------------------------


def region_block(result, y, table):
    regions = cv.region_errors(result, y, table.m49_subregion)
    eligible = regions[regions.n >= MIN_REGION_N]
    return {'worst_region': {'region': str(eligible.index[0]), 'rmse': float(eligible.iloc[0].rmse),
                             'n': int(eligible.iloc[0].n), 'min_region_n': MIN_REGION_N},
            'region_rmse': {str(k): float(v) for k, v in regions.rmse.items()},
            'region_bias': {str(k): float(v) for k, v in regions.bias.items()},
            'region_mae': {str(k): float(v) for k, v in regions.mae.items()}}


def protocol_card(frame, table, result, one_step, station_w, area_w):
    y = frame[OUTCOME_COL].to_numpy(float)
    card = cv.scorecard(frame, one_step, result, station_w, n_permutations=MORAN_PERMUTATIONS, shares=False)
    card['in_sample_one_step_r2'] = card.pop('in_sample_r2')
    card['in_sample_one_step_rmse'] = card.pop('in_sample_rmse')
    errors = np.abs(y - result.yhat)
    card['fold_error_iqr'] = float(np.subtract(*np.quantile(errors, [0.75, 0.25])))
    card['residual_moran_in_sample'] = asdict(morans_i(y - one_step, station_w))
    card['residual_moran_cv'] = asdict(morans_i(y - result.yhat, station_w))
    card['recorded_land_centroid_moran_cv'] = morans_i(y - result.yhat, area_w).statistic
    card.update(region_block(result, y, table))
    card['worst_fold_country'] = str(table.iloc[card['worst_fold']].Country) if card['n_folds'] == len(y) else None
    return card


def graph_descriptors(run, dist, table):
    everyone = np.arange(len(table))
    neighbours = ms.training_neighbours(dist, everyone)
    d = np.take_along_axis(dist, neighbours, axis=1)
    out = {'full_sample': {'first_neighbour_km_median': float(np.median(d[:, 0])),
                           'first_neighbour_km_max': float(np.max(d[:, 0])),
                           'eighth_neighbour_km_median': float(np.median(d[:, -1])),
                           'eighth_neighbour_km_max': float(np.max(d[:, -1]))}}
    for protocol, records in run.records.items():
        first, eighth = [], []
        for record in records:
            att = record.get('attachments')
            if att is None:
                continue
            near = dist[record['test'][:, None], att]
            first.extend(near[:, 0].tolist())
            eighth.extend(near[:, -1].tolist())
        out[protocol] = {'attachment_first_km_min': float(np.min(first)), 'attachment_first_km_median': float(np.median(first)),
                         'attachment_first_km_max': float(np.max(first)), 'attachment_eighth_km_min': float(np.min(eighth)),
                         'attachment_eighth_km_median': float(np.median(eighth)),
                         'attachment_eighth_km_max': float(np.max(eighth))}
    out['spatial_range'] = 'not applicable: a fixed kNN8 graph has no estimated range'
    return out


def dependence_summary(run):
    out = {'full_sample': {'theta': run.full.theta, 'at_domain_bound': run.full.at_domain_bound,
                           'se': run.interval.se, 'wald_lower': run.interval.lower, 'wald_upper': run.interval.upper,
                           'interval_reason': run.interval.reason,
                           'interval_note': 'asymptotic, conditional on W and the Gaussian model; not spatially robust'},
           'reading': 'with the frozen static designs theta-hat is pulled strongly toward negative values under no '
                      'dependence (M3_FEASIBILITY_AUDIT.md §3); its sign and size are not read as dependence strength'}
    for protocol, records in run.records.items():
        thetas = np.array([r['fit'].theta for r in records])
        out[protocol] = {'n_fits': len(thetas), 'median': float(np.median(thetas)), 'min': float(thetas.min()),
                         'max': float(thetas.max()), 'strictly_positive_n': int((thetas > 0).sum()),
                         'at_domain_bound_n': int(sum(r['fit'].at_domain_bound for r in records))}
    primary = np.array([r['fit'].theta for r in run.records[PRIMARY]])
    out['same_sign_as_full_fraction_primary'] = float(np.mean(np.sign(primary) == np.sign(run.full.theta)))
    return out


def qualification(card, static_card, paired, computable):
    delta = float(paired['delta_rmse'])
    lo, hi = (float(v) for v in paired['country_bootstrap_95_interval'])
    m49 = float(card['secondary']['cv_rmse']) - float(static_card['secondary']['cv_rmse'])
    region = float(card['worst_region']['rmse']) - float(static_card['worst_region']['rmse'])
    for name, value in (('delta_rmse', delta), ('interval_low', lo), ('interval_high', hi), ('m49_rmse_change', m49),
                        ('worst_region_rmse_change', region)):
        if not np.isfinite(value):
            raise ValueError(f'qualification input {name} is not finite')
    q = {'Q1_delta_below_zero': bool(delta < 0), 'Q2_interval_upper_below_zero': bool(hi < 0),
         'Q3_m49_veto_passed': bool(m49 <= VETO), 'Q4_worst_region_veto_passed': bool(region <= VETO),
         'Q5_computable': bool(computable)}
    return {**q, 'qualifies': bool(all(q.values())), 'delta_rmse': delta, 'interval': [lo, hi],
            'm49_rmse_change': m49, 'worst_region_rmse_change': region, 'veto_threshold': VETO,
            'worsened_generalization': bool(delta > 0 and lo > 0),
            'not_applied': ['0.002 practical threshold', 'any sign criterion', 'any likelihood criterion',
                            'any overfitting veto']}


def family_card(run, frame, frozen, static, station_w, area_w, dist):
    y = frame[OUTCOME_COL].to_numpy(float)
    q = run.quantities
    card = protocol_card(frame, frozen.table, run.results[PRIMARY], q.one_step, station_w, area_w)
    card['secondary'] = protocol_card(frame, frozen.table, run.results['m49_subregion_lo'], q.one_step, station_w, area_w)
    card['random_reference_only'] = protocol_card(frame, frozen.table, run.results['random10'], q.one_step,
                                                  station_w, area_w)
    tss = float(np.sum((y - y.mean()) ** 2))
    card['trend_r2'] = float(1.0 - np.sum((y - q.trend) ** 2) / tss)
    card['structural_residual_moran_in_sample'] = morans_i(q.structural_residual, station_w, MORAN_PERMUTATIONS).statistic
    card['effective_degrees_of_freedom'] = int(run.full.p + 1)
    card['sigma2'] = run.full.sigma2
    return card


def accounting_block(run, frame, static_scored):
    y = frame[OUTCOME_COL].to_numpy(float)
    shares = static_scored.lmg['shares']
    rss_static = ms.static_rss(y, run.full_design)
    parts = ms.accounting(y, rss_static, run.quantities.innovation, trend=run.quantities.trend, static_shares=shares)
    groups = ev.column_groups(frame, run.full_columns)
    blocks = {g: run.full_design[:, [j for j, h in enumerate(groups) if h == g]] for g in ev.GROUPS
              if g in groups}
    filtered = ms.filtered_group_shares(run.full, y, run.full_design, run.full_graph, blocks)
    static_r2 = sum(shares.values())
    return {
        'estimands': {'a_static': parts.a_static, 'a_dependence': parts.a_dependence, 'a_innovation': parts.a_innovation,
                      'absorbed_fraction_of_static_residual': parts.absorbed_fraction_of_static_residual,
                      'identity_sum': parts.a_static + parts.a_dependence + parts.a_innovation,
                      'tss': parts.tss, 'rss_static': parts.rss_static, 'rss_innovation': parts.rss_innovation,
                      'one_step_r2': parts.one_step_r2, 'trend_r2': parts.trend_r2},
        'static_lmg_shares': dict(shares), 'static_lmg_share_sum_equals_a_static': True,
        'filtered_trend_shares': {'shares': filtered.shares, 'r2_full': filtered.r2_full,
                                  'residual_share': filtered.residual_share, 'n_coalitions': filtered.n_coalitions,
                                  'full_coalition_max_gap': filtered.full_coalition_max_gap,
                                  'denominator': 'variance of the filtered outcome M(theta-hat) y at the full-sample estimate'},
        'composition': {'static': {g: v / static_r2 for g, v in shares.items()}, 'filtered': filtered.composition},
        'material_change_of_non_spatial_part': bool(ms.material_change(filtered.shares)),
        'meaning': 'A_dependence is the in-sample reduction in squared one-step error from the spatial ML fit and its '
                   'neighbour-conditional predictor; it is not an LMG share, not a variance component and not a '
                   'mechanism, and it is never added to any named group. Filtered shares use a different denominator '
                   'and are never summed with the static shares.',
    }


def diagnostics(card, static_card, paired):
    gain_in = card['in_sample_one_step_r2'] - static_card['in_sample_r2']
    gain_cv = card['cv_r2'] - static_card['cv_r2']
    df_rise = card['effective_degrees_of_freedom'] - static_card['effective_degrees_of_freedom']
    clause_i = bool(gain_in - gain_cv > OVERFIT_GAP)
    clause_ii = bool(card['cv_rmse'] > static_card['cv_rmse']
                     and card['in_sample_one_step_rmse'] < static_card['in_sample_rmse'])
    clause_iv = bool(df_rise > OVERFIT_DF_RISE and float(paired['delta_rmse']) > PRACTICAL)
    return {'clause_i_one_step_gain_minus_cv_gain_gt_0.05': clause_i, 'clause_ii_cv_worse_while_in_sample_better': clause_ii,
            'clause_iii': 'theta same-sign fraction reported under dependence_parameter, without a cutoff',
            'clause_iv_df_rise_gt_5_with_gain_lt_0.002': clause_iv, 'effective_df_rise': int(df_rise),
            'signal_from_clauses_i_ii_iv': bool(clause_i or clause_ii or clause_iv),
            'in_sample_one_step_r2_gain': gain_in, 'cv_r2_gain': gain_cv,
            'delta_moran_in_sample_innovation_vs_static': card['residual_morans_i_in_sample']
            - static_card['residual_morans_i_in_sample'],
            'delta_moran_cv': card['residual_morans_i_cv'] - static_card['residual_morans_i_cv'],
            'note': 'diagnostics only; the one-step in-sample R2 uses neighbours\' observed outcomes and is not an OLS R2'}


def paired_block(y, run, static_scored, m0star_scored=None):
    out = {'vs_static': paired_rmse_interval(y, run.results[PRIMARY].yhat, static_scored.results[PRIMARY].yhat)}
    abs_change = np.abs(y - run.results[PRIMARY].yhat) - np.abs(y - static_scored.results[PRIMARY].yhat)
    out['countries'] = {'improved': int((abs_change < 0).sum()), 'worsened': int((abs_change > 0).sum()),
                        'tied': int((abs_change == 0).sum())}
    for protocol in ('m49_subregion_lo', 'random10'):
        ours = float(np.sqrt(np.mean((y - run.results[protocol].yhat) ** 2)))
        theirs = float(np.sqrt(np.mean((y - static_scored.results[protocol].yhat) ** 2)))
        out[f'{protocol}_rmse_change_vs_static'] = ours - theirs
    if m0star_scored is not None:
        out['vs_m0star_descriptive'] = paired_rmse_interval(y, run.results[PRIMARY].yhat,
                                                            m0star_scored.results[PRIMARY].yhat)
    return out


# ---------------------------------------------------------------------
# Equivalence and naming
# ---------------------------------------------------------------------


def _metric(card, dotted):
    value = card
    for part in dotted.split('.'):
        value = value[part]
    return value


def equivalence(runs, entries):
    report, passes = {}, True
    for family in FAMILIES:
        a, b = runs[('primary_total_co2', family)], runs[('per_capita', family)]
        if a.status != b.status:
            return {'passes': False, 'reason': f'{family}: computable in one representation only'}
        if a.status != 'computable':
            report[family] = {'status': a.status}
            continue
        predictions = {'in_sample_one_step': float(np.max(np.abs(a.quantities.one_step - b.quantities.one_step)))}
        for protocol in PROTOCOLS:
            predictions[protocol] = float(np.max(np.abs(a.results[protocol].yhat - b.results[protocol].yhat)))
        thetas = [abs(a.full.theta - b.full.theta)] + [abs(ra['fit'].theta - rb['fit'].theta)
                                                      for ra, rb in zip(a.records[PRIMARY], b.records[PRIMARY])]
        ea, eb = entries[('primary_total_co2', family)], entries[('per_capita', family)]
        scores = {m: abs(float(_metric(ea['card'], m)) - float(_metric(eb['card'], m))) for m in EQUIVALENT_SCORES}
        scores['delta_rmse'] = abs(ea['paired']['vs_static']['delta_rmse'] - eb['paired']['vs_static']['delta_rmse'])
        scores['interval'] = max(abs(x - z) for x, z in zip(ea['paired']['vs_static']['country_bootstrap_95_interval'],
                                                            eb['paired']['vs_static']['country_bootstrap_95_interval']))
        parts = {p: abs(ea['accounting']['estimands'][p] - eb['accounting']['estimands'][p]) for p in ACCOUNTING_PARTS}
        ok = (all(v <= PREDICTION_ATOL for v in predictions.values()) and max(thetas) <= THETA_ATOL
              and len(thetas) == 1 + len(a.records[PRIMARY])
              and all(v <= PREDICTION_ATOL for v in scores.values()) and all(v <= PREDICTION_ATOL for v in parts.values()))
        report[family] = {'predictions': predictions, 'theta_max': float(max(thetas)), 'scores': scores,
                          'accounting_parts': parts, 'passes': bool(ok)}
        passes &= ok
    report['tolerances'] = {'predictions': PREDICTION_ATOL, 'theta': THETA_ATOL, 'scores_and_accounting': PREDICTION_ATOL}
    report['allocations_may_differ'] = 'filtered-trend shares and static LMG shares differ by representation by construction'
    report['passes'] = bool(passes)
    return report


def final_naming(static, entries):
    qualifying = [extension_name(f, static) for f in FAMILIES
                  if entries[('primary_total_co2', f)].get('qualification', {}).get('qualifies') is True]
    if not qualifying:
        final, reason = static, 'no station-weight spatial extension qualifies; the retained static specification'
    elif len(qualifying) == 1:
        final, reason = qualifying[0], ('the unique spatial extension retained for prediction under the improvement '
                                        'rule (af97bd2 M3; DOWNSTREAM_COMPLETION_SPEC.md §2.14)')
    else:
        final, reason = None, 'two qualifying spatial extensions; no historical unique-family selection rule'
    return {'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying,
            'final_primary_predictive_model': final, 'reason': reason,
            'order_note': 'qualifying extensions are listed SEM then SAR; the order is not a ranking'}


# ---------------------------------------------------------------------
# Evidence tables
# ---------------------------------------------------------------------


def prediction_rows(runs, frozen, weights):
    rows, table = [], frozen.table
    for (representation, family), run in runs.items():
        if run.status != 'computable':
            continue
        y = frozen.frames[representation][OUTCOME_COL].to_numpy(float)
        base = {'weights': weights, 'family': family, 'representation': representation}
        for i in range(frozen.n):
            rows.append({**base, 'protocol': 'in_sample', 'iso3': table.iso3.iloc[i], 'm49_subregion': table.m49_subregion.iloc[i],
                         'observed': y[i], 'prediction': run.quantities.one_step[i],
                         'error': y[i] - run.quantities.one_step[i], 'trend': run.quantities.trend[i], 'fold_id': -1,
                         'n_train': frozen.n, 'nearest_train_km': np.nan, 'unseen_levels': 0, 'adjustment': 0.0})
        for protocol in PROTOCOLS:
            result = run.results[protocol]
            for i in range(frozen.n):
                rows.append({**base, 'protocol': protocol, 'iso3': table.iso3.iloc[i],
                             'm49_subregion': table.m49_subregion.iloc[i], 'observed': y[i], 'prediction': result.yhat[i],
                             'error': y[i] - result.yhat[i], 'trend': np.nan, 'fold_id': int(result.fold_id[i]),
                             'n_train': int(result.n_train[i]), 'nearest_train_km': float(result.nearest_train_km[i]),
                             'unseen_levels': int(result.unseen_levels[i]),
                             'adjustment': float(run.adjustments[protocol][i])})
    return pd.DataFrame(rows)


def fit_rows(runs, frozen, weights):
    rows, iso3 = [], frozen.table.iso3.to_numpy()
    for (representation, family), run in runs.items():
        fits = []
        if run.full is not None:
            fits.append(('full_sample', 'all', run.full, np.arange(frozen.n), None, None))
        for protocol, records in run.records.items():
            fits += [(protocol, r['label'], r['fit'], r['train'], r['encoder'], r['state']) for r in records]
        for protocol, label, fit, train, encoder, state in fits:
            rows.append({'weights': weights, 'family': family, 'representation': representation, 'protocol': protocol,
                         'fit': label, 'n': fit.n, 'p': fit.p, 'theta': fit.theta, 'loglik': fit.loglik,
                         'sigma2': fit.sigma2, 'at_domain_bound': fit.at_domain_bound,
                         'grid_local_maxima': fit.n_grid_local_maxima, 'evaluations': fit.n_evaluations,
                         'columns': '|'.join(fit.columns), 'beta': json.dumps([float(v) for v in fit.beta]),
                         'training_iso3': '|'.join(iso3[train]),
                         'levels': json.dumps(encoder.levels, sort_keys=True) if encoder is not None else '',
                         'latitude_state': state.to_json() if state is not None else ''})
    return pd.DataFrame(rows)


def graph_rows(runs, frozen, weights, dist):
    rows, iso3 = [], frozen.table.iso3.to_numpy()
    seen = set()
    for (representation, family), run in runs.items():
        if representation != 'primary_total_co2':
            continue
        fits = [('full_sample', 'all', np.arange(frozen.n), None)] if run.full is not None else []
        for protocol, records in run.records.items():
            fits += [(protocol, r['label'], r['train'], r) for r in records]
        for protocol, label, train, record in fits:
            if (protocol, label) in seen:
                continue
            seen.add((protocol, label))
            neighbours = ms.training_neighbours(dist, train) if record is None else record['neighbours']
            for node, nb in zip(train, neighbours):
                rows.append({'weights': weights, 'protocol': protocol, 'fit': label, 'role': 'training',
                             'iso3': iso3[node], 'neighbours': '|'.join(iso3[nb])})
            if record is not None and record.get('attachments') is not None:
                for node, nb in zip(record['test'], record['attachments']):
                    rows.append({'weights': weights, 'protocol': protocol, 'fit': label, 'role': 'held_out',
                                 'iso3': iso3[node], 'neighbours': '|'.join(iso3[nb])})
    return pd.DataFrame(rows)


def grid_rows(runs, weights):
    rows = []
    grid = ms.theta_grid()
    for (representation, family), run in runs.items():
        if run.full is None:
            continue
        for theta, value in zip(grid, run.full.grid_loglik):
            rows.append({'weights': weights, 'family': family, 'representation': representation,
                         'theta': float(theta), 'loglik': float(value)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


def arm_distance(frozen, weights):
    if weights == 'station':
        table = pd.read_csv(ROOT / ev.COUNTRY_TABLE, usecols=['iso3', 'station_lon', 'station_lat'],
                            float_precision='round_trip').set_index('iso3').loc[frozen.table.iso3]
        dist = haversine_matrix(table.station_lon.to_numpy(), table.station_lat.to_numpy())
        expected = frozen.identity['station_weights_sha256']
    else:
        geom = pd.read_csv(ROOT / ev.GEOMETRY, usecols=['iso3', 'centroid_lon', 'centroid_lat'],
                           float_precision='round_trip').set_index('iso3').loc[frozen.table.iso3]
        dist = haversine_matrix(geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy())
        expected = frozen.identity['area_weights_sha256']
    if prov.array_digest(knn_weights(dist, 8)) != expected:
        raise prov.ExecutionRefused(f'the {weights} kNN8 matrix differs from the recorded identity')
    return dist


def evaluate(frozen, out_dir, *, weights, static, provenance, primary=M2_PRIMARY):
    out_dir = prov.require_fresh(out_dir)
    static_model = STATIC_MODEL[static]
    models = [static_model] if static_model == ev.COMPARATOR else [ev.COMPARATOR, ev.CANDIDATE]
    reproduction = reproduce_static(frozen, models, primary)
    dist = arm_distance(frozen, weights)
    runs, entries, integrity = {}, {}, []
    for representation in ev.REPRESENTATIONS:
        frame = frozen.frames[representation]
        y = frame[OUTCOME_COL].to_numpy(float)
        static_scored = reproduction['scored'][(representation, static_model)]
        m0star = reproduction['scored'].get((representation, ev.COMPARATOR)) if static_model == ev.CANDIDATE else None
        for family in FAMILIES:
            run = run_family(family, static_model, frame, frozen, dist, representation)
            runs[(representation, family)] = run
            entry = {'status': run.status, 'failure': run.failure, 'name': extension_name(family, static),
                     'weights': weights}
            if run.full is not None and run.quantities is not None:
                try:
                    entry['accounting'] = accounting_block(run, frame, static_scored)
                except ms.IntegrityFailure as error:
                    integrity.append(f'{representation} {family}: {error}')
                entry['dependence_parameter_full_sample'] = {'theta': run.full.theta,
                                                             'at_domain_bound': run.full.at_domain_bound,
                                                             'se': run.interval.se, 'reason': run.interval.reason}
            if run.status == 'computable':
                entry['card'] = family_card(run, frame, frozen, static, frozen.station_w, frozen.area_w, dist)
                entry['paired'] = paired_block(y, run, static_scored, m0star)
                entry['dependence_parameter'] = dependence_summary(run)
                entry['graph_descriptors'] = graph_descriptors(run, dist, frozen.table)
                entry['diagnostics'] = diagnostics(entry['card'], static_scored.card, entry['paired']['vs_static'])
            try:
                conditions = qualification(entry['card'], static_scored.card, entry['paired']['vs_static'], True) \
                    if run.status == 'computable' else {'qualifies': False, 'Q5_computable': False,
                                                        'reason': 'non-computable family'}
            except ValueError as error:
                integrity.append(f'{representation} {family}: {error}')
                conditions = {'qualifies': None, 'refused': str(error)}
            if weights == 'station' and representation == 'primary_total_co2':
                entry['qualification'] = conditions
            else:
                entry['arm_conditions'] = {**conditions, 'descriptive_only': True,
                                           'note': 'cannot change the qualifying set or the final naming'}
            entries[(representation, family)] = entry
    equiv = equivalence(runs, entries)
    if not equiv['passes']:
        integrity.append('representation identity outside the prespecified tolerances')
    naming = final_naming(static, entries) if weights == 'station' and not integrity else None
    out_dir.mkdir(parents=True)
    result = {
        'specification': SPEC, 'evaluator_specification': EVALUATOR_SPEC, 'weights': weights,
        'branch': 'B' if static == 'M2' else 'A', 'retained_static_specification': static,
        'static_reproduction': reproduction['record'],
        'families': {f'{r}/{f}': e for (r, f), e in entries.items()},
        'representation_identity': equiv, 'final_naming': naming,
        'interpretation': 'a fall in residual Moran\'s I or a rise in in-sample fit from a spatial term accounts for '
                          'spatial covariance; it does not explain a mechanism',
        'integrity_failures': integrity, 'integrity_passed': not integrity,
    }
    prov.write_json(result, out_dir / 'm3_scorecard.json')
    prov.write_csv(prediction_rows(runs, frozen, weights), out_dir / 'm3_country_predictions.csv')
    prov.write_csv(fit_rows(runs, frozen, weights), out_dir / 'm3_fits.csv')
    prov.write_csv(graph_rows(runs, frozen, weights, dist), out_dir / 'm3_graphs.csv')
    prov.write_csv(grid_rows(runs, weights), out_dir / 'm3_full_sample_grid.csv')
    prov.write_json({**provenance, 'identity': frozen.identity, 'weights_distance_source': weights},
                    out_dir / 'm3_provenance.json')
    manifest = {'result': f'M3 spatial families, {weights}-centroid kNN8', 'specification': SPEC,
                'evaluator_commit': provenance.get('code', {}).get('commit'), 'weights': weights,
                'retained_static_specification': static,
                'family_status': {f'{r}/{f}': e['status'] for (r, f), e in entries.items()},
                'final_naming': naming, 'integrity_passed': not integrity,
                'artifact_sha256': {n: prov.sha256(out_dir / n) for n in DETERMINISTIC_ARTIFACTS},
                'non_deterministic_artifacts': [RUN_METADATA]}
    prov.write_json(manifest, out_dir / RESULT_MANIFEST)
    result['result_manifest'] = manifest
    return result


def reproducibility_record(run_a, run_b, *, code=None):
    a, b = Path(run_a), Path(run_b)
    artifacts, identical = {}, True
    for name in (*DETERMINISTIC_ARTIFACTS, RESULT_MANIFEST):
        da, db = prov.sha256(a / name), prov.sha256(b / name)
        artifacts[name] = {'run_a_sha256': da, 'run_b_sha256': db, 'identical': da == db}
        identical &= da == db
    return {'comparison': 'unchanged-code rerun in a separate process and output root', 'run_a': str(a),
            'run_b': str(b), 'artifacts': artifacts, 'excluded_as_run_metadata': [RUN_METADATA],
            'byte_identical': bool(identical), 'code': code}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score the M3 spatial families under the pre-specified protocol.')
    parser.add_argument('--weights', choices=('station', 'land'), required=True)
    parser.add_argument('--out')
    parser.add_argument('--compare', nargs=2, metavar=('RUN_A', 'RUN_B'))
    parser.add_argument('--record')
    args = parser.parse_args(argv)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    if args.compare:
        record = reproducibility_record(*args.compare, code=prov.frozen_code_state(code_files()))
        if args.record:
            prov.write_json(record, args.record)
        print(prov.json_text({'byte_identical': record['byte_identical']}))
        return record
    prov.require_hash_seed()
    code = prov.frozen_code_state(code_files())
    out_dir = prov.require_fresh(args.out or OUT[args.weights])
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    m2 = verified_m2()
    station = verified_station() if args.weights == 'land' else None
    digests = ev.verify_pins()
    frozen = ev.build_frozen()
    provenance = {'documents_sha256': {d: prov.sha256(ROOT / d) for d in DOCUMENTS},
                  'code': {k: v for k, v in code.items() if k != 'live_remote_commit'},
                  'inputs': digests, 'm2_result_manifest_sha256': m2['manifest_sha256'],
                  'm2_conditional_disposition': m2['conditional_disposition'],
                  'station_package_manifest_sha256': station['manifest_sha256'] if station else None,
                  'estimator_version': ms.ESTIMATOR_VERSION, 'software': ev.software()}
    result = evaluate(frozen, out_dir, weights=args.weights, static=m2['retained_static_specification'],
                      provenance=provenance)
    prov.write_json({'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     'wall_seconds': time.perf_counter() - started, 'host': platform.node(),
                     'executable': sys.executable, 'argv': sys.argv, 'output_root': str(out_dir),
                     'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'live_remote_commit': code['live_remote_commit']},
                    out_dir / RUN_METADATA)
    print(prov.json_text({'integrity_passed': result['integrity_passed'], 'final_naming': result['final_naming'],
                          'families': {k: v['status'] for k, v in result['families'].items()}}))
    return result


if __name__ == '__main__':
    sys.exit(ev.cli_exit_code(main()))
