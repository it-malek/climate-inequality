"""Machine-readable final Model V2 record, assembled only from committed result manifests and records.

It computes nothing scientific: it copies identities, labels and dispositions from the committed M1a, M1b, M2,
M3 and M4 evidence, binds each source by SHA-256 and records the commits that introduced each result. Writes
``outputs/v2_final/v2_final_record.json``:

    uv run python -m research.model_v2.v2_final_record
"""
from __future__ import annotations

import json
import subprocess
import sys

from research.model_v2 import v2_provenance as prov

ROOT = prov.ROOT
OUT = 'research/model_v2/outputs/v2_final/v2_final_record.json'
OUTPUTS = 'research/model_v2/outputs'
SOURCES = {
    'm0_corrected_scorecard': f'{OUTPUTS}/m0_scorecard_territory_corrected.json',
    'rank_audit': f'{OUTPUTS}/rank_audit.json',
    'm1a_scorecard': f'{OUTPUTS}/m1a_scorecard.json',
    'm1b_result_manifest': f'{OUTPUTS}/m1b_primary/m1b_result_manifest.json',
    'm2_result_manifest': f'{OUTPUTS}/m2_primary/m2_result_manifest.json',
    'm2_conditional_not_run': f'{OUTPUTS}/m2_conditional/m2_conditional_not_run.json',
    'm3_station_manifest': f'{OUTPUTS}/m3_station/m3_result_manifest.json',
    'm3_land_centroid_manifest': f'{OUTPUTS}/m3_land_centroid/m3_result_manifest.json',
    'm4_result_manifest': f'{OUTPUTS}/m4_final/m4_result_manifest.json',
    'm4_final_specification': f'{OUTPUTS}/m4_final/m4_final_specification.json',
    'm4_bootstrap_summary': f'{OUTPUTS}/m4_final/m4_bootstrap_summary.json',
    'm4_products': f'{OUTPUTS}/m4_final/m4_products.json',
    'm4_sensitivities': f'{OUTPUTS}/m4_final/m4_sensitivities.json',
    'm3_station_scorecard': f'{OUTPUTS}/m3_station/m3_scorecard.json',
    'm3_land_centroid_scorecard': f'{OUTPUTS}/m3_land_centroid/m3_scorecard.json',
    'm2_scorecard': f'{OUTPUTS}/m2_primary/m2_scorecard.json',
    'downstream_specification': 'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md',
    'm4_evidence_inventory': 'research/model_v2/M4_EVIDENCE_INVENTORY.md',
    'm2_pre_score_resolutions': 'research/model_v2/M2_PRE_SCORE_RESOLUTIONS.md',
}
VERIFICATION = {
    'm2_independent_check': f'{OUTPUTS}/m2_primary_verification/m2_independent_check.json',
    'm2_reproducibility': f'{OUTPUTS}/m2_primary_verification/m2_reproducibility.json',
    'm3_station_independent_check': f'{OUTPUTS}/m3_station_verification/m3_independent_check.json',
    'm3_station_reproducibility': f'{OUTPUTS}/m3_station_verification/m3_reproducibility.json',
    'm3_land_independent_check': f'{OUTPUTS}/m3_land_centroid_verification/m3_independent_check.json',
    'm3_land_reproducibility': f'{OUTPUTS}/m3_land_centroid_verification/m3_reproducibility.json',
    'm4_independent_check': f'{OUTPUTS}/m4_final_verification/m4_independent_check.json',
    'm4_reproducibility': f'{OUTPUTS}/m4_final_verification/m4_reproducibility.json',
}


def load(rel):
    return json.loads((ROOT / rel).read_text())


def introduced(rel):
    done = subprocess.run(['git', 'log', '--format=%H', '--diff-filter=A', '--', rel], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    commits = done.stdout.split()
    return commits[-1] if commits else None


def record() -> dict:
    final = load(SOURCES['m4_final_specification'])
    m2 = load(SOURCES['m2_scorecard'])
    station = load(SOURCES['m3_station_scorecard'])
    land = load(SOURCES['m3_land_centroid_scorecard'])
    products = load(SOURCES['m4_products'])
    bootstrap = load(SOURCES['m4_bootstrap_summary'])
    checks = {}
    for name, rel in VERIFICATION.items():
        payload = load(rel)
        passed = payload.get('all_checks_pass', payload.get('byte_identical'))
        controls = payload.get('negative_controls_all_detected')
        checks[name] = {'path': rel, 'sha256': prov.sha256(ROOT / rel), 'passes': passed,
                        'negative_controls_all_detected': controls}

    def family(scorecard, fam, deciding):
        entry = scorecard['families'][f'primary_total_co2/{fam}']
        conditions = entry['qualification'] if deciding else entry['arm_conditions']
        return {'status': entry['status'], 'qualifies': conditions['qualifies'], 'descriptive_only': not deciding,
                'delta_rmse_vs_static': conditions['delta_rmse'], 'interval': conditions['interval'],
                'm49_rmse_change': conditions['m49_rmse_change'],
                'worst_region_rmse_change': conditions['worst_region_rmse_change'],
                'theta_full_sample': entry['dependence_parameter']['full_sample']['theta'],
                'accounting': {k: entry['accounting']['estimands'][k]
                               for k in ('a_static', 'a_dependence', 'a_innovation')}}

    return {
        'record': 'Model V2 final specification record (research line; V1 v1.3.0 unchanged)',
        'retained_static_specification': final['retained_static_specification'],
        'qualifying_spatial_extensions': final['qualifying_spatial_extensions'],
        'final_primary_predictive_model': final['final_primary_predictive_model'],
        'final_naming_reason': final['final_naming_reason'],
        'stages': {
            'M0': 'frozen V1 baseline; corrected 500 km primary CV row',
            'M0*': 'approved full-rank development baseline (drop per-capita); retained static specification',
            'M1a': 'not promoted; registered area-consistent geography measurement sensitivity',
            'M1b': load(SOURCES['m1b_result_manifest'])['verdict'],
            'M2': m2['verdict']['verdict'],
            'M2_conditional_arms': 'not executed (no predictive support)',
            'M3_station': {fam.upper(): family(station, fam, True) for fam in ('sem', 'sar')},
            'M3_land_centroid_sensitivity': {fam.upper(): family(land, fam, False) for fam in ('sem', 'sar')},
        },
        'static_stopping_indicator': m2['static_stopping_indicator'],
        'm2_delta_rmse_and_interval': [m2['verdict']['delta_rmse'], m2['verdict']['interval']],
        'm2_gamma': m2['representations']['primary_total_co2']['M2']['gamma'],
        'static_share_uncertainty_total_co2': {
            kind: {g: [v['point'], v['ci_low'], v['ci_high']] for g, v in bootstrap['primary_total_co2'][kind]['groups'].items()}
            for kind in ('country', 'block')},
        'material_change_to_v1_conclusion': final['material_change_to_v1_conclusion'],
        'product_sensitive_spatial_extensions': {k.split('/', 1)[1]: v for k, v in products.items()
                                                 if k.startswith('product_sensitive/')},
        'not_pursued_not_rejected': ['optional geostatistical residual model', 'C1 and C3 physical covariates',
                                     'alternative latitude forms (H1-only, spline-only, NH-only, other knots/df)',
                                     'two-product-mean outcome', 'H8 decadal-variability, H9 unit-of-analysis and H10 '
                                     'station-density diagnostics', 'spatial terms on rejected candidates'],
        'commits': {**final['commits'],
                    'm4_result': introduced(f'{OUTPUTS}/m4_final/m4_result_manifest.json'),
                    'downstream_specification_amendment_A1': 'ef04958'},
        'sources': {name: {'path': rel, 'sha256': prov.sha256(ROOT / rel), 'introduced_by': introduced(rel)}
                    for name, rel in SOURCES.items()},
        'verification': checks,
        'interpretation_limits': [
            'descriptive and associational; geography is not causal attribution',
            'responsibility allocation is not greenhouse-gas forcing\'s physical contribution',
            'residual variance is not an unknown climate cause',
            'spatial covariance accounting is not mechanism identification',
            'rejected operational models do not disprove their motivating mechanisms',
            'same-dataset hypothesis generation and follow-up are not independent confirmation',
            'product disagreement is not an identified decomposition of observational error',
        ],
    }


def main():
    payload = record()
    path = ROOT / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    prov.write_json(payload, path)
    print(prov.json_text({k: payload[k] for k in ('retained_static_specification', 'qualifying_spatial_extensions',
                                                  'final_primary_predictive_model')}))
    return payload


if __name__ == '__main__':
    main()
    sys.exit(0)
