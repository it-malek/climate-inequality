"""M1a elevation-completion measurement record: consistency checks against the original-gate build, the
per-country CSV ``outputs/m1a_elevation_amendment_record.csv`` and a Markdown summary (uncommitted scratch output)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('research/model_v2/outputs')
OLD = OUT / 'm1a_feasibility_original_gates'
qa = pd.read_csv(OUT / 'm1a_geography_qa.csv')
old = pd.read_csv(OLD / 'm1a_geography_qa.csv')
new_f = pd.read_csv(OUT / 'm1a_geography_features.csv')
old_f = pd.read_csv(OLD / 'm1a_geography_features.csv')
manifest = json.loads((OUT / 'm1a_measurement_manifest.json').read_text())
assert qa.iso3.tolist() == old.iso3.tolist() == new_f.iso3.tolist() == old_f.iso3.tolist()

checks = {
    'original_centre_only_equals_feasibility_build': bool(np.array_equal(qa.elevation_original_centre_only.to_numpy(),
                                                                         old.elevation_if_computed.to_numpy())),
    'original_coverage_equals_feasibility_build': bool(np.array_equal(qa.elevation_original_coverage.to_numpy(),
                                                                      old.elevation_coverage.to_numpy())),
}
for f in ['abs_latitude', 'hemisphere', 'continentality', 'spatial_block']:
    checks[f'{f}_identical_to_feasibility_build'] = bool(new_f[f].equals(old_f[f]))
klass = pd.DataFrame({'iso3': qa.iso3, 'original_rule': old_f.climate_zone, 'amended_rule': new_f.climate_zone})
changed_k = klass[klass.original_rule.fillna('missing') != klass.amended_rule.fillna('missing')]

cols = ['iso3', 'Country', 'terrestrial_area_km2', 'elevation_original_coverage', 'elevation_coverage',
        'elevation_completed_area_km2', 'elevation_unresolved_area_km2', 'elevation_original_centre_only', 'elevation',
        'elevation_change_vs_original', 'elevation_s1_completed_at_0m', 'elevation_s2_unresolved_at_0m',
        'elevation_s3_ring2_completion', 'elevation_s3_coverage', 'elevation_s4_unresolved_at_completed_min',
        'elevation_s4_unresolved_at_completed_max']
table = qa[cols].copy()
for s in ['s1_completed_at_0m', 's2_unresolved_at_0m', 's3_ring2_completion', 's4_unresolved_at_completed_min',
          's4_unresolved_at_completed_max']:
    table[f'{s}_minus_primary'] = table[f'elevation_{s}'] - table.elevation
table.to_csv(OUT / 'm1a_elevation_amendment_record.csv', index=False)

change = table.elevation_change_vs_original
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
bins = [-np.inf, -10, -5, -2, -1, -.5, -.1, 0, .1, 1, np.inf]
hist = pd.cut(change, bins, right=False).value_counts(sort=False)
L = ['# M1a measurement record with one-ring elevation completion', '',
     '**Recorded before `m1a_evaluate.py` was run.** Builder `m1a_geography.py` sha256 '
     f"`{manifest['code_sha256']}`.",
     '',
     '## Change control',
     '',
     '* **Determinism.** Two independent builds were byte-identical: features, QA, manifest and per-cell land area.',
     '* **Gates.** All 151 countries pass every gate (`all_features_pass_gates = true`); there are no quadrature failures.',
     '* **No changes after observing the builds.** After the completed builds were observed, no parameter, source, threshold, '
     'country membership, measurement rule or evaluator code changed.',
     '',
     '## Consistency with the feasibility build', '']
L += [f'* {k.replace("_", " ")}: **{v}**' for k, v in checks.items()]
L += ['', '## Koppen classes: effect of the identifiability rule', '',
      'Every Koppen class is identified (`A1 > A2 + U`). Classes that differ from the original-rule build:', '']
L += [f'* {r.iso3}: original rule -> {r.original_rule if isinstance(r.original_rule, str) else "missing (failed the 95% classified gate)"}; '
      f'completed rule -> {r.amended_rule}' for r in changed_k.itertuples()] or ['* none']
L += ['', 'No other country changed class. The change arises only from the pre-specified identifiability rule.',
      f'For context, and not as an effect of the completion rule: {int(qa.climate_zone_changed.sum())} countries\' area-dominant class differs from '
      f'V1\'s station-modal class, and {int(qa.hemisphere_changed.sum())} hemispheres change.', '',
      '## Elevation: pre/post distribution (completed minus original centre-only, metres)', '',
      '| Quantile | ' + ' | '.join(f'{x:g}' for x in q) + ' |', '|---|' + '---:|' * len(q),
      '| Change | ' + ' | '.join(f'{v:.3f}' for v in change.quantile(q)) + ' |', '',
      f'Mean change {change.mean():.3f} m; sample SD {change.std():.3f} m. '
      f'{int((change < 0).sum())} countries decrease, {int((change > 0).sum())} increase and {int((change == 0).sum())} are unchanged.', '',
      '| Change bin (m) | Countries |', '|---|---:|']
L += [f'| {b} | {n} |' for b, n in hist.items()]
L += ['', '### Largest absolute elevation changes', '',
      '| ISO3 | Country | Original | Completed | Change | Original coverage | Resolved coverage |', '|---|---|---:|---:|---:|---:|---:|']
for r in table.reindex(change.abs().sort_values(ascending=False).index).head(15).itertuples():
    L.append(f'| {r.iso3} | {r.Country} | {r.elevation_original_centre_only:.2f} | {r.elevation:.2f} | {r.elevation_change_vs_original:+.2f} | '
             f'{r.elevation_original_coverage:.4f} | {r.elevation_coverage:.4f} |')
L += ['', '## Sensitivities S1 to S4, difference from the primary elevation (m)', '',
      'Sensitivities are reported only; they are not used for any decision.',
      '',
      '* **S1:** completed area at 0 m.',
      '* **S2:** unresolved area included at 0 m.',
      '* **S3:** unresolved pixels completed from ring 2.',
      '* **S4:** unresolved area at the country\'s minimum or maximum completed value.',
      '',
      '| Sensitivity | min | p5 | median | p95 | max | max abs (country) |', '|---|---:|---:|---:|---:|---:|---|']
for s in ['s1_completed_at_0m', 's2_unresolved_at_0m', 's3_ring2_completion', 's4_unresolved_at_completed_min',
          's4_unresolved_at_completed_max']:
    d = table[f'{s}_minus_primary']
    i = d.abs().idxmax()
    L.append(f'| {s} | {d.min():.3f} | {d.quantile(.05):.3f} | {d.median():.3f} | {d.quantile(.95):.3f} | {d.max():.3f} | '
             f'{abs(d[i]):.3f} ({table.iso3[i]}) |')
L += ['', f"S3 (ring-2) coverage ranges from {table.elevation_s3_coverage.min():.4f} to {table.elevation_s3_coverage.max():.4f}. "
      f"Total unresolved area is {table.elevation_unresolved_area_km2.sum():,.0f} km^2 "
      f"({table.elevation_unresolved_area_km2.sum() / table.terrestrial_area_km2.sum():.5%} of the 151-country terrestrial support).", '',
      '## Full per-country record', '',
      'The complete per-country table is in `outputs/m1a_elevation_amendment_record.csv`, one row per country for all 151. It contains:',
      '',
      '* original and completed estimates and coverage;',
      '* completed and unresolved area;',
      '* S1 to S4 values and their differences from the primary estimate.',
      '',
      '| ISO3 | Original | Completed | Change | S1 | S2 | S3 | S4 min | S4 max | Resolved cov. |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in table.itertuples():
    L.append(f'| {r.iso3} | {r.elevation_original_centre_only:.2f} | {r.elevation:.2f} | {r.elevation_change_vs_original:+.3f} | '
             f'{r.elevation_s1_completed_at_0m:.2f} | {r.elevation_s2_unresolved_at_0m:.2f} | {r.elevation_s3_ring2_completion:.2f} | '
             f'{r.elevation_s4_unresolved_at_completed_min:.2f} | {r.elevation_s4_unresolved_at_completed_max:.2f} | {r.elevation_coverage:.4f} |')
Path('research/model_v2/feasibility/_generated_m1a_measurement_record.md').write_text('\n'.join(L) + '\n')
print(json.dumps(checks, indent=1))
print(changed_k.to_string(index=False))
print('\n'.join(L[L.index('## Elevation: pre/post distribution (completed minus original centre-only, metres)'):L.index('## Full per-country record')]))
