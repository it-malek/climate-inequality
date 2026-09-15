"""Generate the original-gate feasibility report tables (see M1A_FEASIBILITY_AUDIT.md)."""
from pathlib import Path

import pandas as pd
AUDIT = 'research/model_v2/outputs/m1a_feasibility_original_gates/'
qa = pd.read_csv(AUDIT + 'm1a_geography_qa.csv')
rep = pd.read_csv(AUDIT + 'm1a_gate_coverage_report.csv')
wide=rep.pivot(index='iso3', columns='feature', values='achieved').loc[qa.iso3]
passes=rep.pivot(index='iso3', columns='feature', values='passes').loc[qa.iso3]
qa=qa.set_index('iso3')
L=['# M1a measurement gate report — scoring blocked', '',
   '**Status: pre-result. No M1a model has been fitted or scored.** Frozen gates (M1A_MEASUREMENT_SPEC.md §3.3) are unchanged.',
   'Generated from `outputs/m1a_geography_qa.csv` and `outputs/m1a_gate_coverage_report.csv` (build 1; build 2 identical).', '',
   f"Countries passing all six gates: **{int(passes.all(axis=1).sum())} / 151**. Failed country-feature gates: **{int((~passes).sum().sum())}**.", '',
   '## Complete 151 × 6 coverage table', '',
   'Coverage = valid terrestrial support / terrestrial support. Required: latitude and hemisphere 1.00 (geometry); elevation and continentality ≥ 0.98; climate ≥ 0.95 classified **and** winner margin > unclassified area; spatial block nonmissing. ✗ marks a failed gate.', '',
   '| ISO3 | Country | Terrestrial km² | abs_latitude | hemisphere | elevation | continentality | climate_zone (classified) | Köppen margin/unclassified | spatial_block |',
   '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for iso in wide.index:
    q=qa.loc[iso]
    w=wide.loc[iso]
    p=passes.loc[iso]
    ratio = 'no unclassified' if q.koppen_unclassified_km2==0 else f'{q.koppen_winner_margin_km2/q.koppen_unclassified_km2:.1f}'
    def cell(f, w=w, p=p):
        return '{:.4f}'.format(w[f]) + ('' if p[f] else ' ✗')
    L.append(f"| {iso} | {q.Country} | {q.terrestrial_area_km2:,.0f} | {cell('abs_latitude')} | {cell('hemisphere')} | {cell('elevation')} | {cell('continentality')} | {cell('climate_zone')} | {ratio} | {cell('spatial_block')} |")
L += ['', '## Near-threshold passes', '']
near_e=rep[(rep.feature=='elevation')&rep.passes&(rep.achieved<0.985)].sort_values('achieved')
near_k=rep[(rep.feature=='climate_zone')&rep.passes&(rep.achieved<0.96)]
marg=qa[(qa.koppen_unclassified_km2>0)&(qa.koppen_winner_margin_km2<3*qa.koppen_unclassified_km2)]
L.append('Elevation coverage in [0.980, 0.985): ' + ', '.join(f'{r.iso3} {r.achieved:.4f}' for r in near_e.itertuples()))
L.append('')
L.append('Classified Köppen coverage in [0.95, 0.96): ' + (', '.join(f'{r.iso3} {r.achieved:.4f}' for r in near_k.itertuples()) or 'none'))
L.append('')
L.append('Köppen winner margin < 3 × unclassified area (passes, but least robust): ' + (', '.join(f'{i} margin {r.koppen_winner_margin_km2:,.0f} km² vs unclassified {r.koppen_unclassified_km2:,.0f} km²' for i, r in marg.iterrows()) or 'none'))
Path('research/model_v2/feasibility/_generated_gate_tables.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L[:8]))
