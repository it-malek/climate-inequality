"""Determinism comparison of two M1a builds and a gate-failure report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

OUT, REP = Path(sys.argv[1]), Path(sys.argv[2])  # two independent build output directories

for name in ['m1a_geography_features.csv', 'm1a_geography_qa.csv']:
    same = (OUT / name).read_bytes() == (REP / name).read_bytes()
    print(name, 'byte-identical' if same else 'DIFFERS')
a, b = np.load(OUT / 'm1a_cell_land_area_km2.npz'), np.load(REP / 'm1a_cell_land_area_km2.npz')
print('cell land area arrays equal:', np.array_equal(a['cell_land_area_km2'], b['cell_land_area_km2']))
m1, m2 = (json.loads((d / 'm1a_measurement_manifest.json').read_text()) for d in (OUT, REP))
print('manifest identical:', m1 == m2)

qa = pd.read_csv(OUT / 'm1a_geography_qa.csv')
pd.set_option('display.width', 250)
print('\nall pass:', m1['all_features_pass_gates'], '| quadrature failures:', m1['quadrature_failures'])
print('walk stats', m1['walk_stats'], '| coast', m1['coast'])
fail = qa[qa.missing_reasons != '{}']
print(f'\n{len(fail)} countries with gate failures')
cols = ['iso3', 'Country', 'terrestrial_area_km2', 'gpw_footprint_area_km2', 'elevation_coverage',
        'elevation_missing_fraction', 'koppen_unclassified_share', 'koppen_winner_margin_km2',
        'koppen_unclassified_km2', 'mixed_pixel_area_km2', 'missing_reasons']
print(fail[cols].round(4).to_string(index=False))
print('\nnear-threshold passes (elevation coverage < 0.985 or classified < 0.96):')
near = qa[(qa.missing_reasons == '{}') & ((qa.elevation_coverage < .985) | (qa.koppen_unclassified_share > .04))]
print(near[cols[:-1]].round(4).to_string(index=False))
print('\nquadrature', qa.continentality_quadrature.value_counts().to_dict(),
      'max 3min-5min', qa.continentality_diff_3min_vs_5min_km.max(),
      'max 3min-60s', qa.continentality_diff_3min_vs_60s_km.max())
print(qa[qa.continentality_60s_pass.notna()][['iso3', 'continentality_60s_pass', 'continentality_3min',
      'continentality_5min', 'continentality_60s_check', 'continentality_quadrature']].round(4).to_string(index=False))
