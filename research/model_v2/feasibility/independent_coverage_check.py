"""Independent brute-force coverage for the countries that failed the original M1a gates.

No recursion and no block fast path: every 0.25 deg cell is intersected directly."""
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from research.model_v2 import m1a_geography as g
qa = pd.read_csv(g.m0.ROOT / 'research/model_v2/outputs/m1a_feasibility_original_gates/m1a_geography_qa.csv').set_index('iso3')
codes = pd.read_csv(g.OUTPUT_DIR_TABLE).iso3.tolist()
coc, _, _ = g.gpw_country_grid(codes)
levels, _ = g.load_gshhg()
tree = {L: shapely.STRtree(levels[L]) for L in levels}
with xr.open_dataset(g.KOPPEN_PATH) as ds:
    kg = g.koppen_group_index(ds.koppen_code.to_numpy())[::-1]
area_rows = g.pixel_area_rows(1)
print(qa.loc['JAM', ['continentality_3min', 'continentality_5min', 'continentality_60s_check', 'continentality_diff_3min_vs_5min_km']])
for code in ['BHS', 'DNK']:
    i = codes.index(code)
    rows, cols = np.nonzero(coc == i)
    tot = valid = uncls = 0.0
    for r, c in zip(rows, cols):
        box = shapely.box(-180 + c / 4, -90 + r / 4, -180 + (c + 1) / 4, -90 + (r + 1) / 4)
        parts = {L: shapely.intersection(levels[L][tree[L].query(box)], box) for L in levels}
        land = shapely.union(shapely.difference(shapely.union_all(parts[1]), shapely.union_all(parts[2])),
                             shapely.difference(shapely.union_all(parts[3]), shapely.union_all(parts[4])))
        if land.is_empty:
            continue
        pr, pc = np.meshgrid(np.arange(r * 15, r * 15 + 15), np.arange(c * 15, c * 15 + 15), indexing='ij')
        pr, pc = pr.ravel(), pc.ravel()
        x0, y0 = -180 + pc / 60, -90 + pr / 60
        a = shapely.area(shapely.intersection(shapely.box(x0, y0, x0 + 1 / 60, y0 + 1 / 60), land)) * 3600 * area_rows[pr]
        centre = shapely.intersects_xy(land, x0 + 1 / 120, y0 + 1 / 120)
        tot += a.sum()
        valid += a[centre].sum()
        uncls += a[kg[pr // 30, pc // 30] == 5].sum()
    print(code, 'independent area', round(tot, 3), 'build', round(qa.loc[code, 'terrestrial_area_km2'], 3),
          '| elevation coverage', round(valid / tot, 6), 'build', round(qa.loc[code, 'elevation_coverage'], 6),
          '| koppen unclassified', round(uncls / tot, 6), 'build', round(qa.loc[code, 'koppen_unclassified_share'], 6))
