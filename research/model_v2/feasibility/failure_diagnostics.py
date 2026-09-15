"""M1a feasibility audit under the ORIGINAL gates (builder sha256 920c0577…, commit recording this audit).

Read-only per-pixel diagnostics for the four gate-failing countries (no outputs written to the repo)."""
import json
import numpy as np
import pandas as pd
import xarray as xr
from research.model_v2 import m1a_geography as g
FAIL = ['PHL', 'DNK', 'NOR', 'BHS']
codes = pd.read_csv(g.OUTPUT_DIR_TABLE).iso3.tolist()
coc, _, _ = g.gpw_country_grid(codes)
levels, _ = g.load_gshhg()
with xr.open_dataset(g.KOPPEN_PATH) as ds:
    kraw = ds.koppen_code.to_numpy()[::-1]
kg = g.koppen_group_index(kraw)
area_rows = g.pixel_area_rows(1)
out = {}
for code in FAIL:
    i = codes.index(code)
    sub = np.where(coc == i, 0, -1).astype(np.int16)
    rec = g.Records()
    g.walk_globe(levels, sub, lambda k, a, b, c, d, geom: g.emit_into(rec, k, a, b, c, d, geom), type('C', (), {'stats': {}})())
    r, c = np.concatenate(rec.r), np.concatenate(rec.c)
    frac, centre = np.concatenate(rec.frac), np.concatenate(rec.centre_land)
    keep = sub[r // 15, c // 15] >= 0
    r, c, frac, centre = r[keep], c[keep], frac[keep], centre[keep]
    area = frac * area_rows[r]
    rmin, rmax, cmin, cmax = r.min(), r.max(), c.min(), c.max()
    with xr.open_dataset(g.ETOPO_PATH) as ds:
        z = ds.z[rmin:rmax + 1, cmin:cmax + 1].to_numpy()[r - rmin, c - cmin].astype(float)
    miss = ~centre
    full = frac >= 1
    partial_valid = centre & ~full
    d = {'total_pixels': int(len(r)), 'total_area': float(area.sum()),
         'valid_pixels': int(centre.sum()), 'valid_area': float(area[centre].sum()),
         'missing_pixels': int(miss.sum()), 'missing_area': float(area[miss].sum()),
         'missing_pixel_mean_land_fraction': float(frac[miss].mean()),
         'valid_partial_pixels': int(partial_valid.sum()), 'valid_full_pixels': int((centre & full).sum())}
    # concentration of missing area over 1° cells
    key = (r // 60) * 360 + c // 60
    m_by_cell = pd.Series(area[miss]).groupby(key[miss]).sum().sort_values(ascending=False)
    t_by_cell = pd.Series(area).groupby(key).sum()
    d['one_degree_cells_with_land'] = int(len(t_by_cell))
    d['one_degree_cells_with_missing'] = int(len(m_by_cell))
    d['missing_share_top5_1deg_cells'] = float(m_by_cell.head(5).sum() / m_by_cell.sum())
    d['missing_share_top10pct_1deg_cells'] = float(m_by_cell.head(max(1, len(m_by_cell) // 10)).sum() / m_by_cell.sum())
    d['missing_fraction_by_1deg_cell_p50_p90_max'] = [float(x) for x in np.quantile((m_by_cell / t_by_cell.loc[m_by_cell.index]).to_numpy(), [.5, .9, 1])]
    lat_c = -90 + (r + .5) / 60
    d['missing_lat_range'] = [float(lat_c[miss].min()), float(lat_c[miss].max())]
    # elevation: how the missing support differs
    w = area[centre]
    mean_valid = float(np.sum(w * z[centre]) / w.sum())
    coastal = float(np.sum(area[partial_valid] * z[partial_valid]) / area[partial_valid].sum())
    interior = float(np.sum(area[centre & full] * z[centre & full]) / area[centre & full].sum())
    f = d['missing_area'] / d['total_area']
    d.update(elev_mean_valid=mean_valid, elev_mean_valid_coastal_partial_pixels=coastal, elev_mean_valid_full_pixels=interior,
             etopo_z_at_missing_pixel_centres_median=float(np.median(z[miss])), etopo_z_at_missing_share_below_0=float(np.mean(z[miss] < 0)),
             elev_if_missing_equal_coastal_valid=(1 - f) * mean_valid + f * coastal,
             elev_if_missing_at_0m=(1 - f) * mean_valid)
    # Köppen composition
    kk = kg[r // 30, c // 30]
    d['koppen_area_by_group'] = {('ABCDE' + 'U')[j]: float(area[kk == j].sum()) for j in range(6)}
    uncl = kk == 5
    d['koppen_05deg_pixels_total'] = int(len(np.unique((r // 30) * 720 + c // 30)))
    d['koppen_05deg_pixels_unclassified'] = int(len(np.unique(((r // 30) * 720 + c // 30)[uncl])))
    if uncl.any():
        # nearest classified group among the 8 neighbouring 0.5° pixels, for context only
        kr, kc = r[uncl] // 30, c[uncl] // 30
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neigh.append(kg[np.clip(kr + dr, 0, 359), (kc + dc) % 720])
        neigh = np.stack(neigh, 1)
        groups = [np.bincount(row[row < 5], minlength=5) for row in neigh]
        best = np.array([np.argmax(gc) if gc.sum() else 5 for gc in groups])
        d['unclassified_area_by_neighbour_majority_group'] = {('ABCDE' + 'none')[j] if j < 5 else 'no classified neighbour': float(area[uncl][best == j].sum()) for j in range(6)}
        by = pd.Series(area[uncl]).groupby(((r // 30) * 720 + c // 30)[uncl]).sum().sort_values(ascending=False)
        d['unclassified_share_top5_05deg_pixels'] = float(by.head(5).sum() / by.sum())
    out[code] = d
print(json.dumps(out, indent=1))
