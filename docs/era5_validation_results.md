# ERA5 cross-check of the area-weighting result

Switching each country's warming from the mean over its stations to a
cos-latitude area-weighted mean over the Berkeley Earth 1° grid collapses the
warming–responsibility rank correlation from ρ = +0.36 to +0.01. Because that
rests on one gridded product, it was re-run on ERA5, ECMWF's model-assimilated
reanalysis, which has no station-sampling gaps and is an independent estimate
of the same field. The collapse reproduces: the station coupling is not
significant under either product after area weighting, and the rise in the
inequality coefficient reproduces almost exactly.

## Method

The ERA5 lens uses the same per-cell Theil–Sen estimator, the same 1950-01 to
2013-09 window, the same GPW band-11 country assignment and the same
cos(latitude) weighting as the Berkeley area lens (`src.era5_weighting` reuses
`src.area_weighting.cell_trends`). Only the temperature grid changes. ERA5's
absolute Kelvin values need no anomaly conversion (a slope is offset-invariant);
its CF time axis is snapped to first-of-month, and its 0–360° longitudes are
normalised only when sampling the country grid. The comparison uses the
coupling stage's Spearman ρ and inequality coefficient directly. It is a
cross-check artifact, not a registered projection.

## Coupling with responsibility on the common 153-country set

Every lens is scored on the identical set of 153 countries (those for which all
three lenses are defined), so the comparison is not a coverage artefact.

| Lens | Spearman ρ | p-value | Significant? | Gini |
|---|---:|---:|:--:|---:|
| Station-weighted        | **+0.364** | < 0.0001 | **yes** | 0.571 |
| Area-weighted (Berkeley)| +0.011 | 0.892 | no | 0.606 |
| Area-weighted (**ERA5**)| +0.118 | 0.148 | **no** | 0.603 |

- The significant station coupling (ρ = +0.364) drops to +0.118 under ERA5
  area weighting and is no longer significant (p = 0.15); Berkeley collapses
  further (ρ = +0.011, p = 0.89).
- The rise in the inequality coefficient reproduces: 0.571 (station) → 0.606
  (Berkeley) / 0.603 (ERA5).

## World-land sanity check

| Product | cos(lat) world-land mean (°C/decade) |
|---|---:|
| Berkeley Earth (this pipeline) | 0.193 |
| **ERA5** (this pipeline) | **0.222** |
| Berkeley Earth documented global-land reference | ~0.19 |

Both are in the same 0.2 °C/decade range, which validates the ERA5 ingest. ERA5
runs about 0.03 °C/decade warmer over land than Berkeley, a known
reanalysis-versus-station difference.

## Rank agreement

| Comparison | Spearman ρ | n |
|---|---:|---:|
| ERA5-area vs Berkeley-area | +0.618 | 153 |
| ERA5-area vs station       | +0.639 | 154 |

The two area-weighted products agree on country ordering at ρ = 0.62: solid
but not tight. ERA5-area is about equally similar to Berkeley-area and to the
station ranking, so ERA5's area weighting moves the ranking less far from the
station picture than Berkeley's does. Area-weighted country trends carry real,
product-dependent uncertainty.

## Caveats

1. ERA5's residual coupling (+0.118) is higher than Berkeley's (+0.011). It is
   not significant, but the point estimate keeps a faint positive trace. The
   robust claim is that the coupling collapses to weak and non-significant, not
   that it vanishes.
2. Cross-product rank agreement is moderate (ρ = 0.62), so individual-country
   area-weighted trends should be read with product uncertainty in mind.
3. ERA5's land mean runs about 0.03 °C/decade warmer than Berkeley's.

The qualitative conclusion is robust to the data source; the magnitude of the
residual coupling and the exact country rankings are product-sensitive.

## Reproduce

```
uv sync --extra era5
uv run python scripts/fetch_era5.py          # ERA5 monthly 2m T, 1950–2013, 1° (gitignored)
uv run python -m src.era5_validation         # prints the table above
```

The ERA5 grid needs a free Copernicus account and is not committed; the
artifacts (`era5_area_trends.parquet`, `era5_validation_summary.json`) are
built into `app/data/` by `python -m src.app_assets` when the grid is present
and are shown on the dashboard's validation page.
