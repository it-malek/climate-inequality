# ERA5 cross-check of the area-weighting result

Version 2.1.0 compares area-weighted Berkeley Earth trends with ERA5 trends
aligned to each product's own 1951–1980 monthly climatology. The rank
association with responsibility is near zero under both products. The static
geography result survives the product check, while the spatial models'
predictive advantage does not reproduce under ERA5. This is a comparison of
temperature products with different error structures, not independent
observational confirmation or a causal attribution.

## Released anomaly-aligned comparison

| Area-weighted product | Countries | Spearman ρ versus responsibility | p-value |
|---|---:|---:|---:|
| Berkeley Earth | 154 | +0.011 | 0.896 |
| ERA5, anomaly-aligned | 154 | −0.012 | 0.880 |

Source: `app/data/residual_structure_summary.json`,
`responsibility_comparison`. Nonsignificance does not establish equal warming
or equivalence between products. The spatial-model product check is described in
[`research/model_v2/RESULTS.md`](../research/model_v2/RESULTS.md#7-robustness).

## Historical absolute-temperature sensitivity

The following tables preserve the earlier construction and country set. ERA5
was fitted in absolute temperature, while Berkeley Earth supplied anomalies.
Those results remain useful for showing sensitivity to the outcome definition;
they are distinct from the released anomaly-aligned comparison above.

## Method

The ERA5 lens uses the same per-cell Theil–Sen estimator, the same 1950-01 to
2013-09 window, the same GPW band-11 country assignment and the same
cos(latitude) weighting as the Berkeley area lens (`src.era5_weighting` reuses
`src.area_weighting.cell_trends`). This historical construction fits absolute
ERA5 temperature, while Berkeley Earth supplies anomalies. A constant Kelvin-to-Celsius
offset leaves slopes unchanged; twelve calendar-month climatological offsets
can change a Theil–Sen slope. The final comparison therefore also subtracts
ERA5's own 1951–1980 monthly climatology before fitting. In this historical path,
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

Both are in the same 0.2 °C/decade range, a broad magnitude check rather than
a validation of the ingest. In this historical construction, ERA5 runs about
0.03 °C/decade warmer over land than Berkeley Earth.

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

This historical sensitivity gives a weaker rank association under area
weighting than under the city-record mean. The association and country rankings
depend on the temperature product and outcome construction.

## Historical reproduction reference

```
uv sync --extra era5
uv run python scripts/fetch_era5.py          # ERA5 monthly 2m T, 1950–2013, 1° (gitignored)
uv run python -m src.era5_validation         # prints the table above
```

The ERA5 grid needs a Copernicus account and is not committed. These commands
document the historical pipeline and its separate sensitivity artifacts. The
released 2.1.0 reconstruction is documented in
[`current_validation.md`](current_validation.md). The dashboard identifies the
anomaly-aligned comparison separately from historical sensitivities.
