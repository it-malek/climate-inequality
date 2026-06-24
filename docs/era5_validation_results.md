# ERA5 reanalysis cross-check: does the area-weighted coupling collapse hold?

**Verdict: yes.** The v1.2 headline — that switching country warming means from
*station-weighted* to **area-weighted** collapses the apparent
warming↔responsibility coupling — reproduces on an independent gridded product.
The statistically significant station coupling does **not** survive area-weighting
under *either* Berkeley Earth **or** ERA5; both fall to a level indistinguishable
from zero, and the inequality rise reproduces almost exactly.

## Why this test

v1.2's central result (`src.area_weighting`,
[`docs/future_work.md`](future_work.md) §2) overturned the project's main coupling
finding by weighting every km² equally instead of every station. Because that rests
on a single gridded product (Berkeley Earth), it needed an independent check.
**ERA5** — ECMWF's model-assimilated reanalysis — has no station-sampling gaps and
is a fully independent estimate of the same field.

## Method (identical operator; only the data source differs)

The ERA5 lens uses the **same** per-cell Theil–Sen operator, the **same**
1950-01..2013-09 window, the **same** GPW band-11 ISO3 country assignment, and the
**same** cos(latitude) area-weighting as the Berkeley area lens
(`src.era5_weighting` reuses the generalized `src.area_weighting.cell_trends`). The
only thing that changes is the temperature grid. ERA5-specific handling: absolute
Kelvin (irrelevant for a *slope*), the CF datetime axis, and 0–360° longitudes
normalized for the GPW sampler. The comparison reuses the standalone
`src.coupling` operators (Spearman ρ and the Gini-style inequality coefficient);
ERA5 is a **cross-check artifact, not a PCS projection** — `PCS_V2` stays frozen at
six projections.

## Headline: coupling vs responsibility (common 153-country set)

Every lens is scored on the *identical* set of 153 countries (every lens defined),
so this is an apples-to-apples comparison, not a coverage artifact.

| Lens | Spearman ρ | p-value | Significant? | Gini |
|---|---:|---:|:--:|---:|
| Station-weighted        | **+0.364** | < 0.0001 | **yes** | 0.571 |
| Area-weighted (Berkeley)| +0.011 | 0.892 | no | 0.606 |
| Area-weighted (**ERA5**)| +0.118 | 0.148 | **no** | 0.603 |

- The significant station coupling (**ρ +0.364**) drops by ~68% under ERA5
  area-weighting (**ρ +0.118**) and is **no longer statistically significant**
  (p = 0.15). Berkeley collapses further still (ρ +0.011, p = 0.89).
- **Both** area-weighted products therefore agree on the qualitative conclusion:
  *no meaningful warming↔responsibility coupling once land area is weighted
  honestly.*
- The **inequality rise reproduces near-exactly**: Gini 0.571 (station) →
  0.606 (Berkeley) / 0.603 (ERA5).

## World-land sanity check

| Product | cos(lat) world-land mean (°C/decade) |
|---|---:|
| Berkeley Earth (this pipeline) | 0.193 |
| **ERA5** (this pipeline) | **0.222** |
| Berkeley Earth documented global-land reference | ~0.19 |

Both land in the same ~0.2 °C/decade ballpark, validating the ERA5 ingest. ERA5
runs ~0.03 °C/decade warmer over land than Berkeley — a known reanalysis-vs-station
difference, not a pipeline error.

## Rank agreement

| Comparison | Spearman ρ | n |
|---|---:|---:|
| ERA5-area vs Berkeley-area | +0.618 | 153 |
| ERA5-area vs station       | +0.639 | 154 |

The two area-weighted products agree on country ordering at ρ = 0.62 — solid, but
not tight. Notably, ERA5-area is about *equally* similar to Berkeley-area and to the
station ranking, i.e. ERA5's area-weighting shifts the ranking somewhat **less** far
from the station picture than Berkeley's did. Area-weighted country trends carry
real, product-dependent uncertainty.

## Honest caveats (the publishable nuance)

1. **ERA5's residual coupling is higher than Berkeley's** (+0.118 vs +0.011). It is
   not statistically significant, but the point estimate retains a faint positive
   trace. The robust claim is *"collapses to weak / non-significant,"* not *"vanishes
   to exactly zero"* — the latter is Berkeley-specific.
2. **Moderate cross-product rank agreement** (ρ 0.62): the two products genuinely
   differ on some country rankings, so individual-country area-weighted trends should
   be read with product uncertainty in mind.
3. **ERA5 land mean runs warm** (+0.03 °C/decade vs Berkeley).

**Bottom line:** the central conclusion is robust to the data source. The *magnitude*
of the residual coupling and the exact country rankings are product-sensitive — which
is itself the honest result.

## Reproduce

```
uv sync --extra era5
uv run python scripts/fetch_era5.py          # ERA5 monthly 2m T, 1950–2013, 1° (gitignored)
uv run python -m src.era5_validation         # prints the table above
```

Numbers above are from the run on the full ERA5 grid (153-country common set);
artifacts: `data/processed/era5_area_trends.parquet` +
`era5_validation_summary.json` (gitignored), surfaced on the dashboard's
validation page when the bundle carries the ERA5 assets.
