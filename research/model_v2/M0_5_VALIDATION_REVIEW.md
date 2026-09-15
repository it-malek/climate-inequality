# M0.5 preflight: validation defect requiring review

Status: **M0.5 paused, incomplete. No protocol correction has been applied.**
The current session explicitly requires documenting a genuine validation defect
and stopping for review before changing the frozen protocol. This report records
that gate; it is not the completed rank or product-stability audit.

## Verified repository and M0 state

- Branch: `research/model-v2-residual-structure`.
- Immutable `v1.3.0`: `a6733cb78292b9b27466e7ebfe7c4fdc6b59592f`.
- Session 1 / current HEAD: `af97bd21c30bc7b09708e0bd7c8d71f4d61a334c`.
  Its sole parent is the tagged baseline; all its changes are under `research/`.
- Initial working tree was clean. No existing research output was overwritten.
- `git diff v1.3.0 -- src app outputs scripts tests pyproject.toml uv.lock`
  is empty, so the current production code and bundle are the tagged versions.
- `verify_against_bundle()` independently refits the baseline: n=151,
  R²=0.6364460027547281, residual=0.36355399724527193,
  geography=0.5130328657008392, emissions=0.021392838794997177,
  socioeconomic=0.06932302425611893, population=0.03269727400277295,
  residual Moran's I=0.26868901614174107. Differences from the serialized
  bundle are at most 4.53e-11. Design rank is 20 of 21.
- No push was attempted. Session 1 is local according to the locally available
  remote refs and its publishing policy; live remote state was not queried.

## Defect: the territorial exclusion guarantee is not implemented

`borders.border_distance_matrix()` computes minimum distances between country-
assigned **1° cell centres**, not between land territories or cell footprints.
This approximation is explicitly documented in Session 1, so it is not an
undisclosed code/document mismatch within that commit. However, it does not
satisfy this session's required **500 km minimum inter-territory exclusion**.
The protocol's statement that the nearest training territory is at least 500 km
away overstates what its implementation ensures. Small islands may also vanish
from the 1° sampling.

The existing 15-arc-minute GPW national identifier grid already used by
`geometry.load_national_grid()` supplies a direct check without downloading new
data, using outcomes, or changing folds. For each unordered pair retained by the
frozen table (distance >500 km), find the nearest pair of GPW-assigned centres on
the finer grid. There are **88 pairs spanning 91 countries/LOCO folds** with
finer-grid distances <500 km. Their coordinates and both distances are in
`outputs/m0_5_buffer_witnesses.csv` (176 directed training/test relations).

| Pair | Frozen 1° distance, km | 15′ centre witness, km |
|---|---:|---:|
| Nicaragua–Colombia | 792.4 | 77.528 |
| Italy–Spain | 514.7 | 322.467 |
| Bosnia and Herzegovina–Germany | 500.5 | 360.839 |
| Austria–Belgium | 561.4 | 386.498 |
| Finland–Lithuania | 556.0 | 389.183 |

These are **raster-assignment witnesses**, not surveyed border distances. GPW
itself has finite resolution and country-assignment limitations. The 88 pairs
are not an exhaustive inventory of true boundary violations: even 15′ centres
can miss boundaries and islands. The discrepancy is nevertheless material on
the repository's own finer territorial representation, including mainland pairs
well away from the threshold. Merely rounding the cached table more precisely
cannot resolve it.

## Consequences for existing scores

- V1 in-sample fitted values, R², group decomposition and residual Moran's I
  are unaffected. This is a research validation issue, not a V1 model defect.
- The saved primary CV R² (~0.212), RMSE (~0.0414 °C/decade), regional errors,
  calibration, training sizes, and unseen-level counts remain outputs of the
  documented **1° centre-distance protocol**. They are not verified scores
  under a guaranteed minimum 500 km territorial buffer.
- They therefore cannot certify the required primary validation. Revised folds
  require a separately versioned M0 scorecard before comparing representations
  under a corrected primary implementation. The direction and magnitude of
  score changes are unknown; no corrected scores have been fit here.
- The 1000 km territorial sensitivity and territorial correlogram use the same
  approximate distance metric. Their numerical results are retained, but their
  literal border-distance interpretation has the same limitation.
- The unbuffered M49 holdout and random reference fold memberships do not depend
  on this threshold and are not invalidated by this defect. No claim is made
  that other aspects of validation have received a complete audit.

## Smallest justified correction proposed for review, not implemented

Keep 500 km, LOCO, the country sample, estimator, train-only categorical encoding,
mean-training-level-effect fallback, secondary M49 holdout, random-reference role,
and scorecard definitions fixed. Replace only the distance/fold-membership
implementation with a declared land-territory geometry and a geodesic
minimum-distance calculation that includes islands and multipart territories.
Specify source/vintage and territory-to-ISO mapping before computing scores.
Exclude a pair whenever its conservative distance lower bound is <=500 km;
where geometric approximation uncertainty straddles 500 km, exclude it rather
than asserting clearance. Store unrounded distances and a geometry-error bound.

A finer cell-centre table alone is not an adequate correction: it still cannot
guarantee boundary clearance. A raster implementation needs cell footprints and
conservative error bounds, plus explicit handling of omitted islands. Test shared
borders, small islands, multipart territories, the dateline and threshold cases.
Preserve the Session 1 table and scores as historical artifacts; publish any
approved replacement under distinct research filenames with fold differences.
Recompute the M0 reference CV scores before running the rank sensitivity rows.
This is a correction to match the declared exclusion target, not performance-
driven tuning. Do not change the 500 km width based on subsequent scores.

## Reproduction of the witness table

Run from the repository root with the existing environment and local GPW data.
This code reads existing artifacts and prints the witness table; it does not
replace cached distances or fit any models.

```python
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from research.model_v2.geometry import load_national_grid, EARTH_RADIUS_KM

iso, lats, lons = load_national_grid()
d = pd.read_csv(
    'research/model_v2/outputs/country_border_distance_km.csv', index_col=0
)
points, coords = {}, {}
for code in d.index:
    r, c = np.where(iso == code)
    lat, lon = lats[r], lons[c]
    coords[code] = np.column_stack([lat, lon])
    p, l = np.radians(lat), np.radians(lon)
    points[code] = np.column_stack([
        np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)
    ])
trees = {c: cKDTree(p) for c, p in points.items()}
rows = []
for i, a in enumerate(d.index):
    for b in d.index[i + 1:]:
        if d.loc[a, b] <= 500:
            continue
        ds, js = trees[b].query(points[a])
        k = ds.argmin()
        km = 2 * EARTH_RADIUS_KM * np.arcsin(min(ds[k] / 2, 1))
        if km < 500:
            rows.append(dict(
                a=a, b=b, frozen_km=d.loc[a, b], witness_km=float(km),
                a_lat=coords[a][k, 0], a_lon=coords[a][k, 1],
                b_lat=coords[b][js[k], 0], b_lon=coords[b][js[k], 1]
            ))
print(pd.DataFrame(rows).to_csv(index=False))
```

## Review decision and remaining work

Approve the scoped geometry correction above, or explicitly accept the existing
1° centre-distance approximation and withdraw the strict territorial-clearance
guarantee. Recommendation: approve the correction; do not select an option by
its resulting CV score.

The requested complete rank sensitivity scorecard, product-stability audit,
M1a specification and single completed M0.5 commit are **pending**. This review
note and evidence are deliberately uncommitted so they do not consume that
single commit boundary. No M1/M1a work, predictor expansion, aerosol retrieval,
production edits, protocol changes, nonlinear fits or spatial-model fits occurred.
``mean_effect`` remains the existing arithmetic mean of training category-level
effects (including the reference level), not an outcome-dependent encoding.

## Verification completed before the review stop

- Existing research tests plus full project suite:
  `.venv/bin/python -m pytest research/model_v2/tests tests -q`:
  **604 passed, 5 skipped, 3 warnings** (18.52 seconds).
- `.venv/bin/ruff check .`: all checks passed.
- Witness distances independently recomputed with `spatial.haversine_matrix`:
  all 88 pairs pass both threshold checks; largest numerical difference from
  the unit-sphere tree calculation is 6.26e-13 km; 91 distinct folds verified.
- `git diff --exit-code af97bd2`: no tracked file changed. Only this new review
  note and its witness CSV are untracked. No executable research module was
  added or modified; the reproduction snippet is documentation of the check.
- Tag SHA is unchanged. No commit, push, dashboard regeneration or production
  bundle write occurred. Complete M0.5 validation and focused tests remain
  required when its actual audit code is implemented after review.

## Follow-up after approval: territory membership needs review

The user approved correcting the implementation while freezing the intended
500 km territorial LOCO protocol, and explicitly required another stop if the
geometry work exposed a material ambiguity in what territory belongs to a
country. **That gate is now reached, before any corrected folds or scores.**
The correction will receive its own research commit after this is resolved;
the earlier instruction to reserve a single M0.5 commit is superseded by that
approved two-commit boundary.

### Inventory and evidence

The project data tree contains Natural Earth `ne_110m_land.zip`, a physical
land mask without country ownership, and the 15′ GPW national identifier grid.
The installed pyogrio test fixtures additionally contain
`naturalearth_lowres/naturalearth_lowres.shp` (177 records, EPSG:4326).
No higher-resolution country polygon dataset was found in the project trees
or the local package cache inspected for this check. The fixture is not an
adequate production geometry source: it is generalized, has `iso_a3=-99` for
France and Norway, and its Colombia and Nicaragua geometries each contain
only a mainland polygon. It therefore cannot enforce all-island separation
for the discovered Nicaragua–Colombia case. It has not been adopted.

There is also a material territorial-unit mismatch, independent of numerical
geodesic accuracy:

| Country/territory | Existing GPW assignment | Available country polygon convention |
|---|---|---|
| France / French Guiana | `FRA` and `GUF` are separate | French Guiana is part of the France multipolygon |
| Denmark / Greenland | `DNK` and `GRL` are separate | Greenland is a separate country record |
| Norway / Svalbard and Jan Mayen | `NOR` and `SJM` are separate | Requires an explicit mapping; ISO code for Norway in the fixture is `-99` |

These GPW assignments are verified from the local grid, not inferred from
political labels. For example, `FRA` has 1,122 assigned 15′ cells between
41.375°N and 51.125°N; `GUF` has 116 assigned cells between 2.375°N and 5.625°N.
The fixture's France multipolygon includes a component spanning approximately
54.525°W–51.658°W and 2.053°N–5.757°N. Shapely reports that this France
multipolygon intersects Brazil. The unchanged Session 1 table instead reports
France–Brazil = 6,429.0 km and France–Suriname = 6,668.0 km.

Thus choosing a country polygon layer without an explicit mapping could change
country membership as well as correct geometry. The operational phrase "any
land territory belonging to A" does not settle whether to preserve the V1
ISO-coded observational units or aggregate territories under their sovereign
state. A source's default country labels are not a sufficient decision rule.

### Decision requested, before implementation

Recommendation: **preserve the existing GPW/ISO country-unit membership** for
the validation geography, while replacing the sampled centres with suitable
polygon boundaries and retaining every represented island/exclave within each
unit. French Guiana would remain `GUF`, rather than silently being added to the
`FRA` observation. Greenland would similarly remain `GRL`. This matches the
geographic units used to construct the outcomes and does not aggregate new
territories into existing observations.

The alternative is **sovereign-state extent for exclusion only**: include
separately coded dependencies/overseas territories under the corresponding
sovereign state (for example French Guiana under France, Greenland under
Denmark), while retaining the frozen country outcomes. That is a different
exclusion geography and would need an explicit territory-to-state crosswalk.
No outcome or CV performance has been used to recommend either mapping.

After membership is settled, obtain and pin a country/territory polygon source
with adequate island coverage. The existing low-resolution fixture alone is
insufficient; merely replacing it with finer GPW centres is also insufficient.
Document the source vintage, checksum, crosswalk, CRS, geodesic algorithm and
any conservative numerical error bound before generating corrected scores.
This source work remains pending; no final geometry method is claimed here.

### State at this second stop

- No distance implementation, folds, model fits or scorecards were changed.
- No correction commit or substantive M0.5 commit has been created.
- The original witness CSV is preserved byte-for-byte (SHA-256
  `82365359492513fff9df8bce2468a3c5847a79d97c69624f668f208728938822`).
- Only this previously untracked review document was extended. No new executable
  code was added, so no new test result or corrected geometric guarantee is
  claimed. Earlier test results above refer to the earlier preflight.
- No M1a implementation, predictor addition, aerosol retrieval or public-project
  change occurred. The 500 km threshold, M49 secondary holdout, unseen-level
  handling and frozen scorecard definitions remain unchanged.

## Approved ontology resolved; correction implemented

The owner approved preserving V1 GPW/ISO units. The membership decision above
is resolved and is now explicit in `SPATIAL_CV_PROTOCOL.md`.
The final method is the conservative WGS84 full-footprint clearance certificate
in `territory.py`, documented and proved in `TERRITORIAL_CV_CORRECTION.md`.
It uses all assigned 15′ footprints, not bare raster centres or sovereign polygons.
The 39.489785 km two-footprint radius allowance plus a 1 mm numerical allowance
guarantees exclusion of <=500 km neighbours within the adopted GPW geometry.
It does not claim exact real-world shoreline or coverage of land missing in GPW.

Corrected primary R²=0.153079913 (legacy 0.211980511); RMSE=0.042872581
(legacy 0.041354892). The correction changes 100 folds, adds 218 directed
exclusions and retains no previously excluded pair. Minimum retained certified
clearance is 500.682949 km. All 88 original witness pairs are excluded.
M49 metrics are unchanged. Full score comparison and determinism results are in
`outputs/m0_scorecard_territory_corrected.json`; all legacy artifacts remain intact.
The prior stops above are retained as the decision history, not current blockers.
