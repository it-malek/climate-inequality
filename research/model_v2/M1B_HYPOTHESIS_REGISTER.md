# M1b hypothesis register — design only

**Status (2026-09-15): C2 approved as the sole M1b hypothesis; C1 and C3 held.**

No model has been fitted. *(Update 2026-09-15: CRU TS v4.10 was acquired and C2 was first built under the original coverage rule; see `M1B_FEASIBILITY_AUDIT.md` and contract Amendment 1.)* Each entry follows the owner's required fields. The design is in `M1B_DESIGN.md`; the frozen measurement and evaluation rules are in `M1B_EVALUATION_CONTRACT.md`.

**Notation.**
* **M0\*:** the approved full-rank station-geography baseline. It is row C0 of `M1A_REPORT.md`: primary representation (total CO₂ + population), 151 countries, corrected 500 km LOCO, spatial-CV RMSE 0.042873.
* **D_c:** the frozen M1a terrestrial support (GPW band-11 cells ∩ GSHHG 2.3.7 land).
* **Contract metrics:** the `e3e5901` scorecard.

**Residual evidence allowed.** Only product-robust patterns from `PRODUCT_STABILITY_AUDIT.md` §5 may be cited:
* presence of positive residual spatial autocorrelation;
* the Southern Africa negative regional residual;
* the single-country extremes Canada, Estonia, Indonesia, Kenya (+) and Argentina, Central African Republic, South Korea (−).

The Session 1 residual descriptions for H3/H5/H6 were Berkeley-only and are **not** used as evidence.

---

## C1. Synoptic-scale maritime exposure (land–sea thermal inertia) — HELD

**Held by the owner (2026-09-15).**

* **Why held:** it is a static geography extension overlapping continentality, and its rationale implies opposite effects in different latitude/ocean regimes. A single global linear coefficient is therefore hard to interpret.
* **Future use:** possibly later M2 nonlinear or interaction work. If revisited, it belongs to **geography**, not to its own group.
* **Status of the table below:** the specification is retained for record only.

| Field | Specification |
|---|---|
| Physical mechanism | The ocean's mixed-layer heat capacity and heat uptake make sea-surface temperatures warm more slowly than land, so land warms faster than ocean. Land that is persistently ventilated by marine air masses is tied to the slower-warming ocean surface over synoptic advection scales (~10³ km). |
| Concept represented | How much of a country's atmospheric neighbourhood is ocean. This is exposure to a large thermal reservoir, not proximity to the nearest shoreline. |
| Exact observable | For every 3′ land-area centroid in D_c: the ocean fraction of the spherical cap of great-circle radius **R = 1000 km** (proposed). Country value = land-area-weighted mean over D_c. "Land" for the cap = GSHHG L1 ∪ L5 (Antarctic ice front); lakes (L2) count as land, not ocean. |
| Source and version | GSHHG 2.3.7 full resolution, already local and hash-pinned in `m1a_measurement_manifest.json`. No download. |
| Temporal definition | Static. |
| Spatial aggregation | Area-weighted over D_c with the M1a quadrature. This measures the new variable on the outcome's support; the existing M0\* geography is unchanged. |
| Expected relationship | Negative (more ocean exposure → slower warming), conditional on M0\*. Moderate confidence (see confounding). |
| Type | State / exposure (static geographic exposure to a thermal reservoir). Not a forcing. |
| Overlap with M0 predictors | High with `continentality` (the same family at a much smaller scale) and with `hemisphere` (the SH is ocean-dominated). Moderate with `spatial_block`, maritime Köppen C and latitude. |
| New information beyond M0 | Continentality is the distance to the nearest coast. It cannot separate a narrow isthmus or peninsula surrounded by ocean from a coastal strip on a large continent at the same coast distance, and it has no ~1000 km scale. |
| Potential confounding | The ocean is not one reservoir. Near the Arctic, ocean exposure includes sea-ice loss and Arctic amplification, which pushes toward *faster* warming; the Southern Ocean pushes toward slower warming. The static fraction mixes these, so the conditional sign may be weakened or reversed at high NH latitudes. Separating them would need sea-ice or SST data (in-window, co-response), so no such variant is proposed. |
| Measurement uncertainty | Geometric; negligible numerically. The structural uncertainty is the radius choice. |
| Product dependence | None: purely geographic, independent of Berkeley and ERA5. |
| Leakage risk | None: no temperature information, static. |
| Feature-group assignment | Geography, if ever revisited (owner decision). |
| Degrees of freedom | +1 (linear fraction, identity transform). |
| Test against M0\* | Not tested (held). |
| Acceptance interpretation | Transferable out-of-fold information about cross-country warming differences, consistent with the specified exposure mechanism. Not proof that ocean heat uptake causes the differences. |
| Rejection interpretation | No detectable transfer gain at n = 151 beyond M0\*. That includes the possibility that continentality, hemisphere and the continent effects already carry the signal, or that the Arctic/Southern Ocean contrast cancels. Not evidence that land–sea contrast is absent. |
| Product-robust support | Weak and mixed. Argentina (−), Southern Africa (−) and South Korea (−) are consistent. Indonesia (+) and Estonia (+) are inconsistent. The entry rests on independent rationale. |

## C2. Baseline hydroclimatic dryness — APPROVED sole M1b hypothesis (re-specifies H3)

| Field | Specification |
|---|---|
| Physical mechanism | Where evaporation is water-limited, extra surface energy goes mostly into sensible heat rather than latent heat, so surface temperature responds more strongly. Soil-moisture–temperature coupling is strongest in dry and transitional regimes. |
| Concept represented | Climatological water availability relative to atmospheric demand **before** the trend is measured. |
| Exact observable | Per 0.5° CRU cell j: AI_j = (30-year mean annual precipitation total) / (30-year mean annual PET total), where the PET total uses days-in-month. x_j = log10(AI_j). Country value C2 = terrestrial-area-weighted mean of x_j over D_c. Exact equations, validity and coverage rules: `M1B_EVALUATION_CONTRACT.md` §2 and Amendment 1. |
| Source and version | CRU TS **v4.10** (released 2026-06-25; data directory `cruts.2604091129.v4.10`), full-length 1901–2025 `pre` and `pet` NetCDF files. The exact URLs, sizes and hash-recording rule are in the contract. No DOI existed at review. Not downloaded. |
| Temporal definition | **1920-01 through 1949-12** (360 months), with no temporal overlap with the outcome, which starts in 1950. |
| Spatial aggregation | Terrestrial area of D_c within each 0.5° CRU cell. Uses the frozen M1a support: `m1a_cell_land_area_km2.npz` with the GPW country grid; each CRU cell spans exactly 2×2 GPW cells. |
| Expected relationship | Negative coefficient on log10(AI): wetter → slower warming, drier → faster, conditional on M0\*. |
| Type | State (baseline climate regime). |
| Overlap with M0 predictors | High with Köppen B (categorical, contemporary 1980–2016 classification). Moderate with continentality, latitude 15–35°, elevation. |
| New information beyond M0 | A continuous, pre-window moisture gradient within and across Köppen classes. Köppen A–E cannot distinguish semi-arid from hyper-arid, or seasonally dry from humid, within its classes. |
| Potential confounding | Dryland dust and aerosol, irrigation expansion in drylands, and sparse deserts (station-based products smooth there) all co-occur with aridity. So does low income. |
| Measurement uncertainty and provenance | The CRU TS absolute fields are station anomalies against 1961–1990 normals, gridded and then added to the CRU CL v1.0 1961–1990 climatology. Consequences: (1) cells with no precipitation station within 450 km receive exactly the 1961–1990 climatology; (2) only stations with adequate 1961–1990 records can contribute anomalies; (3) PET is derived (Penman–Monteith) from gridded TMP/TMN/TMX, partly synthetic VAP and CLD, and a static 1961–1990 wind climatology, and has no station counts. The 1920–1949 fields are therefore not purely independent pre-1950 observations: their spatial pattern is anchored to an in-window climatological framework. This is not target leakage from Berkeley and does not disqualify the data; it is quantified by a PRE station-support QA statistic. Gauge undercatch and the PET formulation add uncertainty. |
| Product dependence | Moderate. CRU shares station archives (GHCN-lineage) with Berkeley, and PET uses temperature levels. Drylands are where Berkeley and ERA5 disagree most (Gulf and Northern Africa are product-sensitive), so this candidate is the most exposed to product artefacts. |
| Leakage risk | No outcome leakage: C2 uses no Berkeley data and no 1950–2013 trends. Residual climatological dependence (see above) comes from the 1961–1990 normals built into the absolute values. |
| Feature-group assignment | New declared conceptual group **baseline hydroclimate** (H_hydroclimate). |
| Degrees of freedom | +1. |
| Test against M0\* | M1b = M0\* + C2, under `M1B_EVALUATION_CONTRACT.md` §4. No Bonferroni, since there is one hypothesis. No saturation or spline term: curvature is M2's question. |
| Acceptance interpretation | Transferable information consistent with moisture-limited amplification, subject to the product caveat. The post-freeze ERA5 check matters most for this candidate. |
| Rejection interpretation | No detectable gain beyond Köppen and the other M0\* geography. Not evidence against land–atmosphere coupling. |
| Product-robust support | Mixed or contrary. Kenya (+, partly semi-arid) is consistent. Southern Africa (−, largely semi-arid) and Indonesia (+, humid) are inconsistent. The arid-region clusters that motivated H3 are product-sensitive. The entry rests on independent rationale only. |

## C3. Seasonal-snow albedo exposure — HELD

**Held by the owner (2026-09-15).**

* **Why held:** substantial overlap with latitude, Köppen and elevation. The observed snow series lies inside the outcome window, and the pre-1950 proxy is not observed snow.
* **Status of the table below:** retained for record only.

| Field | Specification |
|---|---|
| Physical mechanism | Snow-albedo feedback: warming shortens and reduces seasonal snow cover, raising absorbed shortwave radiation during high-insolation spring months and amplifying surface warming. |
| Concept represented | Baseline exposure of the land surface to melt-season snow cover. |
| Exact observable | Two mutually exclusive options; the owner picks one before computation (D4). (a) Observed NOAA CDR Northern Hemisphere snow cover extent (Rutgers GSL): mean fraction of March–June weeks snow-covered over 1967–1980, area-weighted over D_c; SH set to structurally missing, and it cannot be zero-filled without an explicit rule. (b) Pre-window proxy: climatological count of months with CRU TS mean temperature < 0 °C, 1921–1950, area-weighted over D_c. |
| Source and version | (a) NOAA CDR NH snow cover extent v01 (Rutgers), 1966–. (b) CRU TS `tmp`, same pinned version as C2. Neither downloaded. |
| Temporal definition | (a) 1967–1980, inside the window. (b) 1921–1950, pre-window. |
| Spatial aggregation | Area-weighted over D_c. |
| Expected relationship | Positive (more baseline snow exposure → faster warming). |
| Type | (a) State, observed, in-window. (b) Proxy: a temperature-level-derived snow-season proxy. |
| Overlap with M0 predictors | Very high with `abs_latitude`, Köppen D/E, elevation and hemisphere. This is the most redundant candidate. |
| New information beyond M0 | The snow-line position is not linear in latitude: maritime and continental regimes differ at the same latitude, and elevation matters. The same curvature is claimed by M2's pre-specified latitude spline, so a gain for C3 is hard to attribute to snow rather than functional form. |
| Potential confounding | NH high-latitude amplification also includes permafrost, sea-ice and circulation changes. Seasonal snow co-varies with all of them. |
| Measurement uncertainty | (a) Coarse weekly charts (~190 km early cells), with known early-record inhomogeneity; no SH coverage. (b) A temperature threshold is a crude snow proxy. |
| Product dependence | (a) Low (satellite charting, independent of station temperatures). (b) Moderate (CRU shares Berkeley's station lineage). |
| Leakage risk | (a) Moderate to high: 1967–1980 snow cover is partly a response to warming inside the window. (b) Low (pre-window). |
| Feature-group assignment | Not assigned (held). |
| Degrees of freedom | +1. |
| Test against M0\* | Not tested (held). |
| Acceptance interpretation | Transferable information consistent with snow-albedo amplification. Cannot be separated from nonlinear latitude effects without M2. |
| Rejection interpretation | No gain beyond the latitude and Köppen structure. Not evidence against the feedback. |
| Product-robust support | Partial. Canada (+) and Estonia (+) are consistent. South Korea (−, seasonal snow) is weakly inconsistent. Northern Europe as a region is unresolved and preprocessing-dependent. |

---

## Considered and not shortlisted

| Candidate | Reason not shortlisted now | Could it re-enter? |
|---|---|---|
| Regional aerosol forcing (H4) | The cluster premise is not product-robust (M0.5). The emission inventories co-vary with the responsibility axis, which touches the V1 guardrail. | Only under a separately justified physical hypothesis, with an independently specified forcing or proxy definition (for example a pre-specified aerosol-optical-depth history) and its own declared group. Not on residual clusters. |
| Land-use / land-cover change, irrigation | The biogeophysical sign is latitude-dependent for deforestation. Confounded with development (income, population). Irrigation cooling is absent from ERA5's land model, so it would be product-sensitive by construction. In-window forcing history. | Possibly later, as a forcing group with a pre-stated sign (irrigation only) and an explicit product caveat. |
| Urbanization / urban heat island | An observational-network artefact rather than regional land climate. Overlaps station density and population. Berkeley homogenization already targets it. | As an observational diagnostic (H10), not a physical predictor. |
| Vegetation greening, soil-moisture trends, SST trends, snow or sea-ice trends | In-window co-responses to warming (mediators): leakage and circularity. | No, as predictors of the same trend. |
| Circulation and variability indices (PDO, AMO, NAO) | Unforced variability changes the outcome's definition (H8); not a cross-sectional physical state. | Only as a separate time-series diagnostic arm. |
| Cloud or surface-radiation trends | In-window responses; strongly product-dependent (reanalysis or satellite era). | No. |
| Additional global forcings (solar, volcanic, well-mixed GHG) | Nearly uniform across countries over 64 years, so no cross-sectional information. | No. |
| Elevation-dependent warming (relief, high-altitude fraction) | Overlaps M0 elevation and C3's snow mechanism; M1a showed elevation measurement is itself unstable (V1–M1a correlation 0.684). | Subsumed by C3 or M2. |
