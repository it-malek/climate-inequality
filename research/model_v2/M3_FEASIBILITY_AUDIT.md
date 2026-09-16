# M3 structural feasibility audit (graphs and synthetic outcomes only)

**Recorded 2026-09-16, before any M2 candidate fit and before any M3 or M4 computation on the warming
outcome. No warming outcome was read.** The audit shows that the frozen M3 method
([`DOWNSTREAM_COMPLETION_SPEC.md`](DOWNSTREAM_COMPLETION_SPEC.md) §2) is executable on the frozen sample for
both possible retained static specifications. It says nothing about whether any spatial term is useful.

## 1. What was run

```
uv run python -m research.model_v2.m3_feasibility
```

* **Code.** Canonical record written from pushed commit **`99f2c3d`** (the tool refuses a canonical record
  from uncommitted or unpushed code). Its code closure digests are in the summary.
* **Inputs** (SHA-256 pinned): `m0_countries.csv` (identifiers, M49 labels, station and land centroids,
  categorical labels), `country_geometry.csv` (land centroids), the `4263429` territorial lower bounds, the
  M1b fold artifact and the `M0star` rows of the M1b encoded design (exact predictor values).
* **Identities.** The full-sample station and land-centroid kNN8 matrices equal the frozen M1b identity
  digests; the primary LOCO memberships equal the frozen fold artifact.
* **Record:** `outputs/m3_feasibility/` (`m3_feasibility_graphs.csv`, `m3_feasibility_synthetic_fits.csv`,
  `m3_feasibility_summary.json`, `m3_feasibility_manifest.json`).

## 2. Graphs: pass everywhere

968 graphs: two weights arms × {full sample, 151 primary 500 km LOCO, 20 M49, 10 random, 151 territorial
1000 km LOCO, 151 land-centroid 1500 km LOCO}.

* **Size.** Minimum training set 107 (1000 km buffer); 124 under the primary protocol. Every graph has at
  least 9 nodes, zero diagonal and rows summing to exactly 1.
* **Log-determinant.** log|I − θW| has sign +1 and is finite at all 199 grid points of every graph.
  Spectral radius 1 (to 1e-14); the most negative real eigenvalue is −0.39.
* **Ties.** Station centroids produce exact distance ties at the eighth/ninth-neighbour boundary (4 in the
  full-sample graph; 544 training-node occurrences across the primary fits), including one pair of
  countries with identical station centroids (Armenia and Georgia, distance 0). The frozen rule, distance
  then canonical index, resolves every tie deterministically. Land centroids have no boundary ties.
* **Held-out attachments.** Under the primary protocol the nearest attached training node is at least 644 km
  (station) or 746 km (land) away; the farthest eighth neighbour is about 9,200 km away.

## 3. The frozen estimator on synthetic outcomes

**Executability.** On the real encoded designs, with outcomes drawn synthetically from each family (true θ
0.4, σ 0.03, fixed seed), all 728 station-weight fits (M0\* and M2 designs × SEM and SAR × full sample, 151
primary, 20 M49 and 10 random fits) are computable. Every grid is unimodal, no fit lies on the domain bound,
held-out predictions are bit-identical when the held-out outcomes are changed, and the median fit takes
about 0.04 s.

**Shrinkage Monte Carlo** (full sample, intercept-only synthetic mean, 100 draws per cell):

| Design | Family | True θ | Mean θ̂ | SD θ̂ | Draws on the domain bound |
|---|---|---:|---:|---:|---:|
| intercept only | SEM | 0 | −0.064 | 0.182 | 0 |
| intercept only | SEM | 0.4 | 0.354 | 0.161 | 0 |
| intercept only | SAR | 0 | −0.023 | 0.179 | 0 |
| intercept only | SAR | 0.4 | 0.357 | 0.161 | 0 |
| M0\* | SEM | 0 | −0.532 | 0.334 | 14 |
| M0\* | SEM | 0.4 | 0.019 | 0.346 | 1 |
| M0\* | SAR | 0 | −0.346 | 0.226 | 0 |
| M0\* | SAR | 0.4 | 0.075 | 0.237 | 0 |
| M2 | SEM | 0 | −0.601 | 0.317 | 13 |
| M2 | SEM | 0.4 | −0.200 | 0.378 | 3 |
| M2 | SAR | 0 | −0.425 | 0.237 | 3 |
| M2 | SAR | 0.4 | −0.019 | 0.248 | 0 |

**Reading, recorded before any result.**
* With an intercept-only mean the estimator behaves as expected (a small downward small-sample bias).
* The frozen static designs contain continent, climate and income dummies that are spatially smooth. They
  absorb spatially smooth error and induce negative autocorrelation in the residual, so the ML dependence
  estimate is pulled strongly toward negative values: about −0.5 under no true dependence. **A negative or
  small θ̂ from M3 is therefore not, by itself, evidence of negative or absent spatial dependence in the
  outcome**, and θ̂ magnitudes are not read as dependence strength.
* Constrained estimates on the lower bound occur in 13–14% of null SEM draws. This is why the spec treats a
  boundary estimate as a flagged constrained estimate, not a structural failure (spec §2.5 step 5). The
  change was made at commit `99f2c3d`, before the specification was frozen and before any M2 or M3 fit; the
  first draft of the audit (commit `c27c7fd`) stopped on such a draw and wrote no record.

## 4. Decision

The M3 method is executable as specified, in both branches, for both weights arms and every reported
protocol. No graph, estimator, domain or rule was changed other than the boundary disposition above.
