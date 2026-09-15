# M1b design — baseline hydroclimatic dryness

**Status (2026-09-15): design approved by the owner.** Nothing has been fitted.

**Status update (2026-09-15).**
* CRU TS v4.10 was acquired and hash-pinned.
* The original-rule C2 build failed the coverage gate in three countries (`M1B_FEASIBILITY_AUDIT.md`).
* The contract's Amendment 1 adds a one-ring structural land-mask harmonization, adopted after the feasibility audit and before any amended value, redundancy diagnostic or score.

The original measurement and evaluation rules were frozen in `M1B_EVALUATION_CONTRACT.md` (`60111ae`) before any CRU acquisition; Amendment 1 is appended there. The candidate specification is in `M1B_HYPOTHESIS_REGISTER.md`.

## 1. Definition

```
M1b = M0* + baseline hydroclimatic dryness (C2)
M1b: y = G + H_hydroclimate + R + S + P + error
```

* **M0\*** is the approved full-rank station-geography baseline, identical to row C0 of `M1A_REPORT.md`:
  * log total cumulative CO₂ (R);
  * log population and station density (P);
  * six station-based geography features (G);
  * income group (S);
  * 151 countries;
  * in-sample R² 0.636446, spatial-CV R² 0.153080, RMSE 0.042873 °C/decade, residual Moran's I 0.268689.
* **H_hydroclimate** is a new declared conceptual group, *baseline hydroclimate*, holding one scalar per country (C2).

Frozen and unchanged:
* outcome, window (1950-01 to 2013-09) and temperature product;
* the 151 countries and the corrected 500 km footprint-certified LOCO;
* M49 secondary; random CV as reference only;
* the unseen-level rule and scorecard definitions;
* V1 and all public artifacts.

M1b tests **one** hypothesis. There is no candidate search and no joint model.

## 2. Why C2 only (owner decision, 2026-09-15)

* **C2, baseline dryness: approved.** It represents genuinely new physical state information beyond the geography block, with a coherent linear hypothesis: drier baseline regimes are expected to warm faster.
* **C1, maritime exposure: held.** It is a static geography extension that overlaps continentality. Its own rationale implies opposite effects in different regimes (Arctic sea-ice amplification versus Southern Ocean heat uptake), so a single global linear coefficient is hard to interpret. It is kept for possible later M2 nonlinear or interaction work, and if revisited it belongs conceptually to **geography**, not to a candidate-specific group.
* **C3, seasonal-snow albedo: held.** It overlaps substantially with latitude, Köppen and elevation. The observed snow series lies inside the outcome window, and the pre-1950 temperature proxy is not observed snow.
* **Aerosols (H4) and the other considered candidates** stay not shortlisted, for the reasons in the register.

## 3. Selection basis and evidence

C2 rests on an independently specified physical rationale:
* in water-limited regimes, extra surface energy goes preferentially into sensible rather than latent heat;
* soil-moisture–temperature coupling is strongest in dry and transitional regimes.

It has **no** clean product-robust residual support (see the register). Its dryland-region motivation from Session 1 was product-sensitive and is not used.

## 4. Measurement (frozen in the contract)

* **Support:** each country's terrestrial area, the frozen M1a support D_c. Not station locations.
* **Pre-outcome temporal window:** 1920-01 through 1949-12, 360 months, with no temporal overlap with the 1950 outcome start. C2 is not a predictor built exclusively from pre-1950 information (see the provenance limitation).
* **Source:** CRU TS v4.10 PRE and PET, full-length 1901–2025 NetCDF files.
* **Formula (mathematically unambiguous):**
  1. Per 0.5° cell, take the ratio of the 30-year mean annual precipitation total to the 30-year mean annual PET total.
  2. Take log10 of that ratio per cell.
  3. Average the cell values over D_c, weighted by terrestrial area.

  The exact equations, validity and coverage rules are in the contract, §2 and Amendment 1.

**Provenance limitation, stated accurately.**
* **Not independent observations.** The 1920–1949 CRU absolute fields are not purely independent pre-1950 observations.
* **How the values are built.** CRU TS converts station series to anomalies against 1961–1990 station normals, grids the anomalies, and adds the CRU CL v1.0 1961–1990 climatology to produce absolute values.
* **Where there are no stations.** Cells with no station within the correlation decay distance receive the climatology itself.
* **PET.** It is derived with Penman–Monteith from gridded temperature, vapour pressure (partly synthetic), cloud cover (partly synthetic) and a static 1961–1990 wind climatology.
* **Consequence.** The spatial pattern of the 1920–1949 values is anchored substantially to a 1961–1990 climatological framework that lies inside the outcome window.

This is **not** target leakage from the Berkeley warming outcome, and it does not disqualify the dataset. It limits how strongly C2 may be described as an independently observed pre-1950 state. It is quantified by a pre-outcome support-quality checkpoint, committed before the outcome is loaded.

## 5. Evaluation (frozen in the contract)

**Comparator.** The comparator is M0\*; its values must reproduce C0 to 1e−10. The existing frozen conditions are retained without weakening:
* 95% paired resampling interval for ΔRMSE (the existing country bootstrap of paired OOF errors);
* practical RMSE-improvement requirement;
* pre-stated sign and ≥ 80% sign stability;
* corrected 500 km spatial CV;
* M49 veto and worst-region veto;
* residual spatial diagnostics.

**Multiplicity.** No Bonferroni: one hypothesis.

**Two-level verdict** (contract §4.4, frozen before C2 is seen):
* **Linear hydroclimate association supported:** paired ΔRMSE interval entirely below zero, and the pre-stated negative sign in the full fit and in ≥ 80% of training fits. M49 and worst-region results are reported prominently.
* **Promoted to the primary V2 baseline:** A1–A5, including ΔRMSE ≤ −0.002 °C/decade and both vetoes.
* **"Supported but sub-material / not promoted"** is a legitimate outcome. Improvement without the pre-stated sign is not support for the mechanism.

**Uncertainty.** The paired resampling interval is the existing i.i.d. country bootstrap of paired out-of-fold errors. It does not resolve spatial dependence; the 500 km spatial CV, M49 holdout and worst-region veto remain the geographic safeguards.

**Order.** A pre-outcome support-quality checkpoint and redundancy diagnostic (C2 regressed on M0\*'s predictors) are computed and committed before the outcome is loaded.

**Sensitivities.** Only after the M1b result is frozen, and only if the linear association is supported (at either level), M1b is replicated under the registered sensitivities:
* M1a area geography;
* per-capita responsibility;
* the preprocessing-aligned ERA5 outcome.

These never determine selection. No alternative dryness window, transform, definition, dataset or threshold is tested after the primary result.

*(Contract Amendment 3 A3.6, dated 2026-09-15: the per-capita representation is reported unconditionally whenever M1b is scored, as `V2_PREDICTOR_CONTRACT.md` requires. Primary acceptance uses the total-CO₂ representation alone. The M1a-geography, aligned-ERA5 and Amendment 2 five-country replications stay conditional.)*

## 6. Order of work

1. This design commit (amended, pushed).
2. `M1B_EVALUATION_CONTRACT.md` commit, before any CRU download.
3. After the amended contract is on the remote:
   * acquire the pinned CRU files and record hashes;
   * construct C2 twice (determinism);
   * apply the coverage gate, stopping on any failure;
   * run the support checkpoint and the redundancy diagnostic without loading the outcome;
   * commit the measurements and both pre-fit records.
4. If all hard stops pass: score M1b against M0\* under the contract; freeze the result.
5. Only if the association is supported: run the registered sensitivities. *(Amendment 3 A3.6: the per-capita representation is reported with any M1b score unconditionally.)*

## 7. What M1b must not do

* Test C1 or C3, or start M2 or M3.
* Search windows, transforms, datasets or thresholds; compute the pooled national P/PET ratio; add station-count thresholds.
* Use product-sensitive residual clusters as motivation.
* Test under both geography measurement sets and keep the better one.
* Change M0\*, M1a or any frozen output.
* Use causal language for any share.
