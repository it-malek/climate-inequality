# Layer 1 (physical drivers) — toolchain / physics-validation survey

**Status: backlog / forward-looking.** Layer 1 is *specified but not implemented*
(`src/forcings.py` and `src/physical_model.py` do not exist). This memo records the
toolchain **decision** for when the L1 build is scheduled; it ships no code and binds
nothing in the current pipeline. The L1 specification itself lives in the
`climate-inequality-instructions` spec repo (`03-models.md`, `07-data-schemas.md`).

## Decision (one line)

**Build L1 on the existing Python scientific stack (NumPy + SciPy), with statsmodels as a
test-time cross-check and Wolfram as a design-time symbolic oracle. Reject Julia and the
climate-emulator stack (FaIR / Hector / MAGICC / climlab / two-layer) for the specified
model, and reject the MCMC frameworks (PyMC / NumPyro / Turing / brms / rstanarm).**

## What L1 actually is (the constraint that decides everything)

The spec's L1 is **not a climate model** — it is a feasible-GLS regression:

> `T(t) = b₀ + Σ bᵢ·Fᵢ(t−ℓᵢ) + εₜ`,  `εₜ = ρ·εₜ₋₁ + uₜ`

fit by a **closed-form Normal-Inverse-Gamma posterior** on the **AR(1)-whitened
(Prais–Winsten)** design, with a ridge prior and `ρ` estimated by deterministic
fixed-point iteration. ~150–275 annual rows, ~7 regressors. **No MCMC, no sampler** (the
spec excludes PyMC/GP explicitly). Outputs: per-driver sensitivities (°C·W⁻¹·m², 95% CrI),
a Student-t predictive band, and hindcast skill (train ≤ 2013 / test > 2013).

Three consequences follow:

- **Performance is irrelevant.** 275×7 linear algebra is microseconds anywhere, so speed
  cannot discriminate between tools.
- The right stack expresses **closed-form linear algebra deterministically with the fewest
  dependencies** — matching the repo's determinism-first, byte-stable philosophy
  (`round_floats`, `write_typed_parquet`).
- Tools built for *sampling* or for *forward physical simulation* are solving a different
  problem and are out of scope for the specified estimator.

## Evaluation (suitability to the L1 spec)

| Tool | Role under the spec | Verdict |
|---|---|---|
| **NumPy + SciPy** | Prais–Winsten whitening + ridge-GLS normal equations + closed-form NIG moments; `scipy.stats.t` for the predictive band | **PRIMARY (runtime)** — already a dependency, deterministic, exact |
| **statsmodels** | `GLSAR` / `SARIMAX(exog, order=(1,0,0))` to independently reproduce `bᵢ` and `ρ` | **TEST-TIME cross-check** — already a dependency; MLE ≠ the NIG posterior, so not the runtime path |
| **Wolfram** | derive/confirm the NIG posterior, σ²→Student-t marginalization, Prais–Winsten algebra, stationarity `\|ρ\|<1`; freeze results as Python literals | **DESIGN-TIME oracle only** — never in CI (network/version → not byte-stable) |
| Julia (DifferentialEquations / Turing / StateSpaceModels / DataFrames) | — | **REJECTED** (see below) |
| Climate emulators (FaIR / Hector / MAGICC / climlab / openscm-twolayermodel) | — | **REJECTED for the spec** (see below); reserved for the expansion scenario |
| MCMC (PyMC / NumPyro / Turing / brms / rstanarm) | — | **REJECTED** — sampler-based; violates the closed-form / no-MCMC determinism requirement |
| JAX | — | **REJECTED** — autodiff/XLA solves a problem we don't have; heavy dep; XLA float reassociation |
| ArviZ | — | **REJECTED** — diagnoses posterior *samples*; a conjugate posterior is analytic |
| PySAL / spatial HAC | — | **REJECTED** — spatial tooling; L1 is a *global temporal* series with no spatial dimension (spatial structure lives in L2) |

## Why Julia is rejected

The spec has no ODE/PDE (DifferentialEquations.jl has nothing to integrate), no sampler
(Turing.jl is MCMC), and StateSpaceModels.jl would fit the model only via Kalman-filter
MLE — not the conjugate NIG posterior the contract specifies. Adopting any of them drags a
**second language, runtime and lockfile**, plus a **second determinism surface**, into a
Python/uv repo for microsecond-scale arithmetic. Integration and long-term maintenance cost
dominate every potential benefit. They are excellent tools; this is the wrong fit.

## Why climate emulators are rejected (for the spec)

FaIR, Hector, MAGICC, climlab and the OpenSCM two-layer model are **forward physical
emulators**: given forcings (or emissions) they *produce* a temperature trajectory. L1 is
the inverse problem — it *learns* per-driver sensitivities with credible intervals from
observed temperature. They are a different paradigm and cannot implement the specified
estimator. (MAGICC additionally ships as a non-commercial-licensed binary; Hector needs a
C++/Boost build — both at odds with the dependency-light, byte-stable philosophy.)

## Expansion scenario (only if L1 later becomes a serious climate model)

Per the locked architecture (layers never merge; artifact-only communication; "a two-box
energy-balance variant as a *separate* model artifact"):

- Add **openscm-twolayermodel** (pure-Python, JOSS-reviewed two-layer EBM — the physical
  generalization of the AR(1) inertia term) and/or **FaIR v2** as a **separate** forward
  artifact, never merged into the closed-form L1.
- Bayesian *calibration* of such an emulator is the one place a sampler is defensible —
  **NumPyro** or **PyMC**, run offline with recorded seeds, as its own artifact.
- MAGICC / Hector only for IPCC-grade assessment, accepting the binary / licensing / build
  burden.

## Revisit trigger

Re-open this decision when (a) the L1 build is actually scheduled, or (b) the L1 scope
changes from "interpretable forcing→temperature regression" to "physical emulator." Until
then this is recorded backlog, not an active dependency.
