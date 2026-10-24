# Distinguishing magma recharge from crustal memory during repeated volcanic unrest

Working research plan, revised 24 October 2026. The filename is retained for
continuity; the primary source is now a chamber/sill at Svartsengi. The companion
proposal is [DIKE_MEMORY_PAPER_DRAFT.typ](DIKE_MEMORY_PAPER_DRAFT.typ). Markdown
owns the experiment specification; Typst presents the scientific argument.
Neither reports a completed chamber inference study. Plane-strain chamber
prototype results (S0/S1 qualification and a first S2 pilot) are summarized
[below](#plane-strain-chamber-prototype-results-october-2026) and detailed in
[CHAMBER_MEMORY.md](examples/stokes/volcano/CHAMBER_MEMORY.md); they are
provisional while the S0 mesh/domain gate is open.

## Geometry strategy (agreed 22 October 2026)

**Use 2-D plane strain for benchmarks, development, and synthetic inference.**
It is the deliberately cheaper model in which to establish numerical correctness,
mechanical memory, identifiability, and the observing experiments. Develop those
pieces to completion before a targeted 3-D campaign. Axisymmetry/3-D is not an
entry gate for the 2-D synthetic study. Keep plane-strain rates in m²/s and
volumes as areas per unit strike; do not interpret them as calibrated Svartsengi
volumes or claim quantitative transfer of amplitudes or relaxation times to 3-D.

## Question and proposed contribution

**When can geodesy distinguish changing magma supply from crustal memory,
and which observations make that distinction possible?**

Delayed deformation, history dependence, and elastic-inversion bias are
established phenomena. The proposed contribution is joint recovery of supply
and uncertain rheology across repeated recharge/withdrawal cycles, explicit
non-identifiable regimes, and observing strategies evaluated by actual recovery
and prediction. A claim of priority needs a broader literature audit.

Do not assume that relaxation explains the decline inferred at Svartsengi,
that relocating stations resolves the ambiguity, or that recovered recharge
predicts an eruption. These are separate tests.

| ID | Hypothesis | Test and useful negative outcome |
|---|---|---|
| H1 | Mechanical memory can mimic part of an apparent within-cycle supply decline. | Invert constant-supply viscoelastic synthetics elastically. Reject this explanation where plausible parameters cannot reproduce temporal curvature and spatial deformation together. |
| H2 | Repeated withdrawals constrain shared rheology sufficiently to recover changing supply. | Jointly estimate supply, rheology, compressibility, initial memory, and uncertain withdrawals; predict a withheld cycle. Persistent tradeoffs imply bounds, not a unique supply history. |
| H3 | Designs targeting recent supply outperform amplitude-based or elastic designs. | Compare actual noisy inversions at equal observing cost under uncertain and misspecified physics. Improvement with fixed true rheology alone is conditional evidence. |
| H4 | A small set of physical ratios predicts recoverability. | Map selected timescale, compressibility, geometry, and noise slices. Report failure of a single-timescale description if necessary. |

## Literature position: checked starting points

Downloaded papers, version/provenance notes, and outstanding access gaps are
indexed in [the local reference collection](literature/dike_memory_review/README.md).

| Source | Established result relevant here | Role in this study |
|---|---|---|
| [Segall (2016), JGR, 10.1002/2016JB013597](https://doi.org/10.1002/2016JB013597) | Viscoelastic aureole, compressibility, and pressure-dependent recharge; some regimes reinflate without recharge. Sudden withdrawals probe transient response. | Analytical chamber benchmark and coupled-budget baseline. |
| [Liao, Karlstrom & Erickson (2023), GRL, 10.1029/2022GL101172](https://doi.org/10.1029/2022GL101172) | History dependence and frequency-dependent response with thermally varying viscoelastic rheology. | Closest history/transfer-function precedent; compare one shell with broader relaxation spectra. Issue year 2023; first online December 2022. |
| [Spang et al. (2026), JGR, 10.1029/2026JB034071](https://doi.org/10.1029/2026JB034071) | 3D feeding, thermal weakening, compressibility, and departures from elastic volume/pressure interpretations. | Closest recent influx-to-deformation precedent; test its first-order elastic corrections where applicable. Published 3 September 2026. |
| [Parks et al. (2025), EPSL 658, 119324](https://doi.org/10.1016/j.epsl.2025.119324) | Svartsengi cycles, fixed-geometry automated GNSS volume series, and evolving volume-based forecast approaches. | Observation workflow and empirical comparison, not independent physical recharge measurements. |
| [Lanzi et al. (2026), EGU26-12710](https://doi.org/10.5194/egusphere-egu26-12710) | Conference abstract comparing spherical, sill, and ellipsoidal sources; more stable inferred properties after March 2024. | Separate early/late-cycle analyses; preliminary conference evidence, not a peer-reviewed constraint. |
| [Alshembari et al. (2023), JGR, 10.1029/2023JB026625](https://doi.org/10.1029/2023JB026625) | Poroviscoelastic mush and thermo-viscoelastic crust jointly affect deformation. | Delimit the single-phase baseline; a mechanical shell is not a complete hydraulic mush model. |
| [Weber, Biggs & Annen (2025), Nature Communications 16, 532](https://doi.org/10.1038/s41467-024-55443-z) | Thermal evolution controls contrasting deformation patterns. | Thermal-state uncertainty; current unrest duration does not establish reservoir thermal maturity. |
| [Hickey et al. (2024), GRL, 10.1029/2024GL108812](https://doi.org/10.1029/2024GL108812) | Topography and heterogeneity affect signal intensity and GPS configuration. | Spatial-model robustness; maximum amplitude need not maximize discrimination. |
| [Strutz & Curtis (2025), Seismica, 10.26443/seismica.v4i1.1452](https://doi.org/10.26443/seismica.v4i1.1452) | Bayesian design for seismic source monitoring. | Design-method precedent; not a direct geodetic recharge study. |

Additional contextual references from the previous plan remain useful but need
targeted reading before supporting new claims: Rivalta & Segall (2008), Segall
(2010), Sigmundsson et al. (2022, 2024), Parks et al. (2023), Pedersen et al.
(2022), and Sathiakumar et al. (2017). Liao et al. provide code through
[Dryad](https://doi.org/10.5061/dryad.3bk3j9knq); inspect it before writing a
new analytical-response implementation.

## Model, observations, and inference target

### Budget and physical scope

Start with a compressible chamber, Maxwell shell, and elastic host under a
prescribed thermal field and small displacements. Add yielding or evolving
thermal fields only to test a specific discrepancy. The shell is an effective
mechanical region, not a demonstrated poroviscoelastic mush model.

For a well-mixed reservoir at fixed temperature/composition and small density
perturbations, mass conservation gives

```math
\dot m_{in}-\dot m_{out}=\rho\dot V+\rho\beta_m V\dot p,
\qquad Q_{in}-Q_{out}=\dot V+\beta_m V\dot p.
```

The second form defines volumetric fluxes at the same reference density.
Track injected mass/reference volume, geometric chamber volume, pressure, and
apparent elastic-source volume separately. Do not impose pressure and volume
as independent chamber controls. Verify the discrete budget with pressure
quadrature; cancellation of integrated transfer sources alone is not a complete
magma mass budget.

Test constant recharge, an externally imposed supply change, and
pressure-dependent recharge `Q_in = C_f (p_deep - p)`. Feeding conductivity
`C_f` is distinct from cohesion. Under pressure feedback, declining inflow
does not by itself imply declining deep pressure or conductivity.

### Observations, target, and nuisance parameters

- Primary data: daily GNSS displacement components. Include offsets,
  reference-frame terms, temporal correlation, and common-mode errors as
  justified by data; begin with a small explicit covariance model.
- Use InSAR to constrain spatial patterns/geometry when available; it is not
  required for the synthetic pilot. Velocity comparisons must propagate the
  covariance introduced by differentiation/filtering.
- Primary target: mean recent positive recharge over a fixed physical window,
  initially 30 days. Also test a persistent 20% supply change. These are pilot
  choices, not observed thresholds or promises of detectability.
- Use a coarse time basis or correlated prior on positive recharge and vary
  its correlation length. Treat signed withdrawals separately: log flux is
  undefined at zero or negative supply.
- Joint nuisance parameters: shell viscosity/thickness, magma compressibility,
  source depth/shape, withdrawal magnitudes, and initial mechanical memory.
  Use physically motivated low-dimensional initial-history modes before
  introducing per-integration-point unknowns.

### Kernels and identifiability

For positive recharge bins `q_j = ln(Q_in,j / Q_ref)`, define the discrete
Jacobian `K_ij = ∂u_i/∂q_j`, with displacement units and bin width included
in the forward map. A continuous functional-derivative density has units of
displacement/time and integrates over the perturbed interval. A two-run step
response is a finite perturbation approaching the linear kernel sum only for
small perturbations.

Whiten by the observation covariance and scale parameter columns by declared
physical scales or prior covariance. Inspect singular values, supply/nuisance
correlations, and prior-to-posterior contraction. Distinguish data information
from regularization. In a local Gaussian approximation, use the Schur complement
of the joint precision matrix's nuisance block for marginal supply precision;
fixing nuisance parameters overstates information.

Explicitly test spatial separability: if `u(x,t) ≈ g(x) a(t)`, moving stations
may improve signal-to-noise without adding an independent signature of memory.
Repeated withdrawals do not guarantee uniqueness when arbitrary supply histories
remain free. Retain indistinguishable alternative histories and bounds.

## Stages and acceptance gates

Thresholds below are proposed pilot gates, to be frozen before production.
A failed gate narrows the claim or prompts a documented model revision; do not
silently relax a tolerance to retain a preferred conclusion.

### S0 — Qualify numerical responses and derivatives

1. Reproduce the old dike step at common physical times with plasticity off,
   fixed physical `η_vp`, and `η_vp = G Δt`, separately. Refine time step,
   mesh, source footprint, and domain; resolve ramps/withdrawals explicitly.
2. Fixed-parameter refinement tests discretization. Tied regularization also
   changes the constitutive model. Persistence at fixed `η_vp` does not prove
   this value describes natural crust; a rate-independent limit needs its own
   convergence evidence. Use an identical reference value when comparing the
   fixed and tied sequences, rather than treating nearby viscosities as equal.
3. Converge the difference of paired forward runs. Compare ±5% and ±20%
   perturbations and FD step sizes, with absolute errors near zero sensitivity.
   Repeat from controlled initial states; do not confuse spin-up with memory.
4. Audit the actual compressible residual, storage block, augmentation, pressure
   objective, multiplier mapping, and history commits against an independently
   assembled tiny full-system transpose. The working tree contains storage and
   `objective_P` changes. Historical guide warnings and old FD numbers are not
   current validation certificates; check the actual bulk-storage convention.
5. Run multi-step directional FD/Taylor checks for supply and rheology and both
   displacement components. Include active/inactive yielding if retained.
   Frozen temperature, pore/lithostatic pressure, or geometry must also be
   frozen in the FD experiment. Fully coupled gradients need their dependence.

**Gate:** response changes below 2% under successive refinements and numerical
error below 10% of the smallest claimed physical effect; nonzero gradients
within 1% of an independent FD plateau with expected Taylor behavior. Report
achieved limits. Rerun relevant adjoint tests after current solver edits. No
publication figure from unqualified dike memory.

**Status (24 October 2026, fixed-mesh linear Maxwell plane-strain chamber):**

- FD step plateau (item 3, Jacobian used by S2): **passes**. At
  `fd_step ∈ {0.20, 0.05, 0.02, 0.01}` the two finest steps agree in marginal
  `σ(log recharge)` to 0.006% in all sampling experiments; the default
  `fd_step = 0.02` lies on the plateau. ±5%/±20% paired-perturbation
  comparisons and controlled-initial-state repeats are not yet run.
- Time step: **passes** (see S1).
- Mesh/domain refinement: **open**. With the station-region `surface_size`
  co-scaled with `√max_area`, successive mesh levels change the 9-station L2
  displacement by 3.7–5.5% (bounded but non-monotone); the central-station
  `uz` sequence is monotone (99.68 → 99.46 → 99.33 → 98.96 mm). Domain widths
  of 20–160 km change it by 3.1–5.6%, non-monotone, with the widest domain
  swinging back toward the smallest. Neither meets the 2% gate; all
  quantitative chamber results are provisional until it closes.
- Items 1 (old dike step controls), 4 (independent tiny full-system transpose),
  and 5 (multi-step directional FD/Taylor checks of the adjoint) are not yet run.
  The chamber pilots use central finite differences of full forward runs, not
  the adjoint.

### S1 — Plane-strain chamber benchmark and synthetic baseline

Use the existing plane-strain circular/elliptical cavity solutions and Mogi
line-source limit for elastic checks, with finite-domain and finite-source
errors stated separately. Benchmark integrated source, chamber expansion,
pressure storage, and accepted stress/pressure history. Linear Maxwell controls
and paired recharge/withdrawal histories establish the initial synthetic model.
A fixed, undeformed mesh with zero thermal expansion, gravity, and plasticity
isolates mechanical memory before adding other processes.

Reuse the existing concentric-interface mesher, finite-K Stokes solver,
quadrature-normalized signed sources, and diagnostics. Use a function-only setup
shared by new runnable drivers and tests; avoid including legacy plotting drivers
or copying their thermal, plastic, and adjoint orchestration.

With `G = 3e10 Pa`, `η = 1e17–1e18 Pa s` implies nominal Maxwell times of about
39–386 days: an illustrative bracket, not an Icelandic measurement. Distinguish
`η/G` from coupled decay times involving compressibility, geometry, and feeding.
Choose time steps by the shortest resolved timescale. Report area in m² and rate
in m²/s; no implicit conversion to a 3-D magma volume is allowed.

**Gate:** qualify the 2-D elastic reference, discrete budget, and time/mesh/domain
response before interpreting 2-D synthetic recovery. A spherical Segall benchmark
and the targeted 3-D comparison follow after the synthetic workflow is settled;
they do not block S2–S4 in plane strain.

**Status (24 October 2026):** a fixed-mesh linear Maxwell chamber (circular
chamber at 4 km depth, concentric shell, nearly elastic host; three cycles of
40, 50, and 30 days, each ending in a ten-day withdrawal) is implemented in
module `ChamberMemory2D`
([chamber_memory_setup2D.jl](examples/stokes/volcano/chamber_memory_setup2D.jl))
and tested in `test/test_chamber_memory.jl`.

- Discrete continuity budget (pressure quadrature, divergence-integrated
  expansion): **passes**, maximum relative error ≈ 5e-6, decreasing with mesh
  refinement (5.40e-6 → 4.48e-6).
- Time step: **passes**; halving Δt changes the vertical displacement history
  by 0.48% at fixed physical withdrawal duration.
- Elastic reference: the elastic-host FEM displacement differs from a Mogi line
  source given the same measured area change by ≈13%. The line source is an
  approximate oracle; finite chamber size and domain truncation are not yet
  separated, so this is a consistency check, not a passed benchmark. The
  plane-strain circular-cavity solution is not yet compared.
- Mesh/domain: **open** (see S0).

### S2 — Identifiability before design

Compare three model families: E-P (elastic, pressure-dependent recharge),
V-C (viscoelastic, constant recharge), and V-P (viscoelastic,
pressure-dependent recharge). Include no-recharge and known-rheology controls.
Use consistent geometry, compressibility, observations, noise, and priors;
include a cheap corrected-elastic baseline where Spang et al.'s approximation
applies.

Compare one cycle with repeated cycles having varied withdrawal sizes and
repose intervals. Estimate shared rheology jointly with low-dimensional supply,
withdrawal uncertainty, and initial memory. Inspect both sensitivity rank and
actual noisy inversions. Variable supply and viscoelasticity may coexist.

**Gate:** classify regimes as data-resolved, prior-dominated, or unresolved.
Demonstrate separation with uncertain rheology before promising design gains.
In unresolved cases retain alternative histories and supply bounds.

**Status (24 October 2026): open; first plane-strain classification only.**
With shell viscosity, withdrawal amplitude, and magma bulk modulus uncertain,
the recent supply change (last 30 days relative to before) is data-resolved
and the absolute recharge amplitude is prior-dominated (see
[prototype results](#plane-strain-chamber-prototype-results-october-2026)).
This rests on one synthetic configuration with known geometry and initial
state, constant (not pressure-dependent) recharge, and local Gaussian or
Gauss–Newton inference. The E-P/V-C/V-P comparison, no-recharge and
known-rheology controls, uncertain initial memory, and varied repose intervals
remain to be run.

### S3 — Recovery, withheld cycles, and Svartsengi

Generate independent synthetic realizations for constant, feedback-controlled,
and externally changed supply. Fit competing models, including plausible wrong
geometry/rheology. Share rheology across training cycles; test both
leave-one-cycle-out reconstruction and chronological prediction of the next
cycle. A forecast cannot use future observations or the true future withdrawal;
integrate over its uncertainty. Known-withdrawal tests are conditional
reconstructions, not operational forecasts.

Report recent-supply bias/RMSE, interval coverage, detection delay/false alarms,
and held-out predictive score. Share noise realizations and target windows
across comparisons. Begin with a few deterministic 2-D FEM recovery cases. After the forward
workflow is qualified, use cached linear responses or a verified reduced model
for 100 cheap noise realizations; increase if sampling uncertainty prevents a
comparison, and cross-check against independent FEM runs.

For real data, reproduce Parks' fixed-geometry Mogi/sill observation workflow.
Use full displacement patterns, not digitized source volume alone. Compare early
cycles with the post-March-2024 subset motivated by Lanzi et al. Test normalized
spatial-pattern evolution alongside temporal curvature. Permit geometry changes
only with evidence/model comparison. Consider tectonic, fault/dike, and geothermal
contributions where relevant.

**Gate:** positive recovery claims need better held-out error/score than elastic
baselines with uncertainty reported and near-nominal coverage within sampling
uncertainty. A Svartsengi attribution also needs spatial residual improvement
and robustness to plausible competing mechanisms.

### S4 — Observing-system design after identifiability

Compare existing/even spacing, maximum amplitude, elastic designs, and joint
physics-based designs. Optimize recent supply after marginalizing nuisance
parameters. D-optimality is secondary; Gaussian linear expected information
gain is

```math
\frac12\log\det\left(I+C_{prior}^{1/2}J^T C_{noise}^{-1}J C_{prior}^{1/2}\right).
```

Report this in nats; distinguish it from the unhalved log determinant. For
supply-specific information integrate out nuisance uncertainty consistently.

Compare added stations with timing/cadence around withdrawals and horizontal
versus vertical components under a declared cost. Daily GNSS is already
continuous: distinguish using existing epochs better from acquiring more data.
Use uncertain-rheology/geometry ensembles, correlated noise, outages, and
feasible locations once real metadata are available.

**Gate:** judge designs with repeated actual inversions on independent
synthetics: bias, coverage, detection delay/false alarms, and held-out error.
Covariance evaluated with the true FEM Jacobian is conditional precision, not
the actual error of an elastic estimator. Report equivalent and no-gain designs.

### S5 — Regimes, forecast implications, and targeted realism

Complete the regime and observing experiments in plane strain first. Map
selected slices over cycle length/nominal Maxwell time, hydraulic
recharge/relaxation time, relative magma/chamber compressibility `β_m/β_c`,
shell radius/source radius, source size/depth, and signal-to-noise. Define
`β_c` from instantaneous elastic compliance so it does not absorb viscous memory.
Estimate effective decay times from verified responses; test one distributed
thermal-viscosity profile before claiming a single time suffices.

Separate supply, geometric-volume recovery, pressure recovery, and event timing.
First evaluate crossing times of a declared proxy threshold with threshold
uncertainty. Eruption forecasts need a separately validated failure/propagation
model; prescribed withdrawals cannot validate eruption timing. Parks' changing
empirical thresholds are uncertainty, not a known universal failure volume.

Background strain, yielding, and broad dike-dip sweeps are targeted later
extensions. Existing dike runs do not establish a universal dip rule.

### S6 — Targeted 3-D transfer after the 2-D workflow is settled

After S0–S5 establish the mechanics and synthetic inference in plane strain,
benchmark a spherical chamber against the applicable Segall solution and repeat
selected resolved and ambiguous cases in 3-D, followed by a sill/ellipsoid where
needed. Assess changes in spatial pattern, compliance, timescales, and observing
strategy. Do not transfer 2-D amplitudes by an arbitrary strike length. A real
Svartsengi attribution requires this geometry assessment and actual observations;
a clearly labeled 2-D synthetic methods study can be completed independently.

## Data and reproducibility

`literature/svartsengi_sundhnukur/` contains the Parks PDF, digitization scripts,
and Fig. 6 volume/rate CSVs. They are elastic-model products, not raw GNSS or
independent flux truth. Digitization uncertainty is approximately ±1 day and
0.13 million m³ per pixel; reproducing rate bars checks transcription only.

Request three-component GNSS series, covariance/processing metadata, station
coordinates/histories, event estimates with covariance, and suitable InSAR
products. No request has been sent. The paper's data statement says available
on request; it does not mean all metadata are closed.
[SENG metadata](https://gnss-metadata.eu/MOID/station.68a48f7bd3bfb21dbc0e97d6)
are public; access to the exact processed study series remains unverified.
Withdrawals estimated with elastic geodesy must carry model uncertainty rather
than enter a viscoelastic experiment as exact physical magma losses.

Archive revision plus dirty diff, environments, meshes, quadrature-normalized
sources, accepted time grids, initial histories, convergence diagnostics, budgets,
random seeds, covariance/prior definitions, inference settings, and figure scripts.
Preserve existing user changes and exploratory outputs.

## Figures and paper decisions

| Figure | Evidence required |
|---|---|
| 1: Budget and repeated-event mechanism | S0–S1 plane-strain verification; physical versus apparent source area. |
| 2: What is identifiable? | S2 degeneracies and competing fits, including an unresolved case. |
| 3: Supply recovery and withheld cycles | S3 joint inference, coverage, and chronological prediction. |
| 4: What to observe and when | S4 empirical target error at equal cost and uncertain physics. |
| 5: Recoverability map | S5 physical ratios and explicitly limited forecast proxies. |

A main paper combines S0–S5 only if inference and prediction justify it. A
synthetic identifiability study is defensible without a real-data attribution.
A methods companion needs independently verified compressible/history derivatives,
clear scope, and measured computational benefit; AD alone is not novelty.
Choose GRL/JGR/GJI or a broader journal after the results. Keep the broad dike-dip
campaign outside the minimum paper.

## Immediate next actions

The fixed-mesh linear plane-strain chamber pilot and a first S2 pilot with
uncertain viscosity, withdrawals, and compressibility exist. In order:

1. Close the S0 mesh/domain gate: finer nested mesh levels (keeping
   `surface_size ∝ √max_area`) and domains of hundreds of km, against the 2%
   gate. Re-check the elastic reference against the plane-strain cavity
   solution, separating finite-source and finite-domain error.
2. Add uncertain initial memory as low-dimensional spin-up modes and repeat
   the supply-change identifiability test.
3. Add pressure-dependent feeding `Q_in = C_f (p_deep − p)` and run the
   E-P / V-C / V-P model-family comparison with known-rheology and no-recharge
   controls.
4. Repeat the supply-change Gauss–Newton test for several truths and noise
   seeds, and for event-end and vertical-only sampling.
5. Then S3 withheld-cycle prediction. S6 (3-D) stays after the 2-D workflow.
6. Separately qualify the compressible/history adjoint (S0 items 4–5) before
   using it in place of finite differences, and run the old dike-step controls
   (S0 item 1) if dike results are to be used.
7. Obtain real observation products and define the training/held-out split.
8. Freeze target windows, priors/noise, metrics, and gates before production;
   retain failed gates and negative results.

### Continuing the work

- Run commands for every chamber driver are listed at the top of
  [CHAMBER_MEMORY.md](examples/stokes/volcano/CHAMBER_MEMORY.md). Run from the
  repository root with `julia --project=examples ...`. Tests:
  `julia --project=. -e 'include("test/test_chamber_memory.jl")'`.
- Drivers in `examples/stokes/volcano/`: `chamber_memory2D.jl` (budget,
  time step, elastic vs Mogi), `chamber_memory_convergence2D.jl` (mesh and
  domain sweeps), `chamber_memory_fd_plateau2D.jl`,
  `chamber_memory_identifiability2D.jl`, `chamber_memory_cycle_sweep2D.jl`,
  `chamber_memory_supply_change2D.jl`, `chamber_memory_gauss_newton2D.jl`, and
  `chamber_memory_figures2D.jl` (plots saved output; no simulation). Reusable
  functions (`build_chamber_memory_mesh`, `chamber_cycle_history` with
  per-cycle `withdrawal_fraction`, `run_chamber_memory`, `pilot_setup`,
  `chamber_identifiability_pilot`, `chamber_gauss_newton`) live in module
  `ChamberMemory2D` in `chamber_memory_setup2D.jl`.
- Parallelism: on the laptop used so far, Julia threads made these small 2-D
  CPU runs about 3× slower (one 4-step forward run: 152 s with 1 thread, 481 s
  with 4, 436 s with 10), because KernelAbstractions per-iteration overhead
  dominates on the ~1221-cell mesh and dynamic relaxation needs ~16,000
  pseudo-iterations per step. Those absolute timings may be inflated by
  machine state, but a threaded sweep independently ran ~3× slower per step.
  Run each forward solve single-threaded and parallelize independent forward
  runs across processes: `chamber_identifiability_pilot` and
  `chamber_gauss_newton` take a `mapper` keyword, and the supply-change and
  Gauss–Newton drivers use `Distributed.pmap` over `CHAMBER_WORKERS` workers
  (default 4). On a larger machine set `CHAMBER_WORKERS` to the physical core
  count, and re-measure threaded performance (especially on the finer meshes
  needed for item 1) before changing this advice.

## Plane-strain chamber prototype results (October 2026)

Detail, tables, and caveats are in
[CHAMBER_MEMORY.md](examples/stokes/volcano/CHAMBER_MEMORY.md). These are
prototype results on one synthetic configuration with known geometry and
initial state, constant recharge, and local Gaussian or Gauss–Newton
inference. The S0 mesh/domain gate is open, so every number below is
provisional. Rates are m²/s and areas m² per unit strike.

- **Qualification.** Budget closure ≈ 5e-6 relative, 0.48% change on halving
  Δt, FD plateau to 0.006%, elastic host vs same-area Mogi line source ≈13%
  (approximate oracle). Mesh L2 change 3.7–5.5% and domain (20–160 km) change
  3.1–5.6%, both non-monotone: the mesh/domain gate is open. Earlier
  non-monotone mesh behavior was caused by `surface_size` not being co-scaled
  with `max_area`; the sweep now co-scales it.
- **Setup.** Central full-history finite-difference Jacobians in log
  parameters; 3 mm Gaussian noise; 100 noise realizations; priors σ = 0.5 in
  log except shell viscosity (σ = 1.0); truth ×1.2 recharge, ×1.5 viscosity,
  ×1.3 withdrawals, ×0.8 magma bulk modulus, ×0.8 recent supply ratio.
- **Fixing nuisance parameters is overconfident.** With uncertain viscosity
  and withdrawal amplitude, fixing them at nominal understates σ(log recharge
  amplitude) by ≈8× (dense sampling) to >60× (event-end sampling), with ~0%
  interval coverage versus 96–100% when marginalized.
- **Repetition alone does not separate supply from withdrawal.** Identical
  repeated cycles shrink the fixed-nuisance σ but leave recharge and
  withdrawal amplitude correlated near −1; varied per-cycle withdrawal
  fractions at equal total withdrawal do not help.
- **Absolute recharge amplitude is prior-dominated** once magma compressibility
  is uncertain: marginal σ ≈ 0.25 against a 0.5 prior for one cycle and ≈ 0.2
  for five. Displacement constrains roughly recharge times the fraction that
  expands the chamber rather than compressing magma.
- **The recent supply change is data-resolved.** For a 20% decline in the
  final 30 days, σ(log ratio) is 0.030–0.035 under every nuisance subset,
  whereas σ(log amplitude) ranges from 0.004 (nothing marginalized) to 0.248
  (all marginalized). The one-step linear estimate under-covers (0.88, bias
  +0.027); an iterated Gauss–Newton MAP fit converges in three iterations and
  gives bias +0.002, σ 0.042, RMSE 0.041, coverage 0.96.
- **Observationally equivalent alternatives.** The converged nuisance
  parameters differ from the truth (less supply, stiffer magma; log offsets
  recharge −0.02 vs +0.18, bulk modulus +0.04 vs −0.22) yet fit the data to
  1.7% of the noise: concrete alternative absolute-supply histories the data
  cannot exclude.

Implications: these results support a *supply change* rather than absolute
amplitude as the primary target and give a first plane-strain S2
classification (recent change data-resolved; absolute amplitude
prior-dominated with uncertain compressibility). They do not yet test
pressure-dependent feeding, the E-P/V-C/V-P families, uncertain initial memory,
other truths or noise seeds, or prediction of withheld cycles; H2–H4 remain
untested beyond this configuration.

## Historical exploratory evidence

Retained from the previous plan for provenance; not rerun in this revision and
not qualified chamber results. Physical interpretations remain conditional on
S0. Covariance experiments are not empirical estimator-error measurements.


### Elastic limit and numerics (dike, 2D)
- Elastic-limit check (η = 1e25, C = 1e10, g = 0, one step, Q on − off):
  - 40 × 20 km box is too small: 2D uplift decays as 1/x².
  - 200 × 100 km: ϵ_tol 1e-6 not converged (center uplift 2.3× Okada);
    1e-8 → misfit 0.09, 1e-10 → 0.08. Near field = Okada × 0.7 (magma
    compression takes 26 % of Q at ΔP = 6.8 MPa).
  - Far field (|x| > 15 km) below Okada and drifting with tolerance:
    long-wavelength modes converge last in DR. Unresolved.
  - ε̇_bg = 0 exactly does not converge; use 1e-30. G_dike barely matters.
- Mogi fitted to Okada (`elastic_sources2D_plot.jl`): a 5 km-deep vertical
  dike fits as a line source at 14 km with 0.75 of the volume; at 60°,
  7.8 km / 0.67; at 30°, 3.6 km / 0.70; a sill, 3.5 km / 0.82.

### Step experiment (dike; δ = +20 % at 10 yr, Δt = 1 yr)
- First-year response 0.4–0.6 × Okada near field; grows over ~5 yr to
  1.3–1.5 × Okada and keeps evolving to 30 yr.
- Baseline still transient at onset; Mogi depth 9 km vs true 5 km.
- ~5 yr growth is far below crustal (1 kyr) and dike (300 yr) Maxwell
  times → S1.

### Time-dependent adjoint (`dike_memory_adjoint2D.jl`)
- History map: stress update with return map, Jaumann rotation, P_old;
  element ForwardDiff Jacobians; per-step `solve_stokes_adjoint_dyrel!`
  with a pressure load (`objective_P`, new in `DR_adjoint.jl`; adjoint
  unit tests not yet rerun).
- Previously reported central-FD agreement (3 steps, 2 stations): 0.07–0.5 %;
  not revalidated against the present working tree or a full coupled derivative.
- T, Pf, P_litho and mesh motion held at forward values.

### Network design (`dike_memory_design2D.jl`; 30 yr, epochs 10/20/30)
- Vertical dike: old injection keeps adding uplift above the dike
  (60 → 94 → 120 mm at t = 10/20/30 vs Okada 9); top-segment kernel
  ~10× Okada; last 3–6 yr under-weighted.
  - Greedy 4-station D-optimal: unhalved logdet score: FEM 65.0 vs Okada-designed 56.6,
    even spacing 59.8.
  - Goal-oriented: old supply σ 0.104 (FEM) vs 0.126 (Okada); Okada
    claims σ 0.32 on recent top-segment supply where the FEM-Jacobian conditional covariance gives 0.40 (not empirical error).
- 45° dike: memory weaker (old injection 1.3–1.9× Okada, latest year ≈
  Okada); FEM and Okada pick the same 4 stations (unhalved logdet score 84.5); goal
  designs differ by 2–4 %.
- Historical interpretation: these two configurations suggest dip-dependent
  design differences; S0 and robust inference must pass before generalization.
- Recent-supply targets are weak with only 3 epochs: need dense epochs
  and a time-correlated prior.

### Svartsengi data (`literature/svartsengi_sundhnukur/`)
- Parks et al. 2025 (EPSL 658, 119324): inflow is into the *magma domain*
  (melt + mush + hot rock, 4–5 km), from GNSS with a Mogi source at 4 km
  (sill at 4.35 km in Suppl. S3). Dikes are modeled separately (Okada
  patches); dike inflow peaks (>7000 m³/s 10 Nov 2023, ~800 m³/s
  18 Dec 2023) are short outflow events.
- Volume change ≠ magma inflow (compressibility); during eruptions the
  system is open and inflow is unconstrained.
- Digitized Fig. 6 (daily volume change, 14 rate bars; slopes reproduce
  the bars to 0–4 %): inflow 7.5 → 2.4 m³/s from Nov 2023 to Jan 2025,
  and within cycles 3.99 → 3.15, 3.67 → 3.00 → 2.61, 3.21 → 2.43 m³/s.
- Paper data statement: "available on request"; processed study-series access
  is unresolved, while some station metadata are public.
