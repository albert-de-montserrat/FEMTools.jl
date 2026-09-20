# Reykjanes thermo-mechanical memory: FEMTools.jl implementation plan

Status: audited implementation proposal · 2026-09-20 · branch `adm/volcanos`
at `759153a`, including the existing dirty working tree. M0 cavity evidence is
reported below; M1 and the compressible-adjoint gate remain open.

Science plan (canonical): `reykjanes/reykjanes_thermomechanical_memory_project_v3_adjoint.typ`,
which adds the adjoint work (Q5 to Q7, Phases 5A and 5B, gates G4A and G4B) to
`reykjanes/reykjanes_thermomechanical_memory_project_v2.typ`. The v2 file and the
`.md` beside it are earlier versions and are not used here.

Basis: static reading of `src/`, `test/`, `examples/`, `docs/` and `.agents/`.
The original plan sections were written without running anything; the historical
implementation section records runs reported afterwards. The original adjoint sections (1.6, GAP-17 to GAP-24,
milestones A0 to A5) come from reading `src/stokes/solvers/DR_adjoint.jl`,
`src/stokes/assemblers/adjoint_operator.jl`, `.agents/solver.md`, the adjoint tests
and miniapps, and the deleted `ADJOINT_PERF_PLAN.md` (`git show b5fd5b5^:ADJOINT_PERF_PLAN.md`).
The earlier authors reported running two 2-D adjoint test files. This audit did
not rerun numerical tests or benchmarks. Statements marked *(verify)* are
inferences from reading the code and need a run before anything relies on them.
Historical timings are evidence for their recorded configuration, not forecasts.

## Audit findings and release blockers

The existing plan has a useful scope and gap register. Its main weaknesses were
overstated adjoint coverage, incomplete event-state semantics, and scientific
choices presented as implementation details. This revision retains the GAP/M/A
identifiers and historical measurements, but makes the following corrections
prerequisites for the affected results.

| Priority | Finding and evidence | Required resolution |
|---|---|---|
| Blocking for adjoints | `FrozenAdjointOperator` stores only A/B/C; `DR_adjoint.jl` sets the pressure adjoint residual from the momentum pullback and discards the pressure pullback's `dP_scratch`. Finite `ηb` gives a nonzero `∂R_P/∂P`. The end-to-end gradient test uses `ηb = K = G = Inf`. | GAP-25: derive and test the full compressible transpose before extending objectives or interpreting Reykjanes gradients. |
| Blocking for failure | `τ − P I` is tension-positive Cauchy stress, despite the former “compression positive” label. Plane strain also has an out-of-plane principal stress. | Use the convention contract below and analytic three-principal-stress tests in GAP-6. |
| Blocking for constitutive calibration | Science yield `C + μ(1−λ_f)P` differs from code yield `C_code cosϕ + P sinϕ`; weakening sine and cosine independently is not a friction-angle law. | Record an explicit coefficient mapping; test damaged and pore-pressure cases. |
| Blocking for cycles | Bisection and intrusion trials can change history, pressure, thermal state, mesh and counters. A Boolean solver flag is insufficient to validate a threshold. | Transactional stepping, bounded retries, residual and conservation checks, then commit once (GAP-11/12). |
| Blocking for thermal results | A one-element numerical band can be tens of times wider than the physical dike. Setting the whole band to magma temperature creates excess heat and mass. | Use physical opening to determine injected enthalpy; resolve the dike or use a conservative sub-grid treatment (GAP-4/11). |
| Blocking for attribution | Resetting fields independently can destroy equilibrium and make factorial arms differ in supply or elapsed loading. | Explicit interventions and matched controls, including a true reset-and-replay null experiment (GAP-12). |
| Planning | Old HEAD, universal 2–4% adjoint costs, full-block memory and per-step timing estimates were being used beyond their evidence. | Separate source facts, historical runs and pending gates; budget measured representative cycles and operator variants. |

**Source provenance.** The two v3 Typst files are different. This audit follows
the canonical `reykjanes/` copy named above (SHA256
`C98E7C2C1F5BB23FAEC08065D767B0EBA7EB77550A68A6FA813F373FDFFCDB0D`).
The root-level copy has SHA256
`AA31CC8C975C463A971C94D3AB33CC598A142BAB8D4D48D7B6E7830B916C2977`.
Do not silently substitute or merge them. The implementation corrections here
are proposed clarifications to the science document, not edits to that document.

**Evidence policy.** “Present” means inspected source exists; “verified” requires
a named test/run and its configuration. The 2026-09-19 measurements below are
retained as historical reports. Their cavity Julia version disagrees with
`.agents/solver.md` (1.12 versus 1.13); resolve from saved logs on the next run.
A run record must include commit plus dirty-diff hash, Julia/dependency versions,
hardware/backend/thread count, mesh and parameter hashes, command, tolerances,
convergence status, diagnostics and output location. No numerical gate was closed
by this documentation audit.

## Summary

- **Already in place.** A compressible visco-elasto-plastic host (Maxwell plus
  Drucker-Prager with viscoplastic regularisation) on unstructured 2-D T7/P1-disc
  meshes, with gravity, a linearised EOS, a volumetric source `Q` in the
  continuity equation, integration-point stress history, coupled thermal--Stokes,
  and backend-neutral kernels. `examples/stokes/volcano/volcano_thermal_stokes.jl`
  is already close to the Level 1 baseline.
- **Missing physics in `src/` (kernel level).** Plastic-state outputs and a
  per-integration-point history container (the foundation for everything below),
  damage weakening with healing, temperature- and strain-rate-dependent creep,
  latent heat and shear heating, and failure diagnostics on the stress field.
  A constitutive tensile cutoff is conditional (GAP-5).
- **Missing workflow (drivers, `examples/reykjanes/`).** A cross-rift mesher, a
  spun-up baseline model, the event-protocol engine, state snapshot/restore and
  memory switches, the memory diagnostics, and run management.
- **Key shortcut.** The eigenstrain dike opening needs *no new solver kernel*. It
  is an exact rewrite of two caller-owned inputs, `τ_old` and `Q` (GAP-11).
- **Adjoints (Phases 5A, 5B, 7).** A 2-D discrete adjoint on T7/P1-disc exists,
  with an incompressible viscous end-to-end gradient test and plastic operator
  checks. The finite-compressibility transpose is incomplete (GAP-25).
  Historical sinking-block adjoint costs were 2 to 4% of a forward solve;
  Reykjanes cost and correctness remain to be established. Also missing is
  everything around it: smooth failure and geodetic objectives with their
  `∂J/∂(v, P)` (the solver has no pressure-objective term), parameter and
  state contractions in `src/` (today only density and viscosity, inside a
  miniapp), adjoint-compatible damage and creep laws, a gradient-verification
  harness, the event-local kernel driver, a geodetic observation operator, and
  for 3-D surface load an external-load hook plus a 3-D adjoint (GAP-17 to
  GAP-24, Section 1.6).
- **Adjoint shortcut, after GAP-25.** An *event-local* adjoint (one converged step from a
  restored snapshot, history frozen) answers Q5 and Q7 with no time-dependent
  machinery, because every kernel is a function of the inherited state. It does
  not cover multi-step geodetic objectives; Section 1.6 says where that matters.
- **Riskiest unknown.** Dynamic-relaxation behaviour in the elastic-dominated,
  high-contrast regime and the noise floor of the threshold measurement, since
  the science claim rests on small differences between thresholds. Gradients
  inherit the same noise floor. This is why Milestone M0 is a spike that uses
  only existing features.
- **Not needed.** MPI, thermal advection, Winkler base, on-disk checkpoints, a
  thermal-solver adjoint, an adjoint through the event protocol's discrete steps,
  and any frontend. 3-D is only needed from month ~10.
- **Phases 0 and 1 (event table, Level 0, inference) need no FEMTools change.**

## Historical implementation and run reports (2026-09-19)

**Done: first slice of GAP-9 and GAP-10.** `examples/reykjanes/reykjanes_thermal_stokes.jl`
is the 2-D volcano driver without the cone: a 40 × 20 km cross-rift section with a
flat surface and a thin elliptical sill (`build_triangulate_t7_sill_mesh` in
`examples/triangulate_meshing.jl`), T7/P1-disc elements, geotherm, hot sill, wall
extension, and the coupled thermal--Stokes solver with Maxwell plus Drucker-Prager
rheology. It is one script, not the helper/driver split proposed under GAP-10; split
it when the event engine needs to reuse the set-up.

**Current GAP-6 slice.** `examples/reykjanes/dike_functions/dike_eigenstrain.jl`
now computes host-side per-integration-point `s3`, Drucker--Prager `F`, hydraulic
margin, principal orientation, degeneracy, and separate shear/tensile flags. The
forward harness interpolates pressure at T7 velocity quadrature points, reduces
the flags to elements, and logs shear/tensile counts, the active component, and a
shallow-target path after each accepted solve. `generate_element_adjacency` now
provides deterministic face/node graphs, and the dike helper keeps component and
shortest-path BFS routines separate. This is diagnostic-only: it is not yet a
KernelAbstractions kernel, device reduction, or event trigger.

**Not yet in it:** tectonic pre-stress (the start is `τ = 0`; deviatoric stress
builds from the wall extension), an EOS-consistent lithostatic start (the spin-up
steps absorb the mismatch, visible as −8 MPa at the base relative to the
constant-density column), a validated intrusion protocol, dike swath refinement,
and any failure diagnostics beyond an element yield ratio. The current driver has
only an opt-in first-step dike state-injection smoke path under
`examples/reykjanes/dike_functions/`; it is not a scientific event result.

**Measured** on the earlier Gmsh mesh (Julia 1.12, 16 threads, defaults: 2637 elements, 8001 velocity nodes,
Δt = 10 kyr, 10 steps): every step converges to 1e-6 with 6 200 to 10 700 inner
iterations, about 70 s including compilation. Mean τII settles at 2.94 MPa (viscous
steady state 2ηε̇ ≈ 3.2 MPa), no element yields (largest τII/yield is 0.32), and the
mean sill pressure reaches 127 MPa against 128 MPa lithostatic. With crustal
η = 1e21 the same 2 cm/yr extension gave a 28 MPa mean stress and 11 % of crustal
elements yielding, which is why the default is η = 1e20. These are the first
GAP-15 data points, all in the viscous-dominated regime (Δt far above the Maxwell
time); the elastic-dominated regime that the event cycles need is measured on the
cavity benchmark below.

The driver has two options beyond that baseline: `isCUDA` (top level) runs the solve on an
NVIDIA GPU, and `advect_mesh = true` moves the mesh with the material (Lagrangian advection
through `update_geometry!(::Mesh)`, stress rotation with the vorticity, a guard on the per-step
edge strain, and an `edge_strain_total` diagnostic that bounds the small-strain error). The
100-step advected run over about 50 % extension has not been completed, so the behaviour of the
distorted mesh at large strain is unverified. On the GPU, viscosity contrast 100 with workgroup
128 and CFL 0.99 diverged at step 1, while workgroup 64 or 256, or CFL 0.9, converged.

**M0 pressurised-cavity benchmark (done).** `examples/reykjanes/elliptical_cavity.jl` (the driver;
`solve_elliptical_cavity` is in `elliptical_cavity_setup.jl`) injects a
soft compressible elliptical inclusion (2.5 × 0.5 km, shear modulus 1e-3 of the host's, `K = 1e10`)
through `Q` in a Maxwell host and solves one elastic step on a disc whose outer circle carries the
infinite-plane displacement. This removes the truncation error for the exact
hole problem; finite inclusion stiffness and discretisation still need a radius study. The reference is
Muskhelishvili's closed form for a pressurised elliptical hole, `cavity_analytic.jl`, checked
here against the circle and crack limits, the wall traction (1e-5, the finite-difference limit),
and the area change by contour integral. The sweeps behind the numbers below are the benchmark
`examples/benchmarks/stokes/elliptical_cavity/elliptical_cavity.jl`, and the guards are
`test/test_elliptical_cavity.jl`. The benchmark itself needed no `src/` change; its one finding
that did is the `converged` fix below.

- On 2 563 elements the cavity pressure, area change and opening are within 0.3 % of the closed
  form, against a 2 % gate, for `Δt` from 1 hour to 1 kyr; the continuity balance `P = K_m (ε_inj − ΔA/A)`
  closes to 1e-6, which fixes the sign and scale of `Q`. The 0.3 % is the inclusion's own stiffness
  (it falls in proportion to its shear modulus); at 13 316 elements and a ratio of 1e-5 the three
  quantities are within 0.02 % and the displacement error is 6e-4.
- First GAP-15 measurements (tables in `.agents/solver.md`): 4 600 to 11 100 iterations from 376
  to 13 316 elements; flat while `Δt` is far below the Maxwell time (8 000 to 9 600 up to 10 yr),
  then 24 100 at 100 yr and 47 800 at 1 kyr as the material becomes nearly incompressible; about
  21 000 at a shear-modulus contrast of 1e5. Each decade of `ϵ_tol` costs about 1 750 iterations,
  and `ϵ_tol = 1e-4` already fixes the pressure to about 1e-5. Five identical runs differ by 7e-13.
- **`converged` was unreliable at the iteration cap (fixed).** `solve_stokes_dyrel!` returned
  `converged = err < ϵ`, but the inner loop overwrites `err` with the inner velocity residual, so a
  run that ended on `total_iterMax` could report `converged = true` with the outer error above `ϵ`
  (seen at 13 316 elements with contrast 1e5, and at three of twelve caps of a 3 × 3 buoyancy
  case). It is now set where the outer test passes, so a capped run reports `false`; the change is
  in `src/stokes/solvers/DR.jl` with a regression sweep in `test/test_solver_convergence_api.jl`,
  and the coupled, adjoint and cavity tests were reported passing. The flag means
  the solver stopping test passed, not that a physical threshold is valid. The
  event engine must also enforce the acceptance checks in GAP-11.
- **The stopping test depends on the units.** The absolute branch of `min(err_abs, err_rel)` lets
  the same problem stop at a relative residual reduction anywhere from 6e-6 to 0.58 as the stress
  scale moves from 1e6 to 1e11 Pa. The pressure moved by 1e-3 here only because the far-field seed
  starts close. The event engine should scale so that the increment of a step is of order one
  (`σ_c` near the overpressure, `L_c` the body size, `t_c = G Δt / σ_c`) or use a relative-only
  stop.

**Adjoint baseline (historically measured, restricted coverage).** On Julia 1.12 with the four repairs below,
`test/test_stokes_adjoint_api.jl` passes 26 of 26 (the adjoint gradient against a
central finite difference, 69 s) and `test/test_adjoint_operator.jl` passes 32 of 32
(the frozen operator against the Enzyme transpose, warm starts and spectral
bounds). The adjoint miniapps (`sinking_block_adj`,
`stokes_2D_pure_shear_triangle_adj`) and the 3-D `test_stokes_3d_reference.jl` were
not run. From the deleted performance plan: the frozen 2-D operator costs about
2.24 kB per element (T7/P1-disc, `Float64`), the adjoint takes 850 to 3000
iterations on the 1/32² and 1/64² sinking-block meshes, and one adjoint solve costs
2 to 4 % of the forward solve.

**Historical repository repair report (not a current failure inventory).** The `194c23a` merge of main into
`adm/volcanos` left the 2-D Stokes path inconsistent: `solve_stokes_dyrel!` typed its
geometry as `AbstractMatrix` while every mesh constructor stores a vector of
per-element tuples, two solver call sites used the old low-level assembler name,
`_stokes_λmin` was called with its old arity, and the pressure kernels call a
two-argument `element_geometry` that did not exist. All four are fixed in this
change, and the repository's own `test_solver_convergence_api.jl` and
`test_coupled_solver.jl` now pass. Other failures that predate it remain
(`test_stokes.jl` continuity test, four `test_type_stability.jl` cases, the 3-D
`MixedMeshCache` `geometry_precision` keyword in `test_mixed_mesh.jl` and the 3-D
volcano mesh test, and the T6 case of `test_vtk.jl`). The volcano driver still calls
`solve_coupled_dyrel!` and `assemble_viscosity_weighted_pressure_scaling!` with a
`cache` argument that no current method accepts. `AGENTS.md`, `.agents/solver.md`
and `.agents/documentation.md` referenced `ADJOINT_PERF_PLAN.md`, deleted in
`b5fd5b5`. Its numbers remain historical; current planning lives here. Recheck
the listed failures before assigning repair work; this audit did not rerun them.

## 0. Scope and ground rules

| Level | Needs finite elements? | Where it lives |
|---|---|---|
| L0 reduced-order, inference, event table | No | `reykjanes/` with its own `Project.toml`. `reykjanes_figures/level0.jl` is the seed. |
| L1 2-D cross-rift, thermo-visco-elasto-plastic | Yes | Kernels in `src/`; drivers in `examples/reykjanes/` |
| L1 adjoint: kernels, geodesy, screening (Phases 5A, 5B) | Yes | Contractions and objectives in `src/`; drivers in `examples/reykjanes/adjoint/`; inversion and statistics in `reykjanes/` |
| L2 3-D orientation selection, surface-load kernel | Yes, later | Same split |

Notation: "Gate G0 to G7" (with G4A and G4B for the adjoint phases) are the
science-plan gates; "GAP-n" are gaps in this document; "M0 to M7" are the build
milestones of Part 2 and "A0 to A5" the adjoint-track milestones. Effort is
relative for one developer including tests and docs: S is days, M is one to two
weeks, L is several weeks. These are rough and not calendar commitments.

Rules from `AGENTS.md` that shape the designs below:

1. Fix the shared cause once. Constitutive and thermal mechanics go in `src/`;
   study-specific orchestration goes in `examples/reykjanes/` and is promoted
   only when a second consumer exists.
2. Preserve existing call paths with the smallest compatible API change. An
   optional argument or typed companion is a design option, not an AGENTS rule
   or a guarantee of AD/backend compatibility. Audit constructors, dispatch,
   residuals, preconditioners and transpose paths together.
3. Backend-neutral, statically sized, `Float32`/`Float64`-preserving. No host
   scalar indexing in solver loops. Host graph work happens per event, not per
   iteration.
4. One gap is one to three small PRs, each with a focused regression check,
   docstrings, and the affected `.agents/` notes (Part 3, last section).
5. Non-convergence and invalid input fail loudly. The plan's claims depend on
   never mistaking an unconverged state for a threshold.

### Numerical and scientific convention contract

- **Stress.** `P > 0` denotes compression; `σ = τ − P I` is tension-positive.
  Define compression-positive stress `S = P I − τ`, ordered principal values
  `s1 ≥ s2 ≥ s3`, and hydraulic margin `P_f − s3 − T0`. In 2-D evaluate the
  full plane-strain tensor, including `τzz = −τxx − τyy`. Report the least
  in-plane stress separately when constraining the dike normal to the section.
  Principal directions are undefined at repeated eigenvalues: report degeneracy.
  Closure stress is `nᵀ S n`, not `nᵀ σ n`. `P_f` is an explicitly prescribed
  hydraulic connection assumption; reservoir pressure is not automatically
  available throughout every dry host element.
- **Strength.** Give science coefficients distinct names `c_y`, `μ_y` and
  `b_y = μ_y(1−λ_f)`. To reproduce `c_y + b_y P` in the current code, use
  `sinϕ = b_y`, `C_code = c_y/cosϕ`, for `0 ≤ b_y < 1`. Unsupported slopes
  need a coefficient-based law rather than an invalid angle. Recompute the
  mapping after damage; do not multiply `sinϕ` and `cosϕ` by separate weakening
  factors. If `μ` instead means `tanϕ`, document that different calibration.
  Keep dilation independent. Initially use zero dilation: the return denominator
  includes `sinΨ`, but the inspected continuity residual has no plastic-dilation
  source; nonzero dilation needs a coupled constitutive/continuity oracle first.
- **Units.** Inputs/outputs are SI; 2-D source and dike volumes are area per unit
  strike length (m²), rates m²/s, `Q` s⁻¹, specific enthalpy J/kg and heating
  W/m³. Never write m² into an observed m³ column. A 2-D/3-D comparison records
  the assumed strike length and its uncertainty. Creep temperature is kelvin.
- **Scaling.** With `x=L_c x̂`, `t=t_c t̂`, `P=σ_c P̂`, use
  `η̂=η/(σ_c t_c)`, `Ĝ=G/σ_c`, `K̂=K/σ_c`, `Q̂=Q t_c`.
  The `ηb` field enters `(P−P0)/(ηb Δt)` and acts as a bulk modulus in this
  workflow, so it scales as `K`, despite its legacy bulk-viscosity name.
  Pick scales once per run and record them; rescaling a snapshot requires all
  histories, loads and tolerances to be converted consistently.
- **Time.** Physical time, DR pseudo-time and instantaneous intrusion time are
  distinct. Only accepted physical steps update old fields, cumulative volume,
  displacement and event counters. A failed solve or trial advances none of them.
- **Pressure.** Distinguish absolute model pressure, fixed equilibrated reference
  pressure, and recharge increment since the previous arrest. Finite storage
  fixes pressure through history; do not remove its mean as an incompressible
  gauge operation. Incompressible benchmarks must specify their gauge and
  whether the objective is invariant to it.

## Part 1. Gap analysis and how to implement it

### 1.1 What the science plan asks of the library

| ID | Requirement | Plan reference |
|---|---|---|
| REQ-1 | Compressible visco-elastic host (finite `K`, `ν ≈ 0.25`), plane strain, gravity, free surface, regional extension, lithostatic pre-stress | Level 1 equations, Geometry |
| REQ-2 | Reservoir driven by volumetric injection with emergent pressure; pressurised soft inclusion for benchmarks | Approaches B and A |
| REQ-3 | Emergent failure: Drucker-Prager shear, tensile criterion `P_f ≥ σ3 + T0`, plastic connectivity from reservoir to shallow crust; regularised | Emergent failure criteria |
| REQ-4 | Damage weakening of `C` and `μ`, healing time `t_h`, accumulated plastic strain | Eq. damage |
| REQ-5 | Power-law `η(T, P, ε̇)`, brittle-ductile transition, flow-law choices | Eq. viscosity |
| REQ-6 | Conduction, latent heat, shear heating, hot dike insertion, resolved thermal aureole | Eq. energy, Phase 4 |
| REQ-7 | Event protocol: recharge, detection, path, mass-conserving intrusion, heating and material assignment, log; full state carried between events | Event protocol |
| REQ-8 | Memory switches (2^4 factorial) and branching from a common state | Phase 5 |
| REQ-9 | Diagnostics: `M_n`, `ΔP_crit`, `V_crit`, overlap `R_n`, clamping stress, stress budget, mass balance, recurrence, surface displacement | Phases 3 and 5, Q4 |
| REQ-10 | Meshes: cross-rift domain with sill and dike swath, at least three refinements; a 3-D sill mesh later | Geometry, Phase 2 |
| REQ-11 | Verification benchmarks with stated pass criteria | Code verification table |
| REQ-12 | Throughput and run management: 10-20 cycles per run, 16 factorial runs, 15+-parameter screening, GPU, provenance | Phases 5-6, Reproducibility |
| REQ-13 | 3-D: same rheology, failure function over all orientations, a few cycles; real topography, bathymetry and coastline with equilibrated initial stress; flat, no-water, hydrostatic-ocean and full-model runs | Phase 7 |
| REQ-14 | Event protocol specified outside the code, so a second code can reproduce it | Codes |
| REQ-15 | Smooth failure-proximity and pathway objectives (`J_fail`, `J_path`) with their loads `∂J/∂(v, P)`; hard `ΔP_crit` kept as the forward diagnostic | Adjoint objectives, Q5 |
| REQ-16 | Sensitivity kernels of `J_fail` to `η`, `C`, `μ`, `T`, `D` and inherited-state fields before selected events; localisation `L_n`, centroid, width, overlap with earlier dikes, seismicity and damage; comparison across factorial arms | Evolving sensitivity kernels, Phase 5A |
| REQ-17 | Gradient verification for every objective and parameter pair: directional-derivative test, several solver tolerances, consistency with the hard threshold, documented non-smooth switches | Adjoint verification |
| REQ-18 | Geodetic misfit `J_geo` (GNSS and InSAR observation operator, data covariance), source-parameter sensitivities, synthetic-truth bias experiment against an elastic source model, low-dimensional inversion with identifiability analysis | Q6, Phase 5B, Inversion hierarchy |
| REQ-19 | Surface-load kernel `K_q(x, y)` in 3-D, with the offshore load `q_s = ρ_w g h_w` applied in the forward model | Q7, Phase 7 |
| REQ-20 | Adjoint gradients as a local screen of 15+ parameters, compared with Morris or Sobol indices | Phase 6 |

### 1.2 What already exists

- **Meshes.** `MixedMesh` with T7 velocity / P1-disc pressure in 2-D, T11 or
  Hex27 with cell-local P1 in 3-D. Gmsh ingestion helpers in
  `src/mesh/utils.jl` and `examples/gmsh_meshing.jl`. Element-wise phase arrays
  (`NV × nels`) are accepted by `_gather_phase` and used by the volcano example.
- **Rheology.** Maxwell viscoelasticity and Drucker-Prager return mapping with
  viscoplastic regularisation `η_reg` (`src/stokes/assemblers/rheology.jl`).
  Stress history is stored at integration points with `stress_size=(nq, nels)`.
  `rotate_stress!` provides the objective rate in 2-D.
- **Compressibility.** Setting `ηb = K` gives an elastic bulk response, because
  the pressure residual uses `(P − P0)/(ηb Δt)` (`pressure_residual.jl`). The
  volcano example uses `K = 5e10`, `G = 3e10`, which is `ν = 0.25`.
- **Injection.** `StokesDR.Q` is the continuity source; covered by the
  `continuity volumetric source` test in `test/test_stokes.jl`.
- **Thermal.** `ThermalDiffusionDR` with per-phase `k, Cp, ρ0`, a nodal
  `source`, and `solve_coupled_dyrel!` (one thermal DR step per inner velocity
  step, thermal and Stokes on the same node numbering).
- **Boundary conditions.** Per-direction Dirichlet sets (`v_nodes`), so free-slip
  on axis-aligned walls is a Dirichlet on one component. The free surface is the
  natural traction-free boundary. `TractionBoundaryCondition` and
  `TangentialFreeSlipBoundaryCondition` are unused stubs.
- **Workflow recipes.** Characteristic-unit scaling, geotherm, lithostatic
  warm-start, pure-shear boundary seeding, `η_reg ≥ ηve`, and VTK output are all
  worked out in the 2-D volcano example. Lagrangian mesh advection exists
  (`update_geometry!`, `stokes_2D_pure_shear_triangle_adv`).
- **Adjoints.** `solve_stokes_adjoint_dyrel!` (2-D, `src/stokes/solvers/DR_adjoint.jl`)
  targets `R_uᵀ λ = −J_u` for one *converged* forward step on T7/P1-disc, with the
  same Powell-Hestenes/DYREL iteration and homogeneous adjoint Dirichlet
  conditions. The linearisation is frozen at that state (`τ_old`, `plastic`,
  phases and `T` are fixed inputs). Three operators: `:blocks` (default; dense
  per-element `A`, `B`, `C` from ForwardDiff, 2.24 kB per element for full blocks,
  1.176 kB for the current packed symmetric layout, handles a plastic
  tangent), `:matrix_free` (needs `plastic === nothing`) and `:enzyme` (stores
  nothing, handles plastic, slowest). `StokesAdjointWorkspace` reuses scratch and
  warm-starts across solves. The caller supplies `objective_vx` and `objective_vy`;
  there is **no pressure-objective term** (`ResλP` is overwritten with `Bᵀλ_v`),
  and no pressure-storage transpose term (GAP-25). Existing operator agreement
  does not establish a correct compressible adjoint.
  The 3-D adjoint is only the caller-owned, linear viscous Hex27 path
  (`stokes_material_gradient_3d`, density and viscosity per phase); the
  `StokesDR`/`MixedMesh` T11 visco-elasto-plastic path has none. Per-element
  density and viscosity sensitivities exist only as `material_sensitivities` inside
  `examples/miniapps/stokes/sinking_block_adj/`, differentiating the momentum
  residual with Enzyme and no plastic model. Sign convention:
  `dJ/dm = J_m + λᵀ R_m`.

### 1.3 Equation-by-equation coverage

| Plan item | FEMTools today | Status |
|---|---|---|
| Momentum balance, gravity, EOS body force | `StokesDR` momentum residual, `eos_density` | Covered |
| Mass balance with finite `K` and source `Q_m` | Pressure residual with `ηb = K` and `dr.Q` (2-D) | Covered in 2-D; 3-D `Q` *(verify)* |
| Strain-rate partition (elastic, viscous, plastic) | Maxwell coefficients plus DP return | Covered |
| Yield with pore-pressure ratio `λ_f` | `F = τII − C cosϕ − P sinϕ`; science coefficients differ | Requires the explicit convention mapping above |
| Tensile cutoff | None | Missing (GAP-5, GAP-6) |
| Damage `D`, weakening, healing | `C`, `ϕ` are per-phase constants | Missing (GAP-1, GAP-2) |
| Accumulated plastic strain | Plastic multiplier `λ` is computed and discarded | Missing (GAP-1) |
| Power-law `η(T, P, ε̇)` | `η` is a per-phase constant tuple | Missing (GAP-3) |
| Conduction | `ThermalDiffusionDR` | Covered |
| Shear heating | Nodal `source` exists; no `τ:ε̇` projection | Partial (GAP-4) |
| Latent heat | None | Missing (GAP-4) |
| Advection | None; the plan drops it while motion is small | Deferred |
| Extension BC, free surface, free-slip base | Dirichlet per direction, natural surface | Covered |
| Winkler base | None | Deferred (GAP-8) |
| Lithostatic initial state | Analytic column or `LithostaticPressureDR`; the deviatoric stress starts at zero | Workflow (GAP-10) |
| Reservoir injection | `Q` | Covered |
| Dike opening (eigenstrain) | None as a term | Missing, but needs no kernel (GAP-11) |
| Failure diagnostics on stress | Element-averaged `ε̇`, `τ`, `τII` only | Missing (GAP-6) |
| Surface displacement | Velocity only | Missing (GAP-13) |
| Adjoint of a converged 2-D VEP step | Incompressible viscous gradient test; plastic operator checks | Finite-compressibility coverage blocked by GAP-25 |
| Smooth objective loads `∂J/∂v`, `∂J/∂P` | Caller-assembled `objective_vx`, `objective_vy`; box-integral loader is miniapp-local; no `∂J/∂P` | Missing (GAP-17) |
| Sensitivity to `η`, `ρ0` | Per element, miniapp-local, momentum residual only, no plastic | Partial (GAP-18) |
| Sensitivity to `G`, `K`, `ηb`, `α`, `C`, `ϕ`, `T`, `τ_old`, `Q`, `D` | None; `K`, `ηb`, `α`, `Q` also need the pressure-residual term `λ_Pᵀ ∂R_P/∂m` | Missing (GAP-18) |
| Adjoint with damage or creep laws | Adjoint keeps a constant-`η` contract; `D` and creep do not exist yet | Blocked on GAP-2, GAP-3 (GAP-19) |
| Gradient tests | Central finite difference on a 3 × 3 mesh (69 s); no Taylor-remainder utility | Partial (GAP-20) |
| Thermal adjoint | None | Not needed event-locally |
| Multi-step adjoint | None; the solver linearises one step | Deferred (GAP-24) |
| Surface traction, external load | 2-D/mixed `StokesDR` has none; the caller-owned 3-D path has `load`; traction types are stubs | Missing (GAP-8, GAP-23) |
| 3-D adjoint on the T11 VEP path | None | Missing (GAP-23) |

### 1.4 Gap register

| Gap | Title | Layer | Size | Needed for | Needed by (month) |
|---|---|---|---|---|---|
| GAP-1 | Plastic-state outputs and per-IP history | `src/stokes` | M | REQ-3, 4, 9 | 4 |
| GAP-2 | Damage weakening and healing | `src/stokes` | M | REQ-4 | 5 |
| GAP-3 | Temperature and strain-rate-dependent creep | `src/stokes` | M-L | REQ-5 | 8 |
| GAP-4 | Latent heat and shear heating | `src/heat_diffusion`, `src/postprocess` | M | REQ-6 | 8 |
| GAP-5 | Tensile cutoff (conditional) | `src/stokes` | M | REQ-3 | 5 if triggered |
| GAP-6 | Failure diagnostics kernel and element adjacency | `src/postprocess`, `src/mesh` | M | REQ-3, 7 | 3 |
| GAP-7 | 3-D enablement | `src/stokes`, examples | L | REQ-13 | 10 |
| GAP-8 | Surface-load hook (S, needed) and Winkler base (deferred) | `src/stokes` | S / M | REQ-13, 19; REQ-1 (Winkler) | 11 (load); Winkler not scheduled |
| GAP-9 | Cross-rift mesher | `examples/reykjanes` | M | REQ-10 | 2 |
| GAP-10 | Level 1 baseline set-up and spin-up | `examples/reykjanes` | M | REQ-1, 2 | 2 |
| GAP-11 | Event-protocol engine | `examples/reykjanes` | L | REQ-7, 14 | 5 |
| GAP-12 | Snapshot/restore and memory switches | `examples/reykjanes` | S-M | REQ-7, 8 | 5 (switches 10) |
| GAP-13 | Memory diagnostics and outputs | `examples/reykjanes` | M | REQ-9 | 5 |
| GAP-14 | Run management, sweeps, Level 0 promotion | `reykjanes/` | M-L | REQ-12 | 12 |
| GAP-15 | Solver behaviour study | study | M-L | all | starts month 1 |
| GAP-16 | Verification suite | `test/`, examples | L | REQ-11 | 5 |
| GAP-17 | Adjoint objectives: `J_fail`, `J_path`, `objective_P` | `src/stokes`, `src/postprocess` | M | REQ-15 | 10 |
| GAP-18 | Parameter and state contractions in `src/` | `src/stokes/assemblers` | M | REQ-16, 18 | 10 |
| GAP-19 | Adjoint contract for damage, creep and cutoff | rides with GAP-2, 3, 5 | S-M | REQ-16, 17 | with each |
| GAP-20 | Gradient verification harness and threshold consistency | `test/`, examples | M | REQ-17 | 10 |
| GAP-21 | Event-local adjoint driver and kernel diagnostics | `examples/reykjanes/adjoint` | M-L | REQ-16, 20 | 13 |
| GAP-22 | Geodetic operator, synthetic truth, low-dimensional inversion | `examples/reykjanes`, `reykjanes/` | M-L | REQ-18 | 11 (first stage) |
| GAP-23 | 3-D surface-load kernel: 3-D adjoint and traction | `src/stokes`, examples | L | REQ-19 | 14, optional |
| GAP-24 | Windowed (multi-step) adjoint | `src/stokes`, examples | L | REQ-18 (if needed) | decide at G4A; not scheduled |
| GAP-25 | Complete compressible 2-D adjoint and independent oracle | `src/stokes`, `test/` | M-L | REQ-15 to REQ-20 | A0, before GAP-17/18 |

"Needed by" is taken from the Gantt in the science plan (Phase 2 starts month 2,
Phase 3 month 5, Phase 4 month 8, Phase 5 month 10, Phases 5A and 5B month 11,
Phase 6 month 12, Phase 7 month 11) and means the work must be *finished* before
that phase can start. The adjoint gaps are needed earlier than Phase 5A itself
because the science plan builds the adjoint tooling alongside Phases 2 to 4.

### 1.5 Gap details

#### GAP-1. Plastic-state outputs and per-integration-point history (foundation)

**First slice implemented.** The 2-D return is factored through one internal
stress-and-multiplier path, and the public unexported `plastic_multiplier`
diagnostic returns the same `λ` that drives the existing stress correction.
Per-integration-point history allocation and the once-per-step update now have
an internal foundation: `IntegrationPointPlasticHistory` owns `(λ, εpl, D)`
arrays and `update_plastic_history!` updates `εpl` after an accepted step. The
container is not yet owned by `StokesDR`, and the current slice deliberately
does not change the inner DR loop.

*Have.* `τ` and `τ_old` live at integration points. The DP return computes the
multiplier `λ` and throws it away (`deviatoric_stress` in `rheology.jl`).
`IntegrationPointStressOutput` already shows how a kernel writes per-IP data.

*Need.* Add the damage update. The diagnostic output now writes `λ` and
accumulates `Δt ε̇_pl` using the explicit plane-strain `J₂`-equivalent
conversion, pinned by simple shear. The standalone history updater remains a
fallback for callers that only have raw multiplier fields; `D` is carried
unchanged until GAP-2.

*Design.*
- Factor the DP return so one internal function returns `(τ, λ)`; the existing
  `deviatoric_stress` methods call it and drop `λ`. One implementation, no
  duplicated mechanics, preserving existing pointwise behavior.
- Add an isbits history bundle pinned to an element, mirroring
  `IntegrationPointStressOutput(τ, iel)`. It is passed **only** in the single
  diagnostic pass that `update_stokes_current_stress!` already runs after a
  converged step, never inside the DR iterations. This avoids write races and
  keeps the iteration cost unchanged.
- A small KernelAbstractions kernel, run once per converged step, updates
  `ε_pl` from `λ` and the flow direction, and `D` (GAP-2). Use the package's
  invariant convention when converting `λ` to a plastic strain-rate invariant,
  and pin it with a simple-shear test.
- Memory: two or three `nq × nels` arrays.

*Check.* `λ = 0` below yield; `λ` equals the hand-computed return for a single
simple-shear point; existing DP tests unchanged; zero allocation; `Float32`; the
2-D and 3-D adjoint tests stay green.

*API.* Start with the narrowest type/helper visibility needed by the driver;
export a history type or update function only if the API review justifies it.
Separate read-only inherited `D` used during residual evaluation from diagnostic
output and once-per-step history mutation. The device-side element view may be
isbits; a host bundle owning ordinary arrays need not be.

#### GAP-2. Damage weakening and healing

**First foundation slice implemented.** `DamageLaw` validates the lagged
per-phase damage parameters, `weakened_drucker_prager_parameters` implements
the cohesion/friction residual-strength mapping, and `update_damage!` applies
the implicit clamped healing/update formula. The 2-D momentum diagnostic path
now accepts optional per-IP `damage_old`, and `update_stokes_current_stress!`
can difference accepted `ε_pl` and update `D` with matching `(εc, th)` arrays;
ordinary calls remain unchanged. Focused tests now cover pure healing and
no-healing saturation. `damage_update_parameters` derives those arrays from a
validated phase-index matrix, and the optional lagged weakening now has 2-D and
3-D pointwise/residual dispatch coverage.

*Need.* `C = C0 [1 − (1 − f_C) D]`, `μ = μ0 [1 − (1 − f_μ) D]`,
`dD/dt = ε̇_pl/ε_c − D/t_h`, with `D ∈ [0, 1]`.

*Design.*
- The yield law reads inherited `D` at the IP, weakens the science cohesion and
  friction coefficients, then applies the convention mapping above. Per-phase
  parameters `(ε_c, f_C, f_μ, t_h)` travel as an
  optional companion of `DruckerPrager`, so the current constructor is unchanged.
  `f_C = f_μ = 1` recovers today's stress exactly.
- Use **lagged** `D`: the value from the last converged step is used inside the
  DR iterations, so the inner problem gains no new nonlinearity. Update `D` once
  per step with the implicit form
  `D⁺ = (D + Δε_pl/ε_c) / (1 + Δt/t_h)`, then clamp to `[0, 1]`.
  `t_h = ∞` gives plain plastic-strain weakening.
- Mirror the change in the 2-D and 3-D `deviatoric_stress` methods; they use
  different flow directions (documented in the 3-D docstring).
- Adjoint contract (GAP-19): the lagged `D` is a fixed input of the step, so its
  kernel `K_D` is one more contraction (GAP-18) and the `D` update stays outside
  the differentiated residual.

*Check.* For constant `r = ε̇_pl/ε_c`, compare step refinement with
`D(t) = D(0) exp(−t/t_h) + r t_h [1−exp(−t/t_h)]` before clipping;
verify first-order accuracy of the proposed implicit update, pure healing,
`t_h = ∞`, and saturation at 1. Validate `ε_c > 0`, `t_h > 0` or infinity,
and residual strength fractions in `[0,1]`. Weakening off reproduces current
stress; ForwardDiff element Jacobian
against finite differences at `D ≠ 0`; one-element softening test.

*Risk.* Softening localises and is mesh dependent, which is exactly Gate G1.
Treat GAP-15 as part of this gap.

#### GAP-3. Temperature and strain-rate-dependent creep

*Have.* `η` is a per-phase `NTuple` interpolated at IPs. `freeze_jacobian`
defaults to `plastic === nothing`.

*Need.* A calibrated viscous flow law and clamps `η_min, η_max`, with per-phase
flow-law tables (wet and dry diabase). Freeze the stress/strain invariant and
prefactor convention first: if `ε̇_vis,II = A τII^n exp(−(E+PV)/(RT))` and
`τ = 2η ε̇_vis`, then
`η = ½ A^(−1/n) ε̇_vis,II^((1−n)/n) exp((E+PV)/(nRT))`.
The science document's missing `½` must be absorbed into its definition of `A`
or corrected; do not combine that formula with laboratory coefficients blindly.

*Design.*
- First audit `StokesMaterial` and `StokesDR`, which currently store `η` as an
  `NTuple`; a new viscosity object cannot simply be inserted into those fields.
  Prefer a typed optional rheology companion if that preserves the existing path.
  Dispatch inside `viscoelastic_coefficients_phase` on the viscosity model:
  the existing `NTuple` path is untouched; a new `PowerLawCreep{nphases, FP}`
  evaluates the law. `T_q` and `P_q` are already computed in the momentum kernel
  for the EOS and only need forwarding.
- Solve the nonlinear Maxwell constitutive relation, not merely the viscous law
  evaluated at total effective strain rate. For a coaxial no-plastic point,
  `e_eff = ε̇' + τ_old/(2GΔt)` obeys
  `|e_eff|_II = τII/(2GΔt) + A τII^n exp(−(E+PV)/(RT))`.
  A bounded scalar local solve gives `τII`; obtain a consistent tangent by
  implicit differentiation or a verified differentiable local solve. Plastic
  coupling needs its own residual/return consistency check. Simply substituting
  `|e_eff|` into the dashpot viscosity is an approximation, requiring validation.
  Specify the zero-rate limit, positive temperature, pressure convention, clamp
  derivatives and overflow-safe evaluation of the Arrhenius factor.
- The problem becomes nonlinear, so the Jacobian is no longer frozen when a
  creep phase is present. `assemble_viscosity_weighted_pressure_scaling!` takes
  `η` per phase and needs a positive representative value. Use a documented
  lagged effective viscosity for preconditioning, then test convergence and
  rebuild when coefficients or `Δt` change. It must not redefine the physical law.
- The adjoint must differentiate the creep law, because Phase 5A asks for `K_T`
  through the rheology. Evaluated inside the element residual, the law is seen by
  `:blocks` and `:enzyme` only after local-solve AD and tangent checks. `:matrix_free` keeps requiring
  `plastic === nothing` and must also reject a creep phase until its tangent is
  shown symmetric. Until this lands, `K_T` acts only through the EOS density
  (GAP-19).
- The alternative is a lagged IP viscosity array refreshed at each `ncheck`.
  Prefer inline evaluation first; switch only if convergence demands it (D5).

*Check.* Analytic power-law simple shear; `n = 1` limit; residual against AD
Jacobian; inference and allocations; a DR convergence run with a 10³ to 10⁶
temperature-viscosity contrast before this feeds any result.

#### GAP-4. Latent heat and shear heating

*Have.* Nodal `source` in the thermal residual; constant per-phase `k`, `Cp`.

*Need.* Freezing of the dike with latent heat, and `H_sh = τ_ij ε̇_ij` for the
inelastic strain rate. Advection stays dropped, as in the plan.

*Design.*
- Latent heat: optional per-phase `(L, T_solidus, T_liquidus)` in
  `ThermalMaterial`, default `L = 0` selecting today's residual. Use the
  **enthalpy form** `ρ(H(T) − H(T0))/Δt`, not a temperature-dependent `Cp`
  multiplying `(T − T0)`, because only the enthalpy form conserves energy
  across the freezing interval. ForwardDiff already differentiates the thermal
  residual, so the Jacobian follows.
- Shear heating: compute viscous plus plastic dissipation, excluding recoverable
  elastic power. A pure elastic loading test must not generate viscous heat.
  Project IP power conservatively using the thermal residual's actual source
  weights; high-order lumping can have zero/negative weights and must be checked.
  Freeze a lagged source for each physical step or iterate a documented coupled
  fixed point while keeping old histories fixed. A one-step lag needs temporal
  convergence and a discrete energy-budget check.
- The current thermal residual normalises by `ρ Cp` and uses nodal transient
  values, so an enthalpy formulation is a discretisation change, not a new `Cp`
  field alone. Specify mass weighting, phase-interface treatment and boundary
  heat flux; account for changing density consistently in the energy ledger.
- Hot dike insertion is a state assignment (GAP-11), not a solver feature.

*Check.* Stefan (Neumann) freezing-front position within 2%; energy conservation
in an insulated box; `L = 0` leaves all thermal tests unchanged.

*Deferred.* Thermal advection, temperature-dependent `k` and `Cp`. Record a
Péclet/displacement estimate before dropping advection in a paper figure.

#### GAP-5. Tensile cutoff (conditional)

*Have.* Nothing; the DP surface alone caps tension at `C cotϕ`.

*Options.*
1. **Event-level only.** The plan defines `T0` as an initiation criterion
   (`P_f ≥ σ3 + T0`). Evaluate it as a diagnostic in GAP-6 with no constitutive
   change.
2. **Constitutive.** Two-surface DP with a tension cap and corner return.

*Recommendation.* Option 1 for Phase 2. Do option 2 only if Gate G1 shows
tensile stresses above `T0` controlling `ΔP_crit` or its mesh sensitivity. It is
the highest-risk numerical item in `rheology.jl` because of the corner regime
and the plane-strain flow direction (see the 3-D docstring). It also adds a second
kink to the differentiated law (GAP-19); option 1 leaves the adjoint untouched.

*Check for option 2.* Uniaxial-tension point test; continuity of the return at
the DP-cap corner; Jacobian against AD.

#### GAP-6. Failure diagnostics kernel and element adjacency

*Have.* Element-averaged strain-rate and stress diagnostics on the host
(`compute_strain_rate_stress_postprocess`), and `generate_node2element`.

*Need.* Stress-derived quantities at every IP, on the device, at every recharge
step: tension-positive `σ = τ − P I` and compression-positive `S = −σ`, least
compressive principal stress `s3` with the failure-plane orientation, the yield function
`F`, a `yielded` flag, and the hydraulic margin `P_f − s3 − T0` for a supplied
magma pressure.

*Design.*
- A KernelAbstractions kernel over IPs writing `nq × nels` arrays. Return trial
  margin, post-return margin and plastic multiplier separately: post-return
  `F ≥ 0` alone is not a reliable branch indicator. Set an explicit tolerance
  and compare IP maximum with yielded-area-fraction reduction under refinement.
  Define combined detection (`shear OR tensile`, or a stricter joint/path rule)
  as a named protocol option; the science plan does not specify its Boolean logic.
- Add `generate_element_adjacency(el2n; shared = :face)` in `src/mesh/`. The
  graph search should default to shared faces so a path cannot leak through a
  corner.
- **Two-stage detection.** Every step, a device reduction answers "is any
  element yielded or at tensile margin?" Only when it is true, copy the flags to
  the host and run a breadth-first search from the reservoir-boundary elements
  toward the target depth or surface. Keep the full connected component and the
  selected opening path as distinct outputs. BFS certifies connectivity; its
  shortest-hop path is not a fracture-energy criterion. Use deterministic
  geometric tie-breaking and audit alternatives. Transfer/search can occur at
  every accepted recharge step after yielding begins, not only once per event.
  Transfers and the graph search then happen per event, not per iteration.
- The extent criterion ("reaches the shallow crust") needs a parameter,
  `reach_depth`, fixed in Phase 2 and reported with the results.

*Check.* Analytic stress states (uniaxial, hydrostatic, pure shear) for `σ3` and
`F`; BFS on a tiny structured mesh with a known path; face versus corner
adjacency; zero kernel allocations; host copy behaviour on a non-CPU backend
when hardware is available.

#### GAP-7. 3-D enablement (Phase 7)

- **Injection.** `Q` is documented as not carried by the *caller-owned* Hex27
  residual. The `StokesDR`/`MixedMesh` 3-D path calls the generic pressure
  kernel, which does take `Q`, but its meaning for the four modal P1 pressure
  coefficients needs a test *(verify)*.
- **Constitutive parity.** Repeat GAP-1, GAP-2, GAP-3 in the 3-D
  `deviatoric_stress`.
- **Orientation failure function.** A kernel that evaluates normal and shear
  traction on a fan of plane normals and returns the Coulomb margin maximum and
  its normal. This answers Q3 without a prescribed trajectory.
- **Mesh.** Adapt `examples/stokes/volcano/volcano_mesh_3D.jl` (T10 fragmented by
  an ellipsoid, then T11) to an oblate sill. The science plan now also asks for real
  topography, bathymetry and the coastline: build the surface from a DEM in the
  examples environment (GMT and GeophysicalModelGenerator are already dependencies
  of the branch), split the surface boundary group into land and sea faces, and
  keep the sea-face list for the ocean load (GAP-8).
- **Equilibrated initial stress with topography.** A 3-D spin-up solve, as in
  GAP-10, since a lithostatic column is wrong under relief.
- **Run hierarchy.** Options for the four stages of Phase 7: flat surface; real
  geometry without water; real geometry with the hydrostatic ocean load; full
  tectonic plus magmatic model.
- **Budget.** Treat reported divergence at a particular mesh size as
  configuration-specific. Benchmark element quality, domain scaling, contrast,
  spectrum and CFL together; there is no established universal 4.4 km limit.

#### GAP-8. Surface-load hook (needed) and Winkler base (deferred)

*Surface load.* Phase 7 applies the offshore load `q_s = ρ_w g h_w` on sea-covered
surface faces. A load that does not depend on the solution needs no boundary-face
kernel: assemble the nodal force vector `f_ext = −Σ_faces ∫ Nᵢ q_s n dΓ` (outward
normal `n`, so the water pushes into the crust) once on the host from the face
list, and apply it to the momentum residual. The caller-owned 3-D path already has
this shape (`load`, subtracted from the velocity residual in `solve_stokes_dyrel!`).
Give the `StokesDR` 2-D and 3-D paths an optional `load` with the same meaning,
default `nothing`, and fix the sign with a hydrostatic-column test. The adjoint
contraction is then a host gather of `λ` over the face nodes (GAP-23), which is why
this is preferred to a constitutive traction type. Size S plus the host
face-quadrature helper; needed by month 11.

*Winkler base.* The plan tests a free-slip or Winkler base. Free-slip is a Dirichlet
condition today. A Winkler base is solution-dependent, so it needs a boundary-face
integral (a Robin term), which the stub types do not implement. Test the influence
of the base by varying domain depth with a free-slip base; build the boundary-face
machinery only if that test shows a Winkler base matters.

#### GAP-9. Cross-rift mesher (examples)

Reuse `build_triangulate_t7_sill_mesh` (current driver) and the existing Gmsh
sill alternative. Introduce `examples/reykjanes/mesh_2D.jl` only as a small
study-specific wrapper when swath refinement or metadata warrants it:
a 40-60 km by 15-25 km rectangle, an elliptical sill at 4-5 km depth, and a
vertical refinement swath above it graded to 50-100 m, with boundary groups.
Emit three resolutions for the convergence gate. Also emit element region tags
and centroid arrays for the event engine. Follow `test/test_volcano_mesh.jl`
for a conformity check (reservoir interface, phase separation).

#### GAP-10. Level 1 baseline set-up and spin-up (examples)

- Characteristic units from the volcano recipe, with the elastic time scale in
  mind; SI in and out.
- Geotherm and reservoir phase; extension boundary condition ramped from zero.
- **Spin-up to an equilibrated initial state.** The volcano example warm-starts
  `P` and starts `τ = 0`. The plan needs a discretely equilibrated `(τ, P)` under
  gravity and lateral confinement, with the tectonic deviatoric stress as the
  parameter `Π_σ`. Do this with a spin-up solve, not by pasting the lithostatic
  pressure.
- Injection helper: convert a per-length rate `V̇_m` (m²/s) into `Q` (1/s)
  spread uniformly over the reservoir pressure DoFs using the pressure
  geometry weights. **Implemented** in `examples/reykjanes/injection_source.jl`
  as `uniform_pressure_source` and `pressure_source_integral`, the latter
  reproducing the quadrature of the continuity residual. On the 442-element
  cavity mesh, `test/test_elliptical_cavity.jl` checks the reservoir area
  against the corner-triangle shoelace area, the sum of the assembled continuity
  residual at rest against the requested rate (relative difference 2e-16), and
  exact cancellation of an `add_dike_source!` band by its sink, which is the
  GAP-11 sink normalisation. The driver does not call it yet: its only source is
  the first-step dike smoke path. The sign and physical scale of `Q` rest on the
  M0 cavity closure, not on this helper.
- Contract: `main(; kwargs...) -> NamedTuple`, headless, explicit output path.
- Initial velocity satisfies the BCs. A divergence-free seed is useful for a
  compatible no-source benchmark, not a requirement on compressible injection.
  The stopping rule mixes absolute and relative errors; use characteristic
  units and independent residual checks. `η_reg ≥ ηve` is a stabilization recipe
  from examples, not a physical law. Keep physical `η_reg` fixed in a time-step
  convergence study; if stabilization instead requires it to track `Δt`, record
  that altered model and perform a joint limiting study (GAP-15).

#### GAP-11. Event-protocol engine (examples, the core deliverable)

**Steps 1 to 3 run end to end**, in `examples/reykjanes/event_protocol.jl` (the rules),
`event_cycles.jl` (`run_cycles`) and the driver `reykjanes_cycles.jl`, on the
machinery below. Every rule is a field of `EventProtocol`, gathered in one object
so that a run records the protocol it used and the sensitivity of a threshold to
a rule can be measured by changing one field. **All of them are placeholders
until Phase 0 fixes them**, and the values below were chosen to make the
machinery run, not from data. Measured on the coarse 284-element section with a
recharge of 2e-7 m² s⁻¹ and 1 kyr steps: the reservoir rises from 123.1 MPa, a
connected failed path reaches the shallow crust during the third step, the
crossing is bracketed at 2.953 kyr, and a 5-element path 1 544 m long with a
closure stress of 71.09 MPa is opened by 0.154 m, which drains the magma
pressure at the path to 72.05 MPa against a target of 72.09. After it, the failed
path is gone (5 elements to 0) and the count of failed elements falls from 89 to
82: the intrusion relieves the system. Recharge rebuilds it and the second event
follows at 4.001 kyr, with the closure stress now 71.69 MPa. That is a repeated
cycle, not yet a memory measurement: one mesh, no refinement study, no null test.

Three findings from getting it to run, each of which changes a rule rather than
the code:

1. **The detector needs a magmastatic head and a nonzero tensile strength.** With
   the reservoir pressure applied everywhere and `T₀ = 0`, a connected path from
   the sill to the surface exists in the undisturbed state, so the run declares an
   event on its first step. `P_f(y) = P_res − ρ_m g (y − y_res)` and `T₀ = 5 MPa`
   give a failed region that nucleates at the reservoir (38 of 284 elements) and
   has to grow to reach the target depth. Note that `ρ_m < ρ_crust` makes the
   magma buoyant enough to reach the surface on its own, so `T₀` is what holds
   the criterion shut, and the threshold will be sensitive to it.
2. **The arrest condition must be written on the reservoir, not on the band.**
   Driving the band's own closure traction to `s_n + ΔP_arrest` opens almost
   nothing (3 mm) and re-trips at once, because the ambient traction across the
   dike normal already sits within 0.5 MPa of `s_n`: that rule makes the opening a
   function of `ΔP_arrest` and the path compliance alone. Draining the magma
   pressure at the path to the same value opens 0.15 m and relieves the criterion.
   Both are available as `arrest_target`; the second is the default and is what
   the plan's wording says.
3. **The intrusion step is bounded from below by the scaling, not by physics.**
   The model's characteristic time is its tectonic one (2 Myr), so a one-hour
   intrusion step is `Δτ ≈ 6e-11` and the relaxation does not converge at all. One
   year and ten years give the same band traction to 0.3% at 1 900 and 1 700
   iterations; a century differs by 2.4%, where `Δt/t_M ≈ 1` and the response is
   no longer elastic. The temporal-insensitivity check the plan asks for therefore
   has a window, and `intrusion_Δt` defaults into it.

Not done: steps 4 and 5 — the contact rule, the enthalpy of the transferred
magma and the full per-event record — which need the thermal work of M3, and
every acceptance check of the step contract beyond convergence and finiteness.
`test/test_event_protocol.jl` guards the rules; the cycle itself is a miniapp,
because it costs minutes.

**Gate G1 is failing, and the reason is the detector.**
`examples/benchmarks/stokes/first_threshold/first_threshold.jl` runs the cycle to
its first event and reports `ΔP_crit`, the rise of the reservoir pressure over its
spun-up baseline. Across 228, 284, 492 and 928 elements it gives 2.899, 2.877,
5.200 and 6.218 MPa: a spread of 77.7% against the gate's 5%. Table and settings
are in `.agents/solver.md`.

This is not solver noise and not the crossing search: every bracket is
±0.016 kyr, every search resolved, and the two runs at sill refinement 3 agree to
0.8% while finding the *same* path (5 elements, 1 544 m). The jump comes with the
sill refinement, where the path becomes 6 and then 8 elements and its length moves
non-monotonically (1 544, 1 839, 1 105 m). The threshold is the moment a
face-connected chain of failed elements first spans from the reservoir to the
target depth, and that is a property of the mesh graph: refining changes which
elements exist, so it changes when a chain closes, even where the stress field has
converged. Section "Error budget and gate evidence" already required refining
detector aggregation and path selection separately from the PDE mesh; this
measurement shows that term dominates the budget rather than contributing to it.

Consequence for the order of work: **no `M_n`, no repeated-intrusion production and
no factorial arm can be read until the detector is made mesh-independent and this
sweep is repeated.** The candidate replacements are an aggregation over a
mesh-independent measure — a failed-area fraction, or a path through a fixed
geometric corridor — in place of one over element adjacency. The `strength`,
`reach`, `arrest` and `band` sweeps of the same benchmark exist to quote protocol
sensitivity separately, and must not be confused with this convergence failure.

**The replacement: a fixed detection corridor.** `EventProtocol.detector` now names the rule, and
its default is no longer the graph. The corridor is a vertical strip of host rock,
`|x − x_sill| ≤ corridor_width / 2`, running from the reservoir crest up to `reach_depth` and cut
into `corridor_bins` bins of equal height. Every one of those numbers is in metres and fixed before
the mesh exists; the reservoir's own elements are excluded, since they are magma and not rock that
can fail. A bin's occupancy is an area fraction — the failed area the mesh put in the bin over the
area it put there at all — and each element contributes the fraction of *its own quadrature* that
failed, not a Boolean. The element reduction is a coarsening on top of the mesh, and at these
resolutions a bin holds a handful of elements, so a Boolean per element would leave the measure as
quantised as the chain it replaces.

Two rules read that corridor. `:corridor_column`, the default, trips when every bin is filled to
`corridor_fill`: the geometric statement of the same physical claim the chain was making, that
failed rock spans from the reservoir to the target depth. `:corridor_fraction` trips when the
corridor as a whole is that full; it is smoother in time, and it does *not* require the failure to
have reached the shallow end, so it answers a weaker question and is kept as a comparison rather than
a default. `:graph_path` keeps the adjacency rule, so that one sweep can quote all three. The band
the intrusion opens is, under either corridor rule, the failed elements of the bins that counted:
the same kind of set as the old path, chosen by geometry instead of by a graph search.

Two consequences follow from the geometry. A bin no element falls in makes the corridor
`under_resolved`, and an under-resolved corridor never trips: a bin the mesh cannot sample is missing
evidence, not absence of failure, and a run that stalls that way is a resolution failure to be read
as one. And a `reach_depth` at or below the reservoir crest leaves the corridor no height at all;
that is an `ArgumentError` at detector construction, not a row, which is why the `reach` sweep now
stops at 3 km on a section whose crest is at 4 km.

What this did and did not fix, measured. The property the detector was built for holds: the same
failure field on meshes whose element size differs by a factor of four gives identical bin fractions
and the same verdict (`test/test_dike_functions.jl`), the two meshes at sill refinement 3 now agree
to 0.015% where the adjacency rule gave 0.8%, and at 2 684 elements the corridor and graph rules
agree to 0.3% — the two detectors converge to the same threshold, so their disagreement on coarse
meshes was resolution, not rule. **Gate G1 is still failing, at 45.6% against 77.7%**, and the term
that now dominates is not the detector but the conditioning of the observable: the event pressure
moves 0.48% across the four meshes and the spun-up baseline 1.56%, while their difference moves
45.6%, an amplification of `P/ΔP ≈ 25` to `30`. The detector work was necessary and is not
sufficient; the baseline is the next lever, and `.agents/solver.md` carries the table.

One thing the rules do *not* share is what gets opened. At 2 684 elements the graph rule opens a
10-element chain with `s_n` 60.08 MPa needing 0.307 m, the corridor rule a 74-element band with
`s_n` 76.75 MPa needing 0.200 m. The thresholds agree; the intrusions do not. M2 must therefore fix
one rule and keep it, rather than treating the two as interchangeable once the event has been
declared.

One driver, `run_cycles(; kwargs...) -> NamedTuple`, implementing the five steps
of the science plan. Keep it in `examples/reykjanes/` until a second consumer
exists. Dike-specific transformations and reusable dike helpers belong in
`examples/reykjanes/dike_functions/`; the driver includes them explicitly when
the event workflow starts using them. They remain example-level code until a
second maintained workflow needs the same contract.

1. **Recharge and relaxation.** Inject `Q`, advance thermal and Stokes with
   `solve_coupled_dyrel!`, then `update_stokes_current_stress!` and the history
   update. Use adaptive `Δt` (days near expected failure, larger where only slow
   processes evolve). Run the two-stage detection of GAP-6 each step. When the
   criterion first trips, restore the pre-step snapshot (GAP-12) and bisect the
   step until the crossing is bracketed to a set tolerance. Each trial begins
   from the same accepted left state. A connected detector need not be monotone
   under healing, stress redistribution or path switching: first sample/subdivide
   the interval and retain the earliest resolved crossing; report unresolved
   multiple crossings rather than trusting blind bisection. The threshold
   precision must be well below the smallest `M_n` of interest, because a
   discrete detector on a continuous field otherwise leaves a jitter of one step.
2. **Path.** BFS on the yield flags gives `Γ_n`; record its overlap with
   `Γ_{n−1}`.
3. **Intrusion by state injection.** See the derivation below. A bracketed,
   safeguarded scalar solve on nonnegative amplitude `a` drives reservoir
   pressure to `s_n + ΔP_arrest`, restoring the pre-intrusion snapshot between
   trials. Define `s_n` as an area/length-weighted compression-positive closure
   stress on the chosen path, frozen before intrusion for the first protocol.
   If closure is recomputed during intrusion it becomes part of the root function.
   Bound drained volume, opening, iterations and trial duration; a missing root
   is an explicit `no_arrest_bracket` outcome, not a forced intrusion.
   Choose a short intrusion `Δt ≪ t_M` so the response is elastic, and
   check temporal insensitivity without silently changing physical `η_reg`.
4. **Heating and material assignment.** Apply the selected contact rule and
   deposit the enthalpy of the physical transferred magma, including latent heat.
   A numerical band of width `h` representing opening `w` generally has melt
   fraction of order `w/h`; heating its whole volume to magma temperature is
   invalid unless the physical dike is resolved. Use enthalpy mixing or a verified
   sub-grid model, and distinguish mixture properties from an intentional weak
   contact law. Assign inherited `T0` only at commit. Zero transient band/sink
   sources, retain separately configured background recharge, and rebuild phase-
   dependent scaling/caches. Log mass, heat and any imposed stress reset.
5. **Log and repeat.** Append the per-event record (GAP-13) and carry the state.

*Eigenstrain by state injection.* Write the Maxwell increment with an eigenstrain
rate `ε̇*`: `ε̇' = ε̇_el + ε̇_vis + ε̇*'`, `τ = 2η ε̇_vis`,
`τ − τ_old = 2GΔt ε̇_el`. Solving,

```
τ = 2 ηve ( ε̇' − ε̇*' + τ_old/(2GΔt) )
  = 2 ηve ( ε̇' + (τ_old − 2G Δε*') / (2GΔt) ),     Δε* = Δt ε̇*
```

so a deviatoric eigenstrain increment is exactly a change `τ_old ← τ_old −
2G Δε*'` at the band's integration points. The volumetric part enters the mass
balance `−∇·v − (P − P0)/(ηb Δt) + Q = 0` as `Q = tr(Δε*)/Δt`. The plastic
return then acts on the same trial stress, so plasticity in the band is
consistent. The injected `τ_old` is not persistent, because the next step's
history comes from the solved `τ`. `Q_band` must be zeroed afterwards.

For a dike opening `w` across an element of size `h` with plane normal `n`, use
`Δε* = (w/h) n⊗n`. Use the deviatoric convention of `deviatoric_stress`
(`tr = (εxx + εyy)/3`, `τzz = −(τxx + τyy)`).

*Reservoir sink.* Normalise `Q_res < 0` with the same physical quadrature as
`Q_band`, so `∫Q_res dΩ + ∫Q_band dΩ = 0` to integration/roundoff accuracy.
This source cancellation is not itself conservation of magma mass. Separately
check reservoir and band storage, deformation, density, thermal expansion,
boundary flux and any erupted mass. State whether the transported quantity is
mass or reference-density volume; use the corresponding density conversion.

*Preconditions and caveats.*
- Needs integration-point stress storage (`stress_size=(nq, nels)`); with nodal
  stress the injection would smear into neighbours.
- Needs finite `G` in the band. It is always finite in this model.
- Use the same interpolated compliance as the constitutive kernel when forming
  `2G_q Δε*'`; a phase-boundary arithmetic modulus is not generally equivalent.
  Recompute the trial from the pristine history for every amplitude, and commit
  physical solved stress, never the eigenstrain-adjusted trial history.
- For irregular elements define `h = A_band/ℓ_path` (or its local geometric
  counterpart), so the quadrature integral of `tr(Δε*)` equals `∫Γ w ds`.
  A generic edge length and one-element width do not ensure mesh-independent
  opening volume. Keep opening, numerical band width and regularisation length
  as separate parameters.
- The band is a stress-free-strain region, not a fluid. The result must be
  compared with an elastic pressurised-crack solution (Sneddon or Eshelby) and
  the band pressure compared with the reservoir pressure ("hydraulic
  consistency") before the protocol is trusted. If that fails, the fallback is
  approach A: constrain `P` on the band and reservoir DoFs (D2).
  **Done**, in `examples/reykjanes/dike_crack_setup.jl` with the driver
  `dike_crack.jl`, the sweeps in `examples/benchmarks/stokes/dike_crack/` and a
  guard in `test/test_dike_functions.jl`. A flat elliptical band of *ordinary
  host material* is opened by `τ_old` and `Q` alone and compared with the M0
  closed form at `b/a ≪ 1`, where it is Sneddon's crack. The band's thickness
  profile `2b √(1 − (x/a)²)` is the opening profile of a uniformly pressurised
  crack, which is why one uniform eigenstrain can be compared with one uniform
  pressure at all. Measurements are in `.agents/solver.md`. Three results:
  the opening profile converges to the closed form (1.6% at 686 elements to
  0.06% at 6 034, against the 2% gate) and the closure traction to the crack
  pressure (2.7% to 0.03%); the numerical band width is a free parameter, with
  the profile within 0.35% and the traction within 0.2% for `b/a` from 0.1 to
  0.0125, once the injected volume is normalised with the pressure quadrature;
  and the answer is unchanged for `Δt` from an hour to a century and for an
  outer radius from 4a to 20a. D2's eigenstrain mechanism is therefore validated
  at the single-band, single-step level and approach A is not needed for it.
- **The band's mean pressure is not the dike pressure.** `P/p` is near 0.8 and
  moves with the band aspect (0.76 at `b/a = 0.1`, 0.82 at 0.0125), because the
  band is held in compression along its length as well as across it. Only
  `P − τ_yy`, the traction that would close the band, is the crack pressure. The
  hydraulic-consistency condition of step 3, and any reservoir-pressure target,
  must therefore be written on that traction; reading `P` in the band instead is
  wrong by about 20%. `test/test_dike_functions.jl` pins this.
- The band also stores `P/K` of the eigenstrain in its own elastic compression,
  because it has the host's `ηb = K`, so the eigenstrain that delivers a wanted
  opening is not known before the pressure is. One fixed-point pass on the solved
  band pressure settles it; without it the opening is 0.9% low. The band's
  continuity identity `ΔA/A + P/K` over the injected eigenstrain closes to 1e-6
  and is the conservation check for this step.
- **Amplitude solve and sink implemented.** `solve_amplitude!` in
  `event_stepping.jl` finds the smallest nonnegative amplitude at which an
  observed quantity reaches a target, on the same trials as the crossing search:
  it measures the state at amplitude zero, doubles until the target is passed or
  `amplitude_max` is reached, and only then bisects, so the monotonicity the
  physics suggests is safeguarded rather than assumed. A target never reached is
  `:no_arrest_bracket`, after `amplitude_max` itself has been tried; the other
  outcomes are `:amplitude_bracketed`, `:no_intrusion_needed` (already met at
  zero), `:unresolved_amplitude` (the trial budget ran out with a bracket in
  hand, the amplitude counterpart of `:unresolved_crossing`) and
  `:solver_failed`. Its oracle in `test/test_event_stepping.jl` is the linearity
  of one elastic step: a target of 1.5 times the pressure of a unit injection is
  reached at amplitude 1.5. `balance_dike_source!` in `injection_source.jl` adds
  the uniform reservoir sink that cancels the band source to 1e-13 of the
  injection, with the quadrature of the continuity residual.
- Still to write for step 3: the target itself — the closure stress `s_n` on the
  chosen path plus `ΔP_arrest`, which the plan freezes before intrusion — the
  bounds on drained volume, opening and trial duration, the contact and heating
  rules of step 4, and the per-event record. A band whose opening varies along
  its length, a band that is not an ellipse, several bands and plasticity in the
  band are also untouched.

*Insertion-rule controls* from Phase 3 become options of steps 3-4:
`contact ∈ {host, intact, weak}`, `ΔP_arrest`, band width, dike thermal
properties, failure criterion.

*Step acceptance and failure contract.* Require solver convergence, finite fields,
independently assembled momentum/continuity (and thermal) residuals, valid
geometry, bounded history increments, and the source/storage budgets before
using a state in the detector. Record both outer residual branches and their
normalisation. On failure restore the complete snapshot, reduce `Δt` within
configured minimum/retry limits, and return a structured failure if exhausted.
Use outcomes `accepted`, `event`, `no_event_by_horizon`, `solver_failed`,
`invalid_geometry`, `unresolved_crossing`, `no_arrest_bracket`; never assign a
finite threshold to a censored or failed run. Prevent zero-time event loops. The
implementation adds `invalid_state` for a field that is not usable,
`no_intrusion_needed` for a target already met, and `unresolved_amplitude`, the
amplitude counterpart of `unresolved_crossing`.

**First slice implemented** in `examples/reykjanes/event_stepping.jl`
(`attempt_step!`, `StepCheck`, `StepRejection`, `step_failure_message`), on the
GAP-12 snapshot. A step is a transaction: the state is captured, the step runs,
and it is accepted only if every check passes; otherwise the capture is restored,
a caller-supplied `rebuild!` runs, and the step is retried with `Δτ` halved until
the attempt budget or a smallest step is reached. The report carries the outcome,
the accepted `Δτ`, the attempts, which budget ended them, and one entry per
rejected attempt with the check that rejected it. The standard checks are solver
convergence and finite state; a step rejects what only it can see (an inverted or
over-distorted mesh) by throwing a `StepRejection`, while any other exception
propagates as the bug it is. `test/test_event_stepping.jl` covers acceptance,
each rejection path, the two retry budgets, argument validation, and the plan's
forced-failure check: a trial that solved and committed a doubled source and was
only then found to have failed leaves the two accepted steps after it equal to a
clean run to `1e-8`.

**Crossing refinement (step 1) implemented** in the same file: `capture_trials`,
`run_trial!` and `bracket_crossing!`. Every trial runs from one captured left
state and is restored afterwards, so a discarded trial cannot change the next
one. The interval `(0, Δτ]` is first sampled at `samples` equally spaced step
sizes and only the earliest bracket is bisected, because a connected detector
need not be monotone; the number of times the sampled detector changed is
reported, and more than one crossing gives `:unresolved_crossing` with the
earliest bracket retained rather than a number to trust. The outcomes are
`:crossing_bracketed`, `:unresolved_crossing`, `:no_crossing` (the trip was not
reproduced from the restored state) and `:solver_failed`; only the first, and
only with `commit`, leaves the state after the crossing step. Sampling `Δτ`
itself also checks that the trip reproduces from the restored state.
Not covered: the adaptive-`Δt` policy that decides when to refine, the
per-event record, and the detector itself, which is the caller's.

*Finding, detector rule.* The diagnostic rule the driver logs today trips on the
first step: on the coarse 284-element section, step 1 flags 133 of 284 elements,
all connected to the sill, with a four-element path to the shallow target. Its
tensile branch compares the mean sill pressure with `s3` at zero tensile
strength, and shallow lithostatic stress is far below a 4.5 km sill pressure, so
it is a hydraulic criterion with nothing holding it back. Freezing the detector
Boolean rule, `T0` and the path condition (sequence item 2) therefore comes
before any trigger is wired to `bracket_crossing!`; the search itself does not
choose them.

`reykjanes_thermal_stokes.jl` now runs its time loop through it, over
`(; dr, thermal, γP)`, with the reduced step carried into the following steps;
the disabled convergence check it had is now enforced, and `update_pressure_scaling!`
takes the step size, because `γP` and `dr.M_P` depend on `Δt`. Measured on a
coarse 284-element section: two steps at `Δt = 1 kyr` accept on the first attempt,
and forcing rejection with `advect_mesh = true, max_step_strain = 0.005` against a
first-step edge strain of 0.0099 takes three attempts and accepts at 0.25 kyr.
Not covered: the independently assembled residual branches, the bounded
history increments and the source/storage budgets are not yet checks; the
`event`, `no_event_by_horizon`, `unresolved_crossing` and `no_arrest_bracket`
outcomes belong to the detector and the amplitude solve, which are not started;
and the caller still owns what is not in the transaction (coordinates, event
geometry, counters).

*Check.* Reset-and-replay null test (`M_n` zero to measured noise); conservation
under 1%; eigenstrain against the crack solution; repeated runs equivalent within
declared floating-point tolerances; threshold brackets stable under temporal and
detector refinement. Force failure inside a trial and compare the next accepted
step against a clean run. Do not require bit-identical atomic assembly.

#### GAP-12. Snapshot/restore and memory switches (examples)

- **Snapshot/restore.** In-memory copies of every physical state array of `StokesDR`,
  `ThermalDiffusionDR` and the history bundle, plus caller-owned phase arrays,
  coordinates, boundary/load data, event geometry, physical time, cumulative
  injection/displacement, controller state and random state where used. Needed for crossing refinement and
  amplitude root-finding (Phase 3), and for branching factorial arms from one
  spun-up state (Phase 5). This is not an on-disk checkpoint, so it stays out of
  `io.md`'s checkpoint decision. Scratch, frozen operators and preconditioners
  must be restored or deterministically rebuilt, including after a geometry,
  material or time-step change. Test that snapshots do not alias live arrays.
  JLD2 analysis exports are optional and are not restart files until a round-trip
  continuation test exists; durable checkpoint support is deferred.
  **First slice implemented** in `examples/reykjanes/state_snapshot.jl`
  (`capture_state`, `restore_state!`, `physical_state`): the state arrays of a
  `StokesDR`, a `ThermalDiffusionDR`, a plastic-history bundle and any caller
  arrays, with non-array values (time, counters) deep-copied through a separate
  channel. The solvers zero `∂v∂τ`, `Rv0`, `∂T∂τ` and `R0` at the start of a solve
  and recompute the other work arrays, so those are listed as scratch and not
  copied; `M_P` is copied and must be rebuilt by a caller that changes the phases,
  time step or geometry. `test/test_state_snapshot.jl` checks that every array
  field is classified, an exact alias-free round trip, loud failure on a changed
  structure, and a replay on the coarse cavity mesh: a trial with doubled `Q` that
  commits `P0` and `τ_old`, restored and re-solved, matches a clean solve to
  1e-14 in the relative pressure norm. Removing `P0`, `τ_old` or `Q` from the
  state raises that difference to 1.4, 0.68 and 1.0, so the replay does detect
  omissions. Not covered: the thermal solver is checked by round trip only, not by
  a replay; the caller must pass what it owns (moving coordinates, `γP`, boundary
  data, event geometry, controller and random state); the intervention table and
  the memory switches are not started.
- **Switches.** Define an intervention table for elastic stress, damage,
  temperature and frozen-dike material/contact geometry. For each, state what
  is removed, what is retained, the reset reference and the re-equilibration
  procedure. `D → 0` must specify the fate of `ε_pl`; thermal reset changes EOS
  density/expansion and requires consistent `T/T0`; contact reset restores both
  velocity- and pressure-phase properties. Contact geometry is distinct from
  moving mesh coordinates. Stress reset restores a defined equilibrated reference
  including `P/P0`, not simply `τ_old = 0` under continued tectonic loading.
- **Control design.** Fork all 16 arms from the same pre-event baseline, using
  identical external loading/supply. Compare at matched event number and also
  matched elapsed time or injected volume. Log artificial heat/work/mass caused
  by resets. Interventions may interact and may not commute; fix their order
  before runs, and audit alternatives if attribution depends on it. The strict
  null resets the complete baseline and replays identical forcing. Merely
  disabling four memories during an evolving geotherm or tectonic spin-up does
  not imply `M_n = 0`. Separate secular background evolution with a no-intrusion
  control. Undefined/no-event arms are censored, not dropped from attribution.
- Promote to `src/` only if a second physics needs the same copy helper.

#### GAP-13. Memory diagnostics and outputs (examples)

| Quantity | Definition in FEMTools terms |
|---|---|
| `ΔP_crit^(n)` | Quadrature-volume-weighted reservoir pressure minus a declared fixed equilibrated reference; record recharge increment since arrest separately |
| `V_crit^(n)` | External recharge `∫dt ∫reservoir Q_recharge dΩ` since previous arrest (m² in 2-D); do not silently replace injected volume by deformation or storage volume |
| `M_n` | `ΔP_crit^(n)/ΔP_crit^(1) − 1` |
| `R_n` | Physical length/area-weighted overlap of declared path corridors; use fixed physical corridor width across meshes, not raw element counts |
| Clamping | Weighted `n·S·n`, compression positive, adjacent to earlier dike planes |
| Stress budget | `∫σ_xx dz` on a fixed section, before and after each event |
| Mass balance | Reservoir volume loss versus dike volume, compressibility-corrected |
| Surface displacement | `u = Σ v Δt` at surface nodes, for comparison with GNSS |
| Small-strain check | Deformation gradients, accumulated strain/rotation, Jacobian quality and boundary motion; rigid translation must not trigger a strain criterion |

If `ΔP_crit^(1)` is near zero, report absolute changes and mark `M_n` undefined.
Also report both threshold bracket endpoints, detector rule, uncertainty and
the pressure reference. A mesh update is not a complete finite-strain/ALE method:
stress transport, thermal transport, history transfer and conservation must pass
their own tests before extending the small-strain scope.

Write a CSV per run with explicit units and reference conventions. Reuse the
science event-table schema only where semantics and dimensions match; otherwise
export a documented conversion table. Write VTK with `write_stokes_vtk`
using cell data for `D`, `ε_pl`, `F` and phase.

#### GAP-14. Run management, sweeps and Level 0 (`reykjanes/` environment)

- Parameter specification (NamedTuple with a TOML mapping), a provenance record
  (git SHA, `Manifest.toml` hash, mesh hash, seed, wall time), and a per-run
  layout `runs/<id>/{params.toml, provenance.toml, events.csv, status.toml, vtk/}`,
  untracked; optional `state.jld2` is an analysis export. The
  "reproducible batch configuration" workflow named in `frontend.md` is the
  justification for a run specification; keep it in the examples layer.
- Sweeps and inference dependencies (QuasiMonteCarlo, GlobalSensitivity, a GP
  package, Turing or equivalent) go in `reykjanes/Project.toml`, never in
  FEMTools.
- Promote `reykjanes_figures/level0.jl` into a tested module with the H0-H4
  closures. Its fixed-point step for `Δt_n` is already the right structure.
- GPU: reuse the `FEMTOOLS_BACKEND` pattern from the 3-D volcano driver.
- Screening (REQ-20): report signed sensitivity to each scalar parameter,
  `dJ/dθ = Σ_i g_i ∂m_i/∂θ`, scaled by a declared plausible perturbation or prior
  width. Separately report integrated absolute field sensitivity, which measures
  spatial influence but can over-rank a scalar whose positive/negative effects
  cancel. Avoid log sensitivities for zero or signed parameters without a scale.
  Sample several inherited states before discarding a family; a frozen-history
  kernel misses parameters acting mainly through earlier evolution. Validate
  rankings with Morris/Sobol or finite differences on the same observable and
  parameter ranges. Keep failed/censored runs visible, and validate emulators
  on held-out full solves before dense sweeps.

#### GAP-15. Solver behaviour study (not a feature; start at month 1)

1. **Elastic-dominated regime.** With `Δt ≪ t_M`, `ηve ≈ GΔt`. Measure DR
   iteration counts against mesh size and against reservoir/host contrast
   (`G` ratio) and near yield. *Measured without plasticity* on the cavity benchmark
   (`.agents/solver.md`); near yield and at production element counts it is still open.
2. **Noise floor.** The convergence test is `err = min(err_abs, err_rel)`
   against fixed tolerances, and the Stokes momentum assembler is atomic-only
   (heat and lithostatic also have colored assembly). Run the null test to
   measure the run-to-run and tolerance-driven scatter of `ΔP_crit`, and set
   `ϵ_tol` so the noise is well below the smallest `M_n` claimed. *Measured on the
   cavity benchmark:* run-to-run scatter 7e-13, tolerance-driven pressure error 1.5e-4
   at `ϵ_tol = 1e-3` and about 1e-5 at 1e-4, and a scale dependence of the stopping test
   (historical run reports above). The threshold-level test needs the event engine.
3. **Regularisation and `Δt`.** Separate physical rate dependence from numerical
   stabilization. First refine `Δt` at fixed physical `η_reg`; then sweep
   `η_reg` and spatial regularisation length independently. `ΔP_crit^(1)` must converge
   under mesh refinement *and* be insensitive to `η_reg` and `Δt`, or the
   memory signal is an artefact of the numerics. Fallback if it is not: a
   non-local plastic strain by implicit-gradient smoothing through the existing
   scalar DR solver (`ρ0 = Cp = 1`, `Δt = 1`, `k = ℓ²` reduces its residual to
   `T − ℓ²∇²T = T0`) *(verify)*.
4. **Small-strain assumption.** Confirm with GAP-13. If it fails, stop extending
   the result until transport, objective stress updates and moving-mesh
   conservation are validated; geometry refresh alone is insufficient.
5. **Cost.** Estimate steps per cycle times cycles times iterations for one run
   and for the 16-run factorial, to decide when the GPU is required.
6. **Adjoint behaviour.** Repeat items 1 and 2 for the adjoint: iteration counts
   against mesh size and `G` contrast in the elastic-dominated regime (the recorded
   850 to 3000 iterations are sinking-block, viscous), and `:blocks` memory at the
   production element count (currently 1.176 kB packed or 2.240 kB full per
   element, excluding scratch; GAP-25 changes this). Benchmark memory-light
   alternatives rather than assuming `:enzyme` is suitable for new physics.
7. **Gradient noise floor.** The forward state is only converged to the DR
   tolerance, and the stopping test is `err = min(err_abs, err_rel)`. An adjoint
   gradient is exact for the *discrete* residual at an exactly converged state, so
   its error grows with the forward residual. Measure the gradient error against
   `ϵ_tol` on the null case, and fix the tolerance used by gradient tests (GAP-20)
   and by production kernels separately.

Record findings in `.agents/solver.md` with the measured tables.

#### Out of scope

MPI or distributed meshes; thermal advection; Winkler base (GAP-8); on-disk
checkpoints; high-order VTK; any new frontend; and the JustRelax implementation
itself. The protocol is specified by the science plan and by the benchmark cycle
exported from GAP-11, not by shared code.

Adjoint items that are out of scope: a thermal-solver adjoint (not needed by the
event-local design); an adjoint through the discrete steps of the event protocol
(threshold detection, path search, amplitude root-finding, phase reassignment),
where the science plan asks for ensemble or finite-difference statements
instead; inversion of a free 3-D field (the last rung of the inversion
hierarchy); and a multi-event adjoint through the whole sequence, which is
GAP-24 and not scheduled.

### 1.6 Adjoint track: design and gaps (Phases 5A, 5B, 7)

#### What each question needs

| Question | Objective | Sensitivity to | Adjoint level | Gaps |
|---|---|---|---|---|
| Q5, Phase 5A: what controls the next failure | `J_fail`, `J_path` | `η`, `C`, `μ`, `T`, `D`, inherited `τ_old` | EL | 17–21, 25 |
| Q6, Phase 5B: geodesy | `J_geo` | source distribution, geometry and rheology | Full-window forward differences; EL only for a one-step objective, WIN otherwise | 22, 25, (24) |
| Q7, Phase 7: surface load | Smooth `J_fail`/recharge proxy; hard threshold by forward perturbation | `q_s(x, y)` | EL in 3-D | 8, 23, 25 |
| Phase 6: screening | `J_fail` | 15+ parameters | EL, conditional ranking | 21, 14, 25 |

Three scopes of adjoint, in increasing cost. EL/WIN/EVT are scope labels;
A0–A5 remain delivery milestone IDs and do not denote these scopes.

- **EL, event-local.** One converged step from a restored snapshot, history frozen.
- **WIN, windowed.** The `k` steps of one recharge interval from a snapshot, with a
  reverse sweep that propagates through the state updates.
- **EVT, through the event protocol.** Not planned (Out of scope above).

The science plan leaves open whether event-local adjoints suffice. The
recommendation (D9) is EL for Phase 5A, Q7 and conditional screening; WIN when
the objective requires history derivatives and full-window finite differences
are too expensive; EVT is unscheduled. Parameter count affects cost, not whether
an event-local derivative is valid.

#### Event-local formulation (conditional on GAP-25)

Restore the snapshot `s = (τ_old, P0, T, T0, D, ε_pl, phases, geometry, loads)`
at the reference state, with `Q` explicitly an input/control,
and re-solve one step of size `Δt` to a tight tolerance: `R(u; m, s) = 0` with
`u = (v, P)`. Evaluate a smooth `J(u; m, s)`. One adjoint solve
`R_uᵀ λ = −J_u` (after the complete transpose passes GAP-25) gives, in a contraction
pass and for every parameter family at once,

```
dJ/dm = J_m + λᵀ R_m          (parameters: η, ρ0, G, K, C, ϕ, α, Q, ...)
dJ/ds = J_s + λᵀ R_s          (inherited fields: τ_old, T, D)
```

Cost per objective is one forward re-solve, one adjoint solve and contractions.
The number of linear solves is independent of parameter count; contraction
work, output and storage grow with the requested fields/directions. Multiple
objectives require separate adjoint right-hand sides. Measure cost on this model.

*What it answers.* Which fields at the start of the step control failure proximity
now. The kernels differ between events because the inherited state differs, so the
evolution of `K_m` across events tests Q5 but is not alone evidence that the crust
"learns where to break". `K_τold` is the elastic-memory kernel, `K_D` the damage kernel, `K_T` the
thermal kernel. *What it does not answer.* How past events or past parameters shaped
the current state; that requires WIN/EVT or full-history forward perturbations.

*Reference states.* Use a sub-critical state (for example 90 to 99 % of the trigger
volume; the crossing refinement of GAP-11 stores pre-step snapshots). Measure
rather than assume the active plastic fraction; inherited states can already
yield below that recharge fraction. Also sample the trigger state. Report a *kink audit*
with every kernel: the fraction of integration points within `s_f` of yield and the
number on the plastic branch.

*Kernels in units of the observable.* Let `V_step` be the volume injected during the
step (`Q` integrated over the reservoir and `Δt`, per unit length), on top of the
volume `V_snap` already accumulated at the snapshot. `J(V_step, m) = J*` defines a
smooth remaining volume to failure, so `V_crit^J = V_snap + V_step^J(m)` and, by the
implicit function theorem,

```
∂V_crit^J/∂m = −(∂J/∂m) / (∂J/∂V_step)
```

The denominator is one more contraction with the same `λ`, because `Q` is an input
coefficient of the pressure residual. Require a locally unique crossing and a
nonzero `∂J/∂V_step` resolved above its numerical uncertainty. Otherwise report
an ill-conditioned proxy and do not divide. Pressure conversion through
`K_eff/V_res` requires a calibrated local compliance held fixed; otherwise
differentiate reservoir pressure directly, including compliance changes.
`M_n` additionally depends on its first-event denominator. The proxy holds the
inherited state fixed, which is the event-local approximation. `V_crit^J` is a
smooth level-set proxy and *not* the hard threshold, which the science plan bars from
being differentiated. The hard `ΔP_crit` stays the forward diagnostic, and comparing
the two is the consistency check of GAP-20.

#### Contract on the forward physics

These constrain GAP-2, GAP-3, GAP-5 and GAP-11 so the adjoint keeps working:

1. Every constitutive quantity is evaluated inside the differentiated element
   residual (`deviatoric_stress` and the viscosity law), never precomputed where AD
   cannot see it.
2. History variables (`τ_old`, `D`, `ε_pl`) and `T`, `Q` are array inputs of the
   step. They are not captured in closures.
3. Non-smooth switches are listed with their treatment in `.agents/solver.md`: the
   plastic return (a kink at yield: subgradient, evaluated away from yield or
   audited), the `η_min`, `η_max` clamps, the `D ∈ [0, 1]` clamp (outside the
   residual), a tensile cutoff (GAP-5 option 2), and the detector and path search
   (not differentiated).
4. Every `src/` PR runs the adjoint tests, as the regression gates already require.
   A new law adds a case with an independent finite-difference oracle.

#### GAP-17. Adjoint objectives (`J_fail`, `J_path`, pressure objective)

*Have.* `objective_vx` and `objective_vy` as caller-assembled loads; a miniapp-local
box-integral loader.

*Need.* Library objectives with `∂J/∂v` and `∂J/∂P`, and a pressure-objective term
in the adjoint solve.

*Design.*
- `J = (1/∫w dV) ∫ w softplus(F/s_f) dV`, with `F = τII − C cosϕ − P sinϕ` from the
  **post-return** stress at integration points (characteristic units), element
  weights `w` (1 above the sill to start; a corridor indicator for `J_path`), and
  `s_f > 0` a fixed, reported stress scale (with a floor for zero cohesion), tuned so the
  sub-critical reference state has `J` well above the noise. Normalising by `∫w dV`
  makes values comparable across meshes. The post-return `F` stays bounded under
  regularised plastic flow. Compare post-return and trial-margin candidates:
  perfect-plastic return can flatten the former and trial margin can overweight
  yielded regions. Neither is selected as physically valid before GAP-20.
  Use stable softplus. Hold `s_f` and corridor weights fixed during a derivative
  test; if they depend on parameters include that derivative explicitly.
- Loads by differentiating an element-level pure function `Jᵉ(vₑ, Pₑ; τ_old, ...)`
  with ForwardDiff, 17 partials for T7/P1 (14 velocity, 3 pressure), the way the
  frozen blocks are built. Share **one** implementation of `F` with GAP-6 so the
  diagnostic and the objective cannot drift.
- After GAP-25, the solver gains optional `objective_P = nothing`; add it on
  every residual assembly, including final checks, to the complete pressure
  adjoint residual. Validate dimensions/backend and boundary projection;
  adding this load alone does not repair compressibility.
- `J_path` is the same functional with a corridor weight around the previous dike
  (`Γ_{n−1}` from GAP-6, widened by `w_c`). A graph-based soft-connectivity
  functional is a later refinement. The detector's own corridor is the natural
  weight here: `build_dike_corridor` already returns the elements of a fixed
  geometric strip and the area it samples in each bin, all of it independent of
  the mesh and of any graph, so `w` can be its indicator and the objective will
  be weighing the same region the forward detector declares events on. Share that
  one construction, for the reason GAP-6 and the objective share one `F`: a
  corridor defined twice will drift, and a gradient weighted over a region the
  detector does not use answers a different question than the one being asked.
- `J_geo` loads come from the host operator of GAP-22 and need no kernel.

*Check.* Load against a finite difference of `J` for random `(v, P)` on one element
and on a small mesh; `objective_P = nothing` reproduces today's adjoint results
exactly; `Float32`; zero allocation of the load kernel.

#### GAP-18. Parameter and state contractions in `src/`

*Have.* `material_sensitivities`, in the sinking-block miniapp: `η` and `ρ0` per
element, momentum residual only, no plastic model, Enzyme through a
KernelAbstractions kernel.

*Need.* `λᵀ ∂R/∂m` for the families below, promoted into `src/`.

| Family | Enters through | Granularity | Note |
|---|---|---|---|
| `η`, `ρ0` | momentum residual | element | exists in the miniapp |
| `G`, `K`, `ηb`, `α` | Maxwell coefficient, `(P − P0)/(ηb Δt)`, EOS | element | needs the term `λ_Pᵀ ∂R_P/∂m` |
| `C`, `ϕ`, `η_reg` | return map | element, per point once `D` exists | needs `plastic` |
| `T` | EOS density, later `η(T)` | pressure DoF (P1-disc) | host maps to thermal nodes by the transpose of the coupled-solver gather |
| `τ_old` | Maxwell history term | integration point | the elastic-memory kernel |
| `D` | `C(D)`, `μ(D)` | integration point | after GAP-2; `ε_pl` has zero event-local sensitivity unless it independently enters the law |
| `P0`, `T0` | pressure storage and thermal expansion | pressure DoF | inherited inputs, distinct from current `P`, `T` |
| `Q` | continuity source | pressure DoF, or one scalar scale | gives `∂J/∂V_inj` |

*Design.* Keep the miniapp's idea (evaluate the element residual with the parameter
replaced by a per-element value, contract with the element's `λ`) with three
changes: include the pressure residual; pass `plastic`; and use **ForwardDiff on the
scalar `λₑᵀ Rₑ(pₑ)`** with a small partial vector (about eight per element) instead
of deferred Enzyme through a kernel. Forward mode matches how the frozen blocks are
built, avoids differentiating through a KernelAbstractions kernel, and costs a few
residual evaluations once per event. One public, unexported function with a
documented family list in `src/stokes/assemblers/`; the miniapp is switched to it.
Each result declares whether it is a derivative with respect to a phase scalar,
element constant, nodal coefficient or IP value. Define field-kernel mass weights
and check `δJ = Σ g_i δm_i = ∫ K_m δm dΩ`; do not divide nodal/IP derivatives by
element area indiscriminately. Include explicit `J_m` and `J_s`: strength enters
the failure objective directly. If `K`, `ηb` and plastic `Kb` represent one
physical modulus, differentiate all through the shared parameter. Include load
and prescribed-BC derivatives when they vary. Normalise spatial sources by
quadrature and differentiate their normalisation as well as their shape.

*Check.* Every family against a central finite difference of `J` on the small
adjoint test case (extends `test_stokes_adjoint_api.jl`); the miniapp's numbers
reproduced; phase sums against the `stokes_material_gradient_3d` pattern; `Float32`.

#### GAP-19. Adjoint contract for damage, creep and cutoff

Rides with GAP-2, GAP-3 and GAP-5. It is a checklist attached to those PRs, not a
separate feature:

- GAP-2: a `K_D` contraction and a Taylor test at `D ≠ 0`.
- GAP-3: the creep law inside the element residual; `:matrix_free` rejects a creep
  phase with an `ArgumentError`; a Taylor test for `η(T)` on a tiny mesh; `K_T`
  through the rheology.
- GAP-5, option 2 only: the cutoff's kink documented; the corner case audited.
- One table of non-smooth switches and their treatment in `.agents/solver.md`,
  which is the science plan's checklist item.
- An adjoint-mode smooth blend of the return-map branches is *not* planned. Add it
  only if the kernel maps near yield prove noisy in GAP-21. A smoothed law must
  also define the forward residual under test; differentiating a surrogate return
  map is not an exact adjoint of the original solve.

#### GAP-20. Gradient verification harness and threshold consistency

- A Taylor-remainder helper in `test/` (a utility, not exported): for a sequence of
  `ε`, return `R(ε) = |J(m + ε δm) − J(m) − ε ∇J·δm|` and the observed order; pass
  when the order is about 2 over at least two decades or until the solver floor. Keep
  the central-difference oracle beside it.
- Tolerance policy (D14): gradient tests use `Float64`, a tight `ϵ_tol`, and are run
  at two tolerances, as the science plan's checklist asks.
- Budget: the existing 3 × 3-mesh finite-difference test takes 69 s. Reuse forward
  solves across the stencil and keep new cases to one or two small meshes.
- Pairs, in order of availability: source (`Q`), spatial `η`, `C`, `μ` (GAP-18);
  `D` (after GAP-2); `T` through the rheology (after GAP-3); surface load in 3-D
  (GAP-23).
- *Threshold consistency* (example level, needs GAP-11): perturb `η` or `C` with a
  family of fields; compare the sign and size of `δV_crit^J` predicted by the
  kernels with `δV_crit` measured by the hard detector. Report sign agreement for
  every perturbation above the GAP-15 noise floor. This is workflow step 3 of the
  science plan; any numerical criterion is a proposal until the noise floor is known.

#### GAP-21. Event-local adjoint driver and kernel diagnostics (Phase 5A)

`examples/reykjanes/adjoint/`, one function `event_kernels(snapshot; ...) ->
NamedTuple`:

1. Restore the snapshot (GAP-12) at the reference state, re-solve one step at the
   gradient tolerance, freeze.
2. Build the load (GAP-17), solve the adjoint warm-started from the previous event's
   `λ`, contract (GAP-18).
3. Write, per family, the element field `K_m` (integral and per-area density) and
   the log-sensitivity `m K_m`, as VTK cell data beside the forward output, plus the
   `V_crit^J` sensitivities.
4. Diagnostics from the science plan, on unsmoothed element densities: `L_n` (share
   of `|K|` inside the corridor of width `w_c` around `Γ_{n−1}`, reported against
   `w_c`), centroid and width from the first and second moments of `|K|` weighted by
   element area, overlap with earlier dikes, with a host seismicity map on element
   centroids, and with high-`D` or high-`T` elements, and the area fraction where
   `K_C` and `K_η` have opposite sign as the marker for competing weakening and
   clamping. Smooth only for display.
5. Arms: the driver takes any snapshot, so the no-memory, stress-only, plastic-only,
   thermal-only and full-memory arms of GAP-12 pass through unchanged.
6. Screening output for GAP-14: signed scalar chain-rule derivatives, declared
   parameter scales and separate absolute field-influence measures.

Size M-L, after GAP-25, GAP-17 and GAP-18.

*Check.* In reset-and-replay controls, kernels agree within noise. Compare
localisation only with the same corridor definition; if total sensitivity is
below its noise floor, `L_n` is undefined. A symmetric mesh, state, loading and
objective must give a symmetric kernel. `L_n` is stable under mesh refinement
with fixed physical corridor width and area-weighted norms.

#### GAP-22. Geodetic operator, synthetic truth and low-dimensional inversion (Phase 5B)

1. **Displacement.** Accumulate `u = Σ v Δt` at surface nodes each step (the GAP-13
   quantity), so the model and the objective use one definition.
2. **Observation operator.** A host sparse matrix `L`: point location on the
   unstructured mesh (host search giving element, reference coordinates and
   shape-function weights), GNSS components, and InSAR line-of-sight `e·u` on
   downsampled pixels. `J_geo = ½ (L u − d)ᵀ C_d⁻¹ (L u − d)` and
   `∂J/∂v = Δt Lᵀ C_d⁻¹ (L u − d)`, assembled on the host and copied to the backend.
   The covariance is a placeholder until Phase 0 supplies one (open decision).
3. **Source parameters without a moving mesh.** Represent the injection as a smooth
   distribution `Q(x; θ)` (position, depth, width), so `∂J/∂θ` is a chain rule from
   `K_Q`. A change of source *shape* changes the mesh and is done by finite
   differences of mesh builds.
4. **Elastic comparator (D11).** The same FE model in the elastic limit
   (`η → ∞`) at Level 1; analytic Mogi or Okada only for 3-D.
5. **Synthetic-truth study (Adjoint paper B).** A forward VEP run with a
   *stationary* source, synthetic data plus noise, an elastic-limit inversion, and
   the apparent depth, volume and position drift over the interval. This needs
   forward runs only, no adjoint.
6. **Low-dimensional inversion** (science plan stages 1 and 2, about ten
   parameters): forward finite differences (`n + 1` runs), or EL for a one-step
   objective. The baseline took 70 s for ten steps including compilation, averaging
   about 7 s per step on that run only, not an upper bound. Benchmark a complete
   observation window including trial/retry work before budgeting an inversion.
   Use `p+1` forward solves for forward differences or `2p` plus a cached baseline
   for central differences. Multi-step objectives need a windowed adjoint or
   full-window finite differences regardless of parameter count.
7. **Identifiability.** Hessian-vector products by finite differences of adjoint
   gradients (two perturbed forward solves plus their adjoints per central product)
   only after gradients are valid for the full objective window. Start with the
   whitened observation Jacobian, singular values, correlations and prior
   sensitivity; add Lanczos only when scale warrants it.

Size M-L, mostly in `examples/` and `reykjanes/`. The point-location helper is the
only mesh-facing piece; keep it in the examples until a second user appears.

*Check.* `L` reproduces a linear field exactly on a patch of elements and matches an
analytic point evaluation; `∂J/∂v` against a finite difference of `J_geo`; the
elastic-limit run against the pressurised-cavity benchmark (M0).

The displayed load `Δt Lᵀ C_d⁻¹(Lu−d)` applies to a single step with inherited
displacement fixed. Specify observation epochs, reference frame/LOS sign,
missing observations and interpolation error. Apply a covariance factorisation
instead of forming `C_d⁻¹`; require positive definiteness or a documented
projection for removed reference modes. A 2-D section cannot predict along-strike
GNSS motion or arbitrary InSAR scenes without a geometric assumption. Moving
injection within a fixed inclusion changes source distribution, not reservoir
depth/shape; label these parameters separately.

#### GAP-23. Surface-load kernel in 3-D (Phase 7, optional)

*Have.* Nothing on the T11 visco-elasto-plastic path. The Hex27 linear viscous
adjoint exists; `load` exists only in the caller-owned 3-D path.

*Need.* The forward surface load (GAP-8) and an adjoint of the `StokesDR`/`MixedMesh`
3-D solve.

*Options.*
1. **Frozen symmetric operator (conditional spike).** Prove symmetry of the full
   constrained, scaled operator, including pressure storage and EOS. Absence of
   plasticity is insufficient. Only then can the forward operator be reused for
   the transpose solve, with a derived load/sign mapping. This
   is the trick of the existing Hex27 wrapper, which calls the forward solver with
   `load = objective_load` and zero material. It needs the GAP-8 `load` in the mixed
   3-D path. The 3-D objective is the elastic-limit yield excess, or the
   orientation-maximised Coulomb margin of GAP-7 smoothed by a log-sum-exp.
2. **Full 3-D `:blocks` or `:enzyme` adjoint** on the mixed T11 path with
   plasticity. Dense blocks for T11/P1 would be `33 × 33 + 33 × 4 + 4 × 33` doubles,
   about 10.8 kB per element before the missing `4 × 4` storage block, geometry,
   state and workspaces; roughly 11 GB for those blocks alone at 10⁶ elements.
   Measure total peak memory and compare valid matrix-free/AD alternatives. Size L.
3. **Patch superposition**, one forward solve per coastline patch, as a cross-check on
   a coarse patch set only.

`K_q` is the transpose of the face-quadrature load map applied to `λ`, including
shape weights, areas and normals, not an unweighted node gather (sign fixed by the
hydrostatic-column test of GAP-8). On land faces it answers "where would added load
matter" (ice, lava), not a physical `h_w`; convert with `ρ_w g` on water faces only
(D13).

Spike at month 8 together with the 3-D `Q` check: option 1 on a one-block mesh
against the sparse-oracle pattern of `test_stokes_3d_reference.jl`. Only Adjoint
paper D depends on this gap.

#### GAP-24. Windowed multi-step adjoint (not scheduled)

*Why it may be needed.* A geodetic objective over an inflation interval, or an
inversion for many rheological and thermal parameters, depends on `k` steps whose
history chains through `τ_old`, `P0`, `D`, displacement and `T`. EL freezes that history and would give a
biased gradient for such objectives.

*Sketch.* Snapshot at the window start (GAP-12) and store or recompute the per-step
state. The reverse sweep re-assembles frozen blocks through element AD, solves
the adjoint, then propagates `λ` into the
step's inputs: `∂R/∂τ_old` (a GAP-18 contraction) and the adjoints of the state
updates `τ_k = Φ_τ(v_k, P_k, τ_{k−1})` and `D_k = Φ_D(...)`, both from ForwardDiff of
the same element functions. With constant `k` and no latent heat or shear heating the
physical thermal operator may be symmetric, but the implemented row-normalised
residual need not be. Prove the discrete transpose and boundary/load mapping
before reusing its solver; shear heating and latent heat add coupling terms.
For fixed material and amplitude, state injection is affine with identity
derivative with respect to `τ_old`; derivatives with respect to amplitude,
modulus and geometry are not identity, and root selection/phase assignment are
outside the windowed scope. No percentage cost is established. Include forward
recomputation, block assembly (AD, not one residual evaluation), thermal/coupling
transposes, history pullbacks and storage/checkpoint scheduling in the budget.

Size L. Do not start before G4A shows kernels worth chaining, or G4B shows that the
identifiability question needs many parameters.

#### GAP-25. Complete compressible adjoint before Reykjanes sensitivity work

*Source finding.* `integrate_PH_pressure_residual` includes
`−(P−P0)/(ηb Δt)`. Therefore the physical Jacobian contains
`D_P = ∂R_P/∂P = −∫ N_P N_Pᵀ/(ηb Δt) dΩ`, nonzero for finite storage.
`FrozenAdjointOperator(A,B,C)` and its application omit this block. The Enzyme
path computes a pressure pullback into `dP_scratch` but does not add it to
`ResλP`. The present incompressible end-to-end test cannot expose that omission.

*Required mathematics.* Begin with the **unaugmented physical residual** and
constrained unknowns:

```
R_u = [ A_v  B ; C  D_P ]
0 = J_v + A_vᵀ λ_v + Cᵀ λ_P
0 = J_P + Bᵀ λ_v + D_Pᵀ λ_P
```

If retaining Powell-Hestenes augmentation, derive the exact row transformation,
its pressure derivative and the relation between augmented and physical adjoint
multipliers. Include EOS pressure coupling, finite `G`, histories and `Q` in the
primal residual used to freeze the plastic tangent. Augmented material
contractions must use the same residual/transformation; an augmented multiplier
cannot be silently contracted with an unaugmented residual. Symmetric packing
and `C = Bᵀ` require measured identities; `plastic === nothing` alone does not
establish them with pressure-dependent density or new constitutive laws.

*Delivery.* Add/fix the full transpose in supported operator modes, or reject
unsupported configurations explicitly. This is a source change, not an objective
loader. Keep residual definitions shared and avoid differentiating the DR
iteration as a substitute for the converged-equation adjoint.

*Acceptance.* A tiny finite-`K`, finite-`G`, nonzero-`Q` problem with fixed old
stress and `P0` must pass: an independently assembled full Jacobian; random-vector
`aᵀ(J b) = (Jᵀ a)ᵀ b`; adjoint residual against the sparse transpose; and central
finite differences/Taylor tests for velocity and pressure objectives with respect
to `Q` and bulk modulus. Add pressure-dependent EOS and an active plastic point
away from a switch as distinct cases. Use two solver tolerances; preserve the
incompressible gauge-aware tests. Agreement between implementations of the same
omitted term is not an independent oracle.

## Part 2. Build order

| Milestone | Science phase (months) | FEMTools work | Exit check |
|---|---|---|---|
| **M0** Feasibility spike | First four weeks | Existing cavity setup; capture reproducible evidence and separate inclusion error from discretisation. Begin GAP-15. | Relevant cavity observables within 2%; iteration table and run manifest. A documented discrepancy leaves the gate open. Solver fixes discovered by the benchmark are permitted. |
| **M1** Minimal mechanics | Phase 2 (2-5) | GAP-9, GAP-10, GAP-6, GAP-1 (yield map and `λ` output), GAP-15, the Phase 2 benchmarks | Gate G1: benchmarks pass; `ΔP_crit^(1)` changes under 5% under refinement, `η_reg` and `Δt`. **Open, and the gate needs rewording before it can be met**: 77.7% with the adjacency detector, 45.6% with the mesh-independent one, 12.4% once the reservoir polygon is resolved, and 1.9% from an equilibrated baseline — but that last configuration fails at its first time step, so it is not a pass. The invariant is the absolute event pressure, which moves 0.45 MPa (0.35%) across a fourfold refinement; see the error budget |
| **M2** Repeated intrusion | Phase 3 (5-8) | GAP-2, GAP-11, GAP-12, GAP-13, GAP-5 if triggered | Null test passes; `M_n` versus `n` under every control; Gate G2. |
| **M3** Thermo-mechanical | Phase 4 (8-11) | GAP-3, GAP-4, dike heating step, aureole convergence | Stefan and creep benchmarks; Gate G3. |
| **M4** Attribution | Phase 5 (10-13) | Switch application (GAP-12), 16-run factorial runner, main-effect and Shapley analysis | Gate G4. |
| **M5** Sweeps | Phase 6 (12-15) | GAP-14, GPU runs, emulator | Gate G5. |
| **M6** 3-D | Phase 7 (11-16) | GAP-7; start `Q`-in-3-D and mesh checks at month 8 | Gate G6. |
| **M7** Generalisation | Phase 8 (15-18) | Krafla and Afar set-ups reuse the drivers with new meshes and parameters; archive code and data | Gate G7. |

Adjoint track. The science plan builds the tooling alongside Phases 2 to 4 so that
Phases 5A and 5B use it. The track therefore runs in parallel with M0 to M3 and
competes with them for the same developer; months 11 to 15 carry Phases 5, 5A, 5B,
6 and 7 together, which is the tightest stretch.

| Milestone | Science phase (months) | FEMTools work | Exit check |
|---|---|---|---|
| **A0** Compressible adjoint correctness | With M0/M1; before sensitivity features | GAP-25 independent oracle and required transpose repair; preserve incompressible tests; then benchmark the Reykjanes baseline. Settle D9/D10 | Full transpose identity and finite-compressibility objective gradients at two tolerances; measured cost/memory |
| **A1** Objectives and contractions | With M1 and M2 (3-7) | GAP-17, GAP-18, the GAP-20 harness | Taylor test per parameter family in a sub-critical state; kink audit in place |
| **A2** Adjoint-compatible physics | With M2 and M3 (5-11) | GAP-19, attached to the GAP-2, GAP-3 and GAP-5 PRs | Taylor tests at `D ≠ 0` and with a creep phase |
| **A3** Event-local kernels | Phase 5A (11-14) | GAP-21, threshold-consistency check | Gate G4A |
| **A4** Geodesy | Phase 5B (11-15) | GAP-22; GAP-24 only if D9 is revised | Gate G4B |
| **A5** Screening and surface load | Phases 6 and 7 (12-16) | Ranking feed to GAP-14; GAP-8 load; GAP-23 (optional) | Gate G5 ranking; Adjoint paper D is optional |

Dependencies that constrain ordering:

- GAP-25 gates A0 and all Reykjanes adjoint objectives/contractions; it does not
  block forward-only M1/M2. Correctness has priority over optional adjoint papers.
- GAP-1 comes before GAP-2 and GAP-11 (history and `λ` output).
- GAP-6 and GAP-12 come before GAP-11 (detection and crossing refinement).
- GAP-9 and GAP-10 come before any cycle run.
- GAP-15 items 1-3 gate M1: do not run cycles before Gate G1 is met.
- GAP-3 and GAP-4 are needed only from M3, so they do not block M1 and M2. Do
  them after M2's null test so the memory result is first established without
  thermal complexity, as the science plan intends.
- GAP-1 and GAP-6 come before GAP-17, so the yield function is one implementation
  shared by the diagnostic and the objective.
- GAP-17 and GAP-18 come before GAP-20, and GAP-20 before GAP-21. GAP-12 comes before
  GAP-21.
- GAP-19 lands *inside* the GAP-2, GAP-3 and GAP-5 PRs, not after them. Otherwise
  each of those changes can silently break the adjoint.
- GAP-22 items 1 to 5 need no memory physics and can start at month 8; the bias study
  needs forward runs only. GAP-8's load comes before GAP-23.

## Part 3. Verification

### Error budget and gate evidence

Each gate records an owner, configuration, command, measured result, artifact
and pass/fail/blocked status. Calendar months are the science proposal's targets,
not a second parallel developer's capacity. Numerical failure keeps a gate open;
a scientifically negative result may close it with the science plan's fallback.

Before interpreting `M_n`, choose a smallest scientifically relevant signal
`M_min`. Estimate absolute threshold uncertainties from solver tolerance,
time refinement, detector bracket, mesh, regularisation, domain boundaries and
run-to-run variability separately. Do not call repeatability a discretisation
error estimate. For positive reference `p1`, a conservative propagation is
`δM_n ≤ δp_n/|p1| + |p_n| δp1/p1²`; correlated errors may be treated more sharply
only with evidence. Proposed acceptance: numerical `δM_n ≤ 0.1 M_min` and both
individual threshold errors small enough to resolve the claimed sign. If this
fails, report an unresolved effect. The science plan's 5% G1 threshold criterion
alone cannot validate a 1% memory signal.

**The observable the gate is written on is badly conditioned, and that is now measured.** `ΔP_crit`
is the difference of two pressures of about 126 MPa, so its relative error is the relative error of
those pressures multiplied by `P/ΔP ≈ 25` to `30`. The mesh sweep shows exactly that: the event
pressure spreads 0.48% and the spun-up baseline 1.56%, and the difference spreads 45.6%. Three
restatements are available, and the measured numbers already rule out two of them:

- *Absolute event pressure.* Spreads 0.48%, so it would pass a 5% gate at once — and that is the
  objection to it. It is dominated by the lithostatic column at the reservoir; the part of it that
  carries the physics of interest is precisely the small difference being discarded. A gate that is
  passed by the depth of the sill is not a gate.
- *Injected volume to failure.* With a constant recharge this is proportional to the time of the
  event, which spreads 18.2% across the same four meshes (3.359, 3.359, 3.219, 2.781 kyr). Better
  conditioned than `ΔP_crit`, still failing, and it carries the same information — it is the same
  crossing read on a different axis. It is, however, the axis the adjoint kernels are already
  written on (`V_crit^J` in Section 1.6), so the forward and adjoint observables would at least
  match.
- *The memory ratio itself.* `M_n = ΔP_crit^(n)/ΔP_crit^(1) − 1` is built from thresholds of the
  same run measured against the same baseline `B`, and a baseline error does **not** simply cancel
  in it. With `M + 1 = (P_n − B)/(P_1 − B)`, `∂(M+1)/∂B = (P_n − P_1)/(P_1 − B)²`, so the relative
  sensitivity is `(P_n − P_1)/[(P_1 − B)(P_n − B)]`. On the coarse row, with a second event at
  roughly twice the injected volume, that is about 0.12 per MPa: the measured 1.5 MPa baseline shift
  moves `M + 1` by about 18%, against the 35% it moves `ΔP_crit` by. The ratio is better conditioned
  than its parts by roughly a factor of two, and no more. It remains the quantity the science claim
  is about, and `ΔP_crit` was only ever the proxy protecting it, but a factor of two does not make a
  1% memory signal readable off a 45.6% threshold spread.

**Measured, and it settles the first two options.** The `baseline` sweep crosses `spinup_steps` with
the sill refinement (`.agents/solver.md`). The spun-up baseline is *not* an equilibrated state: it
rises monotonically with the number of spin-up steps, by 2.3 to 4.5 MPa over 1 to 4 steps, so
`ΔP_crit` measured from it collapses from 4.283 to 1.876 MPa at fixed mesh. The spread across
spin-up is larger than the spread across mesh. The absolute event pressure, by contrast, barely
moves: 0.08% and 0.17% across the same spin-up change, and 0.44% to 0.52% across everything. Failure
is a stress criterion, so the pressure at which the host fails is a property of the model and not of
how long it was loaded first — the invariant is the absolute pressure, and the drift is in the
reference being subtracted from it. A lithostatic reference does not help either: with `ρ0[1] = 2900`
and the sill at 4.5 km, lithostatic is 128.02 MPa, *above* every measured event pressure, because an
extending section is under-pressured; `P_event − P_lith ≈ −1.5 MPa` is worse conditioned than what
it replaces.

**The consequence for the gate's wording.** The model's accuracy is an absolute quantity, and the
sill and clean-mesh sweeps below put a number on it: 0.09 MPa between the two resolved sill
refinements, 0.45 MPa across a fourfold element refinement at fixed sill, and 0.005 MPa between two
meshes once the section is equilibrated. State the gate that way — an absolute tolerance on the
event pressure, tied to `M_min` — rather than as a relative tolerance on a difference of two large
numbers, which is what makes 45.6% out of 0.48%. And the arithmetic that follows is the number the
science plan needs: with thresholds of about 4 MPa, ±0.45 MPa is roughly ±11% on `M_n`, so a 1%
memory signal needs about 0.04 MPa. That is an order of magnitude beyond the element-refinement
error and about the size of the equilibrated one. Either `M_min` is of order 10%, or the protocol
must be run from an equilibrated state whose own difficulty is described below. Choosing between
those is Phase 0's, not the code's.

**Where the error lives: the reservoir geometry, then the baseline.** The `sill` family was run
(refinement 3, 4, 6, 8 at `max_area = 8.0e6`) and it isolates the first term. The event pressure is
126.0424, 126.0424, 126.5441 and 126.6351 MPa: refinements 3 and 4 are bit-identical, because
`Triangulate` returns the same 284-element mesh for both, so the first honest step is 4 to 6, which
moves 0.50 MPa, and 6 to 8 moves 0.09 MPa. The reservoir is a polygonal ellipse whose pressure is
read on its own elements, and below refinement 6 that polygon, not the element size, is the error.
The `mesh` family varies both at once and cannot say so; its 45.6% is therefore not a refinement
sequence at all. Repeating it at fixed refinement 8 over 502, 847, 1 460 and 2 684 elements gives
`ΔP_crit` 12.4%, still failing, but `P_event` 0.35% — 0.446 MPa — and `P_baseline` 0.82%. The
element size is worth about half the error the sill polygon was worth, and it is the baseline that
carries twice the spread of the event pressure, which is the signature of the conditioning above
rather than of a stress field that has not converged.

**The baseline is a transient, and equilibrating it exposes a different problem.** Printing the
spin-up trajectory (`run_cycles` now records `spinup_history`) shows the mesh dependence of the
baseline is a starting transient, not a converged difference: at 228 and 502 elements the two
sections begin 1.885 MPa apart and are within 0.034 MPa by step 12. Carried to 40 spin-up steps the
two baselines agree to 0.0053 MPa — 0.004% — the event pressures to 0.0072 MPa, and `ΔP_crit`
spreads 1.9%, which is inside the gate as originally written. This is the first configuration that
passes G1, and it must not be reported as one, because of what the equilibrated state is: both runs
fire at the *first* recharge step, 0.016 kyr, with `ΔP_crit ≈ 0.10 MPa`, which is the resolution
floor of the crossing search itself. The section that the spin-up converges to is already at its own
failure criterion. There is nothing left to load, so the threshold is no longer a measurement of
when the host fails — it is a measurement of the first time step.

That is a finding about the configuration, not about the discretisation, and the lever is Phase 0's:
`T₀`, the reach depth of the corridor and the extension rate together decide whether the section
this model relaxes to is marginally stable or has a real margin. The numerical conclusion stands on
its own — the baseline's mesh dependence is a transient that refinement of the *time* axis removes,
and the absolute event pressure is the invariant throughout — but a gate cannot be declared met on a
run whose threshold is one time step wide. Both must be true at once: an equilibrated baseline *and*
a section with a margin to load through. Until the protocol settles that, quote the absolute event
pressure with its 0.45 MPa element-refinement error and treat `ΔP_crit` as a derived number whose
reference must be stated with it.

**What is still unrun.** The `time_step` family now holds the spun-up time fixed at 2 kyr while the
step inside it is refined, because `spinup_steps` steps of `Δt` is `spinup_steps * Δt` of spin-up and
the earlier form confounded the step with the state it measured from. That family and `domain` are
the two numerical sweeps left. Whether the true `M_2` moves as much as the sensitivity estimate
allows remains empirical — no run has measured a second event at two refinements — and that
two-event diagnostic is still the cheapest experiment that settles how the gate should be written.
It is a diagnostic rather than M2 production; the condition that barred repeated intrusion, a
detector defined on the mesh graph, no longer holds. Either outcome is a decision for the science
plan, recorded here with its evidence.

Use three meshes at fixed physical band/regularisation widths, at least two
successive time refinements and two tighter solver tolerances. Repeat the
relevant controls at the first event and a representative late event; convergence
of the first event alone does not establish convergence of accumulated memory.
Refine detector aggregation and path selection separately from the PDE mesh. With the corridor
detector this is structural rather than a discipline: the strip, its bins and its fill threshold are
fixed in metres before a mesh exists, so the PDE mesh enters the verdict only through an area
fraction. What remains to be refined separately is the corridor itself — `corridor_width`,
`corridor_bins` and `corridor_fill` — and the benchmark's `corridor` sweep is where that is
quoted, alongside the `detector` sweep that reads one state under all three rules.
Domain-depth/width checks are independent of mesh refinement.

Conservation diagnostics use the same quadrature as assembly. For fixed geometry
and one step, check the integrated continuity balance

```
∫ div(v) dΩ + ∫ ΔP/(ηb Δt) dΩ − ∫ α ΔT/Δt dΩ − ∫ Q dΩ = 0.
```

Also evaluate boundary flux independently and report a relative defect with a
documented nonzero scale; a zero-net-source experiment needs an absolute scale
based on transferred volume. Source cancellation is not a magma mass budget.
Thermal gates account for sensible/latent energy, source work and boundary heat;
moving geometry requires the corresponding moving-domain conservation law.

Gate artifacts: M0 cavity report; M1 first-threshold refinement matrix; M2
rollback/null/conservation and insertion-rule controls; M3 energy and late-event
thermal refinement; M4 all 16 arm outcomes and intervention definitions; A0/A1
transpose and gradient reports; A3 precursor-versus-hard-threshold perturbations;
M5 representative cycle throughput plus failed/censored run accounting; M6
3-D source/traction and geometry checks before orientation claims.

### Benchmark ladder

The science plan's benchmark table mapped onto FEMTools. Fast, deterministic
versions go in `test/`; full-resolution versions are miniapps under `examples/`. Every
benchmark that measures accuracy or solver cost against an oracle also gets its own
folder, `examples/benchmarks/<physics>/<name>/<name>.jl`, that sweeps mesh, contrast,
`Δt` and tolerance and prints one line per run with its `converged` flag. The first is
`examples/benchmarks/stokes/elliptical_cavity/`, for the elastic reservoir row below.

| Component | Oracle | Pass criterion (plan) | Where |
|---|---|---|---|
| Elastic reservoir | Existing infinite-plane cavity with prescribed outer displacement | Pressure, area change and opening within 2%; separately report displacement norm | Existing test/setup/sweeps; historical 0.3% report. This is not yet a free-surface geodetic benchmark |
| Free-surface deformation | Matched plane-strain half-space reference or independent refined solver; appropriate source geometry for any 3-D analytic comparator | Relevant surface components within 2%, away from singularities | Additional miniapp before geodetic validation |
| Pressure-volume stiffness | Analytic stiffness of a pressurised elliptical cavity | Within 2% | miniapp |
| Visco-elastic relaxation | Maxwell relaxation around a cavity | Relaxation time within 5% | miniapp |
| Thermal | Stefan problem with latent heat | Front within 2% | `test/` 1-D column |
| Dike opening | Sneddon-type crack against the eigenstrain band; hydraulic consistency of band and reservoir pressure | Profile within 2% | **Done**: `examples/reykjanes/dike_crack.jl`, sweeps in `examples/benchmarks/stokes/dike_crack/`, guard in `test/test_dike_functions.jl`. Profile 1.6% to 0.06% under refinement; closure traction within 0.2%; band width free from `b/a` 0.1 to 0.0125 |
| Plasticity | Onset load and band angle on three refinements | `ΔP_crit^(1)` within 5% | miniapp |
| Event protocol | Mass balance; null test with all switches off | Volume error under 1%; `M_n` zero to noise | miniapp plus a tiny `test/` cycle |
| Adjoint gradient | Taylor remainder and central finite difference for every objective and parameter pair, at two solver tolerances | Remainder falls as `ε²` until the solver floor (proposal: order 2 ± 0.1 over two decades) | Tiny mesh in `test/`; full mesh miniapp |
| Threshold consistency | Kernel-predicted `δV_crit^J` against the hard-detector `δV_crit` for a family of `η` and `C` perturbations | Sign agreement above the GAP-15 noise floor | miniapp on the Level 1 mesh |
| Observation operator | Linear-field reproduction and transpose dot-product identity | Scale-aware `Float64` tolerance, nominally `1e-12`; precision-scaled for `Float32` | `test/` |
| Synthetic recovery | Matched-model truth recovery, then deliberately mismatched elastic inversion of VEP truth | Predictive residual, identifiable combinations, interval coverage over noise draws and prior sensitivity; drift reported separately | `reykjanes/` |
| Cross-code | Export the Phase 2-3 benchmark cycle | Agreement within 5% | data file for the second code |

Regression gates for every `src/` gap, following `testing.md`:

- The existing DP, viscoelastic and coupled tests pass unchanged (behaviour with
  new options off preserves pointwise behavior and assembled results within
  declared floating-point tolerances).
- The 2-D and 3-D adjoint tests pass, since they share `deviatoric_stress`. With
  `objective_P = nothing` preserve validated incompressible results; correcting
  an incomplete compressible transpose is expected to change that result.
- Zero-allocation and type-inference tests cover any new hot-path function;
  `Float32` and `Float64` both run; JET and Aqua stay green.
- Element Jacobian (ForwardDiff) against finite differences for each new
  constitutive term.
- Deliberate non-convergence still throws or reports `converged = false`.
- Full `Pkg.test()` after any change to `deviatoric_stress`, kernel argument
  lists or the module load order.

Documentation and guide updates per gap (from `documentation.md` and
`AGENTS.md`): docstrings for every exported name; `docs/src/stokes.md` and
`heat_diffusion.md` for new physics; `.agents/solver.md` for constitutive
additions and the GAP-15 findings; `.agents/API.md` for new exports;
`.agents/meshing.md` for the adjacency helper; `.agents/miniapps.md` and
`test/test_example_paths.jl` for each new maintained example;
`.agents/testing.md` for new oracles (the Taylor test). For the adjoint gaps,
`.agents/solver.md` also records `objective_P`, the contraction family list, the
single-step meaning of the adjoint and the table of non-smooth switches.
`REYKJANES_PLAN.md` is the current planning reference; the removed
`ADJOINT_PERF_PLAN.md` is historical git material, not a required local file.

## Part 4. Decisions and risks

### Decisions

| ID | Decision | Recommendation | What changes if reversed |
|---|---|---|---|
| D1 | Where code lives | Kernels in `src/`; L1 and L2 drivers in `examples/reykjanes/`; Level 0, data and inference in `reykjanes/` with its own environment | A separate package would decouple release cadence but duplicate meshing and I/O helpers |
| D2 | Dike intrusion mechanism | State injection into `τ_old` and `Q` first (GAP-11), validated against a crack solution. **Validated** for one band and one step: profile and closure traction within 0.2% under refinement, band width free | Fallback is pressure constraints on band and reservoir DoFs (approach A), which touches the Powell-Hestenes update |
| D3 | Tensile cutoff | Event-level first (GAP-6); constitutive only if Gate G1 needs it (GAP-5) | Constitutive cutoff moves earlier and becomes a Phase 2 task |
| D4 | Damage update | Lagged within a step (GAP-2) | In-loop update adds inner nonlinearity but removes the one-step lag |
| D5 | Creep viscosity | Consistent local constitutive solve and tangent inside the residual (GAP-3), verified by GAP-19/25 | A lagged viscosity is a different discrete model unless its fixed-point coupling is differentiated; keep preconditioning distinct from physics |
| D6 | Thermal resolution | Same mesh as mechanics, which the coupled solver requires today. Resolve the aureole with local refinement; test convergence in Phase 4 | A separate thermal mesh or 1-D sub-grid model means changing `solve_coupled_dyrel!`'s node-layout requirement |
| D7 | Winkler base | Deferred; vary domain depth with a free-slip base | Adds boundary-face assembly (GAP-8) |
| D8 | Backend | CPU through M4; GPU from M5 for sweeps | Earlier GPU work needs the memory budget of GAP-15 item 5 |
| D9 | Adjoint scope | EL for conditional failure kernels after GAP-25; full-window finite differences or WIN for evolving-window objectives; no EVT | WIN adds GAP-24 and state storage/recomputation; fewer parameters do not make EL a valid history derivative |
| D10 | First failure-proximity functional | Compare normalised softplus of post-return versus trial margin; positive fixed `s_f`; fixed physical corridor; choose with GAP-20 | A flat post-return objective or an over-dominant trial margin can each fail threshold consistency |
| D11 | Elastic source model in the synthetic-bias study | At Level 1 the same FE model with `η → ∞`; analytic Mogi or Okada only for 3-D | An analytic 2-D half-space cavity does not cover an elliptical sill, and a 3-D comparator does not match a 2-D truth |
| D12 | Initial thermal-field parameterisation for inversions | A scalar geotherm plus one Gaussian magmatic anomaly first (three or four parameters); basis functions only after identifiability tests | None in the solver: the field stays a nodal array and the chain rule from `K_T` is host-side |
| D13 | Surface-load parameter | Sensitivity to the traction `q_s` per surface face; convert with `ρ_w g` on water faces only | Sensitivity to `h_w` alone is meaningless on land, where `K_q` instead shows where added load (ice, lava) would matter |
| D14 | Gradient tolerance policy | Verification runs: `Float64` and a tight tolerance, at two tolerances. Production kernels: the tolerance calibrated in GAP-15 item 7 | A production tolerance in a Taylor test measures solver noise, not the gradient |

### Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DR does not converge or is very slow in the elastic-dominated, high-contrast regime | Medium | High | M0 spike and GAP-15 items 1-2 before any cycle; contrast and mesh sweeps |
| Threshold noise floor comparable to `M_n` | Medium | High | Bisection on the crossing; measure noise with the null test; tighten tolerances; fix the assembly order if atomics contribute |
| Regularisation or `Δt` sets `ΔP_crit` | Medium | High | Gate G1 includes `η_reg` and `Δt` sweeps; non-local fallback |
| Eigenstrain injection misrepresents a fluid-filled dike | Medium | High | Crack benchmark and hydraulic-consistency check before Phase 3; approach A fallback |
| Kernel-argument growth breaks the adjoint or the allocation tests | Low-Medium | Medium | Optional trailing `nothing` arguments; adjoint and allocation tests in every PR |
| Run cost (steps × cycles × iterations × 16 arms) exceeds CPU budget | Medium | Medium | Adaptive `Δt`; GAP-15 item 5 estimate; GPU from M5 |
| Fixed-mesh small-strain approximation fails in the dike swath | Low-Medium | High | GAP-13 deformation/transport checks; validate moving-mesh conservation before extending scope |
| 3-D `Q` or DR limits block Phase 7 | Medium | Medium | Verify 3-D source interpolation and storage; budget from mesh quality, scaling, contrast and measured convergence |
| The return map's kink at yield makes kernels noisy exactly where failure is near | Medium | Medium | Sub-critical reference state; kink audit with every kernel; softplus objective; subgradient documented. A smooth return blend only if GAP-21 shows the need |
| Gradient error from the DR tolerance hides in Taylor tests or kernels | Medium | Medium | D14; GAP-15 item 7 measures it on the null case; tight-tolerance verification runs |
| Kernels correlate poorly with the hard `ΔP_crit` change | Medium | High for G4A | Threshold-consistency check (GAP-20) before any kernel is interpreted; report the hard threshold as primary |
| Event-local kernels cannot answer a history question | Medium | Medium | `K_τold`, `K_D`, `K_T` show which inherited fields matter; GAP-24 only if G4A or G4B demands it |
| The adjoint work competes with M0 to M3 for one developer | High | High | Budget GAP-25 explicitly; finish forward gates; defer Phase 5B or GAP-23 before weakening verification |
| 3-D adjoint for surface load is large | High | Low | Optional paper D; prove symmetry or implement full transpose; choose storage/operator from measured peak memory and convergence |
| A forward-physics PR breaks the adjoint silently | Medium | Medium | GAP-19 inside the GAP-2, 3 and 5 PRs; the adjoint tests are part of every regression gate |

## Immediate implementation sequence

These are reviewable slices, not permission checkpoints. Follow dependencies;
the month labels above do not override failing gates.

1. **Reproduce the baseline.** Preserve the existing dirty tree; capture its
   provenance, run the cavity regression and smallest headless baseline, and
   reclassify historical failures. Resolve the Julia-version discrepancy from
   actual logs. Record weighted reservoir pressure, residuals and source balance.
2. **Freeze protocol conventions.** Specify stress/strength units, physical
   versus numerical band width, detector Boolean rule, pressure reference,
   hydraulic-pressure assumption, arrest closure and memory intervention table
   in a versioned configuration. Tie each to its acceptance check here.
3. **Minimal first failure.** Deliver GAP-1 diagnostic plastic outputs plus
   GAP-6 stress/adjacency and GAP-10 equilibrated injection. First add reusable
   function-only setup; drivers still execute their default entry point. Gate on
   hydrostatic/shear/plane-strain sign tests and a source-normalisation oracle.
4. **Safe repeated stepping.** Deliver GAP-12 rollback and GAP-11 bracketed
   trials with a forced-failure replay test. Run first-threshold mesh/time/domain
   studies before repeated intrusion. Add eigenstrain/hydraulic and conservation
   benchmarks before opening M2 cycle production.
   *Done:* the GAP-12 snapshot, the GAP-11 step transaction with its retries and
   the driver's time loop, the crossing search of step 1, the crack benchmark
   that validates the eigenstrain mechanism, the amplitude solve and reservoir
   sink of step 3, and `run_cycles`, which puts them together into repeated
   recharge-and-intrusion cycles with every rule named in `EventProtocol`.
   The first-threshold mesh study of this item is done, and **Gate G1 failed at
   77.7%** because the detector's path criterion lived on the mesh graph (GAP-11
   above). *Done since:* the mesh-independent detector. `EventProtocol` now
   carries `detector`, one of `:corridor_column` (the default), `:corridor_fraction`
   and `:graph_path` (the old adjacency rule, kept for comparison), with the
   corridor's geometry in `corridor_width`, `corridor_bins` and `corridor_fill`.
   The corridor is a fixed vertical strip of host rock from the reservoir crest to
   the target depth, cut into bins of equal height, and a bin's occupancy is the
   *area* of failed material in it over the area the mesh put there, each element
   weighted by the fraction of its own quadrature that failed. Nothing in it is a
   function of the element graph. `test/test_dike_functions.jl` checks the
   property directly: the same failure field on meshes whose element size differs
   by a factor of four gives identical bin fractions and the same verdict.
   The mesh sweep was repeated under the new default and **Gate G1 still fails, at
   45.6%** (was 77.7%). The detector is no longer the dominant term: the two meshes
   at sill refinement 3 now agree to 0.015%, where the graph rule gave 0.8%, and at
   2 684 elements the corridor and graph rules agree to 0.3%, so the two detectors
   converge to the same threshold and their coarse-mesh disagreement was a
   resolution effect. What is left is conditioning. Across the four meshes the
   event pressure moves 0.48% and the spun-up baseline 1.56%, while their
   *difference* `ΔP_crit` moves 45.6%: the amplification is `P/ΔP ≈ 25` to `30`, so
   a 5% gate on `ΔP_crit` demands about 0.2% convergence of two ~126 MPa pressures.
   *Where it ended up.* Three more sweeps were run and they locate the error rather
   than remove it. `baseline`: the spun-up baseline is not a state to converge — it
   drifts upward with every spin-up step (2.3 to 4.5 MPa over 1 to 4 steps), so
   `ΔP_crit` collapses with it while the absolute event pressure moves 0.08% to
   0.17%. `sill`: refinements 3 and 4 return the same 284-element mesh, and the
   event pressure moves 0.50 MPa from 4 to 6 and 0.09 MPa from 6 to 8, so below
   refinement 6 the polygonal reservoir, not the element size, is the error — which
   is why the `mesh` family, varying both at once, was never a refinement sequence.
   Repeating it at fixed refinement 8 over 502 to 2 684 elements gives `ΔP_crit`
   12.4% but `P_event` 0.35%, or 0.45 MPa. `spinup_history` then showed the
   baseline's mesh dependence is a *transient*: two meshes start 1.885 MPa apart and
   agree to 0.034 MPa by step 12, and at 40 spin-up steps their baselines agree to
   0.0053 MPa and `ΔP_crit` spreads 1.9%. That is inside the gate and is still not a
   pass, because the equilibrated section fires at its first recharge step with
   `ΔP_crit ≈ 0.10 MPa`, the crossing search's own floor: the state the spin-up
   relaxes to is already at its failure criterion. So the gate must be restated on
   the absolute event pressure with an absolute tolerance, and the configuration
   needs a margin to load through — `T₀`, the reach depth and the extension rate,
   which is Phase 0's decision and not a numerics fix. Still to run: `time_step`
   (now holding the spun-up time fixed at 2 kyr while the step inside it refines,
   because the earlier form confounded the step with the spin-up it bought) and
   `domain`, then the two-event `M_2` diagnostic at two refinements. The `detector`
   and `corridor` families quote protocol sensitivity and must not be added to the
   G1 budget.
5. **Adjoint correctness track.** Deliver GAP-25's tiny full-system oracle and
   repair/reject unsupported finite-storage configurations. Only then add
   pressure loads and source/bulk-modulus contractions, with Taylor tests. This
   can proceed independently of forward cycles if developer capacity permits.
6. **Memory, then thermal complexity.** Damage and intervention controls follow
   accepted mechanics; enthalpy insertion and nonlinear creep follow M2's null
   test. Defer GPU sweeps, geodetic inversion and 3-D adjoints until their own
   correctness and representative-cost gates are met.

## Audit references and remaining limits

Repository evidence is primary: [pressure residual](src/stokes/assemblers/pressure_residual.jl),
[adjoint solver](src/stokes/solvers/DR_adjoint.jl),
[frozen operator](src/stokes/assemblers/adjoint_operator.jl),
[rheology](src/stokes/assemblers/rheology.jl),
[material/state types](src/stokes/types/stokes_types.jl),
[gradient test](test/test_stokes_adjoint_api.jl),
[operator tests](test/test_adjoint_operator.jl), and
[cavity setup](examples/reykjanes/elliptical_cavity_setup.jl).

The power-law prefactor convention was cross-checked against the
[ASPECT 2.5 material-model documentation](https://aspect-documentation.readthedocs.io/en/v2.5.0/parameters/Material_20model.html).
The quadratic Taylor-remainder criterion follows the
[dolfin-adjoint verification documentation](https://www.dolfin-adjoint.org/en/latest/documentation/verification.html).
These references support numerical conventions, not calibrated Reykjanes inputs.
Phase 0 still owns flow-law selection, geological values, observational provenance,
covariance and the literature review. This audit changes planning and engineering
notes only; it neither repairs GAP-25 nor establishes new numerical performance.

Validation of this revision: local Markdown links, table column counts, code
fences, whitespace and all 25 gap-register IDs checked; `git diff --check` passed.
Package tests, benchmarks and the Documenter build were not run: no executable
code, docstrings or published manual pages were changed by this audit.
