# Testing

## Current reality

`test/runtests.jl` imports shared dependencies, defines `FP32`/`FP64`, and
automatically includes every sorted `test_*.jl` file. New test files therefore
need no manual registration.

The suite covers:

- elements, quadrature, shape functions, coordinates, connectivity, mixed
  meshes, boundary conditions, coloring, sparsity, and external-mesh
  ingestion helpers;
- heat, lithostatic, 2-D/3-D Stokes, coupled thermal--Stokes, rheology,
  assembly, convergence, adjoints, post-processing, and VTK;
- constructor, export/public API, helper, and verbosity contracts;
- `Float32`/`Float64`, type inference, JET optimization checks, Aqua quality
  checks, and selected zero-allocation hot paths;
- parsing of maintained example scripts, including the volcano drivers;
- field-container constructor inference, continuity-source storage, and the
  distinct 3-D Voigt versus Stokes assembler stress orders;
- chamber-interface conformance and phase separation for the unstructured 3-D volcano mesh,
  and interface conformance, area partition and boundary groups of the 2-D sill
  meshes from Gmsh and Triangle;
- exact face-versus-node element adjacency, including a 3-D tetrahedral case,
  in `test/test_mesh_producer_api.jl`;
- a one-element 3-D mixed pressure-scaling check and component-wise AD through
  the 3-D plastic momentum residual;
- sparse and finite-difference reference oracles for selected solver/adjoint
  behavior;
- the 2-D end-to-end adjoint gradient oracle currently uses incompressible,
  purely viscous material (`ηb = K = G = Inf`); plastic operator comparisons
  do not establish a finite-compressibility gradient. `REYKJANES_PLAN.md`
  GAP-25 defines the missing full-system transpose and gradient acceptance gate;
- a closed-form oracle for the compressible elastic solver, source `Q` and soft inclusion:
  `test/test_elliptical_cavity.jl` checks Muskhelishvili's pressurised elliptical hole
  (circle and crack limits, area change, wall traction) and solves one step on a coarse mesh
  against it;
- the source normalisation oracle in the same file: `uniform_pressure_source` and
  `pressure_source_integral` are checked against the shoelace area of the reservoir, against the
  sum of the assembled continuity residual at rest (`Σᵢ ∫ Nᵢ Q dΩ = ∫ Q dΩ`), and for exact
  cancellation of a dike band source by the sink;
- the shared pressure residual checks legacy `ηb` storage, explicit finite-`K`
  storage, and accepted quadrature-point `P_old`;
- the forward solver checks a finite-`K` normalized source step (`ΔP = K Qq Δt`)
  and the residual tests cover nonuniform area-rate integration and invalid support;
- `test/test_state_snapshot.jl` guards the in-memory snapshot: every array field of `StokesDR` and
  `ThermalDiffusionDR` must be classified as state or scratch (a new field fails until it is), the
  round trip is exact and alias-free in both directions, structural changes throw, and a replay
  after a discarded trial that committed `P0` and `τ_old` reproduces a clean solve. The replay
  omission was checked by hand: dropping `P0`, `τ_old` or `Q` from the state breaks it;
- `test/test_event_stepping.jl` guards the transaction built on that snapshot: an accepted step
  keeps what it wrote; a step that reports `converged = false`, leaves a non-finite state or throws
  a `StepRejection` is rolled back to the exact pre-step state; the retries shrink the step and stop
  at the attempt or the step-size budget; and a trial forced to fail after it solved and committed a
  doubled source leaves the two accepted steps that follow equal to a clean run;
- the same file guards the crossing search: on a trial whose state is its own step size, the bracket
  contains the known crossing and is shorter than the tolerance, a detector that switches back off
  is reported as several crossings with the earliest one still resolved, and an absent crossing, an
  exhausted trial budget and a failed trial each report themselves; on the cavity, bracketing a
  pressure threshold leaves exactly the state one step of `hi` reaches from the left state. The
  cavity trial rebuilds `γP` and `dr.M_P` per step size, because with the scaling of another step
  the relaxation does not converge at all;
- and the amplitude solve: on a response of known shape it lands on the root, tries the zero
  amplitude first, doubles before it bisects, and reports a saturating response as
  `no_arrest_bracket` after trying `amplitude_max` itself rather than doubling away. Its oracle is
  the linearity of one elastic step: on the cavity, a target of 1.5 times the pressure of a unit
  injection is reached at amplitude 1.5, and the committed state is the one the reported amplitude
  describes;
- `test/test_event_protocol.jl` guards the rules rather than a run: the protocol validates every
  field and keeps the two defaults that make the detector usable (a nonzero tensile strength and a
  magmastatic head), each Boolean criterion reduces integration-point flags to elements as named
  (`:both` is one point failing twice, not one element failing in two ways), the path averages weigh
  by element area, and the deviatoric normal stress follows a normalised dike normal. The cycle
  itself is a miniapp, not a test: it costs minutes;
- the transfer sink in `test/test_elliptical_cavity.jl`: `balance_dike_source!` reproduces the
  source plus sink built by hand, leaves the band's own source untouched, integrates to zero to
  1e-13 of the injection, and refuses a sink that overlaps its source;
- the dike eigenstrain helper regression in `test/test_dike_functions.jl` checks the
  plane-strain deviatoric split, selected integration-point history updates, and
  invalid opening/normal inputs;
- the same dike helper testset checks plane-strain `s3`, Drucker--Prager `F`,
  hydraulic margin, principal orientation normalization, and degeneracy flags
  on analytic stress states;
- the dike helper testset also checks deterministic active-component and
  shortest-path behavior across an inactive barrier;
- the corridor detector of Gate G1, in the same file: the fixed strip bins only what is inside it
  and tiles it exactly on a uniform mesh, `eligible` keeps the reservoir out, a corridor finer than
  the mesh reports `under_resolved` and refuses to trip however much has failed, and an element is
  weighed by the fraction of its quadrature that failed rather than by a Boolean. The property the
  rule exists for is tested directly: the same failure field on meshes whose element size differs by
  a factor of four gives identical bin fractions, an identical corridor fraction, the same verdict
  under both corridor rules, and a band of identical area. The column and area-fraction rules are
  pinned as *different questions* — half a corridor trips the second and not the first;
- `test/test_event_protocol.jl` also pins that the default detector is not the graph rule, that
  `:graph_path` is still available to compare against, and that the corridor's geometry validates;
- the eigenstrain band is checked against the pressurised crack in the same file, on a coarse mesh:
  the opening profile is within the plan's 2 % gate, the closure traction `P − τ_yy` is the crack
  pressure within 2 %, a band half as thick gives the same traction within 1 %, and the band's
  continuity identity `ΔA/A + P/K` closes to 1e-5. It also pins that the band's *mean* pressure is
  not the dike pressure (`P/p < 0.9`), so that a later change cannot quietly start reading it as one;
- the 2-D Drucker--Prager return test checks the exposed pointwise
  `plastic_multiplier` against the hand-computed simple-shear return;
- `test_plastic_history.jl` checks Float32 per-integration-point history
  updates, shape validation, and element-pinned output writes; the update is
  intentionally exercised as a once-per-step helper rather than inside DR.
- The same test pins the plane-strain `J₂`-equivalent plastic strain-rate
  conversion for a simple-shear flow direction.
- `test_damage_law.jl` checks damage-law validation, lagged weakening,
  implicit healing/update, accepted-history differencing, pure healing,
  no-healing saturation, phase-to-IP parameter mapping, clamping bounds, and
  Float32-compatible parameters; it also exercises the 3-D damaged return.
- the 2-D Stokes `converged` flag implies the outer stopping test passed:
  `test/test_solver_convergence_api.jl` sweeps `total_iterMax` on a small buoyancy case, where
  the inner-loop residual once made a capped run read as converged.

GitHub Actions currently runs package tests on Julia 1.12 for Ubuntu and macOS.
The matrix contains macOS twice, which is redundant unless one entry is later
given distinct architecture or configuration. Docs build separately using
Julia `1`. `Project.toml` declares Julia 1.11 compatibility. No CI job currently
executes CUDA, AMDGPU, Metal, examples, benchmarks, or multi-rank MPI.

The 3-D sparse reference tests include the function-only
`examples/miniapps/stokes/sinking_block/sinking_block_3D_setup.jl` helper.
`test/test_stokes_3d_boundary_values.jl` checks nonzero initial boundary values,
agreement with an already constrained initial guess, and malformed values.
`test/test_example_paths.jl` rejects syntax-error expressions and checks shared
meshing/setup includes and documented paths. It uses only `Test`, so it can
run before package instantiation:
`julia --startup-file=no test/test_example_paths.jl`.

## Commands

Full package test from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Documentation build:

```sh
julia --project=docs docs/make.jl
```

Examples use their own environment:

```sh
julia --project=examples path/to/example.jl
```

Do not treat a direct include of an individual test as universally supported:
many files rely on imports and constants established by `runtests.jl`. During
development, a temporary focused `@testset` or a full `Pkg.test()` is safer than
inventing a second test runner. Keep benchmark runs separate from correctness
tests.

## Test design

- Reproduce a bug with the smallest input that reaches the shared cause.
- Prefer exact checks for topology and API shape; use justified tolerances for
  floating-point numerics.
- Compare optimized/parallel paths with a simple independent oracle, not with
  another wrapper around the same implementation.
- Keep tests deterministic and small. Scientific examples and performance
  sweeps do not belong in the default unit suite.
- Test errors at trust boundaries and convergence failure paths, not only happy
  cases.
- Avoid generated fixture files when an in-memory or temporary-file case is
  enough.
- Do not loosen a global tolerance to cover a localized regression.
- Performance tests must warm compilation, identify hardware/backend/Julia
  version, and report problem size and correctness status.

## Change-to-check map

| Change | Minimum checks |
|---|---|
| Element/shape/quadrature | values, partition of unity, gradients, integration, inference |
| Mesh/connectivity | exact tiny topology, boundary, invalid input, mixed mesh if shared |
| Kernel assembly | element oracle plus assembled result; affected backend |
| Solver iteration | convergence and non-convergence; residual/stat semantics |
| Rheology/material | analytic limit, multi-phase, precision, Jacobian/gradient |
| Adjoint | transpose/operator agreement and objective finite difference |
| Public API/export | API tests, docstring/docs build |
| API tier/signature | export, constructor, inference, and failure-contract checks |
| I/O | temporary-file structure, values, invalid lengths/topology |
| Frontend/example | parse plus small headless execution when maintained |
| 2-D/3-D miniapp | inventory update, parse check, smallest practical headless run |
| MPI | one-rank equivalence, two-rank halo/reduction, timeout-safe failure |
| Performance | correctness gate plus warmed representative benchmark |

Run the full suite for shared element, mesh geometry, dynamic-relaxation,
public API, or module-load changes. A narrowly isolated documentation edit may
need only the docs build and link review.

## CI policy

- Keep the declared minimum Julia version tested when feasible; test a newer
  stable version separately rather than silently replacing the minimum.
- Every duplicate matrix entry must differ in a meaningful dimension.
- Add accelerator CI only with hardware/runner availability and a small stable
  smoke case.
- Add MPI CI with an explicit launcher, small rank count, timeout, and logs that
  identify rank failures.
- Coverage upload failure is currently non-fatal. Do not claim coverage gates
  that CI does not enforce.

## Update this guide when

- test discovery, dependencies, commands, or suite structure changes;
- CI Julia/OS/backend/MPI coverage changes;
- a new supported subsystem needs a standard oracle or test category;
- a flaky or expensive check is moved out of the default suite;
- benchmark methodology becomes part of a performance decision.
