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
- parsing of maintained example scripts;
- sparse and finite-difference reference oracles for selected solver/adjoint
  behavior.
- `test/test_triangulate_mesh_ext.jl` exercises the Triangulate meshing
  extension. On a unit square: element orientation, the exact total area, the
  area constraint, midside and bubble placement, refinement under a smaller
  `max_area`, construction of a `Mesh` from the result, and the rejected
  argument cases. On a square split by an interior segment: that the mesh
  conforms to the interface, that each half carries its own attribute and
  exactly half the area, and that a per-region `max_area` binds. The
  conformity check is the one that matters, because a non-conforming mesh still
  returns plausible attributes. `Triangulate` is in `[extras]` and the `test`
  target, so the extension is loaded for the suite but not for users of the
  package.
- `test/test_drucker_prager_cap.jl` covers cap geometry, scalar coefficients,
  the coupled return-map residuals and derivatives, hydrostatic tension,
  radial shear/rotation, independent branch switches, failure paths, softened
  geometry, and Float32/Float64 behavior. `test/test_stokes.jl` adds a
  three-step homogeneous-extension recurrence on a `(2, 2)` T7 mesh for
  pressure, `τxx`, `γ`, and `θ`; the old trial-pressure handoff misses step 3
  by ~7e-2 in `θ`.
  Nested ForwardDiff is checked against central differences, frozen cap-state
  blocks against the Enzyme transpose, and a work-conjugate non-associated
  constitutive tangent is checked to remain nonsymmetric.
  Sixteen numeric fixtures pinned to JustRelax revision `14ffc40c` cover
  corrected invariant, pressure, multiplier, and volumetric rate without a
  JustRelax or GeoParams test dependency.
- `test/test_solver_convergence_api.jl` ends with a bulk-viscous block pulled
  from its exact velocity solution, guarding the inner-loop absolute escape
  described in the solver guide. It checks the closed-form pressure *and* the
  iteration count, because that defect wastes iterations without changing the
  converged values. Two earlier versions of this test passed with the fix
  reverted and had to be thrown away: one asserted an iteration bound on the
  buoyancy case, which exits at the outer check before the inner loop is ever
  entered, and one asserted convergence on a setup whose `ηb → ∞` imposed
  incompressibility on a velocity field with `∇·v ≠ 0`, so its apparent
  pre-fix failure was the ill-posedness, not the defect. Run any solver
  regression test against the reverted fix before trusting it.
- `test/test_popov_2d_material_point.jl` is the first paper-specific gate for
  the planned 2-D Popov example: it checks elastic behavior, converged,
  regularized tensile-cap pressure return, corrected pressure handoff to
  momentum assembly, and the four-matrix `τ_store` path that records that
  corrected pressure at integration points, using the paper's friction,
  dilation, cohesion, and tensile-strength ratios. The spatial smoke driver is kept outside the unit
  suite; its tiny `(2, 2)` mesh reaches the requested residual in two outer
  pressure updates with
  `iterMax=500` and `total_iterMax=50_000`.

GitHub Actions currently runs package tests on Julia 1.12 for Ubuntu and macOS.
The matrix contains macOS twice, which is redundant unless one entry is later
given distinct architecture or configuration. Docs build separately using
Julia `1`. `Project.toml` declares Julia 1.11 compatibility. No CI job currently
executes CUDA, AMDGPU, Metal, examples, benchmarks, or multi-rank MPI.

Use Julia 1.12 for the current full-suite gate. The local Enzyme 0.13.181
environment fails on Julia 1.13 with `AssertionError: VERSION < v"1.13"` in
the adjoint test; Julia 1.11.9 cannot resolve the project's JET 0.11.5/0.12
constraint. Focused constitutive tests and the docs build pass on Julia 1.13,
but this is not full-suite coverage or proof of compatibility on that version.

The 3-D sparse reference tests include the function-only
`examples/miniapps/stokes/sinking_block/sinking_block_3D_setup.jl` helper.
They mesh through Gmsh, whose artifact can fail up front with "Gmsh has not
been initialized" — seen on Windows, and reproducible on a clean checkout, so
it is an environment fault and not a solver regression. `_gmsh_usable()` in
`test/test_stokes_3d_reference.jl` probes initialize/finalize once and skips the
testset when it fails, rather than erroring the whole suite. The probe is
deliberately narrow: a Gmsh that initializes but then meshes wrongly still
fails loudly. A run that reports this skip has *not* covered the 3-D sparse
oracle; do not read it as a pass.
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
