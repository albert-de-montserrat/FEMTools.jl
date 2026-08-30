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
- chamber-interface conformance and phase separation for the unstructured 3-D volcano mesh;
- a one-element 3-D mixed pressure-scaling check and component-wise AD through
  the 3-D plastic momentum residual;
- sparse and finite-difference reference oracles for selected solver/adjoint
  behavior.

GitHub Actions currently runs package tests on Julia 1.12 for Ubuntu and macOS.
The matrix contains macOS twice, which is redundant unless one entry is later
given distinct architecture or configuration. Docs build separately using
Julia `1`. `Project.toml` declares Julia 1.11 compatibility. No CI job currently
executes CUDA, AMDGPU, Metal, examples, benchmarks, or multi-rank MPI.

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
