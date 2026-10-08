# API simplification baseline

Revision: `babe3b488e63209d070dcc93e95b5f8d0a5e9bf1`, with the existing six
benchmark deletions preserved. No numerical source changes have been made.
Stage 1 is in progress; performance and complete workflow baselines are pending.

## Inventory and migration map

`benchmarks/api_inventory.jl` inventories every public name, runtime method,
and textual reference in source, tests, manual, README, examples, benchmarks,
and extensions. Textual references include comments and definitions: they are
review candidates, not a static call graph or proof that a method is unused.
The initial core inventory has 69 exported names, 38 public unexported names,
and 241 methods. Loading optional extensions can add methods.

| Current family | Methods | Classification and proposed destination |
| --- | ---: | --- |
| `ThermalDiffusionDR` | 4 | Expanded property forms removed; mesh form plus count + `ThermalMaterial` forms remain |
| `LithostaticPressureDR` | 4 | Same, using `ThermalMaterial` for EOS properties |
| `StokesDR` | 4 | Expanded property forms removed; mesh form plus count + `StokesMaterial` forms remain |
| `solver!` | 4 | Two scalar high-level and two expanded methods; canonical `solve!`, separate internal mechanics |
| `solve_stokes_dyrel!` | 6 | Mixed-state and specialized cell-pressure solving; canonical dispatch-based `solve!`, remove redundant forwarding variants |
| `solve_stokes_adjoint_dyrel!` | 3 | Advanced adjoint operation; mesh-owned `solve_adjoint!`, preserve explicit outputs and warm starts |
| `solve_coupled_dyrel!` | 2 | Removed; callers use `solve_coupled!` |
| `solve_stokes_3d!` | 1 | Removed; callers use `solve!` |
| `solve_stokes_adjoint_3d!` | 1 | Removed; callers use `solve_stokes_adjoint_dyrel!` |

Keep reference elements, mesh producers, field containers, post-processing, and
writers initially. Review geometry/cache exports and raw kernel public status
against actual advanced callers before demoting anything. Shared kernels are
implementation mechanics; their mathematical helpers remain available to
independent tests without requiring exported status.

Generate a portable inventory outside the repository:

```sh
julia --project=. benchmarks/api_inventory.jl /path/outside/repo/api-baseline.toml
```

The tool also runs without an output path and prints counts. Its generated
TOML report was executed and checked by parsing it back and comparing the API
records exactly.

## Checks actually run

Environment: Julia 1.13.0, FEMTools 0.2.0, persistent Julia session on Windows.
The installed Juliaup configuration has no Julia 1.12 channel. The stale local
ignored manifest lacked GeoParams; `Pkg.resolve()` restored package loading
without a tracked Project.toml change.

| Existing test file | Result |
| --- | --- |
| `test_heat_diffusion_types.jl` | 100 assertions passed, Float32 and Float64 |
| `test_lithostatic_pressure.jl` | 86 assertions passed |
| `test_solver_convergence_api.jl` | 51 assertions passed, including non-convergence, finite K, histories, and exact initial velocity |
| `test_stokes_quadrature_viscosity.jl` | 24 assertions passed, including Q9/P1 hydrostatics |
| `test_coupled_solver.jl` | 15 assertions passed |
| `test_stokes_3d_boundary_values.jl` | 12 assertions passed |
| `test_stokes_adjoint_api.jl` | 26 assertions passed, gradient versus finite differences |
| `test_plastic_history.jl` | 16 assertions passed |
| `test_heat_diffusion_residual.jl` | First 6 assertions passed; subsequent Tet11 test requires Gmsh, unavailable in the root development environment |
| `test_example_paths.jl` | 101 parse assertions passed; documented-path test had 32 passes and 3 failures referring to existing deleted benchmark files; subsequent testset did not run |

The standalone `benchmarks/api_baseline.jl` reproduces small Q9 thermal and
heterogeneous T7/P1 Stokes cases in both precisions. Its CUDA mode compares
fields with CPU results and disables scalar indexing. These are correctness
smokes, not representative performance benchmarks or full CUDA coverage.

Hardware: NVIDIA GeForce RTX 5070 Ti, CUDA functional. CUDA loading initially
hit the 60-second compilation timeout; a longer retry succeeded.

| Precision and case | CPU residual | CUDA residual |
| --- | ---: | ---: |
| Float32 thermal, relative | 5.177624e-6 | 5.1782777e-6 |
| Float64 thermal, relative | 3.599195901961166e-11 | 3.5991959846923246e-11 |
| Float32 Stokes, absolute | 7.950395788695216e-5 | 7.950021914693742e-5 |
| Float64 Stokes, absolute | 9.550157558712095e-9 | 9.550157568294773e-9 |

CPU/CUDA field agreement passed with `CUDA.allowscalar(false)`. Run the
standalone smoke with the package environment for CPU, or an environment with
CUDA available for GPU:

```sh
julia --project=. benchmarks/api_baseline.jl
julia --project=examples benchmarks/api_baseline.jl --cuda
```

## Remaining stage 1 gates

- Resolve the pre-existing Windows Gmsh failure before claiming the dependent
  tetrahedral thermal, mixed 3-D forward/adjoint, and cap-history gates.
- Resolve documented paths to user-deleted benchmark scripts before claiming
  a clean full suite. Preserve those deletions during this task.
- Collect further representative performance comparisons for later numerical
  changes; constructor workflow timing below is not evidence of a speedup.
- Explicit classification of individual advanced method callers before any
  public signatures are removed.

## First implementation slice

Mesh-aware constructors and two-argument Dirichlet construction are implemented.
Constructor inference, precision, backend placement, 2-D/3-D stress dimensions,
explicit omitted stress history, and input validation passed 53 checks on CPU
and 53 on CUDA with scalar indexing disabled. The smoke runner now uses the new
constructors; its 16 CPU/CUDA checks pass with exactly the residuals above.

The six migrated exact-field scripts passed 46 refinement checks and 24 JLD2
archive checks in the instantiated examples environment (ExactFieldSolutions
0.2.3, KernelAbstractions 0.9.43, Enzyme 0.13.211). The Documenter build passes.
The initial exact-field run overlapped edits and encountered stale methods;
the complete rerun after Revise succeeded. It is not an independent full
pre-change exact-field baseline.

Cold pre-change `Pkg.test()` with proper extras returned 4,381 passes, three
missing-document-path failures, four Gmsh-dependent errors, and one broken test.
Only Julia 1.13.0 is installed locally; this is not CI's Julia 1.12 coverage.

The cold post-change suite returned 4,434 passes, the same three path failures,
the same four Gmsh-dependent errors, and the same broken test. All 53 new CPU
constructor checks passed; no additional failing testsets appeared. Existing
allocation, inference/JET, export/public API, and Aqua checks passed.

Warmed SolKz T7/P1 workflow measurements used resolution 16, viscosity contrast
1e6, 4,738 DoFs, tolerance 1e-10, and three repetitions after a warmup with plots,
archives, and output disabled. Each run converged and passed velocity/pressure
error gates of 0.02/0.05. The legacy variant was reconstructed in memory by
replacing only the two changed constructor calls in the maintained driver.

| Constructor form | Whole-workflow seconds | Allocated bytes |
| --- | --- | --- |
| Legacy | 1.616, 1.622, 1.640 | 116,459,349–116,459,365 |
| Mesh-based | 1.577, 1.619, 1.608 | 116,459,461–116,459,573 |

An initial timing attempt overlapped full package tests and was discarded.
These measurements alternate legacy/mesh calls after all other checks finish.
Runtime is comparable and allocations differ by at most 224 bytes out of about
116 MB for the complete workflow. No speedup or exact allocation equality is
claimed. These constructors run during setup, outside established hot loops.
Removal of old entry points follows caller migration and completed baseline
gates, not the inventory count alone.

## Warmed performance against `main`

Julia 1.12.7, CPU, same script on `main` (`babe3b4`) and on this branch, using
calls that exist in both: a heterogeneous T7/P1 Stokes inclusion (24×24 cells,
viscosity contrast 100, `ϵ_tol = 1e-8`) and one Q9 thermal step (48×48 cells).
Three warmed runs each:

| Case | `main` time (s) | Branch time (s) | `main` bytes | Branch bytes | Iterations / result |
| --- | --- | --- | ---: | ---: | --- |
| Stokes | 20.09, 20.02, 19.12 | 18.45, 19.11, 18.79 | 144 995 592 | 144 995 720 | 29 250 in both; `err_abs` identical |
| Thermal | 0.785, 0.755, 0.769 | 0.847, 0.756, 0.767 | 20 415 008 | 20 415 040 | `maximum(T)` identical |

Timings agree within run-to-run variation; allocations grow by a fixed 128 B
(Stokes) and 32 B (thermal) per call. Both versions allocate about 5 KB per
Stokes pseudo-iteration. An allocation profile (`Profile.Allocs`, 8×8 T7/P1)
attributes all of it to KernelAbstractions CPU launches: about 14 launches per
iteration at roughly 350 B each for the task, `NDRange`, and argument pack.
No FEMTools array is allocated inside the loop; removing the cost would mean
fewer launches, not a local fix.

