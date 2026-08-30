# Miniapps

## Scope

This guide tracks every 2-D and 3-D program under `examples/` that uses
FEMTools directly or through an included driver. Here, a miniapp is a complete
scientific or numerical workflow: it builds a mesh/problem, exercises package
types or solvers, and usually visualizes, writes, or benchmarks a result.

The inventory deliberately excludes 1-D examples, generated `.vtk` output,
and standalone finite-difference sandboxes that do not use FEMTools. Support
files and benchmark wrappers are listed separately from runnable miniapps.

## Current execution model

The intended environment is:

```sh
julia --project=examples path/to/example.jl
```

The example collection is not uniform yet:

- every script executes its default case at file load; including one runs it.
  Code a test needs without that side effect belongs in a shared helper such as
  `examples/gmsh_meshing.jl`, not behind a run guard;
- some legacy scripts call `Pkg.activate` themselves, sometimes selecting the
  package root instead of the examples environment;
- several Poisson/KA experiments import CUDA and select `CUDABackend()` at
  global scope, so they are not portable CPU examples as written;
- plotting commonly uses GLMakie and output commonly uses legacy VTK or
  WriteVTK;
- the test suite parse-checks an explicit subset but does not run the miniapps
  end to end.

`solve_stokes_dyrel!` and `solve_coupled_dyrel!` impose three requirements that
a new driver has to satisfy; each of them diverges the relaxation rather than
reporting a setup error:

- the initial velocity must already satisfy the boundary conditions and be
  divergence-free, not merely be a plausible guess;
- absolute residuals are compared against fixed thresholds, so a dimensional
  crustal-scale problem must be solved in characteristic units;
- the Drucker-Prager regularisation viscosity must stay at or above the
  viscoelastic viscosity `G Δt`.

The thermal half of `solve_coupled_dyrel!` converges on a relative residual. A
time step far below the thermal diffusion time leaves that residual at its
cancellation floor from the first iteration, so its tolerance has to be loosened
to a reachable value; the coupled solve otherwise runs to `total_iterMax` with
the Stokes residual already at machine precision.

The 2-D adjoint miniapps use the solver convention
`R_u^T λ = -J_u` and therefore report material sensitivities as
`J_m + λ^T R_m`; do not import the opposite 3-D contraction sign into them.

Do not infer equal support or maturity from a file living under `examples/`.
The categories below distinguish package-solver workflows from scripts that
use FEMTools as a finite-element building-block library.

## 2-D package-solver miniapps

These exercise FEMTools solver states and public solver entry points.

| File | Mesh / method | Purpose |
|---|---|---|
| `examples/heat_diffusion/2D_heat_diffusion.jl` | structured Q2/Q9 | Multi-phase transient heat diffusion with the DR solver |
| `examples/heat_diffusion/2D_heat_diffusion_triangles.jl` | structured T3 | Linear-triangle variant of the thermal DR problem |
| `examples/heat_diffusion/2D_heat_diffusion_unstructured.jl` | Gmsh T3 with circular holes | Coupled lithostatic initialization and transient heat diffusion |
| `examples/heat_diffusion/2D_heat_diffusion_unstructured_T6.jl` | Gmsh T6 with holes | Quadratic unstructured thermal/lithostatic workflow with node reordering |
| `examples/lithostatic_pressure/lithostatic_pressure2D.jl` | unstructured T3 | Dense-inclusion lithostatic-pressure solve and visualization |
| `examples/stokes/buildup/stokes_2D_elastic_buildup.jl` | structured T7/P1-disc | Viscoelastic stress buildup under imposed deformation |
| `examples/stokes/buildup/stokes_2D_elastic_buildup_hole.jl` | Gmsh T7/P1-disc | Gravitational loading around a tunnel/free surface |
| `examples/stokes/sinking_block/sinking_block.jl` | Gmsh T7/P1-disc | Forward dense-inclusion sinking-block Stokes solve |
| `examples/stokes/sinking_block/sinking_block_adj.jl` | Gmsh T7/P1-disc | Forward plus discrete-adjoint density/viscosity sensitivities |
| `examples/stokes/vevp/stokes_2D_pure_shear.jl` | structured T7/P1-disc | Multi-step visco-elasto-plastic pure shear |
| `examples/stokes/vevp/stokes_2D_pure_shear_triangle.jl` | Gmsh T7/P1-disc | Unstructured pure shear with an inclusion |
| `examples/stokes/vevp/stokes_2D_pure_shear_triangle_adj.jl` | Gmsh T7/P1-disc | Discrete-adjoint sensitivities for the unstructured pure-shear model |
| `examples/stokes/vevp/stokes_2D_pure_shear_triangle_adv.jl` | Gmsh T7/P1-disc | Pure shear with mesh advection |
| `examples/stokes/vevp/stokes_2D_shear_bands_triangle.jl` | Gmsh T7/P1-disc | Drucker-Prager shear-localization experiment |
| `examples/stokes/vevp/stokes_2D_viscous_inclusion_triangle.jl` | Gmsh T7/P1-disc | Viscous-inclusion benchmark against an analytical solution |
| `examples/stokes/stokes_2D_pure_shear_triangle_hole.jl` | Gmsh T7/P1-disc | Pure shear around an empty circular hole |
| `examples/stokes/volcano/volcano_thermal_stokes.jl` | Gmsh T7/P1-disc | Coupled thermal--Stokes volcano cross-section with a magma chamber under pure shear |

## 2-D building-block miniapps

These use FEMTools elements, meshes, geometry, coloring, or kernels while
owning substantial assembly or solve logic in the script.

| File | Purpose |
|---|---|
| `examples/2D_diffusion_FEMTools.jl` | Q4 transient Gaussian diffusion with explicit sparse assembly and analytical comparison |
| `examples/Poisson/2D_Poisson.jl` | Q4 Poisson assembly/solve baseline |
| `examples/Poisson/2D_Poisson_AD.jl` | Q9 Poisson residual/Jacobian experiment comparing serial, colored, and atomic assembly |
| `examples/Poisson/2D_Poisson_AD_KA.jl` | CUDA/KernelAbstractions version of the AD Poisson experiment |
| `examples/elasticity/2D_Elasticiy_DR_KA.jl` | Plane-strain cantilever solved with script-local dynamic relaxation |
| `examples/elasticity/2D_Elasticiy_Direct_KA.jl` | Plane-strain cantilever using sparse direct solution and KA geometry/data movement |

## 3-D package-solver miniapps

| File | Mesh / method | Purpose |
|---|---|---|
| `examples/heat_diffusion/3D_heat_diffusion_unstructured_hex.jl` | Gmsh Tet4 despite the historical `_hex` filename | Box with cylindrical inclusions, lithostatic initialization, and transient heat diffusion |
| `examples/lithostatic_pressure/lithostatic_pressure3D.jl` | Gmsh Hex8 | Three-dimensional dense-inclusion lithostatic-pressure solve and VTK output |
| `examples/stokes/sinking_block/sinking_block_3D.jl` | Gmsh Hex27/Q2 with cell-local P1 pressure | Matrix-free forward 3-D sinking block on CPU or accelerator backend |
| `examples/stokes/sinking_block/sinking_block_3D_adj.jl` | same forward mesh/state | Matrix-free 3-D discrete adjoint and material gradients |
| `examples/stokes/volcano/volcano_thermal_stokes_3D.jl` | unstructured chamber-conforming T11/P1-disc | Coupled thermal--Stokes volcanic edifice with visco-elasto-plastic crust and an ellipsoidal chamber |

## 3-D building-block and experimental miniapps

| File | Purpose |
|---|---|
| `examples/3D_diffusion_FEMTools.jl` | Hex8 transient Gaussian diffusion with explicit sparse assembly and analytical comparison |
| `examples/Poisson/3D_Poisson.jl` | Hex8 Poisson assembly/solve baseline |
| `examples/Poisson/3D_Poisson_AD.jl` | Hex8 AD residual/Jacobian assembly experiment |
| `examples/Poisson/3D_Poisson_AD_KA.jl` | CUDA/KA AD Poisson experiment with chunked element Jacobians |
| `examples/Poisson/KA_sandbox/3D_Poisson_AD_KA.jl` | Earlier CUDA/KA 3-D Poisson sandbox |
| `examples/Poisson/KA_sandbox/3D_Poisson_AD_KA_opt.jl` | Optimized variant of the 3-D Poisson sandbox |

The `KA_sandbox/poisson_1step.jl` and `poisson_2step.jl` files are not in this
inventory: they are 3-D finite-difference/KA experiments but do not use
FEMTools.

## Benchmark and support files

These belong to the miniapp ecosystem but are not independent showcase
applications.

| File | Role |
|---|---|
| `examples/benchmarks/assembly_perf_2D.jl` | 2-D sparse/colored/atomic diffusion assembly comparison |
| `examples/benchmarks/assembly_perf_3D.jl` | 3-D counterpart of the assembly comparison |
| `examples/benchmarks/adjoint_perf.jl` | Includes the 2-D sinking-block adjoint and sweeps mesh size/viscosity contrast |
| `examples/benchmarks/forward_lambda_perf.jl` | Compares Gershgorin and measured forward spectral bounds on the sinking block |
| `examples/benchmarks/forward_lambda_shear_band_perf.jl` | Spectral-bound comparison on the unstructured pure-shear workflow |
| `examples/gmsh_meshing.jl` | Shared Gmsh T3/T6/T7 triangle mesh generation, order conversion, and the volcano cross-section geometry |
| `examples/stokes/sinking_block/mesher.jl` | Sinking-block geometry launch helper and Gmsh Hex27 order conversion |
| `examples/stokes/sinking_block/sinking_block_3D_setup.jl` | 3-D sinking-block forward and adjoint definitions shared by the two drivers and `test/test_stokes_3d_reference.jl` |
| `examples/stokes/volcano/volcano_mesh_3D.jl` | Gmsh T10 mesher fragmented by the chamber ellipsoid and enriched to T11 with cell bubbles |
| `examples/elasticity/2D_Elasticity_stress_postprocess.jl` | Includes the DR cantilever and projects quadrature stress to nodes |

## Miniapp contract

New or polished primary miniapps should follow this shape:

1. Define a callable `main(; kwargs...)` (or a clearly named equivalent) with
   scientific and execution parameters visible at the boundary.
2. Return a `NamedTuple` containing the fields and convergence/output metadata
   needed by tests, benchmarks, and downstream plotting.
3. Run the top-level entry point unconditionally. Reusable code that must be
   included without launching a solve belongs in a support file.
4. Support a headless, no-output mode such as `show_plot=false` and
   `write_output=false`.
5. Accept a backend argument when the implementation is backend-neutral; do not
   import CUDA or select a GPU globally unless the file is explicitly a CUDA
   experiment.
6. Rely on `--project=examples`; do not mutate the active Julia environment from
   inside a maintained script.
7. Keep default problem sizes runnable on a normal workstation. Put expensive
   sweeps in `examples/benchmarks/`.
8. Use public FEMTools APIs for supported workflows. Script-local low-level
   experiments are allowed, but label them as such and avoid presenting them as
   stable package API.
9. Write outputs beneath an explicit path, and do not commit generated VTK,
   images, or time-series data unless they are intentional documentation assets.
10. Report non-convergence explicitly and do not plot or differentiate a failed
    solve as though it were valid.

## Direction

Polish existing miniapps before adding near-duplicates:

1. Make every primary 2-D/3-D package-solver miniapp headless-capable and
   callable through a keyword-driven entry point.
2. Remove environment activation from scripts and use the examples project as
   the single dependency declaration.
3. Move hard-coded accelerator selection behind a backend argument or retain it
   only in clearly named accelerator experiments.
4. Consolidate mesh-generation and plotting helpers only after the same stable
   pattern appears in multiple miniapps.
5. Promote script-local numerical machinery into `src/` only when it is a
   supported reusable package capability with tests and docs.
6. Select a small representative smoke set across 2-D/3-D, structured/
   unstructured, scalar/Stokes, and CPU/backend paths instead of running every
   expensive application in default CI.

## Acceptance checks

For a miniapp change:

- parse the changed script and every support file it includes;
- run its smallest meaningful headless/no-output case when dependencies and
  hardware are available;
- assert convergence or the expected analytical/reference error;
- verify the returned result contract rather than scraping printed output;
- check that support files can be included without launching a simulation;
- verify output in a temporary directory when writing changes;
- run the relevant package regression tests when the miniapp exposed a core
  bug or relies on changed core behavior;
- record exact hardware/backend/problem size for benchmark claims.

The current suite's parse list in `test/test_type_stability.jl` is explicit and
does not cover this entire inventory. Update it when a miniapp becomes part of
the maintained set; add a tiny execution test only when it can remain stable
and reasonably fast.

## Update this guide when

- a 2-D/3-D example, benchmark, or support file is added, removed, renamed, or
  changes role;
- a miniapp changes mesh, element pair, physics, solver, backend, output, or
  execution contract;
- parse/smoke coverage changes;
- a script-local feature moves into the package or a package workflow moves
  into an example;
- the supported miniapp set or standard run command changes.
