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
  WriteVTK. `write_vtk` writes a field given as a tuple of component arrays as a
  VTK vector, which is what the 3-D drivers pass their velocity as so a viewer
  can glyph it directly;
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
  viscoelastic viscosity `G Δt`;
- the velocity relaxation needs a fine enough mesh in absolute terms. The 3-D
  volcano models diverge in the first Powell-Hestenes step at a far-field
  element size of 4.4 km and above, and converge at 2.9 km. Note that a
  per-side element count sets `hmax = max(Lx/nx, Ly/ny, depth/nz)`, so the same
  count means a coarser mesh on a wider domain; compare absolute element sizes,
  not element counts, when carrying settings between models.

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
| `examples/miniapps/thermal/2D_heat_diffusion/2D_heat_diffusion.jl` | structured Q2/Q9 | Multi-phase transient heat diffusion with the DR solver |
| `examples/miniapps/thermal/2D_heat_diffusion_triangles/2D_heat_diffusion_triangles.jl` | structured T3 | Linear-triangle variant of the thermal DR problem |
| `examples/miniapps/thermal/2D_heat_diffusion_unstructured/2D_heat_diffusion_unstructured.jl` | Gmsh T3 with circular holes | Coupled lithostatic initialization and transient heat diffusion |
| `examples/miniapps/thermal/2D_heat_diffusion_unstructured_T6/2D_heat_diffusion_unstructured_T6.jl` | Gmsh T6 with holes | Quadratic unstructured thermal/lithostatic workflow with node reordering |
| `examples/miniapps/stokes/lithostatic_pressure2D/lithostatic_pressure2D.jl` | unstructured T3 | Dense-inclusion lithostatic-pressure solve and visualization |
| `examples/miniapps/stokes/stokes_2D_elastic_buildup/stokes_2D_elastic_buildup.jl` | structured T7/P1-disc | Viscoelastic stress buildup under imposed deformation |
| `examples/miniapps/stokes/stokes_2D_elastic_buildup_hole/stokes_2D_elastic_buildup_hole.jl` | Gmsh T7/P1-disc | Gravitational loading around a tunnel/free surface |
| `examples/miniapps/stokes/sinking_block/sinking_block.jl` | Gmsh T7/P1-disc | Forward dense-inclusion sinking-block Stokes solve |
| `examples/miniapps/stokes/sinking_block_adj/sinking_block_adj.jl` | Gmsh T7/P1-disc | Forward plus discrete-adjoint density/viscosity sensitivities |
| `examples/miniapps/stokes/stokes_2D_pure_shear/stokes_2D_pure_shear.jl` | structured T7/P1-disc | Multi-step visco-elasto-plastic pure shear |
| `examples/miniapps/stokes/stokes_2D_pure_shear_triangle/stokes_2D_pure_shear_triangle.jl` | Gmsh T7/P1-disc | Unstructured pure shear with an inclusion |
| `examples/miniapps/stokes/vevp/stokes_2D_pure_shear_triangle_adj.jl` | Gmsh T7/P1-disc | Discrete-adjoint sensitivities for the unstructured pure-shear model |
| `examples/miniapps/stokes/stokes_2D_pure_shear_triangle_adv/stokes_2D_pure_shear_triangle_adv.jl` | Gmsh T7/P1-disc | Pure shear with mesh advection |
| `examples/miniapps/stokes/vevp/stokes_2D_shear_bands_triangle.jl` | Gmsh T7/P1-disc | Drucker-Prager shear-localization experiment |
| `examples/miniapps/stokes/stokes_2D_viscous_inclusion_triangle/stokes_2D_viscous_inclusion_triangle.jl` | Gmsh T7/P1-disc | Viscous-inclusion benchmark against an analytical solution |
| `examples/miniapps/stokes/stokes_2D_pure_shear_triangle_hole/stokes_2D_pure_shear_triangle_hole.jl` | Gmsh T7/P1-disc | Pure shear around an empty circular hole |
| `examples/stokes/volcano/volcano_thermal_stokes.jl` | Gmsh T7/P1-disc | Coupled thermal--Stokes volcano cross-section with a magma chamber under pure shear |
| `examples/reykjanes/reykjanes_thermal_stokes.jl` | Triangulate T7/P1-disc; CPU, or CUDA via the top-level `isCUDA` flag | Coupled thermal--Stokes cross-rift section with a flat surface and a thin elliptical magma sill under wall extension; the 2-D volcano driver without the cone. Geometry, geotherm and material values are illustrative, not calibrated. Starting point of the Reykjanes project in `REYKJANES_PLAN.md` |
| `examples/reykjanes/elliptical_cavity.jl` | Triangulate T7/P1-disc; CPU | Milestone M0 of `REYKJANES_PLAN.md`: a soft compressible inclusion injected through `Q` inside a Maxwell host, one elastic step (`Δt ≪ η/G`) on a disc whose outer circle carries the infinite-plane displacement, compared with Muskhelishvili's closed form (`cavity_analytic.jl`). The driver runs the default case of `solve_elliptical_cavity`, which returns pressure, area change, displacement error and solver statistics and lives in the function-only `elliptical_cavity_setup.jl` shared with `test/test_elliptical_cavity.jl` and the benchmark. Iteration counts are in `solver.md` |
| `examples/reykjanes/reykjanes_cycles.jl` | Triangulate T7/P1-disc; CPU | The event protocol of GAP-11 of `REYKJANES_PLAN.md` end to end: recharge the sill until a connected failed path reaches the shallow crust, bracket the step at which it first does, and open a dike on that path by the amount that drains the magma pressure to the closure stress of the path plus the arrest overpressure. The driver runs two events on a coarse section through `run_cycles` in `event_cycles.jl`; every rule it applies is a field of `EventProtocol` in `event_protocol.jl` and is a placeholder until Phase 0 of the science plan fixes it |
| `examples/reykjanes/dike_crack.jl` | Triangulate T7/P1-disc; CPU | The dike-opening row of the benchmark ladder of `REYKJANES_PLAN.md`: a flat elliptical band of ordinary host material is opened by the eigenstrain of GAP-11 (`τ_old` and `Q` alone, no hole and nothing soft) and one elastic step is compared with the pressurised elliptical hole of the same semi-axes, whose `b → 0` limit is Sneddon's crack. The driver runs the default case of `solve_dike_crack`, which returns the opening profile, the band's closure traction, the area change, the continuity balance and solver statistics and lives in the function-only `dike_crack_setup.jl` shared with `test/test_dike_functions.jl` and the benchmark. Measurements are in `solver.md` |
| `examples/miniapps/stokes/solvi2D/Solvi2D_triangle.jl` | Gmsh T7/P1-disc | Viscous-inclusion benchmark against an analytical solution |
| `examples/stokes/stokes_2D_pure_shear_triangle_hole.jl` | Gmsh T7/P1-disc | Pure shear around an empty circular hole |

## 2-D building-block miniapps

These use FEMTools elements, meshes, geometry, coloring, or kernels while
owning substantial assembly or solve logic in the script.

| File | Purpose |
|---|---|
| `examples/miniapps/thermal/2D_diffusion_FEMTools/2D_diffusion_FEMTools.jl` | Q4 transient Gaussian diffusion with explicit sparse assembly and analytical comparison |
| `examples/miniapps/thermal/2D_Poisson/2D_Poisson.jl` | Q4 Poisson assembly/solve baseline |
| `examples/miniapps/thermal/2D_Poisson_AD/2D_Poisson_AD.jl` | Q9 Poisson residual/Jacobian experiment comparing serial, colored, and atomic assembly |
| `examples/miniapps/thermal/2D_Poisson_AD_KA/2D_Poisson_AD_KA.jl` | CUDA/KernelAbstractions version of the AD Poisson experiment |
| `examples/miniapps/stokes/2D_Elasticiy_DR_KA/2D_Elasticiy_DR_KA.jl` | Plane-strain cantilever solved with script-local dynamic relaxation |
| `examples/miniapps/stokes/2D_Elasticiy_Direct_KA/2D_Elasticiy_Direct_KA.jl` | Plane-strain cantilever using sparse direct solution and KA geometry/data movement |

## 3-D package-solver miniapps

| File | Mesh / method | Purpose |
|---|---|---|
| `examples/miniapps/thermal/3D_heat_diffusion_unstructured_hex/3D_heat_diffusion_unstructured_hex.jl` | Gmsh Tet4 despite the historical `_hex` filename | Box with cylindrical inclusions, lithostatic initialization, and transient heat diffusion |
| `examples/miniapps/stokes/lithostatic_pressure3D/lithostatic_pressure3D.jl` | Gmsh Hex8 | Three-dimensional dense-inclusion lithostatic-pressure solve and VTK output |
| `examples/miniapps/stokes/sinking_block_3D/sinking_block_3D.jl` | Gmsh Hex27/Q2 with cell-local P1 pressure | Matrix-free forward 3-D sinking block on CPU or accelerator backend |
| `examples/miniapps/stokes/sinking_block_3D_adj/sinking_block_3D_adj.jl` | same forward mesh/state | Matrix-free 3-D discrete adjoint and material gradients |
| `examples/stokes/volcano/volcano_thermal_stokes_3D.jl` | unstructured chamber-conforming T11/P1-disc; CPU or selected KernelAbstractions backend | Coupled thermal--Stokes volcanic edifice with visco-elasto-plastic crust and an ellipsoidal chamber |
| `examples/stokes/volcano/volcano_thermal_stokes_topo_3D.jl` | same element pair and solver, free surface read from a DEM | Etna under its sampled topography, free slip on the walls and base, chamber beneath the summit |

## 3-D building-block and experimental miniapps

| File | Purpose |
|---|---|
| `examples/miniapps/thermal/3D_diffusion_FEMTools/3D_diffusion_FEMTools.jl` | Hex8 transient Gaussian diffusion with explicit sparse assembly and analytical comparison |
| `examples/miniapps/thermal/3D_Poisson/3D_Poisson.jl` | Hex8 Poisson assembly/solve baseline |
| `examples/miniapps/thermal/3D_Poisson_AD/3D_Poisson_AD.jl` | Hex8 AD residual/Jacobian assembly experiment |
| `examples/miniapps/thermal/3D_Poisson_AD_KA/3D_Poisson_AD_KA.jl` | CUDA/KA AD Poisson experiment with chunked element Jacobians |
| `examples/miniapps/thermal/3D_Poisson_AD_KA_sandbox/3D_Poisson_AD_KA.jl` | Earlier CUDA/KA 3-D Poisson sandbox |
| `examples/miniapps/thermal/3D_Poisson_AD_KA_opt/3D_Poisson_AD_KA_opt.jl` | Optimized variant of the 3-D Poisson sandbox |

The `KA_sandbox/poisson_1step.jl` and `poisson_2step.jl` files are not in this
inventory: they are 3-D finite-difference/KA experiments but do not use
FEMTools.

## Benchmark and support files

These belong to the miniapp ecosystem but are not independent showcase
applications.

| File | Role |
|---|---|
| `examples/benchmarks/thermal/assembly_perf_2D/assembly_perf_2D.jl` | 2-D sparse/colored/atomic diffusion assembly comparison |
| `examples/benchmarks/thermal/assembly_perf_3D/assembly_perf_3D.jl` | 3-D counterpart of the assembly comparison |
| `examples/benchmarks/stokes/adjoint_perf/adjoint_perf.jl` | Includes the 2-D sinking-block adjoint and sweeps mesh size/viscosity contrast |
| `examples/benchmarks/stokes/forward_lambda_perf/forward_lambda_perf.jl` | Compares Gershgorin and measured forward spectral bounds on the sinking block |
| `examples/benchmarks/stokes/forward_lambda_shear_band_perf/forward_lambda_shear_band_perf.jl` | Spectral-bound comparison on the unstructured pure-shear workflow |
| `examples/gmsh_meshing.jl` | Shared Gmsh T3/T6/T7 triangle mesh generation and order conversion, including the volcano section and the flat-surface sill section (`build_gmsh_t7_sill_mesh`) |
| `examples/triangulate_meshing.jl` | Triangulate T7 meshes without Gmsh: the flat-surface sill section (`build_triangulate_t7_sill_mesh`, used by the Reykjanes driver) and a disc with a concentric elliptical inclusion (`build_triangulate_t7_cavity_mesh`, used by the cavity benchmark) |
| `examples/reykjanes/cavity_analytic.jl` | Function-only closed form for a pressurised elliptical hole in an infinite plane under plane strain: displacement field and area compliance |
| `examples/reykjanes/elliptical_cavity_setup.jl` | Function-only `solve_elliptical_cavity` and its report, shared by the cavity driver, `test/test_elliptical_cavity.jl` and the cavity benchmark |
| `examples/reykjanes/injection_source.jl` | Function-only host helpers for the continuity source `Q`: `uniform_pressure_source` turns a volume rate into a constant `Q` over a set of elements, and `pressure_source_integral` integrates a `Q` with the quadrature of the pressure residual. `balance_dike_source!` adds, in place, the uniform reservoir sink that cancels the integral of a band source so that the two integrate to zero, which is cancellation of the source terms and not a magma mass budget. Used for recharge and for dike transfers; covered by `test/test_elliptical_cavity.jl` |
| `examples/reykjanes/reykjanes_setup.jl` | Function-only `build_reykjanes_model`: the cross-rift mesh, material, boundary conditions, initial fields and solver states of the Reykjanes driver, built but not run, so the driver, the event engine and the tests share one model |
| `examples/reykjanes/event_stepping.jl` | Function-only transactional stepping on the snapshots: `attempt_step!` captures the state, runs a step, accepts it when every `StepCheck` passes, and otherwise restores the state and retries with a smaller step, reporting the outcome instead of throwing. A step rejects its own trial with a `StepRejection`. `capture_trials`/`run_trial!` repeat trials from one left state, `bracket_crossing!` samples and then bisects the step at which a detector first trips, and `solve_amplitude!` doubles and then bisects the smallest injection amplitude that drives an observed quantity to a target. Covered by `test/test_event_stepping.jl` |
| `examples/reykjanes/event_protocol.jl` | Function-only `EventProtocol`, the rules that turn a stress state into an event and an event into an intrusion, with `build_event_detector` (the two-stage detector: failed integration points, then a face-connected path from the reservoir to the target depth), `failed_elements`, `closure_stress`, `band_closure_traction`, `magma_pressure` and `normal_deviatoric_stress`. Covered by `test/test_event_protocol.jl` |
| `examples/reykjanes/event_cycles.jl` | Function-only `run_cycles`: recharge steps, crossing refinement at the first trip, and the dike amplitude search on the failed path, with the reservoir sink that balances the injection. Steps 4 and 5 of the plan's protocol (contact rule, enthalpy, full event record) wait for the thermal work of M3 |
| `examples/reykjanes/state_snapshot.jl` | Function-only in-memory snapshot of a run: `capture_state` deep-copies the state arrays of a `StokesDR`, a `ThermalDiffusionDR`, a plastic-history bundle and caller arrays (plus non-array `values` such as time), and `restore_state!` copies them back in place as often as needed. `physical_state` lists the state arrays and `STOKES_SCRATCH_FIELDS`/`THERMAL_SCRATCH_FIELDS` the scratch that the solvers rebuild. Covered by `test/test_state_snapshot.jl` |
| `examples/reykjanes/dike_crack_setup.jl` | Function-only `solve_dike_crack` and its report, shared by the dike driver, `test/test_dike_functions.jl` and the dike benchmark. A band of semi-axes `(a, b)` is `2b √(1 − (x/a)²)` thick, so one uniform eigenstrain opens it into the profile of a uniformly pressurised crack; the eigenstrain is set by a fixed point on the band's own elastic storage |
| `examples/benchmarks/stokes/first_threshold/first_threshold.jl` | Runs `run_cycles` to its first event only and reports `ΔP_crit`, when it happened, how wide the crossing bracket was left, the failed path and the opening. Separates numerical sweeps (mesh, time step, crossing refinement, spin-up, domain), where the threshold must not move, from protocol sweeps (`T₀`, reach depth, `ΔP_arrest`, band width), where its sensitivity has to be quoted. Each run is minutes; the mesh result is in `solver.md` |
| `examples/benchmarks/stokes/dike_crack/dike_crack.jl` | Sweeps mesh, band aspect, `Δt`, storage passes, `ϵ_tol` and domain radius of the eigenstrain dike and reports the opening profile, the closure traction, the continuity balance, iterations and wall time against the closed form; the tables in `solver.md` come from it |
| `examples/benchmarks/stokes/elliptical_cavity/elliptical_cavity.jl` | Sweeps mesh, shear-modulus contrast, `Δt`, `ϵ_tol`, `rel_drop0`/`ncheck` and stress scale of the cavity solve and reports accuracy against the closed form, iterations, wall time and the residual reduction reached; the tables in `solver.md` come from it |
| `examples/miniapps/stokes/mesher/mesher.jl` | Sinking-block geometry launch helper and Gmsh Hex27 order conversion |
| `examples/miniapps/stokes/sinking_block/sinking_block_3D_setup.jl` | 3-D sinking-block forward and adjoint definitions shared by the two drivers and `test/test_stokes_3d_reference.jl` |
| `examples/miniapps/stokes/2D_Elasticity_stress_postprocess/2D_Elasticity_stress_postprocess.jl` | Includes the DR cantilever and projects quadrature stress to nodes |
| `examples/stokes/volcano/volcano_mesh_3D.jl` | Gmsh T10 mesher fragmented by the chamber ellipsoid and enriched to T11 with cell bubbles |
| `examples/stokes/volcano/volcano_mesh_topo_3D.jl` | Same mesher with the top surface displaced onto a sampled topography; reuses the cone mesher's helpers |
| `examples/stokes/volcano/Etna_Topo.jld2` | Etna elevation tile: `x`, `y`, `surf` in km on a 256 x 256 Cartesian grid |

## Miniapp contract

New or polished primary miniapps should follow this shape:

1. Define a callable `main(; kwargs...)` (or a clearly named equivalent) with
   scientific and execution parameters visible at the boundary.
2. Return a `NamedTuple` containing the fields and convergence/output metadata
   needed by tests, benchmarks, and downstream plotting.
3. Execute the default case unconditionally. Put definitions needed by tests
   or other drivers in a shared helper without a top-level solve.
4. Support a headless, no-output mode such as `show_plot=false` and
   `write_output=false`.
5. Accept a backend argument when the implementation is backend-neutral; do not
   import CUDA or select a GPU globally unless the file is explicitly a CUDA
   experiment.
6. Rely on `--project=examples`; do not mutate the active Julia environment from
   inside a maintained script.
7. Keep default problem sizes runnable on a normal workstation. Put expensive
   sweeps in `examples/benchmarks/`. A verification miniapp that measures
   accuracy or solver cost against an oracle also gets its own benchmark folder,
   `examples/benchmarks/<physics>/<name>/<name>.jl`: it includes the shared
   function-only setup, keeps the sweep settings in a function so a subset can
   run, warms up first, prints one line per run with its `converged` flag, and
   records a failing case as a row instead of aborting.
8. Use public FEMTools APIs for supported workflows. Script-local low-level
   experiments are allowed, but label them as such and avoid presenting them as
   stable package API.
9. Write outputs beneath an explicit path, and do not commit generated VTK,
   images, or time-series data unless they are intentional documentation assets.
10. Report non-convergence explicitly and do not plot or differentiate a failed
    solve as though it were valid.

The previously documented dike-influx driver is absent from the current tree;
its Stokes manual description is retained as historical workflow context.

## Direction

Polish existing miniapps before adding near-duplicates:

1. Keep headless numerical definitions in shared helpers without default solves;
   runnable drivers execute their default case when run or included.
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

The 3-D volcano driver keeps mesh generation and post-processing on the host,
but moves solver connectivity, phases, boundary data, and state arrays to the
selected backend. It defaults to `CPU()` and can be launched with
`FEMTOOLS_BACKEND=cuda` after CUDA is available in the examples environment.

The Reykjanes dike workflow keeps reusable dike transformations under
`examples/reykjanes/dike_functions/`; these helpers are example-level contracts,
covered by focused tests, and should move to core only after a second maintained
consumer exists.

## Acceptance checks

For a miniapp change:

- parse the changed script and every support file it includes;
- run its smallest meaningful headless/no-output case when dependencies and
  hardware are available;
- assert convergence or the expected analytical/reference error;
- verify the returned result contract rather than scraping printed output;
- check that shared setup helpers do not launch a simulation and that runnable
  drivers execute their default case;
- verify output in a temporary directory when writing changes;
- run the relevant package regression tests when the miniapp exposed a core
  bug or relies on changed core behavior;
- record exact hardware/backend/problem size for benchmark claims.

The current suite's parse list in `test/test_example_paths.jl` is explicit and
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
