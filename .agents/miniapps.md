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
| `examples/miniapps/stokes/popov_extension_2D/popov_extension_2D.jl` | Triangulate T7/P1-disc | Popov tensile-cap restrained-extension case in nondimensional paper units, with the release's weak seed; see below |
| `examples/miniapps/stokes/solvi2D/Solvi2D_triangle.jl` | Gmsh T7/P1-disc | Viscous-inclusion benchmark against an analytical solution |
| `examples/stokes/stokes_2D_pure_shear_triangle_hole.jl` | Gmsh T7/P1-disc | Pure shear around an empty circular hole |

### Popov extension driver: seeded from the authors' release

The driver requires JustRelax constitutive and
physical-step parity without GeoParams before paper reproduction. In
particular, corrected pressure must seed the next step, and stress/history
rates must use the same softened material. The driver carries the accepted corrected
IP pressure as `P_old`; measurements below that mention trial-pressure growth
predate that handoff. A 5-step coarse run (`max_area = 3e-3`) converges with
corrected pressure bounded near `pT` (min `-0.108` for `pT = -0.1`) while the
trial field reaches `-0.41`.
The result also returns the cellwise `χ` profile along a horizontal section;
the default `section_y=0.25` matches Figure 6b's `A–A′` line.
At five nondimensional time units, its peak changes by 0.31% between
`max_area=3e-3` and `1.5e-3`, and by 0.004% when `Δt` is halved on the coarse
mesh. These short runs validate the comparison workflow, not the 100 kyr result.
The paper's embedded 375 dpi Figure 6b raster gives the `max_area=3e-4`
reference profile approximately: far-field `χ=1.83e-2`, peak `χ=6.77e-2` at
`x=0.501 m`, and excess-peak FWHM `4.18e-2 m`. Provisional tolerances are ±10% for
baseline and peak and ±0.01 m for peak position and FWHM; these are
raster-derived targets, not source data.
A 100 kyr coarse run (`max_area=3e-3`, 428 cells, `dt=1`, 2000 steps) completes
in 646.6 s and gives baseline `1.6849e-2`, peak `5.0180e-2` at `x=0.4886 m`,
and FWHM `4.4841e-2 m`. Baseline and width meet the provisional tolerances;
peak and position do not.
At `max_area=1.5e-3` (784 cells, 737 s) the same run gives baseline `1.699e-2`
(−7.1%), peak `5.848e-2` (−13.6%) at `x=0.5137 m`, and FWHM `7.25e-2 m`. The
peak converges toward the reference; the band spans about four ~0.02 m cells, so
position and FWHM remain resolution-bound. That FWHM is measured from the
profile median, interpolated between cell centers.
This `1.5e-3` run is the accepted Figure 6b comparison; the `3e-4` paper-mesh run is
out of scope for cost, so no paper-resolution parity is claimed.

The driver reproduces the setup of `MESH/tensile.py` and `CODE/tensile.py` in
the archived GeoTech2D release (DOI 10.5281/zenodo.15496843), which is where
the localization seed comes from. That seed is a semicircular weak inclusion on
the middle of the bottom boundary: centre `(Lx/2, 0)`, radius `0.025`, nine arc
points from `0` to `π`, closed through the centre point, meshed as its own
Triangle region so the mesh conforms to it. At `max_area = 3e-4` it holds 16 of
3748 elements. Its only difference from the bulk is a ten times smaller shear
modulus, `G = 4e9` against `4e10`; `K`, density, friction, cohesion, softening,
tensile strength, and the regularization viscosity are all equal.

Without that seed the driver deformed uniformly and could not localize at any
step count: a single phase plus fully prescribed boundary data made
`vx = ε̇(x − Lx/2)`, `vy = 0` an exact solution, and the spread in accumulated
volumetric plastic strain sat at `4e-18`, machine zero. With the seed the
spread is `5.3e-3` after 25 steps, with the peak directly above the inclusion
at `(0.506, 0.029)` and seed elements 62 percent above background. The lesson
is that plasticity being active is not evidence that it can localize; check the
spread, not whether the return map fires.

The boundary conditions also had to be corrected against the release. `vy = 0`
belongs on the top and bottom only. The side walls carry the horizontal
velocity and must stay free to move vertically, otherwise the band cannot open
where it meets them. In the release both side velocities are `1e-4 mm/yr`,
which over `L = 1 m` gives exactly the `ε̇xx = 6.338e-15 s⁻¹` of Table 1 and
confirms the domain is metres, not kilometres.

An unstructured mesh is not a substitute for a seed. The driver meshes with
`FEMTools.triangulate_t7_mesh` rather than a structured grid, which matches the
paper's setup and takes its target triangle area directly, but switching to it
reproduced the structured numbers bit for bit: trial pressure
`-0.768046344188`, corrected `-0.146582038`, plastic strain spread `3e-18`. A
linear velocity field is represented exactly by T7 elements, so every element
sees the same uniform strain rate whatever its shape, and there is no
mesh-induced perturbation to grow. Do not expect mesh irregularity to trigger
localization in this class of problem.

Both this driver and `ice_bridge_2D` call `FEMTools.triangulate_t7_mesh` from
the Triangulate extension instead of driving `TriangulateIO` themselves. A new
2-D miniapp that needs its own mesh should do the same and supply only the
boundary polygon and `max_area`; pass `segments` and `regions` when the mesh
must conform to an interior interface, as the seed does, and `ice_bridge_2D`
shows the pattern for a non-rectangular domain, where boundary-node selection
stays in the script.

Two traps this driver already fell into, worth avoiding in any new cap example:

- Refresh `τ_old` from `dr.τ` after `update_stokes_current_stress!`, the way
  `solvi2D` does. Without it the stress history stays zero, every step restarts
  from an unstressed state, and the deviatoric response silently freezes at the
  one-step elastic predictor while the run still converges and reports success.
- Read the corrected pressure from the four-matrix `τ_store`, never from
  `dr.P .+ dr.Pnum`. `Pnum` is a solver relaxation term; the sum is not a
  physical pressure and in this case pointed the wrong way from the cap.

This driver is what exposed the inner-loop convergence defect described in the
solver guide: it spent the full `iterMax` budget every step while reporting
`err ≈ 5e-17`. After the fix it exits at the first check, 25 iterations instead
of 501, with the physics unchanged to 11 significant figures.

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

## Exact-field verification drivers

The root `benchmarks/` tree also contains three maintained exact-field drivers:
`stokes/solkz2D/SolKz2D_triangle.jl`,
`stokes/solcx2D/SolCx2D_triangle.jl`, and
`thermal/thermal_diffusion2D/ThermalDiffusion2D.jl`. Run them with
`--project=examples`; they do not activate an environment internally.
Each benchmark follows the SolVi2D script layout: top-level imports and one
`main` with labeled mesh, geometry, state, boundary, solve, comparison, VTK, and
visualization sections, followed by an unconditional call and completion message.
Numerical sampling helpers are local to `main`; heatmaps are built inline. Running or including it
executes the default solve with numerical/analytical/error heatmaps displayed.
Plots use filled mesh polygons with quadrature-weighted cell averages, like
SolVi2D; error colors average absolute pointwise error. Field panels share a
color range and do not smooth the SolCx interface. Each script displays/saves
one comparison figure: Stokes has pressure, x velocity, and y velocity rows, with numerical,
analytical, and absolute-error columns. Thermal has one temperature row.
Stokes PNG files are named `SolKz.png`/`SolCx.png` or their `_quad.png` variants.
Set `FEMTOOLS_BENCHMARK_PLOTS=false` for headless execution; validation and
refinement runners set this during includes. The local `main` still defaults
to headless/no-output for customized calls. Optional output writes VTK and PNG
and imports GLMakie at file load, like SolVi2D.
`run_refinement.jl` includes the three scripts in isolated modules and records
spatial/temporal errors and rates in CSV plus TOML versions. Validation runners
use the same isolated-module approach and execute the default solves too.
CPU Float64 correctness is validated, including 1e6 Stokes contrasts. CUDA has
an explicit hardware check in `check_cuda.jl`; no local hardware pass is claimed.

## Benchmark and support files

`benchmarks/api_inventory.jl` is a host-only development inventory of public
methods and textual references. `benchmarks/api_baseline.jl` runs small thermal
and heterogeneous Stokes correctness smokes; `--cuda` requires CUDA and compares
CPU/device fields in both precisions with scalar indexing disabled. Both run
their entry points unconditionally and need no plotting dependencies.

These belong to the miniapp ecosystem but are not independent showcase
applications.

| File | Role |
|---|---|
| `examples/benchmarks/thermal/assembly_perf_2D/assembly_perf_2D.jl` | 2-D sparse/colored/atomic diffusion assembly comparison |
| `examples/benchmarks/thermal/assembly_perf_3D/assembly_perf_3D.jl` | 3-D counterpart of the assembly comparison |
| `examples/benchmarks/stokes/adjoint_perf/adjoint_perf.jl` | Includes the 2-D sinking-block adjoint and sweeps mesh size/viscosity contrast |
| `examples/benchmarks/stokes/forward_lambda_perf/forward_lambda_perf.jl` | Compares Gershgorin and measured forward spectral bounds on the sinking block |
| `examples/benchmarks/stokes/forward_lambda_shear_band_perf/forward_lambda_shear_band_perf.jl` | Spectral-bound comparison on the unstructured pure-shear workflow |
| `examples/gmsh_meshing.jl` | Shared Gmsh T3/T6/T7 triangle mesh generation and order conversion |
| `examples/miniapps/stokes/mesher/mesher.jl` | Sinking-block geometry launch helper and Gmsh Hex27 order conversion |
| `examples/miniapps/stokes/sinking_block/sinking_block_3D_setup.jl` | 3-D sinking-block forward and adjoint definitions shared by the two drivers and `test/test_stokes_3d_reference.jl` |
| `examples/miniapps/stokes/2D_Elasticity_stress_postprocess/2D_Elasticity_stress_postprocess.jl` | Includes the DR cantilever and projects quadrature stress to nodes |

The examples environment supports CUDA as its only GPU backend. The
`poisson_1step` thermal experiment selects CUDA.

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

The same benchmark folders contain standalone `SolKz2D_quad.jl`,
`SolCx2D_quad.jl`, and `ThermalDiffusion2D_quad.jl`, with the same one-main
SolVi2D layout. Stokes uses Q2/P1; thermal uses scalar Q2 (no pressure), as its
original Q9 driver already does. Quad outputs use `output_quad/` and `_quad`
filenames, so running variants does not replace triangle outputs.

## Update this guide when

- a 2-D/3-D example, benchmark, or support file is added, removed, renamed, or
  changes role;
- a miniapp changes mesh, element pair, physics, solver, backend, output, or
  execution contract;
- parse/smoke coverage changes;
- a script-local feature moves into the package or a package workflow moves
  into an example;
- the supported miniapp set or standard run command changes.

The six exact-field benchmark scripts also show a separate convergence figure
in a new GLMakie Screen. Stokes overlays vx/vy/p on one logarithmic panel;
thermal plots its scalar T residual across cumulative DR iterations. Both
comparison and `_convergence.png` are written when output is enabled. Stokes
returns records in stats.history; thermal returns convergence_history separately
from its physical-time error history. Zero/tiny values are clamped only for plots.

When showing and saving a figure, display it on its explicit GLMakie Screen
before saving. Saving first can attach an offscreen Screen and GLMakie rejects
subsequent display of that scene on a second Screen.

All six standalone exact-field drivers save `*_convergence.jld2` by default
(`save_history=true`), independently of show_plot/write_output. Dataset names
are convergence_history and metadata; metadata includes portable version/backend
strings, parameters, discretization, and per-field/total DoF counts. Counts
include boundary constraints and cell-local pressure values. Return history_path
is the filename, or nothing when disabled. The default entry point respects
FEMTOOLS_BENCHMARK_HISTORY=false; validation/sweep runners suppress archives
except serialization checks in temporary directories.

The six maintained exact-field drivers use mesh-based state constructors and
two-argument Dirichlet construction. Stokes stress dimensions are inferred.
Material defaults remove explicit unit, zero, and infinite property tuples.
SolKz uses default nodal phases without all-one cell matrices. SolCx shares
one backend-resident cell-phase row across velocity and pressure, preserving
the discontinuous interface. Pressure scaling remains explicit pending later
API work.

## Owned pressure scaling in drivers

SolKz/SolCx (verified by running), Solvi, sinking block (benchmark and miniapp,
`scaling_viscosity` = mean viscosity), shear bands, ice bridge and Popov
extension call `solve!(…; dt, pressure_factor)`. The
Gmsh-based ones were only parse-checked: Gmsh is unavailable on this Windows
setup. The elastic build-up and pure-shear-hole drivers and the adjoint
miniapps still use low-level/positional forms.
