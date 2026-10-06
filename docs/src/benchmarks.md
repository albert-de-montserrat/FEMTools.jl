# Exact-field benchmarks

SolKz, SolCx, and two-dimensional thermal diffusion compare FEMTools against
[ExactFieldSolutions.jl](https://github.com/tduretz/ExactFieldSolutions.jl).
They report quadrature-weighted absolute and relative L² errors and reject
non-converged solves before writing results.

Run from the repository root:

```sh
julia --project=examples benchmarks/stokes/solkz2D/SolKz2D_triangle.jl
julia --project=examples benchmarks/stokes/solcx2D/SolCx2D_triangle.jl
julia --project=examples benchmarks/thermal/thermal_diffusion2D/ThermalDiffusion2D.jl
julia --project=examples benchmarks/check_exact_fields.jl
julia --project=examples benchmarks/run_refinement.jl
```

Stokes uses T7/P1-disc elements, analytical boundary velocities, and purely
viscous incompressible material. SolKz samples its smooth exponential viscosity
at quadrature points. SolCx keeps the `x=0.5` interface conforming and assigns
phases per element. Pressure errors remove the weighted mean independently
from both fields, preserving the pressure discontinuity.

Thermal diffusion uses Q9 elements and backward Euler against a transient
Gaussian. Its initial and time-dependent boundary temperatures come from the
exact solution. Analytical diffusivity `K` equals `k/(ρ Cp)` and is unrelated
to the material bulk modulus. Spatial and temporal sweeps are separate; lower
the timestep when spatial errors reach the temporal floor.

Each benchmark is a standalone script in its own folder, following SolVi2D:
top-level imports, one `main` with setup/solve/visualization sections, and an
unconditional call at the end. Running or including
it executes a default solve and displays numerical, analytical, and absolute-error
plots for pressure/velocity or temperature. Set `FEMTOOLS_BENCHMARK_PLOTS=false`
for headless execution; validation and refinement runners do this automatically. Its local `main` accepts optional plotting,
VTK/PNG output, and a backend argument. Refresh the examples development manifest
after dependency changes with `Pkg.develop(path=".")` and `Pkg.instantiate()`
from the repository root in the examples environment.
`run_refinement.jl` records errors, measured rates, and versions in CSV/TOML
files. Timings include compilation and are not performance measurements.

CPU Float64 validation includes moderate and `10⁶` Stokes viscosity contrasts.
CUDA agreement has a separate `benchmarks/check_cuda.jl` hardware check;
skipping it does not establish CUDA execution coverage.

See the repository's
[benchmark README](https://github.com/albert-de-montserrat/FEMTools.jl/blob/main/benchmarks/README.md)
for parameters, customized runs, forcing conventions, and output controls.

Comparison heatmaps use filled mesh elements, as in SolVi2D. Colors show
quadrature-weighted cell averages; the error panel averages absolute pointwise
error. Numerical and analytical panels share a color range. SolCx pressure
remains discontinuous across the material interface.

## Quadrilateral variants

Run the standalone Q2/P1 Stokes variants in the same folders:

```sh
julia --project=examples benchmarks/stokes/solkz2D/SolKz2D_quad.jl
julia --project=examples benchmarks/stokes/solcx2D/SolCx2D_quad.jl
julia --project=examples benchmarks/thermal/thermal_diffusion2D/ThermalDiffusion2D_quad.jl
```

Stokes uses Q9 velocity and three discontinuous P1 pressure values per cell.
The pressure basis spans `(1, ξ, η)` on the reference square, with positive
Jacobi mass weights and velocity-cell quadrature. Thermal diffusion uses Q2
for temperature; it has no pressure field and its original script already uses Q9.
The variants retain inline heatmaps and write to separate `output_quad/` folders.

Each benchmark displays and saves one comparison figure. Stokes combines
pressure, x velocity, and y velocity as rows, with FEMTools, analytical, and
absolute-error heatmaps as columns. Thermal diffusion keeps temperature in one
row. Stokes PNG names are `SolKz.png` / `SolCx.png` and their `_quad.png` variants.

## Convergence plots

Each script also opens a separate `GLMakie.Screen()` for convergence history.
Stokes overlays `vx`, `vy`, and `p` residuals in one panel against cumulative DR
iterations, with a logarithmic residual axis. Velocity components use the
solver's `norm(Rv[c])/(2sqrt(nnodes))` scaling; pressure uses
`norm(RP/M_P)/sqrt(nnodesP)`. Outer and inner checks record the residuals assembled
at that check, including the final outer convergence check.

Thermal diffusion has no velocity or pressure fields. Its separate panel plots
`T` residuals across cumulative DR iterations, including every physical time
step; residuals restart when each new physical step begins. Histories retain
raw values; plots clamp zeros/tiny values to machine precision for log axes.
`write_output=true` also saves a `_convergence.png` beside the comparison PNG.
Stokes histories are returned in `result.stats.history`; temperature DR history
is returned in `result.convergence_history`, separately from physical-time errors.

## Saved convergence data

Each standalone benchmark saves a `*_convergence.jld2` file in its output folder
by default, even when PNG/VTK output is disabled. Use `save_history=false` in
`main`, or `FEMTOOLS_BENCHMARK_HISTORY=false` for the default script run, to skip
this archive. Check/refinement runners disable saving except explicit round-trip
checks in temporary directories. JLD2 is an examples-environment dependency.

Each file contains two datasets: `convergence_history` (the raw solver records)
and `metadata`. Metadata includes parameters, discretization, version/backend
strings, and `dofs`. Stokes DoFs are `vx`, `vy`, `p`, and `total`; thermal DoFs
are `T` and `total`. Counts include prescribed boundary DoFs. Pressure counts
include all discontinuous cell-local values. The returned `history_path` gives
the saved filename (or `nothing` when disabled).

```julia
using JLD2
saved = JLD2.load(result.history_path)
history = saved["convergence_history"]
dofs = saved["metadata"].dofs
```
