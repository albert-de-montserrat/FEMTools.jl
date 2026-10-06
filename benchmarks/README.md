# Exact-field verification benchmarks

These benchmarks compare FEMTools solves with ExactFieldSolutions.jl. They
measure discretization error, not speed. Existing SolVi and other benchmark
files are left unchanged.

Run from the repository root with the examples environment. After changing
package dependencies, refresh its local development manifest first:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run:

```sh
julia --project=examples benchmarks/stokes/solkz2D/SolKz2D_triangle.jl
julia --project=examples benchmarks/stokes/solcx2D/SolCx2D_triangle.jl
julia --project=examples benchmarks/thermal/thermal_diffusion2D/ThermalDiffusion2D.jl
julia --project=examples benchmarks/check_exact_fields.jl
julia --project=examples benchmarks/run_refinement.jl
```

Each benchmark is a standalone script in its own folder, with mesh setup,
solve, analytical comparison, and plotting defined locally. Running or including
a script executes its default solve and displays comparison plots: FEMTools,
ExactFieldSolutions, and absolute error. Stokes plots pressure and both velocity
components; thermal diffusion plots temperature. For headless execution, set
`FEMTOOLS_BENCHMARK_PLOTS=false`. Check and refinement runners set this automatically.
Call `main` again to customize it:

```julia
include("benchmarks/stokes/solcx2D/SolCx2D_triangle.jl")
r = main(; resolution=16, contrast=1e6,
    show_plot=true, write_output=true, output_dir="/tmp/solcx")

include("benchmarks/thermal/thermal_diffusion2D/ThermalDiffusion2D.jl")
t = main(; resolution=16, nsteps=40,
    show_plot=true, write_output=true, output_dir="/tmp/diffusion")
```

`write_output=true` saves VTK fields and PNG comparisons. Like SolVi2D, each
script imports GLMakie at the top and defines one `main` containing mesh setup,
geometry, state, boundary conditions, solve, comparison, and visualization. Keep generated results outside the tracked source tree.

## Quadrilateral variants

The `_quad.jl` files are standalone scripts with the same SolVi2D layout:

```sh
julia --project=examples benchmarks/stokes/solkz2D/SolKz2D_quad.jl
julia --project=examples benchmarks/stokes/solcx2D/SolCx2D_quad.jl
julia --project=examples benchmarks/thermal/thermal_diffusion2D/ThermalDiffusion2D_quad.jl
```

Stokes uses continuous Q2 velocity (Q9) and discontinuous P1 pressure with three
cell-local values at the cell center and positive x/y edge midpoints. The pressure
basis is `(1-ξ-η, ξ, η)` on the reference square, spanning `(1, ξ, η)`. Positive
Jacobi mass weights avoid zero/signed row sums. SolCx keeps its conforming
`x=0.5` interface. Quadrilateral variants write to `output_quad/` and use `_quad`
output names. Thermal diffusion has a scalar Q2 temperature field and no
pressure unknown; the original thermal script already uses Q9 elements.
The external check runs both Stokes discretizations and checks thermal parity.

## Stokes cases

Both cases use structured T7/P1-disc triangles on `[0,1]²`, purely viscous
incompressible material, and the exact velocity on every external boundary.
The interior starts at zero. No elasticity, plasticity, or pressure storage is
present. The prescribed Dirichlet values reproduce the analytical solution;
these drivers do not test free-slip traction boundary conditions.

- **SolKz:** `Stokes2D_SolKz_Zhong1996`, with viscosity `exp(2By)` and
  `B=log(contrast)/2`. Viscosity and body force are sampled at velocity
  integration points. The default contrast is `1e6`. The installed reference
  fixes the horizontal wavenumber to `3π`; the driver fixes `n=3` accordingly.
- **SolCx:** `Stokes2D_SolCx_Zhong1996`, with viscosities `(1, contrast)`
  separated by `x=0.5`. An even resolution ensures no triangle crosses the
  interface. Element-local phase matrices preserve the discontinuity at shared
  nodes. The default contrast is `2`; `1e6` is also exercised locally.

The references use opposite signs for their supplied density conventions:
the assembled vertical force is `-ρ` for SolKz and `+sin(πy)cos(πx)` for
SolCx. Refinement checks guard these signs. Analytical evaluations remain on
the host; sampled viscosity and assembled loads are transferred once.

Velocity errors use both components. Pressure comparisons subtract the
volume-weighted mean from each field independently. Both absolute and relative
L² errors are integrated at quadrature points; pressure is evaluated through
the discontinuous pressure basis without smoothing across the interface.

The solver's quadrature-viscosity path uses the element maximum only for its
conservative preconditioner; the physical residual and pressure scaling use
the sampled values. This path currently excludes plasticity, viscoelasticity,
coupled solves, measured spectral estimates, and adjoints.

## Thermal diffusion

`Diffusion2D_Gaussian` supplies the exact transient temperature on
`[-0.5,0.5]²`. The mesh uses Q9 elements and the time integrator is backward
Euler. Defaults are amplitude `1`, width `0.2`, diffusivity `1`, and final time
`0.01`. Exact initial conditions and time-dependent Dirichlet values avoid
truncating the Gaussian with artificial zero boundary temperatures.

The analytical parameter `K` means diffusivity `k/(ρ Cp)`. It is unrelated to
the FEMTools material bulk modulus, which is `Inf` here. Density and heat
capacity are one, expansivity is zero, and the heat source is zero.

The caller copies the accepted temperature to `T0` before every physical step.
The error history includes every accepted step. A non-converged DR step throws.

## Refinement and validation

`run_refinement.jl` writes `refinement.csv`, `rates.csv`, and `metadata.toml`
under `benchmarks/output/`. Stokes sweeps use contrasts `10` and `2`, and
resolutions `(4,8,16)`. Thermal spatial refinement uses 2000 time steps; temporal
refinement holds the mesh at 16 cells per side and uses `(4,8,16)` steps.
Decrease the timestep further when thermal spatial error reaches a time-error
floor. Rates are measured, not hard-coded theoretical claims.

For plots or high-contrast refinement:

```julia
include("benchmarks/run_refinement.jl")
main(; stokes_contrasts=(1e6,1e6),
    write_plots=true, output_dir="/tmp/refinement")
```

CPU Float64 checks cover convergence, refinement, pressure-gauge invariance,
interface conformity, input rejection, and explicit non-convergence. Core
tests independently check hydrostatic pressure and the sampled-viscosity path.
Timing includes compilation and must not be presented as a warmed benchmark.

CUDA uses the same drivers after loading its extension:

```julia
using CUDA
CUDA.allowscalar(false)
include("benchmarks/stokes/solkz2D/SolKz2D_triangle.jl")
main(; backend=CUDA.CUDABackend())
```

Run `julia --project=examples benchmarks/check_cuda.jl` on an NVIDIA GPU for
CPU/CUDA agreement. A skipped hardware check is not CUDA execution coverage.

References: [ExactFieldSolutions.jl](https://github.com/tduretz/ExactFieldSolutions.jl),
[SolKz](https://github.com/tduretz/ExactFieldSolutions.jl/blob/main/src/Stokes2D/Stokes2D_SolKz_Zhong1996.jl),
[SolCx](https://github.com/tduretz/ExactFieldSolutions.jl/blob/main/src/Stokes2D/Stokes2D_SolCx_Zhong1996.jl),
[Gaussian diffusion](https://github.com/tduretz/ExactFieldSolutions.jl/blob/main/src/Diffusion2D/Diffusion2D_Gaussian.jl).

Comparison heatmaps use filled mesh elements, as in SolVi2D. Colors show
quadrature-weighted cell averages; the error panel averages absolute pointwise
error. Numerical and analytical panels share a color range. SolCx pressure
remains discontinuous across the material interface.

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
