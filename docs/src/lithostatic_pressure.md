# Lithostatic Pressure

FEMTools.jl includes a lithostatic-pressure solver based on the same
pseudo-transient dynamic-relaxation (DR) scheme as the [heat-diffusion
solver](heat_diffusion.md). It runs on both CPU and GPU via
KernelAbstractions.jl.

## Physical model

The pressure field satisfies the steady-state Poisson problem

```math
\int \nabla P \cdot \nabla v \, \mathrm{d}\Omega = \int \rho(T)\, \mathbf{g} \cdot \nabla v \, \mathrm{d}\Omega,
```

where the density follows the linearised equation of state

```math
\rho_p = \rho_{0,p}\bigl(1 - \alpha_p (T - T_\text{ref}) + P / K_p\bigr).
```

Temperature `T` enters only as a known coefficient, so the state object holds no
time-step arrays. Per-phase properties (`ρ0`, `α`, `K`) are stored as
`NTuple{nphases, FP}` scalars encoded in the type parameters. The reference
temperature `Tref` and body-force vector `g` are passed to `solver!`.

## Solver state

```@docs
LithostaticPressureDR
```

### Constructor

```julia
# CPU (default)
dr = LithostaticPressureDR(nnodes, ρ0, α, K)

# Explicit backend (e.g. GPU)
using CUDA
dr = LithostaticPressureDR(CUDABackend(), nnodes, ρ0, α, K)
```

Fill `dr.T` with the current temperature field via `copyto!(dr.T, ...)` before
calling the solver. The pressure field is advanced by
[`solver!`](@ref) (documented on the [Heat Diffusion](heat_diffusion.md) page),
which dispatches on the solver-state type.

## Assembly

```@docs
FEMTools.assemble_lithostatic_pressure_matrices_atomix!
FEMTools.assemble_lithostatic_pressure_matrices_colored!
FEMTools.lp_element_residual
FEMTools.lp_element_jacobian
FEMTools.lp_integrate_residual
```
