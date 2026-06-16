# Heat Diffusion

FEMTools.jl includes a transient multi-phase heat-diffusion solver based on a
pseudo-transient dynamic-relaxation (DR) scheme. The solver is backend-agnostic
and runs on both CPU and GPU via KernelAbstractions.jl.

## Physical model

The transient heat equation for a mixture of `nphases` phases is

```math
\sum_p \rho_p C_{p,p} \frac{\partial T}{\partial t} = \sum_p k_p \nabla^2 T + s
```

where the per-phase density follows a linearised equation of state

```math
\rho_p = \rho_{0,p} \bigl(1 - \alpha_p (T - T_\text{ref}) + P / K_p\bigr), \quad T_\text{ref} = 273\,\text{K}.
```

Material properties (`k`, `Cp`, `ρ0`, `α`, `K`) are stored as
`NTuple{nphases, FP}` scalars inside the solver state object and are encoded as
type parameters, so the compiler can unroll the per-phase accumulation loop
without dynamic dispatch.

## Solver state

```@docs
ThermalDiffusionDR
```

### Constructor

```julia
# CPU (default)
dr = ThermalDiffusionDR(nnodes, k, Cp, ρ0, α, K)

# Explicit backend (e.g. GPU)
using CUDA
dr = ThermalDiffusionDR(CUDABackend(), nnodes, k, Cp, ρ0, α, K)
```

All keyword arguments (`CFL`, `c_fact`, `ϵ`) are converted to the float type
inferred from the phase-property tuples, so mixing `Float32` properties with
`Float64` literals in keyword arguments is safe.

### Example — two-phase problem

```julia
using FEMTools, DomainSets, KernelAbstractions

FP = Float64
nphases = 2

k  = (FP(3.0),    FP(5.0))     # W m⁻¹ K⁻¹
Cp = (FP(1000.0), FP(800.0))   # J kg⁻¹ K⁻¹
ρ0 = (FP(3000.0), FP(2700.0))  # kg m⁻³
α  = (FP(3e-5),   FP(2e-5))    # K⁻¹
K  = (FP(1e11),   FP(8e10))    # Pa

mesh = Mesh(0.0..1.0, 100)
dr   = ThermalDiffusionDR(CPU(), mesh.nnodes, k, Cp, ρ0, α, K; CFL=0.9)
```

## Assembly

The residual and Jacobian assembly functions live in
`src/heat_diffusion/assembly/residual.jl` and operate on element-local
`SVector`s inside KernelAbstractions kernels:

| Function | Role |
|:-------- |:---- |
| `assemble_diffusion_matrices_atomix!` | Top-level assembler; fills `R`, `∂R∂T`, `PC` using atomic scatter |
| `integrate_residual` | Quadrature-point loop; returns the element residual `SVector` |
| `element_residual` | Gathers element-local fields and calls `integrate_residual` |
| `element_jacobian` | ForwardDiff Jacobian of `integrate_residual`; returns row sums and diagonal |
