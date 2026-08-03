"""
    LithostaticPressureDR{nphases, _T, _TI, FP}

Solver state for the lithostatic-pressure problem solved with a
pseudo-transient dynamic-relaxation (DR) scheme.

The weak form is `∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ` (steady-state Poisson),
so no time-step arrays are needed. Temperature `T` enters only as a known
coefficient for the density EOS `ρ = ρ0(1 − α(T−Tref) + P/K)`.
`Tref` and `g` are passed to `solver!`, so callers can choose the reference
temperature and body-force vector without rebuilding the solver state.

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `_T`       — nodal float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TI`      — nodal integer array type (same backend as `_T`, element type `Int`)
- `FP`       — floating-point precision (`Float32` or `Float64`)

# Nodal arrays (length `nnodes`)
| Field   | Description                               |
|:------- |:----------------------------------------- |
| `R`     | Current residual                          |
| `R0`    | Residual snapshot used for λ_min estimate |
| `∂R∂P`  | Row-sum Jacobian estimate                 |
| `PC`    | Diagonal preconditioner                   |
| `P`     | Pressure (current iterate, the unknown)   |
| `∂P∂τ`  | Pseudo-transient rate                     |
| `T`     | Temperature (input from thermal solver)   |
| `phases`| Per-node phase index (1-based integer)    |

# Material properties
`ρ0`, `α`, and `K` are taken from [`ThermalMaterial`](@ref). This allows the
same material definition to be shared with `ThermalDiffusionDR`.

# Solver parameters
`CFL`, `c_fact`, `ϵ` (convergence tolerance).

# Constructor
    LithostaticPressureDR(backend, nnodes, material::ThermalMaterial; CFL=0.98, c_fact=0.9, ϵ=1e-6)
    LithostaticPressureDR(nnodes, material::ThermalMaterial; kwargs...)  # CPU

The tuple-based constructors remain available for compatibility.

All nodal float arrays are zero-initialised; `phases` is initialised to 1.
`T` should be filled via `copyto!(dr.T, ...)` before calling `solver!`.
"""
struct LithostaticPressureDR{nphases, _T, _TI, FP}
    # preallocated work arrays
    R::_T
    R0::_T
    ∂R∂P::_T
    PC::_T
    # solution field and pseudo-transient rate
    P::_T
    ∂P∂τ::_T
    # temperature: input coefficient, updated from thermal solver each step
    T::_T
    # per-node phase assignment
    phases::_TI
    # physical parameters – one scalar per phase
    ρ0::NTuple{nphases, FP}   # reference density   [kg m⁻³]
    α::NTuple{nphases, FP}    # thermal expansivity [K⁻¹]
    K::NTuple{nphases, FP}    # bulk modulus        [Pa]
    # solver parameters
    CFL::FP
    c_fact::FP
    ϵ::FP

    function LithostaticPressureDR(
        backend, nnodes,
        ρ0::Tuple{FP, Vararg{FP, N}}, α::Tuple{FP, Vararg{FP, N}},
        K::Tuple{FP, Vararg{FP, N}};
        CFL = 0.98, c_fact = 0.9, ϵ = 1e-6,
    ) where {N, FP}
        newvec()  = KernelAbstractions.zeros(backend, FP,  nnodes)
        newivec() = KernelAbstractions.ones(backend,  Int, nnodes)
        new{N + 1, typeof(newvec()), typeof(newivec()), FP}(
            newvec(), newvec(), newvec(), newvec(),  # R, R0, ∂R∂P, PC
            newvec(), newvec(),                      # P, ∂P∂τ
            newvec(),                                # T
            newivec(),                               # phases (default: all phase 1)
            ρ0, α, K,
            FP(CFL), FP(c_fact), FP(ϵ),
        )
    end
end

pressure(dr::LithostaticPressureDR) = dr.P

LithostaticPressureDR(nnodes, ρ0, α, K; kwargs...) =
    LithostaticPressureDR(CPU(), nnodes, ρ0, α, K; kwargs...)

LithostaticPressureDR(backend, nnodes, material::ThermalMaterial; kwargs...) =
    LithostaticPressureDR(backend, nnodes, material.ρ0, material.α, material.K; kwargs...)
LithostaticPressureDR(nnodes, material::ThermalMaterial; kwargs...) =
    LithostaticPressureDR(CPU(), nnodes, material; kwargs...)
