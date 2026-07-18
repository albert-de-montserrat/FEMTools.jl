"""
    ThermalDiffusionDR{nphases, _T, _TI, FP}

Solver state for a transient multi-phase heat-diffusion problem solved with a
pseudo-transient dynamic-relaxation (DR) scheme.

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `_T`       — nodal float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TI`      — nodal integer array type (same backend as `_T`, element type `Int`)
- `FP`       — floating-point precision (`Float32` or `Float64`)

# Nodal arrays (length `nnodes`)
| Field    | Description                               |
|:-------- |:----------------------------------------- |
| `R`      | Current residual                          |
| `R0`     | Residual snapshot used for λ_min estimate |
| `∂R∂T`   | Row-sum Jacobian estimate                 |
| `PC`     | Diagonal preconditioner                   |
| `T`      | Temperature (current iterate)             |
| `T0`     | Temperature at previous time step         |
| `∂T∂τ`  | Pseudo-transient rate                     |
| `P`      | Pressure                                  |
| `source` | Nodal heat source                         |
| `phases` | Per-node phase index (1-based integer)    |

`phases` defaults to all ones (single phase). Overwrite with `copyto!(dr.phases, ...)` to assign multi-phase configurations.

Dirichlet boundary conditions (DOF indices and prescribed values) are
problem-specific and are therefore passed directly to `solver!` rather than
stored here.

# Per-phase scalar tuples (`NTuple{nphases, FP}`)
`k` (conductivity), `Cp` (specific heat), `ρ0` (reference density),
`α` (thermal expansivity), `K` (bulk modulus).
The reference temperature `Tref` for the density EOS is passed to `solver!`.

# Solver parameters
`CFL`, `c_fact`, `ϵ` (convergence tolerance).

# Constructors
    ThermalDiffusionDR(backend, nnodes, k, Cp, ρ0, α, K; CFL=0.98, c_fact=0.9, ϵ=1e-6)
    ThermalDiffusionDR(nnodes, k, Cp, ρ0, α, K; kwargs...)  # defaults to CPU()

All nodal float arrays are zero-initialised; `phases` is initialised to 1.
The float type `FP` is inferred from the element type of the phase-property tuples.
"""
struct ThermalDiffusionDR{nphases, _T, _TI, FP}
    # preallocated work arrays
    R::_T
    R0::_T
    ∂R∂T::_T
    PC::_T
    # solution fields (current and previous time step)
    T::_T
    T0::_T
    ∂T∂τ::_T
    # pressure and source
    P::_T
    source::_T
    # per-node phase assignment
    phases::_TI
    # physical parameters – one scalar per phase (intrinsic material properties)
    k::NTuple{nphases, FP}    # thermal conductivity   [W m⁻¹ K⁻¹]
    Cp::NTuple{nphases, FP}   # specific heat capacity [J kg⁻¹ K⁻¹]
    ρ0::NTuple{nphases, FP}   # reference density      [kg m⁻³]
    α::NTuple{nphases, FP}    # thermal expansivity    [K⁻¹]
    K::NTuple{nphases, FP}    # bulk modulus           [Pa]
    # solver parameters
    CFL::FP
    c_fact::FP
    ϵ::FP

    function ThermalDiffusionDR(
        backend, nnodes,
        k::Tuple{FP, Vararg{FP, N}}, Cp::Tuple{FP, Vararg{FP, N}},
        ρ0::Tuple{FP, Vararg{FP, N}}, α::Tuple{FP, Vararg{FP, N}},
        K::Tuple{FP, Vararg{FP, N}};
        CFL = 0.98, c_fact = 0.9, ϵ = 1e-6,
    ) where {N, FP}
        newvec()  = KernelAbstractions.zeros(backend, FP,  nnodes)
        newivec() = KernelAbstractions.ones(backend,  Int, nnodes)
        new{N + 1, typeof(newvec()), typeof(newivec()), FP}(
            newvec(), newvec(), newvec(), newvec(),  # R, R0, ∂R∂T, PC
            newvec(), newvec(), newvec(),            # T, T0, ∂T∂τ
            newvec(), newvec(),                      # P, source
            newivec(),                               # phases (default: all phase 1)
            k, Cp, ρ0, α, K,
            FP(CFL), FP(c_fact), FP(ϵ),
        )
    end
end

temperature(dr::ThermalDiffusionDR) = dr.T

ThermalDiffusionDR(nnodes, k, Cp, ρ0, α, K; kwargs...) =
    ThermalDiffusionDR(CPU(), nnodes, k, Cp, ρ0, α, K; kwargs...)
