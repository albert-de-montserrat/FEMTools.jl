"""
    StokesDR{nphases, _T, _TI, _TS, FP}

Solver state for an incompressible Stokes flow solved with a pseudo-transient
dynamic-relaxation (DR) scheme using mixed elements (separate velocity and
pressure node sets, e.g. T6/P1 Taylor-Hood-like pair).

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `_T`       — nodal float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TI`      — nodal integer array type (same backend, element type `Int`)
- `_TS`      — stress array type (nodal by default, or integration-point storage)
- `FP`       — floating-point precision (`Float32` or `Float64`)

# Velocity-node arrays (length `nnodes_v`)
| Field      | Description                               |
|:---------- |:----------------------------------------- |
| `vx`       | x-velocity (current iterate)              |
| `vy`       | y-velocity (current iterate)              |
| `∂vx∂τ`   | Pseudo-transient rate for x-velocity       |
| `∂vy∂τ`   | Pseudo-transient rate for y-velocity       |
| `Rv_x`     | x-momentum residual                       |
| `Rv_x0`    | Residual snapshot for λ_min estimate      |
| `∂Rv_x∂vx` | Row-sum Jacobian estimate for x-momentum  |
| `PC_vx`    | Diagonal preconditioner for x-momentum    |
| `Rv_y`     | y-momentum residual                       |
| `Rv_y0`    | Residual snapshot for λ_min estimate      |
| `∂Rv_y∂vy` | Row-sum Jacobian estimate for y-momentum  |
| `PC_vy`    | Diagonal preconditioner for y-momentum    |
| `M_V`      | Positive velocity mass diagonal (`∫ N_i² dΩ`) |
| `phases_v` | Per-node phase index (1-based integer)    |
| `τxx`      | Current deviatoric stress xx              |
| `τyy`      | Current deviatoric stress yy              |
| `τxy`      | Current deviatoric stress xy              |
| `τxx_old`  | Previous-step deviatoric stress xx        |
| `τyy_old`  | Previous-step deviatoric stress yy        |
| `τxy_old`  | Previous-step deviatoric stress xy        |

# Pressure-node arrays (length `nnodes_P`)
| Field      | Description                               |
|:---------- |:----------------------------------------- |
| `P`        | Pressure (current iterate)                |
| `P0`       | Pressure at previous time step            |
| `∂P∂τ`    | Pseudo-transient rate for pressure         |
| `T`        | Temperature (input from thermal solver)   |
| `T0`       | Temperature at previous time step         |
| `Q`        | Volumetric source/sink rate [s⁻¹] (positive = expansion) |
| `RP`       | Pressure residual                         |
| `RP0`      | Residual snapshot for λ_min estimate      |
| `M_P`      | Lumped pressure mass (`∫ N_i dΩ`)         |
| `Pnum`     | Arrow-Hurwicz numerical pressure correction (`γP·RP/M_P`) passed to the momentum equation |
| `phases_P` | Per-node phase index (1-based integer)    |

# Per-phase scalar tuples (`NTuple{nphases, FP}`)
`η` (dynamic shear viscosity), `ηb` (bulk viscosity), `α` (thermal expansivity),
`ρ0` (reference density, default 1), `K` (bulk modulus for EOS, default Inf).

# Global scalar fields
`g::NTuple{2,FP}` (gravity vector, default `(0,0)`), `Tref::FP` (reference
temperature for the linearised EOS, default 0).

# Solver parameters
`CFL_v`, `CFL_P`, `c_fact`, `ϵ` (convergence tolerance).

# Constructor
    StokesDR(backend, nnodes_v, nnodes_P, η, ηb, α;
             ρ0=nothing, K=nothing, g=nothing, Tref=nothing,
             CFL_v=0.98, CFL_P=0.98, c_fact=0.9, ϵ=1e-6,
             stress_size=nothing)
    StokesDR(nnodes_v, nnodes_P, η, ηb, α; kwargs...)  # defaults to CPU()

All nodal float arrays, including `Q`, are zero-initialised; phase arrays are
initialised to 1. Set `Q` on pressure DoFs to prescribe a volumetric source or
sink: positive values produce expansion and negative values contraction.
Stress arrays default to nodal storage of length `nnodes_v`; pass
`stress_size=(nq, nels)` to store current and previous stress directly at
integration points. `T` and `T0` should be filled via `copyto!` before calling
the solver. The time step `Δt` is passed directly to the assembler rather than
stored here.
"""
struct StokesDR{nphases, _T, _TI, _TS, FP}
    # velocity-node solution fields
    vx::_T
    vy::_T
    ∂vx∂τ::_T
    ∂vy∂τ::_T
    # velocity-node residual and DR work arrays
    Rv_x::_T
    Rv_x0::_T
    ∂Rv_x∂vx::_T
    PC_vx::_T
    Rv_y::_T
    Rv_y0::_T
    ∂Rv_y∂vy::_T
    PC_vy::_T
    M_V::_T
    # velocity-node phase assignment
    phases_v::_TI
    # deviatoric stress history
    τxx::_TS
    τyy::_TS
    τxy::_TS
    τxx_old::_TS
    τyy_old::_TS
    τxy_old::_TS
    # pressure-node solution fields
    P::_T
    P0::_T
    ∂P∂τ::_T
    T::_T
    T0::_T
    Q::_T
    # pressure-node residual and DR work arrays
    RP::_T
    RP0::_T
    M_P::_T
    Pnum::_T   # Arrow-Hurwicz numerical pressure correction (γP·RP/M_P) fed into momentum equation
    # pressure-node phase assignment
    phases_P::_TI
    # physical parameters – one scalar per phase
    η::NTuple{nphases, FP}     # dynamic shear viscosity [Pa s]
    ηb::NTuple{nphases, FP}    # bulk viscosity          [Pa s]
    α::NTuple{nphases, FP}     # thermal expansivity     [K⁻¹]
    ρ0::NTuple{nphases, FP}    # reference density       [kg m⁻³]
    K::NTuple{nphases, FP}     # bulk modulus (EOS)      [Pa]
    # global scalar parameters
    g::NTuple{2, FP}           # gravitational acceleration [m s⁻²]
    Tref::FP                   # reference temperature for EOS [K]
    # solver parameters
    CFL_v::FP
    CFL_P::FP
    c_fact::FP
    ϵ::FP

    function StokesDR(
        backend, nnodes_v, nnodes_P,
        η::NTuple{nphases, FP}, ηb::NTuple{nphases, FP}, α::NTuple{nphases, FP};
        ρ0   = nothing,
        K    = nothing,
        g    = nothing,
        Tref = nothing,
        CFL_v = 0.98, CFL_P = 0.98, c_fact = 0.9, ϵ = 1e-6,
        stress_size = nothing,
    ) where {nphases, FP}
        _ρ0  = ρ0  === nothing ? ntuple(_ -> FP(1),   Val(nphases)) : NTuple{nphases, FP}(ρ0)
        _K   = K   === nothing ? ntuple(_ -> FP(Inf), Val(nphases)) : NTuple{nphases, FP}(K)
        _g   = g   === nothing ? (FP(0), FP(0))   : (FP(g[1]), FP(g[2]))
        _Tref = Tref === nothing ? FP(0)           : FP(Tref)
        stress_dims = stress_size === nothing ? (nnodes_v,) :
            stress_size isa Integer ? (stress_size,) : Tuple(stress_size)
        newv()  = KernelAbstractions.zeros(backend, FP,  nnodes_v)
        newP()  = KernelAbstractions.zeros(backend, FP,  nnodes_P)
        newτ()  = KernelAbstractions.zeros(backend, FP,  stress_dims...)
        newiv() = KernelAbstractions.ones(backend,  Int, nnodes_v)
        newip() = KernelAbstractions.ones(backend,  Int, nnodes_P)
        τxx = newτ()
        new{nphases, typeof(newv()), typeof(newiv()), typeof(τxx), FP}(
            newv(), newv(), newv(), newv(),           # vx, vy, ∂vx∂τ, ∂vy∂τ
            newv(), newv(), newv(), newv(),           # Rv_x, Rv_x0, ∂Rv_x∂vx, PC_vx
            newv(), newv(), newv(), newv(),           # Rv_y, Rv_y0, ∂Rv_y∂vy, PC_vy
            newv(),                                    # M_V
            newiv(),                                  # phases_v
            τxx, newτ(), newτ(),                      # τxx, τyy, τxy
            newτ(), newτ(), newτ(),                   # τxx_old, τyy_old, τxy_old
            newP(), newP(), newP(), newP(), newP(),   # P, P0, ∂P∂τ, T, T0
            newP(),                                    # Q
            newP(), newP(), newP(), newP(),           # RP, RP0, M_P, Pnum
            newip(),                                  # phases_P
            η, ηb, α, _ρ0, _K, _g, _Tref,
            FP(CFL_v), FP(CFL_P), FP(c_fact), FP(ϵ),
        )
    end
end

"""
    velocity(dr::StokesDR) -> (vx, vy)

Return the current velocity components of a Stokes solver state.
"""
velocity(dr::StokesDR) = (dr.vx, dr.vy)

"""
    stress(dr::StokesDR) -> (τxx, τyy, τxy)

Return the current deviatoric stress components of a Stokes solver state.
"""
stress(dr::StokesDR) = (dr.τxx, dr.τyy, dr.τxy)

"""
    pressure(dr::StokesDR) -> P

Return the current pressure field of a Stokes solver state.
"""
pressure(dr::StokesDR) = dr.P

"""
    temperature(dr::StokesDR) -> T

Return the current temperature field of a Stokes solver state.
"""
temperature(dr::StokesDR) = dr.T

StokesDR(nnodes_v, nnodes_P, η, ηb, α; kwargs...) =
    StokesDR(CPU(), nnodes_v, nnodes_P, η, ηb, α; kwargs...)

"""
    DruckerPrager{nphases, FP}

Per-phase Drucker-Prager elasto-viscoplastic parameters.

| Field    | Description                               |
|:-------- |:----------------------------------------- |
| `cosϕ`  | cos(friction angle)                        |
| `sinϕ`  | sin(friction angle)                        |
| `sinΨ`  | sin(dilation angle)                        |
| `C`      | cohesion [Pa]                             |
| `η_reg`  | plastic regularization viscosity [Pa s]   |
| `Kb`     | bulk modulus for volumetric correction [Pa]|

All fields are `NTuple{nphases, FP}`.  Pass `nothing` in place of a
`DruckerPrager` wherever plasticity is not needed (the assemblers accept both).
"""
struct DruckerPrager{nphases, FP}
    cosϕ  :: NTuple{nphases, FP}
    sinϕ  :: NTuple{nphases, FP}
    sinΨ  :: NTuple{nphases, FP}
    C     :: NTuple{nphases, FP}
    η_reg :: NTuple{nphases, FP}
    Kb    :: NTuple{nphases, FP}
end

"""
    DruckerPrager(ϕ, Ψ, C, η_reg, Kb) -> DruckerPrager

Construct Drucker-Prager elasto-viscoplastic parameters from friction angle
`ϕ`, dilation angle `Ψ`, cohesion `C`, regularization viscosity `η_reg`, and
volumetric bulk modulus `Kb`. All arguments are `NTuple{nphases, FP}`.

Stores `cos(ϕ)` and `sin(ϕ)` / `sin(Ψ)` precomputed so that yield-function
evaluations inside assembly kernels avoid repeated trigonometric calls.
"""
function DruckerPrager(
    ϕ     :: NTuple{nphases, FP},
    Ψ     :: NTuple{nphases, FP},
    C     :: NTuple{nphases, FP},
    η_reg :: NTuple{nphases, FP},
    Kb    :: NTuple{nphases, FP},
) where {nphases, FP}
    DruckerPrager{nphases, FP}(
        map(cos, ϕ), map(sin, ϕ), map(sin, Ψ), C, η_reg, Kb,
    )
end
