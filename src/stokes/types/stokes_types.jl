"""
    StokesMaterial(; η=(1.0,), ηb=(1.0,), G=(Inf,), α=(0.0,),
                   ρ0=(1.0,), K=(Inf,), g=(0.0, 0.0), Tref=0.0)

Typed per-phase material properties and body-force parameters for `StokesDR`.
All property tuples must have the same length and floating-point type. The
length of `g` selects two or three spatial dimensions.
"""
@kwdef struct StokesMaterial{D, nphases, FP}
    η::NTuple{nphases, FP} = (1.0,)
    ηb::NTuple{nphases, FP} = (1.0,)
    G::NTuple{nphases, FP} = (Inf,)
    α::NTuple{nphases, FP} = (0.0,)
    ρ0::NTuple{nphases, FP} = (1.0,)
    K::NTuple{nphases, FP} = (Inf,)
    g::NTuple{D, FP} = (0.0, 0.0)
    Tref::FP = 0.0

    function StokesMaterial(
        η::Tuple{FP, Vararg{FP}}, ηb::Tuple{FP, Vararg{FP}},
        G::Tuple{FP, Vararg{FP}}, α::Tuple{FP, Vararg{FP}},
        ρ0::Tuple{FP, Vararg{FP}}, K::Tuple{FP, Vararg{FP}},
        g::NTuple{D, FP}, Tref::FP,
    ) where {D, FP}
        nphases = length(η)
        length(ηb) == length(G) == length(α) == length(ρ0) == length(K) == nphases ||
            throw(DimensionMismatch("Stokes material property tuples must have the same length"))
        return new{D, nphases, FP}(η, ηb, G, α, ρ0, K, g, Tref)
    end
end

"""
    StokesDR{D, nphases, _T, _TI, _TS, FP, Nτ}

Solver state for an incompressible Stokes flow solved with a pseudo-transient
dynamic-relaxation (DR) scheme using mixed elements (separate velocity and
pressure node sets, e.g. T6/P1 Taylor-Hood-like pair).

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `D`       — number of spatial dimensions
- `_T`       — nodal float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TI`      — nodal integer array type (same backend, element type `Int`)
- `_TS`      — stress array type (nodal by default, or integration-point storage)
- `FP`       — floating-point precision (`Float32` or `Float64`)

# Velocity-node arrays (length `nnodes_v`)
| Field      | Description                               |
|:---------- |:----------------------------------------- |
| `vx`       | x-velocity (current iterate)              |
| `vy`       | y-velocity (current iterate)              |
| `vz`       | z-velocity in three dimensions            |
| `∂vx∂τ`   | Pseudo-transient rate for x-velocity       |
| `∂vy∂τ`   | Pseudo-transient rate for y-velocity       |
| `∂vz∂τ`   | Pseudo-transient rate for z-velocity       |
| `Rv_x`     | x-momentum residual                       |
| `Rv_x0`    | Residual snapshot for λ_min estimate      |
| `∂Rv_x∂vx` | Row-sum Jacobian estimate for x-momentum  |
| `PC_vx`    | Diagonal preconditioner for x-momentum    |
| `Rv_y`     | y-momentum residual                       |
| `Rv_y0`    | Residual snapshot for λ_min estimate      |
| `∂Rv_y∂vy` | Row-sum Jacobian estimate for y-momentum  |
| `PC_vy`    | Diagonal preconditioner for y-momentum    |
| `phases_v` | Per-node phase index (1-based integer)    |
| `τxx`      | Current deviatoric stress xx              |
| `τyy`      | Current deviatoric stress yy              |
| `τxy`      | Current deviatoric stress xy              |
| `τxx_old`  | Previous-step deviatoric stress xx        |
| `τyy_old`  | Previous-step deviatoric stress yy        |
| `τxy_old`  | Previous-step deviatoric stress xy        |

Three-dimensional states additionally carry `Rv_z`, its work arrays, and the
stress components `τzz`, `τxz`, and `τyz` with matching old-stress arrays.

# Pressure-node arrays (length `nnodes_P`)
| Field      | Description                               |
|:---------- |:----------------------------------------- |
| `P`        | Pressure (current iterate)                |
| `P0`       | Pressure at previous time step            |
| `∂P∂τ`    | Pseudo-transient rate for pressure         |
| `T`        | Temperature (input from thermal solver)   |
| `T0`       | Temperature at previous time step         |
| `Q`        | Volumetric source/sink in continuity      |
| `RP`       | Pressure residual                         |
| `RP0`      | Residual snapshot for λ_min estimate      |
| `M_P`      | Pressure mass diagonal (`∫NᵢdΩ` in 2-D, `∫Nᵢ²dΩ` in 3-D) |
| `Pnum`     | Arrow-Hurwicz numerical pressure correction (`γP·RP/M_P`) passed to the momentum equation |
| `phases_P` | Per-node phase index (1-based integer)    |

# Material properties
Properties are supplied together through [`StokesMaterial`](@ref).

# Global scalar fields
`G` is the shear modulus. `g::NTuple{D,FP}` is the gravity vector (default
`(0,0)`), and `Tref::FP` is the equation-of-state reference temperature.

# Solver parameters
`CFL_v`, `CFL_P`, `c_fact`, `ϵ` (convergence tolerance).

# Constructor
    StokesDR(backend, nnodes_v, nnodes_P, material::StokesMaterial;
             CFL_v=0.98, CFL_P=0.98, c_fact=0.9, ϵ=1e-6,
             stress_size=nothing)
    StokesDR(backend, nnodes_v, nnodes_P, η, ηb, α;
             ρ0=nothing, K=nothing, G=nothing, g=nothing, Tref=nothing,
             CFL_v=0.98, CFL_P=0.98, c_fact=0.9, ϵ=1e-6,
             stress_size=nothing)
    StokesDR(nnodes_v, nnodes_P, η, ηb, α; kwargs...)  # defaults to CPU()

All nodal float arrays are zero-initialised; phase arrays are initialised to 1.
Stress arrays default to nodal storage of length `nnodes_v`; pass
`stress_size=(nq, nels)` to store current and previous stress directly at
integration points. `T`, `T0`, and `Q` should be filled via `copyto!` before
calling the solver. `Q` is the prescribed volumetric production rate (positive)
or sink rate (negative) in the continuity equation. The time step `Δt` is
passed directly to the assembler rather than stored here.
"""
struct StokesDR{D, nphases, _T, _TI, _TS, FP, Nτ}
    # velocity-node solution fields, one array per spatial direction
    v::NTuple{D, _T}
    ∂v∂τ::NTuple{D, _T}
    # velocity-node residual and DR work arrays, one array per direction
    Rv::NTuple{D, _T}
    Rv0::NTuple{D, _T}
    ∂Rv∂v::NTuple{D, _T}
    PC_v::NTuple{D, _T}
    # velocity-node phase assignment
    phases_v::_TI
    # deviatoric stress history, ordered (τxx, τyy, τxy) in plane strain and
    # (τxx, τyy, τzz, τxy, τxz, τyz) in three dimensions
    τ::NTuple{Nτ, _TS}
    τ_old::NTuple{Nτ, _TS}
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
    G::NTuple{nphases, FP}     # shear modulus           [Pa]
    # global scalar parameters
    g::NTuple{D, FP}           # gravitational acceleration [m s⁻²]
    Tref::FP                   # reference temperature for EOS [K]
    # solver parameters
    CFL_v::FP
    CFL_P::FP
    c_fact::FP
    ϵ::FP

    function StokesDR(
        backend, nnodes_v, nnodes_P,
        η::Tuple{FP, Vararg{FP, N}}, ηb::Tuple{FP, Vararg{FP, N}}, α::Tuple{FP, Vararg{FP, N}};
        ρ0   = nothing,
        K    = nothing,
        G    = nothing,
        g    = nothing,
        Tref = nothing,
        CFL_v = 0.98, CFL_P = 0.98, c_fact = 0.9, ϵ = 1e-6,
        stress_size = nothing,
    ) where {N, FP}
        nphases = N + 1
        _ρ0  = ρ0  === nothing ? ntuple(_ -> FP(1),   Val(nphases)) : NTuple{nphases, FP}(ρ0)
        _K   = K   === nothing ? ntuple(_ -> FP(Inf), Val(nphases)) : NTuple{nphases, FP}(K)
        _G   = G   === nothing ? ntuple(_ -> FP(Inf), Val(nphases)) : NTuple{nphases, FP}(G)
        _g   = g   === nothing ? (FP(0), FP(0))   : map(FP, Tuple(g))
        D    = length(_g)
        # Independent components of a symmetric tensor: 3 in plane strain, 6 in 3-D.
        Nτ   = D * (D + 1) ÷ 2
        _Tref = Tref === nothing ? FP(0)           : FP(Tref)
        stress_dims = stress_size === nothing ? (nnodes_v,) :
            stress_size isa Integer ? (stress_size,) : Tuple(stress_size)
        newv()  = KernelAbstractions.zeros(backend, FP,  nnodes_v)
        newP()  = KernelAbstractions.zeros(backend, FP,  nnodes_P)
        newτ()  = KernelAbstractions.zeros(backend, FP,  stress_dims...)
        newiv() = KernelAbstractions.ones(backend,  Int, nnodes_v)
        newip() = KernelAbstractions.ones(backend,  Int, nnodes_P)
        dirs()  = ntuple(_ -> newv(), D)
        new{D, nphases, typeof(newv()), typeof(newiv()), typeof(newτ()), FP, Nτ}(
            dirs(), dirs(),                           # v, ∂v∂τ
            dirs(), dirs(), dirs(), dirs(),           # Rv, Rv0, ∂Rv∂v, PC_v
            newiv(),                                  # phases_v
            ntuple(_ -> newτ(), Nτ),                  # τ
            ntuple(_ -> newτ(), Nτ),                  # τ_old
            newP(), newP(), newP(), newP(), newP(), newP(), # P, P0, ∂P∂τ, T, T0, Q
            newP(), newP(), newP(), newP(),           # RP, RP0, M_P, Pnum
            newip(),                                  # phases_P
            η, ηb, α, _ρ0, _K, _G, _g, _Tref,
            FP(CFL_v), FP(CFL_P), FP(c_fact), FP(ϵ),
        )
    end
end

# Component names for the direction-indexed and stress-history tuples. The
# solver and assemblers work with the tuples directly; these aliases keep the
# per-component names that drivers, diagnostics and the adjoint paths use.
@inline _stress_index(::StokesDR{2}, s::Symbol) =
    s === :τxx ? 1 : s === :τyy ? 2 : s === :τxy ? 3 : 0
@inline _stress_index(::StokesDR{3}, s::Symbol) =
    s === :τxx ? 1 : s === :τyy ? 2 : s === :τzz ? 3 :
    s === :τxy ? 4 : s === :τxz ? 5 : s === :τyz ? 6 : 0
@inline _stress_old_index(::StokesDR{2}, s::Symbol) =
    s === :τxx_old ? 1 : s === :τyy_old ? 2 : s === :τxy_old ? 3 : 0
@inline _stress_old_index(::StokesDR{3}, s::Symbol) =
    s === :τxx_old ? 1 : s === :τyy_old ? 2 : s === :τzz_old ? 3 :
    s === :τxy_old ? 4 : s === :τxz_old ? 5 : s === :τyz_old ? 6 : 0

@inline function Base.getproperty(dr::StokesDR, s::Symbol)
    s === :vx     && return getfield(dr, :v)[1]
    s === :vy     && return getfield(dr, :v)[2]
    s === :vz     && return getfield(dr, :v)[3]
    s === :∂vx∂τ  && return getfield(dr, :∂v∂τ)[1]
    s === :∂vy∂τ  && return getfield(dr, :∂v∂τ)[2]
    s === :∂vz∂τ  && return getfield(dr, :∂v∂τ)[3]
    s === :Rv_x   && return getfield(dr, :Rv)[1]
    s === :Rv_y   && return getfield(dr, :Rv)[2]
    s === :Rv_z   && return getfield(dr, :Rv)[3]
    s === :Rv_x0  && return getfield(dr, :Rv0)[1]
    s === :Rv_y0  && return getfield(dr, :Rv0)[2]
    s === :Rv_z0  && return getfield(dr, :Rv0)[3]
    s === :∂Rv_x∂vx && return getfield(dr, :∂Rv∂v)[1]
    s === :∂Rv_y∂vy && return getfield(dr, :∂Rv∂v)[2]
    s === :∂Rv_z∂vz && return getfield(dr, :∂Rv∂v)[3]
    s === :PC_vx  && return getfield(dr, :PC_v)[1]
    s === :PC_vy  && return getfield(dr, :PC_v)[2]
    s === :PC_vz  && return getfield(dr, :PC_v)[3]
    i = _stress_index(dr, s)
    i > 0 && return getfield(dr, :τ)[i]
    j = _stress_old_index(dr, s)
    j > 0 && return getfield(dr, :τ_old)[j]
    return getfield(dr, s)
end

"""
    velocity(dr::StokesDR) -> NTuple

Return the nodal velocity-component arrays of a Stokes solver state, one per
spatial direction.
"""
velocity(dr::StokesDR) = getfield(dr, :v)

"""
    stress(dr::StokesDR) -> NTuple

Return the current deviatoric-stress arrays of a Stokes solver state, ordered
`(τxx, τyy, τxy)` in plane strain and `(τxx, τyy, τzz, τxy, τxz, τyz)` in three
dimensions.
"""
stress(dr::StokesDR) = getfield(dr, :τ)

"""
    stress_old(dr::StokesDR) -> NTuple

Return the previous-time deviatoric-stress arrays of a Stokes solver state, in
the same component order as [`stress`](@ref).
"""
stress_old(dr::StokesDR) = getfield(dr, :τ_old)

"""
    pressure(dr) -> P

Return the pressure array of a `StokesDR` or `LithostaticPressureDR` solver
state.
"""
pressure(dr::StokesDR) = dr.P

"""
    temperature(dr) -> T

Return the temperature array of a `StokesDR` or `ThermalDiffusionDR` solver
state.
"""
temperature(dr::StokesDR) = dr.T

StokesDR(nnodes_v, nnodes_P, η, ηb, α; kwargs...) =
    StokesDR(CPU(), nnodes_v, nnodes_P, η, ηb, α; kwargs...)

StokesDR(backend, nnodes_v, nnodes_P, material::StokesMaterial; kwargs...) =
    StokesDR(backend, nnodes_v, nnodes_P, material.η, material.ηb, material.α;
        ρ0 = material.ρ0, K = material.K, G = material.G,
        g = material.g, Tref = material.Tref, kwargs...)
StokesDR(nnodes_v, nnodes_P, material::StokesMaterial; kwargs...) =
    StokesDR(CPU(), nnodes_v, nnodes_P, material; kwargs...)

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

# Examples
```jldoctest
julia> dp = DruckerPrager((deg2rad(30),), (0.0,), (1.0e6,), (1.0e19,), (1.0e10,));

julia> dp.sinϕ[1] ≈ 0.5
true
```
"""
function DruckerPrager(
    ϕ     :: Tuple{FP, Vararg{FP, N}},
    Ψ     :: Tuple{FP, Vararg{FP, N}},
    C     :: Tuple{FP, Vararg{FP, N}},
    η_reg :: Tuple{FP, Vararg{FP, N}},
    Kb    :: Tuple{FP, Vararg{FP, N}},
) where {N, FP}
    DruckerPrager{N + 1, FP}(
        map(cos, ϕ), map(sin, ϕ), map(sin, Ψ), C, η_reg, Kb,
    )
end
