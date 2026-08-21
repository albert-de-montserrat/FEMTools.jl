"""
    StokesMaterial(; η=(1.0,), ηb=(1.0,), G=(Inf,), α=(0.0,),
                   ρ0=(1.0,), K=(Inf,), g=(0.0, 0.0), Tref=0.0)

Typed per-phase material properties and body-force parameters for `StokesDR`.
All property tuples must have the same length and floating-point type.

The length of the gravity vector `g` sets the spatial dimension `ndim`, and a
[`StokesDR`](@ref) built from this material inherits it. Pass a three-component
`g` for a three-dimensional problem, `(0.0, 0.0, 0.0)` included.
"""
@kwdef struct StokesMaterial{nphases, ndim, FP}
    η::NTuple{nphases, FP} = (1.0,)
    ηb::NTuple{nphases, FP} = (1.0,)
    G::NTuple{nphases, FP} = (Inf,)
    α::NTuple{nphases, FP} = (0.0,)
    ρ0::NTuple{nphases, FP} = (1.0,)
    K::NTuple{nphases, FP} = (Inf,)
    g::NTuple{ndim, FP} = (0.0, 0.0)
    Tref::FP = 0.0

    function StokesMaterial(
        η::Tuple{FP, Vararg{FP}}, ηb::Tuple{FP, Vararg{FP}},
        G::Tuple{FP, Vararg{FP}}, α::Tuple{FP, Vararg{FP}},
        ρ0::Tuple{FP, Vararg{FP}}, K::Tuple{FP, Vararg{FP}},
        g::NTuple{ndim, FP}, Tref::FP,
    ) where {ndim, FP}
        nphases = length(η)
        length(ηb) == length(G) == length(α) == length(ρ0) == length(K) == nphases ||
            throw(DimensionMismatch("Stokes material property tuples must have the same length"))
        ndim == 2 || ndim == 3 ||
            throw(ArgumentError("gravity must have 2 or 3 components, got $ndim"))
        return new{nphases, ndim, FP}(η, ηb, G, α, ρ0, K, g, Tref)
    end
end

# Storage extents accept either a node count or an explicit dimension tuple, so
# a cell-local layout such as `(4, nels)` is expressible alongside nodal storage.
_storage_dims(n::Integer) = (n,)
_storage_dims(dims) = Tuple(dims)

# The gravity vector fixes the spatial dimension; `Val` keeps the container
# choice a compile-time decision. Callers may supply any 2- or 3-element
# container, so it is normalised to a `Tuple` before the length is read.
_spatial_dimension(::NTuple{N}) where {N} = Val(N)
_dimension_value(::Val{N}) where {N} = N

_zero_vector_field(::Val{2}, new_array) = VectorField2D(new_array(), new_array())
_zero_vector_field(::Val{3}, new_array) =
    VectorField3D(new_array(), new_array(), new_array())

_zero_symmetric_tensor(::Val{2}, new_array) =
    SymmetricTensor2D(new_array(), new_array(), new_array(), new_array())
_zero_symmetric_tensor(::Val{3}, new_array) = SymmetricTensor3D(
    new_array(), new_array(), new_array(), new_array(),
    new_array(), new_array(), new_array(),
)

"""
    StokesDR{nphases, ndim, _TV, _TT, _TIV, _TP, _TIP, FP}

Solver state for an incompressible Stokes flow solved with a pseudo-transient
dynamic-relaxation (DR) scheme using mixed elements (separate velocity and
pressure node sets, e.g. T6/P1 Taylor-Hood-like pair).

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `ndim`     — spatial dimension, `2` or `3`
- `_TV`      — velocity container type ([`VectorField2D`](@ref FEMTools.VectorField2D) or [`VectorField3D`](@ref FEMTools.VectorField3D))
- `_TT`      — stress container type ([`SymmetricTensor2D`](@ref FEMTools.SymmetricTensor2D) or [`SymmetricTensor3D`](@ref FEMTools.SymmetricTensor3D))
- `_TIV`     — velocity-node integer array type (element type `Int`)
- `_TP`      — pressure float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TIP`     — pressure integer array type (element type `Int`)
- `FP`       — floating-point precision (`Float32` or `Float64`)

In two dimensions the stress carries the components `xx`, `yy`, `xy`; in three
dimensions `xx`, `yy`, `zz`, `yz`, `xz`, `xy`. The dimension follows the length
of the gravity vector `g`.

# Velocity-node fields (component length `nnodes_v`)
| Field      | Description                               |
|:---------- |:----------------------------------------- |
| `v`        | Velocity (current iterate)                |
| `∂v∂τ`    | Pseudo-transient rate for velocity         |
| `Rv`       | Momentum residual                         |
| `Rv0`      | Residual snapshot for λ_min estimate      |
| `∂Rv∂v`   | Row-sum Jacobian estimate for momentum     |
| `PC_v`     | Diagonal preconditioner for momentum      |
| `phases_v` | Per-node phase index (1-based integer)    |
| `τ`        | Current deviatoric stress                 |
| `τ_old`    | Previous-step deviatoric stress           |

`v`, `∂v∂τ`, `Rv`, `Rv0`, `∂Rv∂v` and `PC_v` are vector fields whose `.x`, `.y`
(and `.z` when `ndim == 3`) hold the arrays for the corresponding momentum
equation. `τ` and `τ_old` are symmetric tensor fields. `phases_v` is a plain
array.

# Pressure-node arrays (length `nnodes_P`)
| Field      | Description                               |
|:---------- |:----------------------------------------- |
| `P`        | Pressure (current iterate)                |
| `P0`       | Pressure at previous time step            |
| `∂P∂τ`    | Pseudo-transient rate for pressure         |
| `T`        | Temperature (input from thermal solver)   |
| `T0`       | Temperature at previous time step         |
| `RP`       | Pressure residual                         |
| `RP0`      | Residual snapshot for λ_min estimate      |
| `M_P`      | Lumped pressure mass (`∫ N_i dΩ`)         |
| `Pnum`     | Arrow-Hurwicz numerical pressure correction (`γP·RP/M_P`) passed to the momentum equation |
| `phases_P` | Per-node phase index (1-based integer)    |

# Material properties
Properties are supplied together through [`StokesMaterial`](@ref).

# Global scalar fields
`G` is the shear modulus. `g::NTuple{2,FP}` (gravity vector, default `(0,0)`), `Tref::FP` (reference
temperature for the linearised EOS, default 0).

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
Individual components are reached through the field containers, e.g. `dr.v.x`
and `dr.τ.xy`; the invariant slots `dr.τ.II` and `dr.τ_old.II` are allocated but
left to the caller.

Stress components default to nodal storage of length `nnodes_v`; pass
`stress_size=(nq, nels)` to store current and previous stress directly at
integration points. `nnodes_v` and `nnodes_P` likewise accept a dimension tuple
instead of a node count, which is how a cell-local pressure layout such as
`(4, nels)` is expressed. `T` and `T0` should be filled via `copyto!` before
calling the solver. The time step `Δt` is passed directly to the assembler
rather than stored here.

The spatial dimension follows the length of `g`, defaulting to two. Pass a
three-component gravity vector — `g = (0.0, 0.0, -9.81)`, or `(0.0, 0.0, 0.0)`
for a gravity-free three-dimensional problem — to obtain `VectorField3D`
velocity fields and `SymmetricTensor3D` stresses. The two-dimensional
mixed-mesh solvers accept only `StokesDR{<:Any, 2}`.
"""
struct StokesDR{nphases, ndim, _TV, _TT, _TIV, _TP, _TIP, FP}
    # velocity-node solution fields
    v::_TV
    ∂v∂τ::_TV
    # velocity-node residual and DR work arrays
    Rv::_TV
    Rv0::_TV
    ∂Rv∂v::_TV
    PC_v::_TV
    # velocity-node phase assignment
    phases_v::_TIV
    # deviatoric stress history
    τ::_TT
    τ_old::_TT
    # pressure-node solution fields
    P::_TP
    P0::_TP
    ∂P∂τ::_TP
    T::_TP
    T0::_TP
    # pressure-node residual and DR work arrays
    RP::_TP
    RP0::_TP
    M_P::_TP
    Pnum::_TP  # Arrow-Hurwicz numerical pressure correction (γP·RP/M_P) fed into momentum equation
    # pressure-node phase assignment
    phases_P::_TIP
    # physical parameters – one scalar per phase
    η::NTuple{nphases, FP}     # dynamic shear viscosity [Pa s]
    ηb::NTuple{nphases, FP}    # bulk viscosity          [Pa s]
    α::NTuple{nphases, FP}     # thermal expansivity     [K⁻¹]
    ρ0::NTuple{nphases, FP}    # reference density       [kg m⁻³]
    K::NTuple{nphases, FP}     # bulk modulus (EOS)      [Pa]
    G::NTuple{nphases, FP}     # shear modulus           [Pa]
    # global scalar parameters
    g::NTuple{ndim, FP}        # gravitational acceleration [m s⁻²]
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
        _Tref = Tref === nothing ? FP(0)           : FP(Tref)
        dim = _spatial_dimension(_g)
        v_dims = _storage_dims(nnodes_v)
        P_dims = _storage_dims(nnodes_P)
        stress_dims = stress_size === nothing ? v_dims : _storage_dims(stress_size)
        newv()  = KernelAbstractions.zeros(backend, FP,  v_dims...)
        newP()  = KernelAbstractions.zeros(backend, FP,  P_dims...)
        newτ()  = KernelAbstractions.zeros(backend, FP,  stress_dims...)
        newiv() = KernelAbstractions.ones(backend,  Int, v_dims...)
        newip() = KernelAbstractions.ones(backend,  Int, P_dims...)
        newvfield() = _zero_vector_field(dim, newv)
        newτfield() = _zero_symmetric_tensor(dim, newτ)
        new{
            nphases, _dimension_value(dim), typeof(newvfield()), typeof(newτfield()),
            typeof(newiv()), typeof(newP()), typeof(newip()), FP,
        }(
            newvfield(), newvfield(),                 # v, ∂v∂τ
            newvfield(), newvfield(),                 # Rv, Rv0
            newvfield(), newvfield(),                 # ∂Rv∂v, PC_v
            newiv(),                                  # phases_v
            newτfield(), newτfield(),                 # τ, τ_old
            newP(), newP(), newP(), newP(), newP(),   # P, P0, ∂P∂τ, T, T0
            newP(), newP(), newP(), newP(),           # RP, RP0, M_P, Pnum
            newip(),                                  # phases_P
            η, ηb, α, _ρ0, _K, _G, _g, _Tref,
            FP(CFL_v), FP(CFL_P), FP(c_fact), FP(ϵ),
        )
    end
end

"""
    velocity(dr::StokesDR) -> (vx, vy[, vz])

Return the nodal velocity-component arrays of a Stokes solver state, two or
three of them according to its dimension.
"""
velocity(dr::StokesDR) = Tuple(dr.v)

"""
    stress(dr::StokesDR) -> (τxx, τyy, τxy) or (τxx, τyy, τzz, τyz, τxz, τxy)

Return the independent deviatoric-stress component arrays of a Stokes solver
state, in Voigt order and excluding the invariant slot.
"""
stress(dr::StokesDR) = Tuple(dr.τ)

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
