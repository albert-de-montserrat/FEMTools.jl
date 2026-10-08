"""
    StokesMaterial(; η=(1.0,), ηb=(1.0,), G=(Inf,), α=(0.0,),
                   ρ0=(1.0,), K=(Inf,), g=(0.0, 0.0), Tref=0.0)

Typed per-phase material properties and body-force parameters for `StokesDR`.
Properties accept scalars or phase tuples. The first supplied property tuple
determines phase count and precision; without tuples, the first supplied scalar
does. Scalars apply to every phase. Omitted properties keep the documented
defaults in that precision and phase count. Supplied floating-point values,
including gravity and reference temperature, must have matching precision;
incompatible precisions are not promoted. With no phase properties supplied,
`Tref` or gravity determines precision. With no inputs, precision is Float64.

`g` may be any two- or three-element collection, such as a tuple or `SVector`.
The length of the gravity vector `g` sets the spatial dimension `ndim`, and a
[`StokesDR`](@ref) built from this material inherits it. Pass a three-component
`g` for a three-dimensional problem, `(0.0, 0.0, 0.0)` included.
"""
struct StokesMaterial{nphases, ndim, FP}
    η::NTuple{nphases, FP}
    ηb::NTuple{nphases, FP}
    G::NTuple{nphases, FP}
    α::NTuple{nphases, FP}
    ρ0::NTuple{nphases, FP}
    K::NTuple{nphases, FP}
    g::NTuple{ndim, FP}
    Tref::FP

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

function StokesMaterial(;
        η = nothing, ηb = nothing, G = nothing, α = nothing,
        ρ0 = nothing, K = nothing, g = nothing, Tref = nothing,
    )
    gravity = g === nothing ? nothing : Tuple(g)
    reference = _material_reference(
        (η, ηb, G, α, ρ0, K, Tref, gravity === nothing ? nothing : first(gravity)), nothing,
    )
    FP = eltype(reference)
    return StokesMaterial(
        _material_property(η, reference, 1), _material_property(ηb, reference, 1),
        _material_property(G, reference, Inf), _material_property(α, reference, 0),
        _material_property(ρ0, reference, 1), _material_property(K, reference, Inf),
        gravity === nothing ? (zero(FP), zero(FP)) : gravity,
        Tref === nothing ? zero(FP) : float(Tref),
    )
end

# Storage extents accept either a node count or an explicit dimension tuple, so
# a cell-local layout such as `(4, nels)` is expressible alongside nodal storage.
_storage_dims(n::Integer) = (n,)
_storage_dims(dims) = Tuple(dims)

_zero_vector_field(::Val{2}, new_array) = VectorField2D(new_array(), new_array())
_zero_vector_field(::Val{3}, new_array) =
    VectorField3D(new_array(), new_array(), new_array())

# The solver never reads the invariant slot, so it gets a zero-length array of
# the component type: stress history is element-sized and a full slot would add
# one unused array per tensor.
_zero_symmetric_tensor(::Val{2}, new_array, new_empty) =
    SymmetricTensor2D(new_array(), new_array(), new_array(), new_empty())
_zero_symmetric_tensor(::Val{3}, new_array, new_empty) = SymmetricTensor3D(
    new_array(), new_array(), new_array(), new_array(),
    new_array(), new_array(), new_empty(),
)

# Declared ahead of StokesDR so that its docstring stays adjacent to the struct
# it documents; a definition placed between the two silently steals it.
struct CapPlasticHistory{Tγ, Tθ}
    γ::Tγ
    θ::Tθ
end

struct CapPlasticHistoryOutput{Tγ, Tθ}
    γ::Tγ
    θ::Tθ
    iel::Int
end

"""
    StokesDR{nphases, ndim, _TV, _TT, _TIV, _TP, _TIP, FP, _TH}

Solver state for an incompressible Stokes flow solved with a pseudo-transient
dynamic-relaxation (DR) scheme using mixed elements (separate velocity and
pressure node sets, e.g. T6/P1 Taylor-Hood-like pair).

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `ndim`     — spatial dimension, `2` or `3`
- `_TV`      — velocity container type ([`VectorField2D`](@ref FEMTools.VectorField2D) or [`VectorField3D`](@ref FEMTools.VectorField3D))
- `_TT`      — stress container type ([`SymmetricTensor2D`](@ref FEMTools.SymmetricTensor2D) or [`SymmetricTensor3D`](@ref FEMTools.SymmetricTensor3D))
- `_TIV`     — velocity-node integer array type (element type `Int32`)
- `_TP`      — pressure float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TIP`     — pressure integer array type (element type `Int32`)
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
| `γP`       | Pressure scale `γ_eff` owned by the high-level solve |
| `T`        | Temperature (input from thermal solver)   |
| `T0`       | Temperature at previous time step         |
| `Q`        | Volumetric source/sink in continuity      |
| `Pf`       | Fluid pressure; the yield functions see the effective pressure `P − Pf` |
| `RP`       | Pressure residual                         |
| `RP0`      | Residual snapshot for λ_min estimate      |
| `M_P`      | Pressure mass diagonal (`∫NᵢdΩ` in 2-D, `∫Nᵢ²dΩ` in 3-D) |
| `Pnum`     | Arrow-Hurwicz numerical pressure correction (`γP·RP/M_P`) passed to the momentum equation |
| `phases_P` | Per-node phase index (1-based integer)    |

# Material properties
Properties are supplied together through [`StokesMaterial`](@ref).

# Global scalar fields
`G` is the shear modulus. `g::NTuple{ndim,FP}` is the gravity vector (default
`(0,0)`), and `Tref::FP` is the equation-of-state reference temperature.

# Solver parameters
`CFL_v`, `CFL_P`, `c_fact`, `ϵ` (convergence tolerance).

# Constructor
    StokesDR(backend, nnodes_v, nnodes_P, material::StokesMaterial;
             CFL_v=0.98, CFL_P=0.98, c_fact=0.9, ϵ=1e-6,
             stress_size=nothing, plastic_history_size=nothing)
    StokesDR(nnodes_v, nnodes_P, material; kwargs...)  # defaults to CPU()

All nodal float arrays are zero-initialised; phase arrays are initialised to 1.
Individual components are reached through the field containers, e.g. `dr.v.x`
and `dr.τ.xy`; the invariant slots `dr.τ.II` and `dr.τ_old.II` are zero-length
arrays, so they add no per-element storage and are not available as scratch.

Stress components default to nodal storage of length `nnodes_v`; pass
`stress_size=(nq, nels)` to store current and previous stress directly at
integration points, or `stress_size=:none` to allocate no stress history at all.
Pass `plastic_history_size=(nq, nels)` to allocate zeroed integration-point
arrays `dr.plastic_history.γ` and `dr.plastic_history.θ` for the cap's future
history update; it is independent of stress storage and defaults to `nothing`.
`nnodes_v` and `nnodes_P` likewise accept a dimension tuple instead of a node
count, which is how a cell-local pressure layout such as `(4, nels)` is
expressed. `T`, `T0`, and `Q` should be filled via `copyto!` before calling the
solver. `Q` is the volumetric source (positive) or sink (negative) in the
continuity equation. `Pf` is the fluid (pore or magma) pressure on the pressure
DoFs, zero by default. Only the yield functions read it: plasticity is evaluated
at the effective pressure `P − Pf`, while the momentum balance and the equation
of state keep the total pressure `P`. The time step `Δt` is passed directly to
the assembler rather than stored here.

`:none` suits a purely viscous model, where the shear modulus is infinite and the
stress history is never read. `τ` and `τ_old` are then `nothing`,
[`stress`](@ref) returns `nothing`, and the assemblers must be called with
`τ_old = nothing`.

The spatial dimension follows the length of `g`, defaulting to two. Pass a
three-component gravity vector — `g = (0.0, 0.0, -9.81)`, or `(0.0, 0.0, 0.0)`
for a gravity-free three-dimensional problem — to obtain `VectorField3D`
velocity fields and `SymmetricTensor3D` stresses. Mixed-mesh solvers accept
one boundary condition per spatial dimension.
"""
struct StokesDR{nphases, ndim, _TV, _TT, _TIV, _TP, _TIP, FP, _TH}
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
    # optional cap plastic history at integration points
    plastic_history::_TH
    # pressure-node solution fields
    P::_TP
    P0::_TP
    γP::_TP
    T::_TP
    T0::_TP
    Q::_TP
    Pf::_TP
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
            backend, nnodes_v, nnodes_P, material::StokesMaterial{nphases, ndim, FP};
            CFL_v = 0.98, CFL_P = 0.98, c_fact = 0.9, ϵ = 1.0e-6,
            stress_size = nothing,
            plastic_history_size = nothing,
        ) where {nphases, ndim, FP}
        (; η, ηb, α, ρ0, K, G, g, Tref) = material
        stress_size isa Symbol && stress_size !== :none && throw(
            ArgumentError(
                "stress_size must be `nothing`, `:none`, an integer, or a size tuple; got :$stress_size"
            )
        )
        plastic_history_size isa Symbol && plastic_history_size !== :none && throw(
            ArgumentError(
                "plastic_history_size must be `nothing`, `:none`, an integer, or a size tuple; got :$plastic_history_size"
            )
        )
        dim = Val(ndim)
        v_dims = _storage_dims(nnodes_v)
        P_dims = _storage_dims(nnodes_P)
        stress_dims = stress_size === nothing ? v_dims :
            stress_size === :none ? nothing : _storage_dims(stress_size)
        history_dims = plastic_history_size === nothing || plastic_history_size === :none ?
            nothing : _storage_dims(plastic_history_size)
        newv() = KernelAbstractions.zeros(backend, FP, v_dims...)
        newP() = KernelAbstractions.zeros(backend, FP, P_dims...)
        newτ() = KernelAbstractions.zeros(backend, FP, stress_dims...)
        newτ0() = KernelAbstractions.zeros(backend, FP, map(zero, stress_dims)...)
        newiv() = KernelAbstractions.ones(backend, Int32, v_dims...)
        newip() = KernelAbstractions.ones(backend, Int32, P_dims...)
        newvfield() = _zero_vector_field(dim, newv)
        newτfield() = stress_dims === nothing ? nothing : _zero_symmetric_tensor(dim, newτ, newτ0)
        newhistory() = history_dims === nothing ? nothing : CapPlasticHistory(
                KernelAbstractions.zeros(backend, FP, history_dims...),
                KernelAbstractions.zeros(backend, FP, history_dims...),
            )
        return new{
            nphases, ndim, typeof(newvfield()), typeof(newτfield()),
            typeof(newiv()), typeof(newP()), typeof(newip()), FP, typeof(newhistory()),
        }(
            newvfield(), newvfield(),                 # v, ∂v∂τ
            newvfield(), newvfield(),                 # Rv, Rv0
            newvfield(), newvfield(),                 # ∂Rv∂v, PC_v
            newiv(),                                  # phases_v
            newτfield(), newτfield(),                 # τ, τ_old
            newhistory(),                              # optional cap history
            newP(), newP(), newP(), newP(), newP(), newP(), # P, P0, γP, T, T0, Q
            newP(),                                   # Pf
            newP(), newP(), newP(), newP(),           # RP, RP0, M_P, Pnum
            newip(),                                  # phases_P
            η, ηb, α, ρ0, K, G, g, Tref,
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
    stress(dr::StokesDR) -> (τxx, τyy, τxy) or (τxx, τyy, τzz, τxy, τxz, τyz)

Return the independent deviatoric-stress component arrays of a Stokes solver
state, in assembler order and excluding the invariant slot. In three dimensions
this differs from the tensor container's Voigt order; for states built with
`stress_size = :none`, it returns `nothing`.
"""
stress(dr::StokesDR{<:Any, 2}) = Tuple(dr.τ)
stress(dr::StokesDR{<:Any, 3}) =
    (dr.τ.xx, dr.τ.yy, dr.τ.zz, dr.τ.xy, dr.τ.xz, dr.τ.yz)
stress(::StokesDR{<:Any, <:Any, <:Any, Nothing}) = nothing

"""
    stress_old(dr::StokesDR) -> NTuple

Return previous-time stress component arrays in the same assembler order as
[`stress`](@ref).
"""
stress_old(dr::StokesDR{<:Any, 2}) = Tuple(dr.τ_old)
stress_old(dr::StokesDR{<:Any, 3}) =
    (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.zz, dr.τ_old.xy, dr.τ_old.xz, dr.τ_old.yz)

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

StokesDR(nnodes_v, nnodes_P, material::StokesMaterial; kwargs...) =
    StokesDR(CPU(), nnodes_v, nnodes_P, material; kwargs...)

"""
    StokesDR(mesh::MixedMesh, material::StokesMaterial; stress_size, kwargs...)

Allocate velocity and pressure fields on the mesh backend. By default, stress
history has one entry per velocity integration point and element. Override
`stress_size` for a different layout, or use `:none` to omit stress history.
The mesh must have cached geometry; material precision and gravity dimension
must match its coordinates. Remaining keywords control the count-based constructor.
"""
function StokesDR(
        mesh::MixedMesh{D}, material::StokesMaterial;
        stress_size = (length(_mesh_geometry(mesh).element_v.integration_points.ω), mesh.nels),
        kwargs...,
    ) where {D}
    _check_material_precision(mesh, material.η)
    length(material.g) == D || throw(DimensionMismatch("material gravity must have $D components"))
    _mesh_geometry(mesh)
    return StokesDR(KernelAbstractions.get_backend(mesh.coords), mesh.nnodes, mesh.nnodesP,
        material; stress_size, kwargs...)
end

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
    cosϕ::NTuple{nphases, FP}
    sinϕ::NTuple{nphases, FP}
    sinΨ::NTuple{nphases, FP}
    C::NTuple{nphases, FP}
    η_reg::NTuple{nphases, FP}
    Kb::NTuple{nphases, FP}
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
        ϕ::Tuple{FP, Vararg{FP, N}},
        Ψ::Tuple{FP, Vararg{FP, N}},
        C::Tuple{FP, Vararg{FP, N}},
        η_reg::Tuple{FP, Vararg{FP, N}},
        Kb::Tuple{FP, Vararg{FP, N}},
    ) where {N, FP}
    return DruckerPrager{N + 1, FP}(map(cos, ϕ), map(sin, ϕ), map(sin, Ψ), C, η_reg, Kb)
end

"""
    DruckerPragerCap{nphases, FP}

Drucker-Prager parameters closed on the tensile side by a globally continuous
circular cap, after Popov, Berlie and Kaus (2025), Geosci. Model Dev. 18,
7035-7058.

| Field    | Description                                |
|:-------- |:------------------------------------------ |
| `cosϕ`   | cos(friction angle)                        |
| `sinϕ`   | sin(friction angle)                        |
| `sinΨ`   | sin(dilation angle)                        |
| `C`      | cohesion [Pa]                              |
| `pT`     | tensile strength [Pa], compression-positive so `pT ≤ 0` |
| `η_reg`  | plastic regularization viscosity [Pa s]    |
| `Kb`     | bulk modulus for volumetric correction [Pa]|
| `C_min`  | lower cohesion bound [Pa]                |
| `H_C`    | cohesion softening modulus [Pa]          |

All fields are `NTuple{nphases, FP}`. The shear branch is identical to
[`DruckerPrager`](@ref); the extra `pT` adds the cap. Unlike `DruckerPrager`,
`Kb` **must be finite**: every dilatant plasticity model needs a finite elastic
bulk modulus, so the cap cannot be used in the `K = Inf` incompressible gauge.
"""
struct DruckerPragerCap{nphases, FP}
    cosϕ::NTuple{nphases, FP}
    sinϕ::NTuple{nphases, FP}
    sinΨ::NTuple{nphases, FP}
    C::NTuple{nphases, FP}
    pT::NTuple{nphases, FP}
    η_reg::NTuple{nphases, FP}
    Kb::NTuple{nphases, FP}
    C_min::NTuple{nphases, FP}
    H_C::NTuple{nphases, FP}
end

"""
    DruckerPragerCap(ϕ, Ψ, C, pT, η_reg, Kb; C_min=C, H_C=zero) -> DruckerPragerCap

Construct tensile-cap Drucker-Prager parameters from friction angle `ϕ`,
dilation angle `Ψ`, cohesion `C`, tensile strength `pT`, regularization
viscosity `η_reg`, and bulk modulus `Kb`. `C_min` and `H_C` configure optional
linear cohesion softening against accumulated deviatoric plastic strain `γ`;
defaults disable softening. Angles are in radians. All tuple arguments are
`NTuple{nphases, FP}` and must be finite.

Validates, per phase, the conditions under which the cap exists at all:

  - `pT ≤ 0`, because pressure is compression-positive here;
  - `C·cos(ϕ) + sin(ϕ)·pT > 0`, which is what makes the cap radius positive.
    Cohesion must exceed `sin(ϕ)·|pT|`, so a rock cannot be given a tensile
    strength that outruns its shear strength;
  - positive cap radius also at `C_min`, so it survives the entire linear
    cohesion-softening interval;
  - `Kb > 0`, since dilatant plasticity needs positive finite bulk compliance;
  - `η_reg ≥ 0`.

Each is an `ArgumentError` naming the offending phase.

# Examples
```jldoctest
julia> dp = DruckerPragerCap((deg2rad(30),), (deg2rad(10),), (1.0e6,), (-5.0e5,), (0.0,), (2.0e11,));

julia> dp.pT[1] ≈ -5.0e5
true
```
"""
function DruckerPragerCap(
        ϕ::Tuple{FP, Vararg{FP, N}},
        Ψ::Tuple{FP, Vararg{FP, N}},
        C::Tuple{FP, Vararg{FP, N}},
        pT::Tuple{FP, Vararg{FP, N}},
        η_reg::Tuple{FP, Vararg{FP, N}},
        Kb::Tuple{FP, Vararg{FP, N}},
        ; C_min = C,
        H_C = ntuple(_ -> zero(FP), Val(N + 1)),
    ) where {N, FP}
    C_min = NTuple{N + 1, FP}(C_min)
    H_C = NTuple{N + 1, FP}(H_C)
    for i in 1:(N + 1)
        all(isfinite, (ϕ[i], Ψ[i], C[i], pT[i], η_reg[i], Kb[i], C_min[i], H_C[i])) ||
            throw(ArgumentError("phase $i: tensile-cap parameters must all be finite"))
        pT[i] > 0 && throw(
            ArgumentError(
                "phase $i: tensile strength pT must be ≤ 0 with compression-positive \
                 pressure, got $(pT[i])"
            )
        )
        c = C[i] * cos(ϕ[i])
        c + sin(ϕ[i]) * pT[i] > 0 || throw(
            ArgumentError(
                "phase $i: cap radius is not positive; need C·cos(ϕ) + sin(ϕ)·pT > 0, \
                 got $(c + sin(ϕ[i]) * pT[i]). Cohesion must exceed sin(ϕ)·|pT|."
            )
        )
        Kb[i] > 0 || throw(
            ArgumentError(
                "phase $i: dilatant plasticity requires a positive finite bulk modulus, got $(Kb[i])"
            )
        )
        η_reg[i] ≥ 0 || throw(
            ArgumentError("phase $i: η_reg must be ≥ 0, got $(η_reg[i])")
        )
        C_min[i] ≥ 0 || throw(
            ArgumentError("phase $i: C_min must be ≥ 0, got $(C_min[i])")
        )
        C_min[i] ≤ C[i] || throw(
            ArgumentError("phase $i: C_min must be ≤ C, got C_min=$(C_min[i]), C=$(C[i])")
        )
        H_C[i] ≤ 0 || throw(
            ArgumentError("phase $i: H_C must be ≤ 0 for softening, got $(H_C[i])")
        )
        C_min[i] * cos(ϕ[i]) + sin(ϕ[i]) * pT[i] > 0 || throw(
            ArgumentError("phase $i: softened cap radius must remain positive; need C_min·cos(ϕ) + sin(ϕ)·pT > 0")
        )
    end
    return DruckerPragerCap{N + 1, FP}(
        map(cos, ϕ), map(sin, ϕ), map(sin, Ψ), C, pT, η_reg, Kb, C_min, H_C,
    )
end

"""
    CellPressureStokesDR(mesh, material::StokesMaterial; phases=nothing)

State of the viscous 3-D Stokes solver on Hex27 cells: continuous Q2 velocity and
four discontinuous pressure modes `(1, ξ, η, ζ)` per cell.

Precision, backend, and sizes follow `mesh`. The velocity `v` and the
`4 × nels` pressure `P` start at zero and are updated in place by
[`solve!`](@ref). The momentum uses the phase viscosities `material.η`, densities
`material.ρ0`, and the three-component gravity `material.g`. The discretization
is purely viscous and incompressible: finite `G` or `K` and nonzero `α` are
rejected, and `ηb` and `Tref` are not used.

`phases` assigns one material phase per cell and must live on the mesh backend;
it defaults to phase one everywhere and is borrowed, not copied.

The state also owns the residuals, the Jacobi preconditioner, the lumped
pressure mass, and the reference tables that every solve reuses. The
preconditioner and pressure mass are refilled from `η` and `phases` at the start
of each solve.
"""
struct CellPressureStokesDR{T, N, TV, TP, TC, TT}
    v::VectorField3D{TV}
    P::TP
    phases::TC
    η::NTuple{N, T}
    ρ::NTuple{N, T}
    g::NTuple{3, T}
    Rv::NTuple{3, TV}
    RP::TP
    diagonal::NTuple{3, TV}
    pressure_mass::TP
    tables::TT
end

function CellPressureStokesDR(mesh::Mesh, material::StokesMaterial; phases = nothing)
    mesh.element isa ReferenceElement{<:QuadraticElement{3, 27}} || throw(
        ArgumentError("CellPressureStokesDR requires a Hex27 mesh, got $(typeof(mesh.element))")
    )
    _check_material_precision(mesh, material.η)
    length(material.g) == 3 || throw(DimensionMismatch("material gravity must have 3 components"))
    all(isinf, material.G) && all(isinf, material.K) && all(iszero, material.α) || throw(
        ArgumentError("CellPressureStokesDR is purely viscous and incompressible; G and K must be Inf and α zero")
    )
    backend = KernelAbstractions.get_backend(mesh.coords)
    T = eltype(eltype(mesh.coords))
    nphases = length(material.η)
    if phases === nothing
        phases = KernelAbstractions.ones(backend, Int32, mesh.nels)
    else
        length(phases) == mesh.nels ||
            throw(DimensionMismatch("phases has $(length(phases)) entries but the mesh has $(mesh.nels) cells"))
        typeof(KernelAbstractions.get_backend(phases)) === typeof(backend) ||
            throw(ArgumentError("phases must live on the mesh backend $backend"))
        lo, hi = extrema(phases)
        1 ≤ lo && hi ≤ nphases ||
            throw(ArgumentError("phases must lie in 1:$nphases, got $lo:$hi"))
    end
    nodal() = KernelAbstractions.zeros(backend, T, mesh.nnodes)
    cellwise() = KernelAbstractions.zeros(backend, T, 4, mesh.nels)
    return CellPressureStokesDR(
        VectorField3D(nodal(), nodal(), nodal()), cellwise(), phases,
        material.η, material.ρ0, material.g,
        ntuple(_ -> nodal(), 3), cellwise(), ntuple(_ -> nodal(), 3), cellwise(),
        stokes_tables_3d(backend, mesh.element),
    )
end

"""
    StokesAdjointWorkspace(dr, v_nodes...; enzyme=false)

Caller-owned scratch for the mixed-mesh Stokes adjoint DYREL solver, with one
set of constrained velocity nodes per direction in `v_nodes`, e.g.
`StokesAdjointWorkspace(dr, vx_nodes, vy_nodes)` in two dimensions.

The workspace owns the mesh-sized residual, rate, pullback, and homogeneous
boundary-value buffers that would otherwise be allocated on every adjoint
solve. Pass it as the `workspace` keyword of
[`solve_stokes_adjoint_dyrel!`](@ref) to reuse those buffers across an
optimization loop. Set `enzyme=true` when the workspace will be used with
`operator = :enzyme`, which is available in two dimensions only; the block and
matrix-free paths do not allocate those additional reverse-mode buffers.

A workspace belongs to the velocity and pressure layouts and boundary-node
counts from which it was constructed. The solver validates those dimensions
before use and refills all scratch that carries values, so reuse never carries
residual state from one solve into the next.
"""
struct StokesAdjointWorkspace{TC, TE}
    common::TC
    enzyme::TE
end

function StokesAdjointWorkspace(
        dr::StokesDR{<:Any, D}, v_nodes::Vararg{Any, D}; enzyme = false,
    ) where {D}
    enzyme && D != 2 &&
        throw(ArgumentError("the Enzyme adjoint workspace is implemented in two dimensions only, got D = $D"))
    Rv = Tuple(getfield(dr, :Rv))
    v = velocity(dr)
    common = (;
        Resλv = map(zero, Rv),
        Resλv0 = map(zero, Rv),
        ResλP = zero(dr.P),
        λrate = map(zero, v),
        dv = map(zero, v),
        zero_bc = map((vc, nodes) -> fill!(similar(vc, length(nodes)), 0), v, v_nodes),
    )
    enzyme_scratch = enzyme ? (;
            Rv_x_buf = zero(dr.Rv.x),
            Rv_y_buf = zero(dr.Rv.y),
            seed_Rv_x = zero(dr.Rv.x),
            seed_Rv_y = zero(dr.Rv.y),
            seed_RP = zero(dr.RP),
            dP = zero(dr.P),
            dP_scratch = zero(dr.P),
            Pnum = zero(dr.P),
            dPnum = zero(dr.P),
        ) : nothing
    return StokesAdjointWorkspace(common, enzyme_scratch)
end
