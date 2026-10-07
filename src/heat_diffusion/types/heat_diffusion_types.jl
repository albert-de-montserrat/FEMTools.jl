"""
    ThermalMaterial(; k=(1.0,), Cp=(1.0,), ρ0=(1.0,), α=(1.0,), K=(1.0,))

Per-phase thermal conductivity, heat capacity, reference density, thermal
expansivity, and bulk modulus. Properties accept scalars or phase tuples.
The first supplied tuple determines phase count and precision; without tuples,
the first supplied scalar determines precision. Scalars apply to every phase.
Omitted properties are unit-valued in that precision and phase count.
Supplied properties must share a floating-point type; incompatible precisions
are not promoted. Mismatched tuple lengths throw `DimensionMismatch`.

Fields use `NTuple{nphases, FP}` so phase count and precision remain available
to the compiler. The defaults describe one unit-valued `Float64` phase.
"""
struct ThermalMaterial{nphases, FP}
    k::NTuple{nphases, FP}
    Cp::NTuple{nphases, FP}
    ρ0::NTuple{nphases, FP}
    α::NTuple{nphases, FP}
    K::NTuple{nphases, FP}

    function ThermalMaterial(
            k::Tuple{FP, Vararg{FP}}, Cp::Tuple{FP, Vararg{FP}},
            ρ0::Tuple{FP, Vararg{FP}}, α::Tuple{FP, Vararg{FP}},
            K::Tuple{FP, Vararg{FP}},
        ) where {FP}
        nphases = length(k)
        length(Cp) == length(ρ0) == length(α) == length(K) == nphases ||
            throw(DimensionMismatch("material property tuples must have the same length"))
        return new{nphases, FP}(k, Cp, ρ0, α, K)
    end
end

function ThermalMaterial(; k = nothing, Cp = nothing, ρ0 = nothing, α = nothing, K = nothing)
    reference = _material_reference((k, Cp, ρ0, α, K), nothing)
    return ThermalMaterial(
        _material_property(k, reference, 1), _material_property(Cp, reference, 1),
        _material_property(ρ0, reference, 1), _material_property(α, reference, 1),
        _material_property(K, reference, 1),
    )
end

"""
    ThermalDiffusionDR{nphases, _T, _TI, FP}

Solver state for a transient multi-phase heat-diffusion problem solved with a
pseudo-transient dynamic-relaxation (DR) scheme.

# Type parameters
- `nphases` — number of material phases (compile-time constant)
- `_T`       — nodal float array type (e.g. `Vector{Float64}` on CPU, `CuArray` on GPU)
- `_TI`      — nodal integer array type (same backend as `_T`, element type `Int32`)
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
problem-specific and are therefore passed directly to `solve!` rather than
stored here.

# Material properties
Properties are supplied together through [`ThermalMaterial`](@ref) and stored
as `NTuple{nphases, FP}` fields in the solver state.
The reference temperature `Tref` for the density EOS is passed to `solve!`.

# Solver parameters
`CFL`, `c_fact`, `ϵ` (convergence tolerance).

# Constructors
    ThermalDiffusionDR(backend, nnodes, material::ThermalMaterial; CFL=0.98, c_fact=0.9, ϵ=1e-6)
    ThermalDiffusionDR(nnodes, material::ThermalMaterial; kwargs...)  # CPU

The tuple-based constructors remain available for compatibility.

All nodal float arrays are zero-initialised; `phases` is initialised to 1.
The float type `FP` and phase count are inferred from `material`.
"""
struct ThermalDiffusionDR{nphases, _T, _TI, FP} <: AbstractDRProblem
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
            CFL = 0.98, c_fact = 0.9, ϵ = 1.0e-6,
        ) where {N, FP}
        newvec() = KernelAbstractions.zeros(backend, FP, nnodes)
        newivec() = KernelAbstractions.ones(backend, Int32, nnodes)
        return new{N + 1, typeof(newvec()), typeof(newivec()), FP}(
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

dr_fields(dr::ThermalDiffusionDR) =
    (R = dr.R, R0 = dr.R0, ∂R∂u = dr.∂R∂T, PC = dr.PC, u = dr.T, ∂u∂τ = dr.∂T∂τ)
dr_name(::ThermalDiffusionDR) = "thermal diffusion"

ThermalDiffusionDR(nnodes, k, Cp, ρ0, α, K; kwargs...) =
    ThermalDiffusionDR(CPU(), nnodes, k, Cp, ρ0, α, K; kwargs...)

ThermalDiffusionDR(backend, nnodes, material::ThermalMaterial; kwargs...) =
    ThermalDiffusionDR(backend, nnodes, material.k, material.Cp, material.ρ0, material.α, material.K; kwargs...)
ThermalDiffusionDR(nnodes, material::ThermalMaterial; kwargs...) =
    ThermalDiffusionDR(CPU(), nnodes, material; kwargs...)

"""
    ThermalDiffusionDR(mesh::Mesh, material::ThermalMaterial; kwargs...)

Allocate thermal fields on the mesh backend, using its node count. Material
properties must have the same scalar type as the mesh coordinates. Solver
control keywords are forwarded to the count-based constructor.
"""
function ThermalDiffusionDR(mesh::Mesh, material::ThermalMaterial; kwargs...)
    _check_material_precision(mesh, material.k)
    return ThermalDiffusionDR(KernelAbstractions.get_backend(mesh.coords), mesh.nnodes, material; kwargs...)
end
