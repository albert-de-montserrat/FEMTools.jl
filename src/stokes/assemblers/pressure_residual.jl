@inline _pressure_bulk_modulus(::Nothing, K) = nothing
@inline _pressure_bulk_modulus(::DruckerPrager, K) = nothing
@inline _pressure_bulk_modulus(::DruckerPragerCap, K) = K
@inline _select_pressure_bulk(ηb, ::Nothing) = ηb
@inline _select_pressure_bulk(ηb, K) = K

"""
    compute_velocity_divergence(v, ∂N∂x_v) -> ∇V

Compute the volumetric divergence ∇·v at a single quadrature point.

`v` is an `NTuple{N}` of per-component nodal velocity vectors (each
`SVector{M}`), and `∂N∂x_v` is the `M×N` matrix of shape-function spatial
gradients at the point. The result is the scalar `∑ⱼ ∂Nᵢ/∂xⱼ · vⱼ` summed
over all nodes `i` and spatial dimensions `j`. Implemented as a `@generated`
function to unroll all loops at compile time.
"""
@generated function compute_velocity_divergence(v::Tuple{SVector{M}, Vararg{SVector{M}, N}}, ∂N∂x_v) where {N, M}
    quote
        @inline
        ∇V = zero(∂N∂x_v[1, 1] * v[1][1])
        Base.@nexprs $(N + 1) j-> begin
            v_j = v[j]
            Base.@nexprs $M i-> begin
                ∇V += ∂N∂x_v[i,j] * v_j[i]
            end
        end
        return ∇V
    end
end

@kernel function _pressure_source_area_kernel!(area, invalid, @Const(Qq), @Const(el2nP), @Const(geo_P), NqP, ::Val{NP}) where {NP}
    iel = @index(Global)
    nodes = local_nodes_of(el2nP, iel, Val(NP))
    Qloc = _gather_local(Qq, nodes, Val(NP))
    for i in 1:NP
        (!isfinite(Qloc[i]) || Qloc[i] < 0) &&
            Atomix.@atomic invalid[1] += Int32(1)
    end
    for q in eachindex(geo_P[iel])
        qvalue = dot(NqP[q], Qloc)
        if !isfinite(qvalue) || qvalue < 0
            Atomix.@atomic invalid[1] += Int32(1)
        end
        Atomix.@atomic area[1] += qvalue * _pressure_weight(geo_P[iel][q])
    end
end

@kernel function _finite_source_kernel!(invalid, @Const(Qq))
    i = @index(Global, Linear)
    isfinite(Qq[i]) || Atomix.@atomic invalid[1] += Int32(1)
end

function _validate_finite_source!(Qq, backend, workgroup)
    invalid = KernelAbstractions.zeros(backend, Int32, 1)
    _finite_source_kernel!(backend, workgroup)(invalid, Qq; ndrange = length(Qq))
    KA.synchronize(backend)
    Array(invalid)[1] == 0 || throw(ArgumentError("Qq must be finite"))
    return nothing
end

function _normalize_pressure_source!(Q, Qq, Q2D, el2nP, geo_P, nels, element_v, element_P, backend, workgroup)
    size(Qq) == size(Q) || throw(DimensionMismatch("Qq must match the pressure field shape"))
    typeof(KA.get_backend(Qq)) === typeof(backend) ||
        throw(ArgumentError("Qq and Stokes state must use the same backend"))
    isfinite(Q2D) || throw(ArgumentError("Q2D must be finite"))
    NqP = shape_function_values(element_P, element_v.integration_points)
    area = KernelAbstractions.zeros(backend, eltype(Q), 1)
    invalid = KernelAbstractions.zeros(backend, Int32, 1)
    _pressure_source_area_kernel!(backend, workgroup)(
        area, invalid, Qq, el2nP, geo_P, NqP, local_nodes_val(element_P);
        ndrange = nels,
    )
    KA.synchronize(backend)
    Array(invalid)[1] == 0 || throw(ArgumentError("Qq must be finite and nonnegative"))
    area_host = Array(area)[1]
    (isfinite(area_host) && area_host > 0) ||
        (iszero(Q2D) ? (fill!(Q, 0); return nothing) :
         throw(ArgumentError("nonzero Q2D requires nonempty positive Qq support")))
    copyto!(Q, Qq)
    Q .*= eltype(Q)(Q2D / area_host)
    return nothing
end

"""
    IntegrationPointPressure{TP}

Previous-step pressure held at the `NQ` quadrature points of one element, as
gathered from an `NQ × nels` matrix. The pressure rate then reads the old
pressure directly at each point instead of interpolating a nodal field.
"""
struct IntegrationPointPressure{TP}
    P::TP
end

@inline pressure_increment_at_ip(Nv, P_loc, P0loc::SVector, _) = dot(Nv, P_loc - P0loc)
@inline pressure_increment_at_ip(Nv, P_loc, P0::IntegrationPointPressure, q) =
    dot(Nv, P_loc) - P0.P[q]

@inline _gather_old_pressure(P0::AbstractVector, nodes, _, ::Val{NP}, ::Val) where NP =
    _gather_local(P0, nodes, Val(NP))
@inline _gather_old_pressure(P0::AbstractMatrix, _, iel, ::Val, ::Val{NQ}) where NQ =
    IntegrationPointPressure(SVector{NQ}(ntuple(q -> P0[q, iel], Val(NQ))))

@inline _pressure_weight(dΩ::Number) = dΩ
@inline _pressure_weight(dΩ) = last(dΩ)

"""
    integrate_PH_pressure_residual(v, P_loc, P0loc, Tloc, T0loc, Qloc,
                                   geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, Nq; K=nothing) -> RP_e

Integrate the element pressure residual for a P–H (pressure–heat) coupled Stokes formulation.

`v` contains one element velocity vector per spatial dimension. `P_loc`
and `P0loc` are the current and previous pressure values at the `N` pressure
nodes. `Tloc` and `T0loc` are the corresponding temperatures. `α` and `ηb`
are per-phase thermal expansion and bulk viscosity `NTuple`s; `Δt` is the time
step. `Nq` contains pressure shape-function values at pressure quadrature
points.

`P0loc` may instead be an `IntegrationPointPressure`, which supplies the old
pressure at each quadrature point.

`Qloc` is an element-local volumetric source/sink. The optional `K` keyword
selects the trial-pressure formulation's elastic compressibility term: when
supplied, `K` replaces `ηb` in the pressure-rate denominator. Omitting it
preserves the historical bulk-viscosity form.

Weak form per pressure node `i`:

    RPᵢ = ∫ Nᵢ (−∇·v − ∂P/∂t/ηₚ + α ∂T/∂t + Q) dΩ

where `geo_v_el` provides velocity shape-function gradients and `geo_P_el`
provides pressure quadrature weights. `geo_P_el` holds those weights directly:
pressure gradients have no consumer, so they are not stored. Note: velocity gradients are currently
evaluated at velocity integration points rather than pressure points. Pressure
and temperature rates are interpolated from their nodal increments before the
material factors are applied.
"""
@inline function integrate_PH_pressure_residual(v::Tuple{SVector{M}, Vararg{SVector{M}, D}}, P_loc::SVector{N}, P0loc, Tloc, T0loc, Qloc, geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, Nq; K = nothing) where {D, M, N}
    RP_e = zero(P_loc)
    for q in eachindex(geo_P_el)
        ∂N∂x_v, = geo_v_el[q] # velocity NOTE: this should be ∂N∂x_v evaluated at linear 3 ips
        dΩ      = _pressure_weight(geo_P_el[q]) # pressure
        Nv      = Nq[q]
        Qq      = isnothing(Qloc) ? zero(P_loc[1]) : dot(Nv, Qloc)

        # project parameters to integration point
        bulk = _select_pressure_bulk(ηb, K)
        ηbq = interp2ip_phase(Nv, bulk, phase_loc)
        αq  = interp2ip_phase(Nv, α, phase_loc)
        ∂P∂t = pressure_increment_at_ip(Nv, P_loc, P0loc, q) / (ηbq * Δt)
        ∂T∂t = αq * dot(Nv, Tloc - T0loc) / Δt
        # project divergence to integration point
        ∇V = compute_velocity_divergence(v, ∂N∂x_v)
        # compute pressure residual
        RP_e += SVector{N}(ntuple(
            i -> Nv[i] * (-∇V - ∂P∂t + ∂T∂t + Qq) * dΩ,
            Val(N),
        ))
    end
    return RP_e
end

integrate_PH_pressure_residual(v, P_loc, P0loc, Tloc, T0loc,
        geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, Nq; K = nothing) =
    integrate_PH_pressure_residual(
        v, P_loc, P0loc, Tloc, T0loc, nothing,
        geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, Nq; K,
    )

"""
    assemble_pressure_residual_matrices_atomix!(RP, vx, vy, P, P0, T, T0,
                                                el2n_v, el2nP, geo_v, geo_P, nels,
                                                element_v, element_P,
                                                phases, α, ηb, Δt, backend, workgroup)

Assemble the Stokes pressure residual `RP` using Atomix-backed atomic scatter.

`RP` is indexed over pressure DoFs (connectivity `el2nP`). Velocity fields
`vx`, `vy` are indexed over velocity DoFs (`el2n_v`). `P`, `P0`, `T`, `T0`
are current and previous pressure and temperature fields on pressure nodes;
`P0` may instead be an `NQ × nels` matrix of previous pressure at the velocity
quadrature points.
`phases` is a nodal integer array (pressure-node indexed) selecting the phase
for material interpolation. `α` and `ηb` are per-phase thermal expansion and
bulk viscosity `NTuple`s.
"""
function assemble_pressure_residual_matrices_atomix!(
    RP,
    vx::AbstractVector, vy::AbstractVector,
    P, P0,
    T, T0,
    el2n_v, el2nP,
    geo_v, geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    α, ηb,
    Δt,
    backend, workgroup,
    ; Q = nothing, K = nothing,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    # Evaluate pressure shape functions at velocity IPs so that geo_P
    # (precomputed at velocity IPs) and NqP share the same quadrature points.
    NqP = shape_function_values(element_P, element_v.integration_points)
    ∂N∂ξ_v = shape_function_gradients(element_v)

    return assemble_pressure_residual_kernel!(
        RP, vx, vy, P, P0, T, T0,
        el2n_v, el2nP, geo_v, geo_P, nels,
        phases, α, ηb, Δt, NqP, ∂N∂ξ_v,
        Val(NV), Val(NP), workgroup, K; Q,
    )
end

function assemble_pressure_residual_matrices_atomix!(
    RP, v::NTuple{D, <:AbstractVector}, P, P0, T, T0,
    el2n_v, el2nP, geo_v, geo_P, nels,
    element_v::ReferenceElement{TV}, element_P::ReferenceElement{TP},
    phases, α, ηb, Δt, backend, workgroup; K = nothing,
) where {D, TV <: AbstractElement{D, NV}, TP <: AbstractElement{D, NP}} where {NV, NP}
    Q = fill!(similar(P), 0)
    return assemble_pressure_residual_matrices_atomix!(
        RP, v, P, P0, T, T0, Q,
        el2n_v, el2nP, geo_v, geo_P, nels, element_v, element_P,
        phases, α, ηb, Δt, backend, workgroup; K,
    )
end

function assemble_pressure_residual_matrices_atomix!(
    RP, v::NTuple{D, <:AbstractVector}, P, P0, T, T0, Q,
    el2n_v, el2nP, geo_v, geo_P, nels,
    element_v::ReferenceElement{TV}, element_P::ReferenceElement{TP},
    phases, α, ηb, Δt, backend, workgroup; K = nothing,
) where {D, TV <: AbstractElement{D, NV}, TP <: AbstractElement{D, NP}} where {NV, NP}
    return assemble_pressure_residual_matrices_atomix!(
        RP, v[1], v[2], P, P0, T, T0,
        el2n_v, el2nP, geo_v, geo_P, nels, element_v, element_P,
        phases, α, ηb, Δt, backend, workgroup; Q, K,
    )
end

"""
    assemble_pressure_residual_kernel!(RP, vx, vy, P, P0, T, T0,
                                       el2n_v, el2nP, geo_v, geo_P, nels,
                                       phases, α, ηb, Δt, NqP, ∂N∂ξ_v,
                                       Val(NV), Val(NP), workgroup)

Zero `RP`, launch the atomic pressure-residual kernel over `nels` elements,
and synchronize.

Low-level entry point beneath `assemble_pressure_residual_matrices_atomix!`:
the pressure shape-function table `NqP` (evaluated at the velocity quadrature
points), the velocity reference-element gradients `∂N∂ξ_v`, and the local node
counts `Val(NV)`, `Val(NP)` are passed explicitly,
which makes the call differentiable with Enzyme (see
`assemble_pressure_residual_matrices_atomix_adj!`). The backend is inferred
from `RP`.

Both tables may be the `NTuple`s the shape-function accessors return or the
backend arrays [`quadrature_table`](@ref) builds from them. A caller that
launches this repeatedly wants the arrays: a tuple is copied into the kernel
argument pack on every launch.
"""
function assemble_pressure_residual_kernel!(
    RP, vx::AbstractVector, vy::AbstractVector, P, P0, T, T0,
    el2n_v, el2nP, geo_v, geo_P, nels,
    phases, α, ηb, Δt, NqP, ∂N∂ξ_v,
    ::Val{NV}, ::Val{NP}, workgroup, K = nothing,
    ; Q = nothing,
) where {NV, NP}
    Q = isnothing(Q) ? fill!(similar(P), 0) : Q
    fill!(RP, 0)
    backend = KA.get_backend(RP)
    pressure_residual_atomic_kernel!(backend, workgroup)(
        RP, vx, vy, P, P0, T, T0, Q,
        el2n_v, el2nP, geo_v, geo_P,
        phases, α, ηb, Δt, NqP, ∂N∂ξ_v, Val(NV), Val(NP), K;
        ndrange = nels,
    )
    KA.synchronize(backend)
    return nothing
end

function assemble_pressure_residual_kernel!(
    RP, v::NTuple{D, <:AbstractVector}, P, P0, T, T0,
    el2n_v, el2nP, geo_v, geo_P, nels, phases, α, ηb, Δt,
    NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP}, workgroup, K = nothing,
) where {D, NV, NP}
    Q = fill!(similar(P), 0)
    return assemble_pressure_residual_kernel!(
        RP, v, P, P0, T, T0, Q,
        el2n_v, el2nP, geo_v, geo_P, nels, phases, α, ηb, Δt,
        NqP, ∂N∂ξ_v, Val(NV), Val(NP), workgroup, K,
    )
end

function assemble_pressure_residual_kernel!(
    RP, v::NTuple{D, <:AbstractVector}, P, P0, T, T0, Q,
    el2n_v, el2nP, geo_v, geo_P, nels, phases, α, ηb, Δt,
    NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP}, workgroup, K = nothing,
) where {D, NV, NP}
    return assemble_pressure_residual_kernel!(
        RP, v[1], v[2], P, P0, T, T0,
        el2n_v, el2nP, geo_v, geo_P, nels, phases, α, ηb, Δt,
        NqP, ∂N∂ξ_v, Val(NV), Val(NP), workgroup, K; Q,
    )
end

function assemble_pressure_residual_kernel!(
    RP, vx::AbstractVector, vy::AbstractVector, P, P0, T, T0, Q, args...,
)
    return assemble_pressure_residual_kernel!(
        RP, vx, vy, P, P0, T, T0, args...; Q,
    )
end

@kernel function pressure_residual_atomic_kernel!(
    RP,
    @Const(vx), @Const(vy),
    @Const(P), @Const(P0),
    @Const(T), @Const(T0),
    @Const(Q),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v), @Const(geo_P),
    @Const(phases),
    α, ηb, Δt, NqP, @Const(∂N∂ξ_v), ::Val{NV}, ::Val{NP}, K,
) where {NV, NP}
    iel = @index(Global)
    local_nodes_P, Re = pressure_element_residual(vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), Q, K)
    # P is discontinuous here, so element pressure DoFs are not shared.
    _add_local!(RP, local_nodes_P, Re, Val(false))
end

"""
    pressure_element_residual(vx, vy, P, P0, T, T0, el2n_v, el2nP,
                               geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel,
                               Val(NV), Val(NP))

Gather element-local nodal values and integrate the Stokes pressure residual for element `iel`.

Returns `(local_nodes_P, Re)` ready for global scatter into `RP`.
"""
@inline function pressure_element_residual(vx, vy, P, P0, T, T0, el2n_v, el2nP,
        geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel, ::Val{NV}, ::Val{NP}, Q = nothing, K = nothing) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = element_geometry(geo_v, iel, ∂N∂ξ_v)
    geo_P_el  = geo_P[iel]
    vxloc     = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc     = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc     = _gather_local(P,  local_nodes_P, Val(NP))
    P0loc     = _gather_old_pressure(P0, local_nodes_P, iel, Val(NP), quadrature_points_val(geo_v_el))
    Tloc      = _gather_local(T,  local_nodes_P, Val(NP))
    T0loc     = _gather_local(T0, local_nodes_P, Val(NP))
    Qloc      = isnothing(Q) ? nothing : _gather_local(Q, local_nodes_P, Val(NP))
    phase_loc = _gather_phase(phases, local_nodes_P, iel, Val(NP))
    Re = integrate_PH_pressure_residual(
        (vxloc, vyloc), P_loc, P0loc, Tloc, T0loc, Qloc,
        geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, NqP; K,
    )
    return local_nodes_P, Re
end

@inline function pressure_element_residual(
    v::NTuple{D}, P, P0, T, T0, Q, el2n_v, el2nP,
    geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel,
    ::Val{NV}, ::Val{NP}, K = nothing,
) where {D, NV, NP}
    return pressure_element_residual(
        v[1], v[2], P, P0, T, T0, el2n_v, el2nP,
        geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel,
        Val(NV), Val(NP), Q, K,
    )
end

@inline function pressure_element_residual(
    v::NTuple{D}, P, P0, T, T0, el2n_v, el2nP,
    geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel,
    ::Val{NV}, ::Val{NP},
) where {D, NV, NP}
    return pressure_element_residual(
        v, P, P0, T, T0, nothing, el2n_v, el2nP,
        geo_v, geo_P, phases, α, ηb, Δt, NqP, ∂N∂ξ_v, iel,
        Val(NV), Val(NP), nothing,
    )
end

"""
    assemble_stokes_pressure_residual_3d!(RP, v, mesh; workgroup=256, tables=...)

Assemble `-∇·v` against the four cell-local modes `(1, ξ, η, ζ)`.

`tables` supplies the backend-resident reference tables from
[`stokes_tables_3d`](@ref); the default builds a fresh set, so a loop that calls
this repeatedly should hand over one it owns.
"""
function assemble_stokes_pressure_residual_3d!(
    RP::AbstractMatrix, v::NTuple{3}, mesh::Mesh; workgroup = 256,
    tables = stokes_tables_3d(KA.get_backend(RP), mesh.element),
)
    size(RP) == (4, mesh.nels) || throw(DimensionMismatch("RP must be 4 × nels"))
    all(length(u) == mesh.nnodes for u in v) || throw(DimensionMismatch("velocity size must match mesh nodes"))
    backend = KA.get_backend(RP)
    stokes_pressure_residual_3d_kernel!(backend, workgroup)(
        RP, v, mesh.el2n, mesh.geometry, tables.modes, tables.∂N∂ξ; ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function stokes_pressure_residual_3d_kernel!(
    RP, @Const(v), @Const(el2n), @Const(geometry), @Const(NqP), @Const(∂N∂ξ),
)
    cell = @index(Global)
    nodes = local_nodes_of(el2n, cell, Val(27))
    velocity = ntuple(i -> _gather_local(v[i], nodes, Val(27)), 3)
    residual = zero(SVector{4, eltype(RP)})
    geo_el = element_geometry(geometry, cell, ∂N∂ξ)
    for q in eachindex(geo_el)
        gradient, dΩ = geo_el[q]
        residual -= NqP[q] * (compute_velocity_divergence(velocity, gradient) * dΩ)
    end
    for i in 1:4
        RP[i, cell] = residual[i]
    end
end
