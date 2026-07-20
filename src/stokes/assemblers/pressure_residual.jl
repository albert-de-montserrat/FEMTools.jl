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

"""
    integrate_PH_pressure_residual(v, P_loc, P0loc, Tloc, T0loc,
                                   geo_v_el, geo_P_el, phase_loc, α, ηb, ξ, Δt, Nq) -> RP_e

Integrate the element pressure residual for a P–H (pressure–heat) coupled Stokes formulation.

`v` is an `NTuple{2}` of element velocity vectors `(vxloc, vyloc)`. `P_loc`
and `P0loc` are the current and previous pressure values at the `N` pressure
nodes. `Tloc` and `T0loc` are the corresponding temperatures. `α` and `ηb`
are per-phase thermal expansion and bulk viscosity `NTuple`s; `ξ` is the
per-phase compaction viscosity. `Δt` is the time step. `Nq` contains pressure
shape-function values at pressure quadrature points.

Weak form per pressure node `i`:

    RPᵢ = ∫ Nᵢ (−∇·v − ∂P/∂t/ηb − P/ξ + α ∂T/∂t) dΩ

where `geo_v_el` provides velocity shape-function gradients and `geo_P_el`
provides pressure quadrature weights. Note: velocity gradients are currently
evaluated at velocity integration points rather than pressure points. Pressure
and temperature rates are interpolated from their nodal increments before the
material factors are applied.
"""
@inline function integrate_PH_pressure_residual(v::Tuple{<:SVector, <:SVector}, P_loc::SVector{N}, P0loc, Tloc, T0loc, geo_v_el, geo_P_el, phase_loc, α, ηb, ξ, Δt, Nq) where N
    RP_e = zero(P_loc)
    for q in eachindex(geo_P_el)
        ∂N∂x_v, = geo_v_el[q] # velocity NOTE: this should be ∂N∂x_v evaluated at linear 3 ips
        _, dΩ   = geo_P_el[q] # pressure
        Nv      = Nq[q]

        # project parameters to integration point
        ηbq = interp2ip_phase(Nv, ηb, phase_loc)
        ξq  = interp2ip_phase(Nv, ξ, phase_loc)
        αq  = interp2ip_phase(Nv, α, phase_loc)
        ∂P∂t = dot(Nv, P_loc - P0loc) / (ηbq * Δt) + dot(Nv, P_loc) / ξq
        ∂T∂t = αq * dot(Nv, Tloc - T0loc) / Δt
        # project divergence to integration point
        ∇V = compute_velocity_divergence(v, ∂N∂x_v)
        # compute pressure residual
        RP_e += SVector{N}(ntuple(
            i -> Nv[i] * (-∇V - ∂P∂t + ∂T∂t) * dΩ,
            Val(N),
        ))
    end
    return RP_e
end

"""
    assemble_pressure_residual_matrices_atomix!(RP, vx, vy, P, P0, T, T0,
                                                el2n_v, el2nP, geo_v, geo_P, nels,
                                                element_v, element_P,
                                                phases, α, ηb, ξ, Δt, backend, workgroup)

Assemble the Stokes pressure residual `RP` using Atomix-backed atomic scatter.

`RP` is indexed over pressure DoFs (connectivity `el2nP`). Velocity fields
`vx`, `vy` are indexed over velocity DoFs (`el2n_v`). `P`, `P0`, `T`, `T0`
are current and previous pressure and temperature fields on pressure nodes.
`phases` is a nodal integer array (pressure-node indexed) selecting the phase
for material interpolation. `α`, `ηb`, and `ξ` are per-phase thermal expansion,
bulk viscosity, and compaction viscosity `NTuple`s.
"""
function assemble_pressure_residual_matrices_atomix!(
    RP,
    vx, vy,
    P, P0,
    T, T0,
    el2n_v, el2nP,
    geo_v, geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    α, ηb, ξ,
    Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    # Evaluate pressure shape functions at velocity IPs so that geo_P
    # (precomputed at velocity IPs) and NqP share the same quadrature points.
    NqP = shape_function_values(element_P, element_v.integration_points)

    return assemble_pressure_residual_kernel!(
        RP, vx, vy, P, P0, T, T0,
        el2n_v, el2nP, geo_v, geo_P, nels,
        phases, α, ηb, ξ, Δt, NqP,
        Val(NV), Val(NP), workgroup,
    )
end

"""
    assemble_pressure_residual_kernel!(RP, vx, vy, P, P0, T, T0,
                                       el2n_v, el2nP, geo_v, geo_P, nels,
                                       phases, α, ηb, ξ, Δt, NqP,
                                       Val(NV), Val(NP), workgroup)

Zero `RP`, launch the atomic pressure-residual kernel over `nels` elements,
and synchronize.

Low-level entry point beneath `assemble_pressure_residual_matrices_atomix!`:
the pressure shape-function table `NqP` (evaluated at the velocity quadrature
points) and the local node counts `Val(NV)`, `Val(NP)` are passed explicitly,
which makes the call differentiable with Enzyme (see
`assemble_pressure_residual_matrices_atomix_adj!`). The backend is inferred
from `RP`.
"""
function assemble_pressure_residual_kernel!(
    RP, vx, vy, P, P0, T, T0,
    el2n_v, el2nP, geo_v, geo_P, nels,
    phases, α, ηb, ξ, Δt, NqP,
    ::Val{NV}, ::Val{NP}, workgroup
) where {NV, NP}
    fill!(RP, 0)
    backend = KA.get_backend(RP)
    pressure_residual_atomic_kernel!(backend, workgroup)(
        RP, vx, vy, P, P0, T, T0,
        el2n_v, el2nP, geo_v, geo_P,
        phases, α, ηb, ξ, Δt, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function pressure_residual_atomic_kernel!(
    RP,
    @Const(vx), @Const(vy),
    @Const(P), @Const(P0),
    @Const(T), @Const(T0),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v), @Const(geo_P),
    @Const(phases),
    α, ηb, ξ, Δt, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_P, Re = pressure_element_residual(vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, phases, α, ηb, ξ, Δt, NqP, iel, Val(NV), Val(NP))
    # P is discontinuous here, so element pressure DoFs are not shared.
    _add_local!(RP, local_nodes_P, Re, Val(false))
end

"""
    pressure_element_residual(vx, vy, P, P0, T, T0, el2n_v, el2nP,
                               geo_v, geo_P, phases, α, ηb, ξ, Δt, NqP, iel, Val(NV), Val(NP))

Gather element-local nodal values and integrate the Stokes pressure residual for element `iel`.

Returns `(local_nodes_P, Re)` ready for global scatter into `RP`.
"""
@inline function pressure_element_residual(vx, vy, P, P0, T, T0, el2n_v, el2nP,
        geo_v, geo_P, phases, α, ηb, ξ, Δt, NqP, iel, ::Val{NV}, ::Val{NP}) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = geo_v[iel]
    geo_P_el  = geo_P[iel]
    vxloc     = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc     = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc     = _gather_local(P,  local_nodes_P, Val(NP))
    P0loc     = _gather_local(P0, local_nodes_P, Val(NP))
    Tloc      = _gather_local(T,  local_nodes_P, Val(NP))
    T0loc     = _gather_local(T0, local_nodes_P, Val(NP))
    phase_loc = _gather_phase(phases, local_nodes_P, iel, Val(NP))
    Re = integrate_PH_pressure_residual(
        (vxloc, vyloc), P_loc, P0loc, Tloc, T0loc,
        geo_v_el, geo_P_el, phase_loc, α, ηb, ξ, Δt, NqP,
    )
    return local_nodes_P, Re
end
