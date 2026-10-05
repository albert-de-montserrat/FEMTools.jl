"""
    element_residual(...)

Gather element-local values and return the integrated thermal residual with its
global node indices.
"""
@inline function element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, ∂N∂ξ, source_ip, iel, ::Val{N}) where {N}
    nodes = local_nodes_of(el2n, iel, Val(N))
    Tloc = _gather_local(T, nodes, Val(N))
    geo_el = element_geometry(geo, iel, ∂N∂ξ)
    args = (
        Tloc,
        _gather_local(T0, nodes, Val(N)),
        geo_el,
        _gather_local(source, nodes, Val(N)),
        _gather_phase(phases, nodes, iel, Val(N)),
        k, Cp, ρ0, α, K,
        _gather_local(P, nodes, Val(N)),
        Δt, Tref, Nq, Val(N),
        _gather_history(source_ip, iel, quadrature_points_val(geo_el)),
    )
    return nodes, integrate_residual(args...)
end

"""
    element_jacobian(...)

Return element node indices, absolute Jacobian row sums, and absolute diagonal.
"""
@inline function element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, ∂N∂ξ, source_ip, iel, ::Val{N}) where {N}
    nodes = local_nodes_of(el2n, iel, Val(N))
    Tloc = _gather_local(T, nodes, Val(N))
    T0loc = _gather_local(T0, nodes, Val(N))
    sloc = _gather_local(source, nodes, Val(N))
    Ploc = _gather_local(P, nodes, Val(N))
    phase_loc = _gather_phase(phases, nodes, iel, Val(N))
    geo_el = element_geometry(geo, iel, ∂N∂ξ)
    Φloc = _gather_history(source_ip, iel, quadrature_points_val(geo_el))
    J = ForwardDiff.jacobian(Tloc) do u
        integrate_residual(
            u, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc,
            Δt, Tref, Nq, Val(N), Φloc,
        )
    end
    rowsums, diags = jacobian_rowsums_and_diagonal(J)
    return nodes, rowsums, diags
end

"""
    integrate_residual(...)

Integrate one element's transient, source, and diffusion contributions using
the nodal phase assignments and linearised density equation of state.

`sloc` is the nodal heat source. `Φloc`, when given, is an additional heat
source sampled at the element's quadrature points, such as shear heating.
"""
@inline function integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, ::Val{N}, Φloc = nothing) where {N}
    Re = zero(Tloc)
    # Compressibility β = 1/K: safe for K=Inf (β=0) and avoids NaN from
    # interp2ip_phase when quadratic shape functions are negative.
    β = map(inv, K)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        Tq = dot(Nv, Tloc)
        Pq = dot(Nv, Ploc)
        kq = interp2ip_phase(Nv, k, phase_loc)
        αq = interp2ip_phase(Nv, α, phase_loc)
        βq = interp2ip_phase(Nv, β, phase_loc)
        ρq = interp2ip_phase(Nv, ρ0, phase_loc) * (1 - αq * (Tq - Tref) + βq * Pq)
        Δt_ρCp = Δt / (ρq * interp2ip_phase(Nv, Cp, phase_loc))
        KTloc = kq * (∂N∂x * (∂N∂x' * Tloc))
        Φq = _history_at_ip(Φloc, q)
        Hq = isnothing(Φq) ? zero(Δt_ρCp) : Δt_ρCp * Φq
        # Consistent mass and source: the row-sum lumped mass ∫Nᵢ dΩ is
        # negative at the corners of quadratic tetrahedra, which makes the
        # transient operator indefinite.
        rate_q = dot(Nv, T0loc) - Tq + Δt_ρCp * dot(Nv, sloc) + Hq
        Re += SVector{N}(
            ntuple(Val(N)) do i
                rate_q * Nv[i] * dΩ - Δt_ρCp * KTloc[i] * dΩ
            end
        )
    end
    return Re
end

# Argument bundle shared by the atomic and colored thermal assemblers, in the
# order `element_residual` and `element_jacobian` consume it. `source_ip` is an
# optional `nq × nels` heat source at the quadrature points.
@inline diffusion_element_arguments(
    T, T0, source, el2n, geo, phases,
    k, Cp, ρ0, α, K, P, Δt, Tref, element, source_ip = nothing,
) =
    (
    T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref,
    shape_function_values(element), shape_function_gradients(element), source_ip,
)
