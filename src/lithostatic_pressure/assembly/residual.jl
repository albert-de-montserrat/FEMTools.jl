"""
    lp_element_residual(...)

Gather element-local values and return the integrated lithostatic-pressure
residual with its global node indices.
"""
@inline function lp_element_residual(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, ::Val{N}) where N
    nodes = local_nodes_of(el2n, iel, Val(N))
    args = (
        _gather_local(P, nodes, Val(N)),
        _gather_local(T, nodes, Val(N)),
        geo[iel],
        _gather_phase(phases, nodes, iel, Val(N)),
        ρ0, α, K, Tref, g, Nq, Val(N),
    )
    return nodes, lp_integrate_residual(args...)
end

"""
    lp_element_jacobian(...)

Return element node indices, absolute Jacobian row sums, and absolute diagonal.
"""
@inline function lp_element_jacobian(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, ::Val{N}) where N
    nodes = local_nodes_of(el2n, iel, Val(N))
    Tloc = _gather_local(T, nodes, Val(N))
    Ploc = _gather_local(P, nodes, Val(N))
    phase_loc = _gather_phase(phases, nodes, iel, Val(N))
    J = ForwardDiff.jacobian(Ploc) do p
        lp_integrate_residual(p, Tloc, geo[iel], phase_loc, ρ0, α, K, Tref, g, Nq, Val(N))
    end
    rowsums, diags = jacobian_rowsums_and_diagonal(J)
    return nodes, rowsums, diags
end

"""
    lp_integrate_residual(...)

Integrate one element of `∫ (ρ(T, P) ∇Nᵢ⋅g - ∇Nᵢ⋅∇P) dΩ`, where
`ρ = ρ0 (1 - α(T - Tref) + P/K)`.
"""
@inline function lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, Tref, g, Nq, ::Val{N}) where N
    Re = zero(Ploc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        Tq = dot(Nv, Tloc)
        Pq = dot(Nv, Ploc)
        αq = interp2ip_phase(Nv, α, phase_loc)
        Kq = interp2ip_phase(Nv, K, phase_loc)
        ρq = interp2ip_phase(Nv, ρ0, phase_loc) * (1 - αq * (Tq - Tref) + Pq / Kq)
        Re += (ρq * (∂N∂x * SVector(g)) - ∂N∂x * (∂N∂x' * Ploc)) * dΩ
    end
    return Re
end
