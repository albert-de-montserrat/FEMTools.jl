"""
    element_residual(...)

Gather element-local values and return the integrated thermal residual with its
global node indices.
"""
@inline function element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, ::Val{N}) where N
    nodes = local_nodes_of(el2n, iel, Val(N))
    Tloc = _gather_local(T, nodes, Val(N))
    args = (
        Tloc,
        _gather_local(T0, nodes, Val(N)),
        geo[iel],
        _gather_local(source, nodes, Val(N)),
        _gather_phase(phases, nodes, iel, Val(N)),
        k, Cp, ρ0, α, K,
        _gather_local(P, nodes, Val(N)),
        Δt, Tref, Nq, Val(N),
    )
    return nodes, integrate_residual(args...)
end

"""
    element_jacobian(...)

Return element node indices, absolute Jacobian row sums, and absolute diagonal.
"""
@inline function element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, ::Val{N}) where N
    nodes = local_nodes_of(el2n, iel, Val(N))
    Tloc = _gather_local(T, nodes, Val(N))
    T0loc = _gather_local(T0, nodes, Val(N))
    sloc = _gather_local(source, nodes, Val(N))
    Ploc = _gather_local(P, nodes, Val(N))
    phase_loc = _gather_phase(phases, nodes, iel, Val(N))
    J = ForwardDiff.jacobian(Tloc) do u
        integrate_residual(
            u, T0loc, geo[iel], sloc, phase_loc, k, Cp, ρ0, α, K, Ploc,
            Δt, Tref, Nq, Val(N),
        )
    end
    rowsums, diags = jacobian_rowsums_and_diagonal(J)
    return nodes, rowsums, diags
end

"""
    integrate_residual(...)

Integrate one element's transient, source, and diffusion contributions using
the nodal phase assignments and linearised density equation of state.
"""
@inline function integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, ::Val{N}) where N
    Re = zero(Tloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        Tq = dot(Nv, Tloc)
        Pq = dot(Nv, Ploc)
        kq = interp2ip_phase(Nv, k, phase_loc)
        αq = interp2ip_phase(Nv, α, phase_loc)
        Kq = interp2ip_phase(Nv, K, phase_loc)
        ρq = interp2ip_phase(Nv, ρ0, phase_loc) * (1 - αq * (Tq - Tref) + Pq / Kq)
        Δt_ρCp = Δt / (ρq * interp2ip_phase(Nv, Cp, phase_loc))
        KTloc = kq * (∂N∂x * (∂N∂x' * Tloc))
        Re += SVector{N}(ntuple(Val(N)) do i
            (-Tloc[i] + T0loc[i] + Δt_ρCp * sloc[i]) * Nv[i] * dΩ -
            Δt_ρCp * KTloc[i] * dΩ
        end)
    end
    return Re
end

# Argument bundle shared by the atomic and colored thermal assemblers, in the
# order `element_residual` and `element_jacobian` consume it.
@inline diffusion_element_arguments(T, T0, source, el2n, geo, phases,
        k, Cp, ρ0, α, K, P, Δt, Tref, element) =
    (T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref,
     shape_function_values(element))
