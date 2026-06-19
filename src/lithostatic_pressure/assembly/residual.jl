"""
    assemble_lithostatic_pressure_matrices_atomix!(R, ∂R∂P, PC, T, P, el2n, geo, nels,
                                                   element, phases, ρ0, α, K, Tref, g,
                                                   do_∂R∂P, backend, workgroup)

Assemble the lithostatic-pressure residual `R` and (optionally) Jacobian
diagnostics `∂R∂P`, `PC` using Atomix-backed atomic scatter.

The weak form solved is `∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ`, i.e. the stiffness
term is `∇·∇P` with no conductivity prefactor. Density is computed from the
linearised EOS `ρ = ρ0 (1 - α(T-Tref) + P/K)`.
`Tref` is the reference temperature in that EOS and `g` is the body-force
vector in physical coordinates.
"""
function assemble_lithostatic_pressure_matrices_atomix!(
        R, ∂R∂P, PC, T, P, el2n, geo, nels,
        element::ReferenceElement{Te}, phases, ρ0, α, K, Tref, g,
        do_∂R∂P, backend, workgroup) where Te <: AbstractElement{nDim, N} where {nDim, N}
    Nq = shape_function_values(element)

    fill!(R, 0)
    lp_residual_atomic_kernel!(backend, workgroup)(
        R, T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, Val(N); ndrange = nels)
    if do_∂R∂P
        fill!(∂R∂P, 0)
        fill!(PC, 0)
        lp_jacobian_atomic_kernel!(backend, workgroup)(
            ∂R∂P, PC, T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

@kernel function lp_residual_atomic_kernel!(R, @Const(T), @Const(P), @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = lp_element_residual(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
end

@kernel function lp_jacobian_atomic_kernel!(∂R∂P, PC, @Const(T), @Const(P), @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = lp_element_jacobian(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂P[inod] += rowsums[i]
        Atomix.@atomic :monotonic PC[inod]    += diags[i]
    end
end

"""
    assemble_lithostatic_pressure_matrices_colored!(R, ∂R∂P, PC, T, P, el2n, geo,
                                                    el_groups, element, phases, ρ0, α, K, Tref, g,
                                                    do_∂R∂P, backend, workgroup)

Graph-coloring alternative to `assemble_lithostatic_pressure_matrices_atomix!`.
Elements in the same color group share no nodes, so no atomics are needed.
All physical arguments, including `Tref` and `g`, match the Atomix assembler.
"""
function assemble_lithostatic_pressure_matrices_colored!(
        R, ∂R∂P, PC, T, P, el2n, geo, el_groups,
        element::ReferenceElement{Te}, phases, ρ0, α, K, Tref, g,
        do_∂R∂P, backend, workgroup) where Te <: AbstractElement{nDim, N} where {nDim, N}
    Nq = shape_function_values(element)

    fill!(R, 0)
    for group in el_groups
        lp_residual_colored_kernel!(backend, workgroup)(
            R, T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, group, Val(N);
            ndrange = length(group))
    end
    if do_∂R∂P
        fill!(∂R∂P, 0)
        fill!(PC, 0)
        for group in el_groups
            lp_jacobian_colored_kernel!(backend, workgroup)(
                ∂R∂P, PC, T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, group, Val(N);
                ndrange = length(group))
        end
    end
    KA.synchronize(backend)
end

@kernel function lp_residual_colored_kernel!(R, @Const(T), @Const(P), @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq, @Const(group), ::Val{N}) where N
    i = @index(Global)
    iel = group[i]
    local_nodes, Re = lp_element_residual(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N))
    for (j, inod) in enumerate(local_nodes)
        R[inod] += Re[j]
    end
end

@kernel function lp_jacobian_colored_kernel!(∂R∂P, PC, @Const(T), @Const(P), @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq, @Const(group), ::Val{N}) where N
    i = @index(Global)
    iel = group[i]
    local_nodes, rowsums, diags = lp_element_jacobian(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N))
    for (j, inod) in enumerate(local_nodes)
        ∂R∂P[inod] += rowsums[j]
        PC[inod]    += diags[j]
    end
end

"""
    lp_element_residual(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N))

Gather element-local nodal values and integrate the lithostatic-pressure residual.
Returns `(local_nodes, Re)` ready for global scatter.
"""
@inline function lp_element_residual(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el    = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]], Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]], Val(N)))
    phase_loc = SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
    Re = lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, Tref, g, Nq, Val(N))
    return local_nodes, Re
end

"""
    lp_element_jacobian(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N))

Compute per-element Jacobian diagnostics for the lithostatic-pressure residual
via ForwardDiff. Returns `(local_nodes, rowsums, diags)`.
"""
@inline function lp_element_jacobian(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el    = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]], Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]], Val(N)))
    phase_loc = SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
    ∂Re∂Pe = ForwardDiff.jacobian(
        Ploc -> lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, Tref, g, Nq, Val(N)),
        Ploc,
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂Pe[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂Pe[i, i]), Val(N)))
    return local_nodes, rowsums, diags
end

"""
    lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, Tref, g, Nq, Val(N))

Integrate the element residual for the lithostatic pressure equation.

Weak form: `∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ`

Per node `i`: `Rᵢ = ∫ (∇Nᵢ·∇P − ρ(T) ∇Nᵢ·g) dΩ`

Density uses the linearised EOS `ρ = ρ0 (1 − α(T−Tref) + P/K)`.
`g` is the body-force vector in physical coordinates.
"""
@inline function lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, Tref, g, Nq, ::Val{N}) where N
    Re = zero(Ploc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv  = Nq[q]
        Tq  = dot(Nv, Tloc)
        Pq  = dot(Nv, Ploc)
        αq  = interp2ip_phase(Nv,  α, phase_loc)
        Kq  = interp2ip_phase(Nv,  K, phase_loc)
        ρ0q = interp2ip_phase(Nv, ρ0, phase_loc)
        ρq  = ρ0q * (1 - αq * (Tq - Tref) + Pq / Kq)
        ∇P_N = ∂N∂x * (∂N∂x' * Ploc)   # N-vector: ∇Nᵢ · ∇P  (stiffness)
        ρg_N = ρq * (∂N∂x * g)         # N-vector: ρ ∇Nᵢ · g  (body force)
        Re  += (ρg_N - ∇P_N) * dΩ      # sign: f − K·P, consistent with thermal DR convention
    end
    return Re
end
