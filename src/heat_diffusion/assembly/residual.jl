"""
    assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels,
                                        element, phases, k, Cp, ρ0, α, K, P, Δt,
                                        source, Tref, do_∂R∂T, backend, workgroup)

Assemble the transient heat-diffusion residual `R` using Atomix-backed atomic scatter.

`phases` is a nodal integer array whose entries select the phase index (1-based)
for each node. Per-node properties are interpolated to quadrature points by
weighting with shape functions. `k`, `Cp`, `ρ0`, `α`, and `K` are `NTuple`s of
per-phase intrinsic material properties. When `do_∂R∂T` is true, also assemble
row-sum Jacobian estimates into `∂R∂T` and the absolute diagonal into `PC`.
`Tref` is the reference temperature in the density equation of state.
"""
function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels, element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, do_∂R∂T, backend, workgroup) where Te<:AbstractElement{nDim, N} where {nDim, N}
    Nq = shape_function_values(element)

    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, Val(N); ndrange = nels)
    if do_∂R∂T
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂T, PC, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

# k, Cp, ρ0, α, K are NTuples of per-phase scalars — no @Const (not device arrays).
# phases is a nodal integer array — passed as @Const.

@kernel function residual_atomic_kernel!(R, @Const(T), @Const(T0), @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P), Δt, Tref, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
end

@kernel function jacobian_atomic_kernel!(∂R∂T, PC, @Const(T), @Const(T0), @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P), Δt, Tref, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂T[inod] += rowsums[i]
        Atomix.@atomic :monotonic PC[inod]   += diags[i]
    end
end

"""
    assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, el_groups,
                                         element, phases, k, Cp, ρ0, α, K, P, Δt,
                                         source, Tref, do_∂R∂T, backend, workgroup)

Graph-coloring alternative to `assemble_diffusion_matrices_atomix!`.

`el_groups` is a `Vector` of integer arrays (one per color) as produced by
`build_element_groups`. Elements in the same group share no nodes, so the
scatter into `R`, `∂R∂T`, and `PC` is conflict-free and requires no atomics.
The outer loop over colors is sequential; within each color the kernel runs in
parallel without synchronisation barriers between colors.

All other arguments are identical to `assemble_diffusion_matrices_atomix!`,
except that `nels` is replaced by `el_groups`.
"""
function assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, el_groups,
                                               element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, do_∂R∂T, backend, workgroup) where Te<:AbstractElement{nDim, N} where {nDim, N}
    Nq = shape_function_values(element)

    fill!(R, 0)
    for group in el_groups
        residual_colored_kernel!(backend, workgroup)(
            R, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, group, Val(N);
            ndrange = length(group),
        )
    end
    if do_∂R∂T
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        for group in el_groups
            jacobian_colored_kernel!(backend, workgroup)(
                ∂R∂T, PC, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, group, Val(N);
                ndrange = length(group),
            )
        end
    end
    KA.synchronize(backend)
end

@kernel function residual_colored_kernel!(R, @Const(T), @Const(T0), @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P), Δt, Tref, Nq, @Const(group), ::Val{N}) where N
    i = @index(Global)
    iel = group[i]
    local_nodes, Re = element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))
    for (j, inod) in enumerate(local_nodes)
        R[inod] += Re[j]
    end
end

@kernel function jacobian_colored_kernel!(∂R∂T, PC, @Const(T), @Const(T0), @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P), Δt, Tref, Nq, @Const(group), ::Val{N}) where N
    i = @index(Global)
    iel = group[i]
    local_nodes, rowsums, diags = element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))
    for (j, inod) in enumerate(local_nodes)
        ∂R∂T[inod] += rowsums[j]
        PC[inod]   += diags[j]
    end
end

"""
    element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))

Gather element-local nodal values and integrate the multi-phase heat-diffusion residual.

`phases` is the nodal phase-index array. `phase_loc` is gathered element-locally
as an `SVector{N, Int}` and forwarded to `integrate_residual`. Returns
`(local_nodes, Re)` ready for global scatter.
"""
@inline function element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el  = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]],         Val(N)))
    T0loc     = SVector{N}(ntuple(i -> T0[local_nodes[i]],        Val(N)))
    sloc      = SVector{N}(ntuple(i -> source[local_nodes[i]],    Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]],         Val(N)))
    phase_loc = SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
    Re = integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, Val(N))
    return local_nodes, Re
end

"""
    element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))

Compute per-element Jacobian diagnostics for the multi-phase heat-diffusion residual
via ForwardDiff. Returns `(local_nodes, rowsums, diags)`.
"""
@inline function element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el  = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]],           Val(N)))
    T0loc     = SVector{N}(ntuple(i -> T0[local_nodes[i]],          Val(N)))
    sloc      = SVector{N}(ntuple(i -> source[local_nodes[i]],      Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]],           Val(N)))
    phase_loc = SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
    ∂Re∂Te = ForwardDiff.jacobian(
        Tloc -> integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, Val(N)),
        Tloc,
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂Te[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂Te[i, i]), Val(N)))
    return local_nodes, rowsums, diags
end


"""
    integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, Val(N))

Integrate the element residual for a transient multi-phase heat-diffusion equation.

`phase_loc` is an `SVector{N, Int}` of per-node phase indices. At each quadrature
point, material properties are interpolated from the nodal phase assignments using
the shape functions, then combined through the linearised EOS:

    ρ_q = ρ0_q * (1 - α_q*(T_q - Tref) + P_q/K_q)
"""
@inline function integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, ::Val{N}) where N
    Re = zero(Tloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv  = Nq[q]
        Tq  = dot(Nv, Tloc)
        P   = dot(Nv, Ploc)
        tmp = ∂N∂x' * Tloc
        kq  = sum(Nv[p] *  k[phase_loc[p]] for p in eachindex(phase_loc))
        αq  = sum(Nv[p] *  α[phase_loc[p]] for p in eachindex(phase_loc))
        Kq  = sum(Nv[p] *  K[phase_loc[p]] for p in eachindex(phase_loc))
        ρ0q = sum(Nv[p] * ρ0[phase_loc[p]] for p in eachindex(phase_loc))
        Cpq = sum(Nv[p] * Cp[phase_loc[p]] for p in eachindex(phase_loc))
        ρq  = ρ0q * (1 - αq * (Tq - Tref) + P / Kq)
        KTloc = kq * (∂N∂x * tmp)
        Re   += SVector{N}(ntuple(
            i -> (-Tloc[i] + T0loc[i] + Δt / (ρq * Cpq) * sloc[i]) * Nv[i] * dΩ -
                 Δt / (ρq * Cpq) * KTloc[i] * dΩ,
            Val(N),
        ))
    end
    return Re
end
