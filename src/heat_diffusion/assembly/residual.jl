"""
    assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels,
                                        element, phases, k, Cp, ρ0, α, K, P, Δt,
                                        source, do_∂R∂T, backend, workgroup)

Assemble the transient heat-diffusion residual `R` using Atomix-backed atomic scatter.

`phases` is a nodal integer array whose entries select the phase index (1-based)
for each node. Per-node properties are interpolated to quadrature points by
weighting with shape functions. `k`, `Cp`, `ρ0`, `α`, and `K` are `NTuple`s of
per-phase intrinsic material properties. When `do_∂R∂T` is true, also assemble
row-sum Jacobian estimates into `∂R∂T` and the absolute diagonal into `PC`.
"""
function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels, element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, do_∂R∂T, backend, workgroup) where Te<:AbstractElement{2, N} where N
    Nq = shape_function_values(element)

    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, Val(N); ndrange = nels)
    if do_∂R∂T
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂T, PC, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

# k, Cp, ρ0, α, K are NTuples of per-phase scalars — no @Const (not device arrays).
# phases is a nodal integer array — passed as @Const.

@kernel function residual_atomic_kernel!(R, @Const(T), @Const(T0), @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P), Δt, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
end

@kernel function jacobian_atomic_kernel!(∂R∂T, PC, @Const(T), @Const(T0), @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P), Δt, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂T[inod] += rowsums[i]
        Atomix.@atomic :monotonic PC[inod]   += diags[i]
    end
end

"""
    element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, iel, Val(N))

Gather element-local nodal values and integrate the multi-phase heat-diffusion residual.

`phases` is the nodal phase-index array. `phase_loc` is gathered element-locally
as an `SVector{N, Int}` and forwarded to `integrate_residual`. Returns
`(local_nodes, Re)` ready for global scatter.
"""
@inline function element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el  = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]],         Val(N)))
    T0loc     = SVector{N}(ntuple(i -> T0[local_nodes[i]],        Val(N)))
    sloc      = SVector{N}(ntuple(i -> source[local_nodes[i]],    Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]],         Val(N)))
    phase_loc = SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
    Re = integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Nq, Val(N))
    return local_nodes, Re
end

"""
    element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, iel, Val(N))

Compute per-element Jacobian diagnostics for the multi-phase heat-diffusion residual
via ForwardDiff. Returns `(local_nodes, rowsums, diags)`.
"""
@inline function element_jacobian(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el  = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]],           Val(N)))
    T0loc     = SVector{N}(ntuple(i -> T0[local_nodes[i]],          Val(N)))
    sloc      = SVector{N}(ntuple(i -> source[local_nodes[i]],      Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]],           Val(N)))
    phase_loc = SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
    ∂Re∂Te = ForwardDiff.jacobian(
        Tloc -> integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Nq, Val(N)),
        Tloc,
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂Te[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂Te[i, i]), Val(N)))
    return local_nodes, rowsums, diags
end


"""
    integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Nq, Val(N))

Integrate the element residual for a transient multi-phase heat-diffusion equation.

`phase_loc` is an `SVector{N, Int}` of per-node phase indices. At each quadrature
point, material properties are interpolated from the nodal phase assignments using
the shape functions, then combined through the linearised EOS:

    ρ_q = ρ0_q * (1 - α_q*(T_q - Tref) + P_q/K_q),   Tref = 273 K
"""
@inline function integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Nq, ::Val{N}) where N
    Re   = zero(Tloc)
    Tref = 273  # kelvin
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