"""
    assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels,
                                        element, phases, k, Cp, ρ0, α, K, P, Δt,
                                        source, Tref, backend, workgroup;
                                        compute_jacobian=false)

Assemble the transient heat-diffusion residual `R` using Atomix-backed atomic scatter.

`phases` is a nodal integer array whose entries select the phase index (1-based)
for each node. Per-node properties are interpolated to quadrature points by
weighting with shape functions. `k`, `Cp`, `ρ0`, `α`, and `K` are `NTuple`s of
per-phase intrinsic material properties. When `compute_jacobian` is true, also
assemble row-sum Jacobian estimates into `∂R∂T` and the absolute diagonal into `PC`.
`Tref` is the reference temperature in the density equation of state.
"""
function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels, element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, backend, workgroup; compute_jacobian = false) where Te<:AbstractElement{nDim, N} where {nDim, N}
    Nq = shape_function_values(element)

    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, Val(N); ndrange = nels)
    if compute_jacobian
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂T, PC, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels, element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, do_∂R∂T::Bool, backend, workgroup) where Te<:AbstractElement{nDim, N} where {nDim, N}
    Base.depwarn("passing do_∂R∂T as a positional Bool is deprecated; use compute_jacobian = $do_∂R∂T instead", :assemble_diffusion_matrices_atomix!)
    return assemble_diffusion_matrices_atomix!(
        R, ∂R∂T, PC, T, T0, el2n, geo, nels, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        backend, workgroup; compute_jacobian = do_∂R∂T,
    )
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
    element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, Val(N))

Gather element-local nodal values and integrate the multi-phase heat-diffusion residual.

`phases` is the nodal phase-index array. For multiple materials, `phase_loc` is
gathered element-locally as an `SVector{N, Int}` and forwarded to
`integrate_residual`; single-material tuples skip that gather. Returns
`(local_nodes, Re)` ready for global scatter.
"""
@inline function element_residual(T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el  = geo[iel]
    Tloc      = SVector{N}(ntuple(i -> T[local_nodes[i]],         Val(N)))
    T0loc     = SVector{N}(ntuple(i -> T0[local_nodes[i]],        Val(N)))
    sloc      = SVector{N}(ntuple(i -> source[local_nodes[i]],    Val(N)))
    Ploc      = SVector{N}(ntuple(i -> P[local_nodes[i]],         Val(N)))
    phase_loc = local_phase_indices(phases, local_nodes, k, Val(N))
    Re = integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, Val(N))
    return local_nodes, Re
end

@inline local_phase_indices(::Any, ::Any, ::NTuple{1}, ::Val{N}) where N = nothing

@inline function local_phase_indices(phases, local_nodes, k, ::Val{N}) where N
    return SVector{N}(ntuple(i -> Int(phases[local_nodes[i]]), Val(N)))
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
    phase_loc = local_phase_indices(phases, local_nodes, k, Val(N))
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

For multiple materials, `phase_loc` is an `SVector{N, Int}` of per-node phase
indices and properties are interpolated using the shape functions. A
single-material tuple bypasses the phase data. The properties are combined
through the linearised EOS:

    ρ_q = ρ0_q * (1 - α_q*(T_q - Tref) + P_q/K_q)
"""
@inline function integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, Tref, Nq, ::Val{N}) where N
    Re = zero(Tloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv  = Nq[q]
        Tq  = dot(Nv, Tloc)
        Pq  = dot(Nv, Ploc)
        ∇T  = ∂N∂x' * Tloc
        kq, Cpq, ρ0q, αq, Kq = heat_properties(Nv, phase_loc, k, Cp, ρ0, α, K)
        ρq  = ρ0q * (1 - αq * (Tq - Tref) + Pq / Kq)
        source_scale    = Δt / (ρq * Cpq)
        diffusion_scale = source_scale * kq
        Re   += SVector{N}(ntuple(
            i -> ((-Tloc[i] + T0loc[i] + source_scale * sloc[i]) * Nv[i] -
                  diffusion_scale * gradient_projection(∂N∂x, ∇T, i)) * dΩ,
            Val(N),
        ))
    end
    return Re
end


@inline function gradient_projection(∂N∂x::StaticMatrix{N, D}, ∇T, i) where {N, D}
    return dot(SVector{D}(ntuple(d -> ∂N∂x[i, d], Val(D))), ∇T)
end


@inline function heat_properties(Nv, phase_loc, k::NTuple{1}, Cp::NTuple{1}, ρ0::NTuple{1}, α::NTuple{1}, K::NTuple{1})
    return k[1], Cp[1], ρ0[1], α[1], K[1]
end

@inline function heat_properties(Nv, phase_loc, k, Cp, ρ0, α, K)
    return (
        interp2ip_phase(Nv, k, phase_loc),
        interp2ip_phase(Nv, Cp, phase_loc),
        interp2ip_phase(Nv, ρ0, phase_loc),
        interp2ip_phase(Nv, α, phase_loc),
        interp2ip_phase(Nv, K, phase_loc),
    )
end
