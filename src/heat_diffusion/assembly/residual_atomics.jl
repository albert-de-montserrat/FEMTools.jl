"""
    assemble_diffusion_matrices_atomix!(...; compute_jacobian=false)

Assemble the thermal residual and optional Jacobian diagnostics with atomic
scatter. Material tuples are interpolated from the nodal phase indices. Set
`compute_jacobian=true` to also fill the absolute Jacobian row sums `∂R∂T` and
absolute diagonal `PC`.
"""
function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels,
        element::ReferenceElement{<:AbstractElement{nDim, N}}, phases,
        k, Cp, ρ0, α, K, P, Δt, source, Tref, backend, workgroup;
        compute_jacobian = false) where {nDim, N}
    Nq = shape_function_values(element)
    args = (T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, Val(N))
    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, args...; ndrange = nels)
    if compute_jacobian
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂T, PC, args...; ndrange = nels)
    end
    KA.synchronize(backend)
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, T0, el2n, geo, nels,
        element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        compute_jacobian::Bool, backend, workgroup)
    Base.depwarn("passing the Jacobian flag positionally is deprecated; use compute_jacobian=$compute_jacobian", :assemble_diffusion_matrices_atomix!)
    return assemble_diffusion_matrices_atomix!(
        R, ∂R∂T, PC, T, T0, el2n, geo, nels, element, phases,
        k, Cp, ρ0, α, K, P, Δt, source, Tref, backend, workgroup;
        compute_jacobian,
    )
end

@kernel function residual_atomic_kernel!(R, @Const(T), @Const(T0), @Const(source),
        @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P),
        Δt, Tref, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = element_residual(
        T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq,
        iel, Val(N),
    )
    for (i, node) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[node] += Re[i]
    end
end

@kernel function jacobian_atomic_kernel!(∂R∂T, PC, @Const(T), @Const(T0),
        @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0,
        α, K, @Const(P), Δt, Tref, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = element_jacobian(
        T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq,
        iel, Val(N),
    )
    for (i, node) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂T[node] += rowsums[i]
        Atomix.@atomic :monotonic PC[node] += diags[i]
    end
end
