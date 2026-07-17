"""
    assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, el_groups,
                                         element, phases, k, Cp, ρ0, α, K, P, Δt,
                                         source, Tref, backend, workgroup;
                                         compute_jacobian=false)

Graph-coloring alternative to `assemble_diffusion_matrices_atomix!`.

`el_groups` is a `Vector` of integer arrays (one per color) as produced by
`generate_element_groups`. Elements in the same group share no nodes, so the
scatter into `R`, `∂R∂T`, and `PC` is conflict-free and requires no atomics.
The outer loop over colors is sequential; within each color the kernel runs in
parallel without synchronisation barriers between colors.

All other arguments are identical to `assemble_diffusion_matrices_atomix!`,
except that `nels` is replaced by `el_groups`.
"""
function assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, el_groups,
                                               element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, backend, workgroup; compute_jacobian = false) where Te<:AbstractElement{nDim, N} where {nDim, N}
    Nq = shape_function_values(element)

    fill!(R, 0)
    for group in el_groups
        residual_colored_kernel!(backend, workgroup)(
            R, T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq, group, Val(N);
            ndrange = length(group),
        )
    end
    if compute_jacobian
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

function assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, el_groups,
                                               element::ReferenceElement{Te}, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, do_∂R∂T::Bool, backend, workgroup) where Te<:AbstractElement{nDim, N} where {nDim, N}
    Base.depwarn("passing do_∂R∂T as a positional Bool is deprecated; use compute_jacobian = $do_∂R∂T instead", :assemble_diffusion_matrices_colored!)
    return assemble_diffusion_matrices_colored!(
        R, ∂R∂T, PC, T, T0, el2n, geo, el_groups, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        backend, workgroup; compute_jacobian = do_∂R∂T,
    )
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
