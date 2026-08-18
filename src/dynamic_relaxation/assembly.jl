@inline _element_index(::Nothing, i) = i
@inline _element_index(group, i) = group[i]

"""
    dr_residual_kernel!(R, element_residual, args, group, Val(atomic), Val(N))

Scatter element residuals into the global residual `R`.

`element_residual(args..., iel, Val(N))` returns the element's global node
indices and its residual contributions. `group` selects the elements to visit:
`nothing` maps the global index directly onto the element index, and a vector
of element indices restricts the launch to one conflict-free color group.
`atomic` chooses between atomic and plain accumulation.
"""
@kernel function dr_residual_kernel!(R, element_residual, args, group, ::Val{atomic}, ::Val{N}) where {atomic, N}
    i = @index(Global)
    nodes, Re = element_residual(args..., _element_index(group, i), Val(N))
    _add_local!(R, nodes, Re, Val(atomic))
end

"""
    dr_jacobian_kernel!(∂R∂u, PC, element_jacobian, args, group, Val(atomic), Val(N))

Scatter element Jacobian diagnostics into the row-sum estimate `∂R∂u` and the
diagonal preconditioner `PC`. `group` and `atomic` behave as for
[`dr_residual_kernel!`](@ref).
"""
@kernel function dr_jacobian_kernel!(∂R∂u, PC, element_jacobian, args, group, ::Val{atomic}, ::Val{N}) where {atomic, N}
    i = @index(Global)
    nodes, rowsums, diags = element_jacobian(args..., _element_index(group, i), Val(N))
    _add_local_pair!(∂R∂u, PC, nodes, rowsums, diags, Val(atomic))
end

"""
    assemble_dr_matrices_atomix!(R, ∂R∂u, PC, element_residual, element_jacobian,
                                 args, nels, Val(N), backend, workgroup;
                                 compute_jacobian=false)

Assemble a dynamic-relaxation residual over `nels` elements with atomic
scatter, optionally also filling the Jacobian row sums `∂R∂u` and diagonal
preconditioner `PC`.
"""
function assemble_dr_matrices_atomix!(
        R, ∂R∂u, PC, element_residual, element_jacobian, args, nels, ::Val{N},
        backend, workgroup; compute_jacobian = false,
    ) where {N}
    fill!(R, 0)
    dr_residual_kernel!(backend, workgroup)(
        R, element_residual, args, nothing, Val(true), Val(N); ndrange = nels,
    )
    if compute_jacobian
        fill!(∂R∂u, 0)
        fill!(PC, 0)
        dr_jacobian_kernel!(backend, workgroup)(
            ∂R∂u, PC, element_jacobian, args, nothing, Val(true), Val(N); ndrange = nels,
        )
    end
    KA.synchronize(backend)
    return nothing
end

"""
    assemble_dr_matrices_colored!(R, ∂R∂u, PC, element_residual, element_jacobian,
                                  args, groups, Val(N), backend, workgroup;
                                  compute_jacobian=false)

Assemble a dynamic-relaxation residual by launching the conflict-free element
color groups from `generate_element_groups` in sequence. Elements within a
group share no nodes, so accumulation needs no atomics.
"""
function assemble_dr_matrices_colored!(
        R, ∂R∂u, PC, element_residual, element_jacobian, args, groups, ::Val{N},
        backend, workgroup; compute_jacobian = false,
    ) where {N}
    fill!(R, 0)
    for group in groups
        dr_residual_kernel!(backend, workgroup)(
            R, element_residual, args, group, Val(false), Val(N); ndrange = length(group),
        )
    end
    if compute_jacobian
        fill!(∂R∂u, 0)
        fill!(PC, 0)
        for group in groups
            dr_jacobian_kernel!(backend, workgroup)(
                ∂R∂u, PC, element_jacobian, args, group, Val(false), Val(N); ndrange = length(group),
            )
        end
    end
    KA.synchronize(backend)
    return nothing
end
