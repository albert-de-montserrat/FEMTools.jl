"""
    assemble_lithostatic_pressure_matrices_colored!(...; compute_jacobian=false)

Assemble the lithostatic-pressure residual and optional Jacobian diagnostics
by sequentially launching conflict-free groups from `generate_element_groups`.
Set `compute_jacobian=true` to also fill the absolute Jacobian row sums `∂R∂P`
and absolute diagonal `PC`.
"""
function assemble_lithostatic_pressure_matrices_colored!(R, ∂R∂P, PC, T, P,
        el2n, geo, groups, element::ReferenceElement{<:AbstractElement{nDim, N}},
        phases, ρ0, α, K, Tref, g, backend, workgroup;
        compute_jacobian = false) where {nDim, N}
    Nq = shape_function_values(element)
    args = (T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq)
    fill!(R, 0)
    for group in groups
        lp_residual_colored_kernel!(backend, workgroup)(
            R, args..., group, Val(N); ndrange = length(group),
        )
    end
    if compute_jacobian
        fill!(∂R∂P, 0)
        fill!(PC, 0)
        for group in groups
            lp_jacobian_colored_kernel!(backend, workgroup)(
                ∂R∂P, PC, args..., group, Val(N); ndrange = length(group),
            )
        end
    end
    KA.synchronize(backend)
end

function assemble_lithostatic_pressure_matrices_colored!(R, ∂R∂P, PC, T, P,
        el2n, geo, groups, element, phases, ρ0, α, K, Tref, g,
        compute_jacobian::Bool, backend, workgroup)
    Base.depwarn("passing the Jacobian flag positionally is deprecated; use compute_jacobian=$compute_jacobian", :assemble_lithostatic_pressure_matrices_colored!)
    return assemble_lithostatic_pressure_matrices_colored!(
        R, ∂R∂P, PC, T, P, el2n, geo, groups, element, phases, ρ0, α, K, Tref, g,
        backend, workgroup; compute_jacobian,
    )
end

@kernel function lp_residual_colored_kernel!(R, @Const(T), @Const(P),
        @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq,
        @Const(group), ::Val{N}) where N
    i = @index(Global)
    nodes, Re = lp_element_residual(
        T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, group[i], Val(N),
    )
    _add_local!(R, nodes, Re, Val(false))
end

@kernel function lp_jacobian_colored_kernel!(∂R∂P, PC, @Const(T), @Const(P),
        @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq,
        @Const(group), ::Val{N}) where N
    i = @index(Global)
    nodes, rowsums, diags = lp_element_jacobian(
        T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, group[i], Val(N),
    )
    _add_local_pair!(∂R∂P, PC, nodes, rowsums, diags, Val(false))
end
