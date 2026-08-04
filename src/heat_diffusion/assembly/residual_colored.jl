"""
    assemble_diffusion_matrices_colored!(...; compute_jacobian=false)

Assemble the thermal residual and optional Jacobian diagnostics by sequentially
launching conflict-free element color groups from `generate_element_groups`.
Set `compute_jacobian=true` to also fill the absolute Jacobian row sums `∂R∂T`
and absolute diagonal `PC`.
"""
function assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, groups,
        element::ReferenceElement{<:AbstractElement{nDim, N}}, phases,
        k, Cp, ρ0, α, K, P, Δt, source, Tref, backend, workgroup;
        compute_jacobian = false) where {nDim, N}
    Nq = shape_function_values(element)
    args = (T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq)
    fill!(R, 0)
    for group in groups
        residual_colored_kernel!(backend, workgroup)(
            R, args..., group, Val(N); ndrange = length(group),
        )
    end
    if compute_jacobian
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        for group in groups
            jacobian_colored_kernel!(backend, workgroup)(
                ∂R∂T, PC, args..., group, Val(N); ndrange = length(group),
            )
        end
    end
    KA.synchronize(backend)
end

function assemble_diffusion_matrices_colored!(R, ∂R∂T, PC, T, T0, el2n, geo, groups,
        element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        compute_jacobian::Bool, backend, workgroup)
    Base.depwarn("passing the Jacobian flag positionally is deprecated; use compute_jacobian=$compute_jacobian", :assemble_diffusion_matrices_colored!)
    return assemble_diffusion_matrices_colored!(
        R, ∂R∂T, PC, T, T0, el2n, geo, groups, element, phases,
        k, Cp, ρ0, α, K, P, Δt, source, Tref, backend, workgroup;
        compute_jacobian,
    )
end

@kernel function residual_colored_kernel!(R, @Const(T), @Const(T0), @Const(source),
        @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0, α, K, @Const(P),
        Δt, Tref, Nq, @Const(group), ::Val{N}) where N
    i = @index(Global)
    nodes, Re = element_residual(
        T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq,
        group[i], Val(N),
    )
    _add_local!(R, nodes, Re, Val(false))
end

@kernel function jacobian_colored_kernel!(∂R∂T, PC, @Const(T), @Const(T0),
        @Const(source), @Const(el2n), @Const(geo), @Const(phases), k, Cp, ρ0,
        α, K, @Const(P), Δt, Tref, Nq, @Const(group), ::Val{N}) where N
    i = @index(Global)
    nodes, rowsums, diags = element_jacobian(
        T, T0, source, el2n, geo, phases, k, Cp, ρ0, α, K, P, Δt, Tref, Nq,
        group[i], Val(N),
    )
    _add_local_pair!(∂R∂T, PC, nodes, rowsums, diags, Val(false))
end
