"""
    assemble_lithostatic_pressure_matrices_atomix!(...; compute_jacobian=false)

Assemble the lithostatic-pressure residual and optional Jacobian diagnostics
with atomic scatter. Set `compute_jacobian=true` to also fill the absolute
Jacobian row sums `∂R∂P` and absolute diagonal `PC`.
"""
function assemble_lithostatic_pressure_matrices_atomix!(R, ∂R∂P, PC, T, P,
        el2n, geo, nels, element::ReferenceElement{<:AbstractElement{nDim, N}},
        phases, ρ0, α, K, Tref, g, backend, workgroup;
        compute_jacobian = false) where {nDim, N}
    Nq = shape_function_values(element)
    args = (T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, Val(N))
    fill!(R, 0)
    lp_residual_atomic_kernel!(backend, workgroup)(R, args...; ndrange = nels)
    if compute_jacobian
        fill!(∂R∂P, 0)
        fill!(PC, 0)
        lp_jacobian_atomic_kernel!(backend, workgroup)(∂R∂P, PC, args...; ndrange = nels)
    end
    KA.synchronize(backend)
end

function assemble_lithostatic_pressure_matrices_atomix!(R, ∂R∂P, PC, T, P,
        el2n, geo, nels, element, phases, ρ0, α, K, Tref, g,
        compute_jacobian::Bool, backend, workgroup)
    Base.depwarn("passing the Jacobian flag positionally is deprecated; use compute_jacobian=$compute_jacobian", :assemble_lithostatic_pressure_matrices_atomix!)
    return assemble_lithostatic_pressure_matrices_atomix!(
        R, ∂R∂P, PC, T, P, el2n, geo, nels, element, phases, ρ0, α, K, Tref, g,
        backend, workgroup; compute_jacobian,
    )
end

@kernel function lp_residual_atomic_kernel!(R, @Const(T), @Const(P),
        @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq,
        ::Val{N}) where N
    iel = @index(Global)
    nodes, Re = lp_element_residual(
        T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N),
    )
    _add_local!(R, nodes, Re, Val(true))
end

@kernel function lp_jacobian_atomic_kernel!(∂R∂P, PC, @Const(T), @Const(P),
        @Const(el2n), @Const(geo), @Const(phases), ρ0, α, K, Tref, g, Nq,
        ::Val{N}) where N
    iel = @index(Global)
    nodes, rowsums, diags = lp_element_jacobian(
        T, P, el2n, geo, phases, ρ0, α, K, Tref, g, Nq, iel, Val(N),
    )
    _add_local_pair!(∂R∂P, PC, nodes, rowsums, diags, Val(true))
end
