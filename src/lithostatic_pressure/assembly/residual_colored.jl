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
    return assemble_dr_matrices_colored!(
        R, ∂R∂P, PC, lp_element_residual, lp_element_jacobian,
        lithostatic_element_arguments(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, element),
        groups, Val(N), backend, workgroup; compute_jacobian,
    )
end
