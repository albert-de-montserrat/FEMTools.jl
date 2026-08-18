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
    return assemble_dr_matrices_colored!(
        R, ∂R∂T, PC, element_residual, element_jacobian,
        diffusion_element_arguments(T, T0, source, el2n, geo, phases,
                                    k, Cp, ρ0, α, K, P, Δt, Tref, element),
        groups, Val(N), backend, workgroup; compute_jacobian,
    )
end
