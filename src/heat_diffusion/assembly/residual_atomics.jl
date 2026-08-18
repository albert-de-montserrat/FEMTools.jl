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
    return assemble_dr_matrices_atomix!(
        R, ∂R∂T, PC, element_residual, element_jacobian,
        diffusion_element_arguments(T, T0, source, el2n, geo, phases,
                                    k, Cp, ρ0, α, K, P, Δt, Tref, element),
        nels, Val(N), backend, workgroup; compute_jacobian,
    )
end
