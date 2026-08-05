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
    return assemble_dr_matrices_atomix!(
        R, ∂R∂P, PC, lp_element_residual, lp_element_jacobian,
        lithostatic_element_arguments(T, P, el2n, geo, phases, ρ0, α, K, Tref, g, element),
        nels, Val(N), backend, workgroup; compute_jacobian,
    )
end
