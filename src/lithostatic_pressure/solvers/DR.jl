# Low-level lithostatic solve with explicitly supplied geometry, element,
# boundary arrays, backend, and workgroup size; `solve!` is the user-facing
# entry point.
function _solve_lithostatic!(
        dr::LithostaticPressureDR, mesh, geo, element,
        Γ_dofs, Γ_zero, Γ_vals,
        backend, workgroup;
        Tref = eltype(dr.P)(273),
        g = SVector(zero(eltype(dr.P)), -eltype(dr.P)(9.81)),
        kwargs...
    )
    assemble!(compute_jacobian) = assemble_lithostatic_pressure_matrices_atomix!(
        dr.R, dr.∂R∂P, dr.PC, dr.T, dr.P, mesh.el2n, geo, mesh.nels,
        element, dr.phases, dr.ρ0, dr.α, dr.K, Tref, g,
        backend, workgroup; compute_jacobian,
    )
    return solve_dynamic_relaxation!(
        dr, assemble!, mesh.nnodes, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
        kwargs...,
    )
end

"""
    solve!(dr::LithostaticPressureDR, mesh, bc; tolerance=dr.ϵ,
           max_iterations=10_000, check_interval=100, verbose=true,
           collect_history=false, throw_on_failure=true,
           workgroup=256, Tref=273, g=(0, -9.81))

Solve the lithostatic-pressure problem `∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ` with the
element and geometry stored in `mesh` and the Dirichlet values in `bc`. The
backend is inferred from `mesh.coords`. `dr.T` must hold the current
temperature; `dr.P` is updated in place.

`Tref` and `g` control the density equation of state and the body force. `g` is
an `SVector` or a plain `Tuple` of matching length; the default is only
appropriate for 2-D problems. The iteration controls and returned statistics
`(; converged, iterations, residual, history)` are those of
`solve_dynamic_relaxation!`.
"""
solve!(
    dr::LithostaticPressureDR, mesh::Mesh, bc::DirichletBoundaryCondition;
    workgroup = 256, kwargs...
) =
    _solve_lithostatic!(dr, _mesh_solver_arguments(mesh, bc, workgroup)...; kwargs...)
