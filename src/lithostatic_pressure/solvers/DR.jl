"""
    solver!(dr::LithostaticPressureDR, mesh, geo, element,
            Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; kwargs...)

Run the pseudo-transient dynamic-relaxation (DR) solver for the
lithostatic-pressure problem `∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ`.

`dr.T` must be set to the current temperature field before calling. Geometry,
element, boundary arrays, backend, and workgroup size are supplied explicitly.
`Tref` and `g` control the density equation of state and body-force vector.
`g` is the gravitational acceleration vector; either an `SVector` or a plain
`Tuple` of matching length (e.g. `SVector(0, -9.81)` or `(0.0, -9.81)` for
2-D). The default is only appropriate for 2-D problems.
`ncheck` controls how often spectral estimates and convergence are recomputed.
`iterMax` is the maximum number of pseudo-transient iterations before a
non-convergence error.
Set `verbose = false` to suppress per-iteration residual output.

Modifies `dr.P` in-place. Returns `nothing` on convergence.

"""
function solver!(dr::LithostaticPressureDR, mesh, geo, element,
                 Γ_dofs, Γ_zero, Γ_vals,
                 backend, workgroup;
                 Tref = eltype(dr.P)(273),
                 g = SVector(zero(eltype(dr.P)), -eltype(dr.P)(9.81)),
                 kwargs...)
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
    solver!(dr, mesh, bc; workgroup=256, kwargs...)

Solve for lithostatic pressure using the element and geometry stored in `mesh`
and the prescribed values in `bc`. The backend is inferred from `mesh.coords`;
remaining keywords are forwarded to the low-level solver.
"""
solver!(dr::LithostaticPressureDR, mesh::Mesh, bc::DirichletBoundaryCondition;
        workgroup = 256, kwargs...) =
    solver!(dr, _mesh_solver_arguments(mesh, bc, workgroup)...; kwargs...)
