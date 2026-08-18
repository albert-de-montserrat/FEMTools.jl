"""
    solver!(dr::ThermalDiffusionDR, Δt, mesh, geo, element,
            Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; kwargs...)

Run the pseudo-transient dynamic-relaxation (DR) solver on `dr` for one time
step of size `Δt` using explicitly supplied geometry, element, boundary arrays,
backend, and workgroup size.

`ncheck` controls how often spectral
estimates and convergence are recomputed, and `iterMax` limits the number of
pseudo-transient iterations. Set `verbose=false` to suppress residual output.
`Tref` is the reference temperature used by the density equation of state.

The solver modifies `dr.T` in-place. `dr.T0` must be set to the temperature at
the previous physical time step before calling.

"""
function solver!(dr::ThermalDiffusionDR, Δt, mesh, geo, element,
                 Γ_dofs, Γ_zero, Γ_vals,
                 backend, workgroup;
                 Tref = eltype(dr.T)(273),
                 kwargs...)
    assemble!(compute_jacobian) = assemble_diffusion_matrices_atomix!(
        dr.R, dr.∂R∂T, dr.PC, dr.T, dr.T0, mesh.el2n, geo, mesh.nels,
        element, dr.phases, dr.k, dr.Cp, dr.ρ0, dr.α, dr.K, dr.P, Δt, dr.source, Tref,
        backend, workgroup; compute_jacobian,
    )
    return solve_dynamic_relaxation!(
        dr, assemble!, mesh.nnodes, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
        kwargs...,
    )
end

"""
    solver!(dr, Δt, mesh, bc; workgroup=256, kwargs...)

Solve one thermal-diffusion time step using the element and geometry stored in
`mesh` and the prescribed values in `bc`. The backend is inferred from
`mesh.coords`; remaining keywords are forwarded to the low-level solver.
"""
solver!(dr::ThermalDiffusionDR, Δt, mesh::Mesh, bc::DirichletBoundaryCondition;
        workgroup = 256, kwargs...) =
    solver!(dr, Δt, _mesh_solver_arguments(mesh, bc, workgroup)...; kwargs...)
