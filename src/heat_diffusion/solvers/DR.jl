function _assemble_thermal!(
        dr, Δt, mesh, geo, element, Tref, backend, workgroup, compute_jacobian;
        source_ip = nothing,
    )
    return assemble_diffusion_matrices_atomix!(
        dr.R, dr.∂R∂T, dr.PC, dr.T, dr.T0, mesh.el2n, geo, mesh.nels,
        element, dr.phases, dr.k, dr.Cp, dr.ρ0, dr.α, dr.K, dr.P, Δt, dr.source, Tref,
        backend, workgroup; compute_jacobian, source_ip,
    )
end

# Low-level thermal solve with explicitly supplied geometry, element, boundary
# arrays, backend, and workgroup size; `solve!` is the user-facing entry point.
function _solve_thermal!(
        dr::ThermalDiffusionDR, Δt, mesh, geo, element,
        Γ_dofs, Γ_zero, Γ_vals,
        backend, workgroup;
        Tref = eltype(dr.T)(273),
        kwargs...
    )
    assemble!(compute_jacobian) = _assemble_thermal!(
        dr, Δt, mesh, geo, element, Tref, backend, workgroup, compute_jacobian,
    )
    return solve_dynamic_relaxation!(
        dr, assemble!, mesh.nnodes, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
        kwargs...,
    )
end

"""
    solve!(dr::ThermalDiffusionDR, mesh, bc; dt, tolerance=dr.ϵ,
           max_iterations=10_000, check_interval=100, verbose=true,
           collect_history=false, throw_on_failure=true,
           workgroup=256, Tref=273)

Solve one backward-Euler thermal-diffusion step of size `dt` with the element
and geometry stored in `mesh` and the Dirichlet values in `bc`. The backend is
inferred from `mesh.coords`. `dr.T` is updated in place; `dr.T0` must hold the
temperature of the previous physical time step, and this call does not advance
it.

`Tref` is the reference temperature of the density equation of state. The
iteration controls and the returned statistics
`(; converged, iterations, residual, history)` are those of
`solve_dynamic_relaxation!`: spectral estimates and convergence are
refreshed every `check_interval` iterations, `residual` is the relative residual
that `tolerance` bounds, and exhausting `max_iterations` throws unless
`throw_on_failure=false`.
"""
solve!(
    dr::ThermalDiffusionDR, mesh::Mesh, bc::DirichletBoundaryCondition;
    dt, workgroup = 256, kwargs...
) =
    _solve_thermal!(dr, dt, _mesh_solver_arguments(mesh, bc, workgroup)...; kwargs...)
