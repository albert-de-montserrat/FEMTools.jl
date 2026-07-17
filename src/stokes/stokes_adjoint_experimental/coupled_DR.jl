"""
    solve_stokes_adjoint_coupled_experimental!(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_v, phases_P, τ_old, plastic, G, Δt, γP,
        objective_vx, objective_vy, λvx, λvy, λP,
        backend, workgroup; kwargs...)

Experimental single-loop dynamic-relaxation solve of the complete discrete
Stokes adjoint. Unlike [`solve_stokes_adjoint_dyrel!`](@ref), this method does
not alternate a velocity DYREL solve with an Arrow--Hurwicz pressure step.
Every pseudo-time iteration assembles all three residuals and updates
`λvx`, `λvy`, and `λP` together.

The frozen spectral estimates are formed from all nine blocks of the augmented
forward Jacobian. Absolute column sums of that Jacobian are Gershgorin row
bounds for its transpose. The velocity fields use their absolute Jacobian
diagonals as preconditioners; pressure uses `dr.M_P`, because its diagonal is
zero in the incompressible limit. Fieldwise lower eigenvalue estimates use the
same residual-difference Rayleigh/secant estimate as the forward DR solver.

The momentum adjoints step against their residuals, while the pressure adjoint
steps with the opposite sign, matching the signs of the production
DYREL/Arrow--Hurwicz method. The coupled saddle operator is generally
nonsymmetric and can have complex eigenvalues, so the scalar fieldwise bounds
are stabilizing heuristics rather than a proof of convergence. This routine is
intentionally experimental and leaves the production solver unchanged.

`objective_vx`, `objective_vy`, and optional `objective_P` are the components
of `-∂J/∂u`. An identically zero objective gradient is an error. Set
`pressure_gauge=:auto` (the default) to remove the mass-weighted mean of `λP`
only when every bulk viscosity is infinite. Other accepted values are `:none`
and `:mean_zero`.

Convergence is checked every `ncheck` iterations and declared when either the
largest blockwise RMS residual (absolute) or the largest ratio of a block's
RMS residual to its own running peak (relative) drops below `adjoint_tol`.
The per-block peak normalization keeps the relative criterion dimensionless,
so it remains meaningful for problems posed in physical units.
"""
function solve_stokes_adjoint_coupled_experimental!(
    dr,
    mesh_stokes,
    geo_v,
    geo_P,
    element_v,
    element_P,
    phases_v,
    phases_P,
    τ_old,
    plastic,
    G,
    Δt,
    γP,
    objective_vx,
    objective_vy,
    λvx,
    λvy,
    λP,
    backend,
    workgroup;
    vx_nodes,
    vy_nodes,
    objective_P = nothing,
    ncheck = 50,
    adjoint_tol = 1.0e-6,
    iterMax = 50_000,
    verbose = true,
    collect_history = false,
    pressure_gauge = :auto,
    CFL_v = dr.CFL_v,
    CFL_P = dr.CFL_P,
    c_fact = dr.c_fact,
)
    ncheck > 0 || throw(ArgumentError("ncheck must be positive"))
    iterMax > 0 || throw(ArgumentError("iterMax must be positive"))
    pressure_gauge in (:auto, :none, :mean_zero) ||
        throw(ArgumentError("pressure_gauge must be :auto, :none, or :mean_zero"))

    M_P = dr.M_P
    all(isfinite, M_P) && all(>(zero(eltype(M_P))), M_P) ||
        throw(ArgumentError("dr.M_P must contain positive finite pressure masses"))

    objective_P_local = objective_P === nothing ? zero(dr.P) : objective_P
    length(objective_P_local) == length(λP) ||
        throw(DimensionMismatch("objective_P and λP must have the same length"))

    ResλVx = zero(dr.Rv_x)
    ResλVy = zero(dr.Rv_y)
    ResλP = zero(dr.P)
    ResλVx0 = zero(dr.Rv_x)
    ResλVy0 = zero(dr.Rv_y)
    ResλP0 = zero(dr.P)
    λrate_vx = zero(λvx)
    λrate_vy = zero(λvy)
    λrate_P = zero(λP)

    Rv_x_buf = zero(dr.Rv_x)
    Rv_y_buf = zero(dr.Rv_y)
    seed_Rv_x = zero(dr.Rv_x)
    seed_Rv_y = zero(dr.Rv_y)
    seed_RP = zero(dr.RP)
    dvx = zero(dr.vx)
    dvy = zero(dr.vy)
    dP = zero(dr.P)
    dP_scratch = zero(dr.P)
    Pnum = zero(dr.P)
    dPnum = zero(dr.P)

    zero_vx_bc = fill!(similar(λvx, length(vx_nodes)), 0)
    zero_vy_bc = fill!(similar(λvy, length(vy_nodes)), 0)
    one_vx_bc = fill!(similar(λvx, length(vx_nodes)), 1)
    one_vy_bc = fill!(similar(λvy, length(vy_nodes)), 1)
    residual_bc_gradient_vx = zero(zero_vx_bc)
    residual_bc_gradient_vy = zero(zero_vy_bc)
    bc_gradient_vx = zero(zero_vx_bc)
    bc_gradient_vy = zero(zero_vy_bc)

    # Masks describe which forward momentum residual rows survive the
    # Dirichlet overwrite before the transpose is applied.
    free_residual_vx = fill!(similar(λvx), 1)
    free_residual_vy = fill!(similar(λvy), 1)
    apply_dirichlet!(free_residual_vx, vx_nodes, zero_vx_bc, backend, workgroup)
    apply_dirichlet!(free_residual_vy, vy_nodes, zero_vy_bc, backend, workgroup)

    rowsum_vx = zero(λvx)
    rowsum_vy = zero(λvy)
    rowsum_P = zero(λP)
    diag_vx = zero(λvx)
    diag_vy = zero(λvy)
    diag_P = zero(λP)
    assemble_experimental_adjoint_spectral_diagnostics!(
        rowsum_vx, diag_vx, rowsum_vy, diag_vy, rowsum_P, diag_P,
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_v, phases_P, free_residual_vx, free_residual_vy,
        τ_old, plastic, G, Δt, γP, backend, workgroup,
    )

    # Constrained adjoint rows are never updated. Give them harmless unit
    # diagnostics so validation and reductions remain finite.
    apply_dirichlet!(rowsum_vx, vx_nodes, one_vx_bc, backend, workgroup)
    apply_dirichlet!(rowsum_vy, vy_nodes, one_vy_bc, backend, workgroup)
    apply_dirichlet!(diag_vx, vx_nodes, one_vx_bc, backend, workgroup)
    apply_dirichlet!(diag_vy, vy_nodes, one_vy_bc, backend, workgroup)

    λmax_vx = _checked_λmax(rowsum_vx, diag_vx, "experimental adjoint vx")
    λmax_vy = _checked_λmax(rowsum_vy, diag_vy, "experimental adjoint vy")
    λmax_P = _checked_λmax(rowsum_P, M_P, "experimental adjoint P")
    Δτ_vx = 2 / sqrt(λmax_vx) * CFL_v
    Δτ_vy = 2 / sqrt(λmax_vy) * CFL_v
    Δτ_P = 2 / sqrt(λmax_P) * CFL_P
    λmin_vx = zero(λmax_vx)
    λmin_vy = zero(λmax_vy)
    λmin_P = zero(λmax_P)
    α_vx, β_vx = _stokes_cheb(Δτ_vx, λmin_vx, c_fact)
    α_vy, β_vy = _stokes_cheb(Δτ_vy, λmin_vy, c_fact)
    α_P, β_P = _stokes_cheb(Δτ_P, λmin_P, c_fact)

    remove_mean = pressure_gauge === :mean_zero ||
        (pressure_gauge === :auto && all(isinf, dr.ηb))

    function assemble_adjoint_residual!()
        fill!(ResλVx, 0)
        fill!(ResλVy, 0)
        fill!(ResλP, 0)
        fill!(dvx, 0)
        fill!(dvy, 0)
        fill!(dP, 0)
        fill!(dPnum, 0)

        copyto!(seed_Rv_x, λvx)
        copyto!(seed_Rv_y, λvy)
        fill!(residual_bc_gradient_vx, 0)
        fill!(residual_bc_gradient_vy, 0)
        apply_dirichlet_pullback!(
            Rv_x_buf, seed_Rv_x, vx_nodes, zero_vx_bc,
            residual_bc_gradient_vx, workgroup,
        )
        apply_dirichlet_pullback!(
            Rv_y_buf, seed_Rv_y, vy_nodes, zero_vy_bc,
            residual_bc_gradient_vy, workgroup,
        )

        assemble_momentum_residual_matrices_atomix_adj!(
            Rv_x_buf, seed_Rv_x, Rv_y_buf, seed_Rv_y,
            dr.vx, dvx, dr.vy, dvy, dr.P, dP, dr.T, Pnum, dPnum,
            mesh_stokes, geo_v, element_v, element_P,
            phases_v, τ_old, plastic,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            workgroup,
        )
        copyto!(ResλP, dP)

        # Transpose of the augmented pressure correction in momentum.
        @. seed_RP = γP * dPnum / M_P
        fill!(dP_scratch, 0)
        assemble_pressure_residual_matrices_atomix_adj!(
            dr, seed_RP, dvx, dvy, dP_scratch,
            mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_P, Δt, workgroup,
        )
        @. ResλP += dP_scratch

        # Transpose continuity block seeded by λP.
        copyto!(seed_RP, λP)
        fill!(dP_scratch, 0)
        assemble_pressure_residual_matrices_atomix_adj!(
            dr, seed_RP, dvx, dvy, dP_scratch,
            mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_P, Δt, workgroup,
        )
        @. ResλP += dP_scratch + objective_P_local
        @. ResλVx = objective_vx + dvx
        @. ResλVy = objective_vy + dvy

        fill!(bc_gradient_vx, 0)
        fill!(bc_gradient_vy, 0)
        apply_dirichlet_pullback!(
            Rv_x_buf, ResλVx, vx_nodes, zero_vx_bc, bc_gradient_vx, workgroup,
        )
        apply_dirichlet_pullback!(
            Rv_y_buf, ResλVy, vy_nodes, zero_vy_bc, bc_gradient_vy, workgroup,
        )
        return nothing
    end

    apply_dirichlet!(λvx, vx_nodes, zero_vx_bc, backend, workgroup)
    apply_dirichlet!(λvy, vy_nodes, zero_vy_bc, backend, workgroup)
    remove_mean && remove_pressure_mean!(λP, M_P)
    assemble_adjoint_residual!()

    nfree_vx = length(λvx) - length(vx_nodes)
    nfree_vy = length(λvy) - length(vy_nodes)
    rms(r, n) = iszero(n) ? zero(eltype(r)) : norm(r) / sqrt(n)
    err_vx = rms(ResλVx, nfree_vx)
    err_vy = rms(ResλVy, nfree_vy)
    err_P = rms(ResλP, length(λP))
    iszero(max(err_vx, err_vy, err_P)) && error(
        "the adjoint residual vanishes at the initial guess: \
        the objective gradient is identically zero"
    )
    # Each block's relative error is measured against the largest RMS residual
    # that block has reached so far. Blocks the objective does not touch start
    # at exactly zero and only acquire residual through coupling, so their
    # initial value cannot serve as a scale; the running peak keeps every
    # ratio dimensionless, which matters for dimensional (unscaled) problems
    # where the raw residual magnitudes carry physical units.
    err_vx0 = err_vx
    err_vy0 = err_vy
    err_P0 = err_P
    relerr(e, e0) = iszero(e0) ? zero(e) : e / e0
    history = NamedTuple[]
    converged = false
    err = Inf

    if verbose
        @info "Starting experimental coupled adjoint DR" λmax_vx λmax_vy λmax_P Δτ_vx Δτ_vy Δτ_P remove_mean
    end

    for iter in 1:iterMax
        copyto!(ResλVx0, ResλVx)
        copyto!(ResλVy0, ResλVy)
        copyto!(ResλP0, ResλP)

        stokes_update_rate!(λrate_vx, ResλVx, diag_vx, β_vx,
            length(λvx), backend, workgroup)
        stokes_update_rate!(λrate_vy, ResλVy, diag_vy, β_vy,
            length(λvy), backend, workgroup)
        stokes_update_rate!(λrate_P, ResλP, M_P, β_P,
            length(λP), backend, workgroup)

        stokes_update_variable!(λvx, λrate_vx, -α_vx,
            length(λvx), backend, workgroup)
        stokes_update_variable!(λvy, λrate_vy, -α_vy,
            length(λvy), backend, workgroup)
        # Opposite sign for the saddle-point pressure row.
        stokes_update_variable!(λP, λrate_P, α_P,
            length(λP), backend, workgroup)

        apply_dirichlet!(λvx, vx_nodes, zero_vx_bc, backend, workgroup)
        apply_dirichlet!(λvy, vy_nodes, zero_vy_bc, backend, workgroup)
        apply_dirichlet!(λrate_vx, vx_nodes, zero_vx_bc, backend, workgroup)
        apply_dirichlet!(λrate_vy, vy_nodes, zero_vy_bc, backend, workgroup)
        if remove_mean
            remove_pressure_mean!(λP, M_P)
            remove_pressure_mean!(λrate_P, M_P)
        end

        assemble_adjoint_residual!()

        if iter == 1 || iszero(iter % ncheck) || iter == iterMax
            err_vx = rms(ResλVx, nfree_vx)
            err_vy = rms(ResλVy, nfree_vy)
            err_P = rms(ResλP, length(λP))
            err_vx0 = max(err_vx0, err_vx)
            err_vy0 = max(err_vy0, err_vy)
            err_P0 = max(err_P0, err_P)
            err_abs = max(err_vx, err_vy, err_P)
            err_rel = max(
                relerr(err_vx, err_vx0), relerr(err_vy, err_vy0),
                relerr(err_P, err_P0),
            )
            err = min(err_abs, err_rel)

            all(isfinite, (err, err_vx, err_vy, err_P)) ||
                error("non-finite residual in experimental coupled adjoint DR at iteration $iter")

            λmin_vx = _stokes_λmin(α_vx, λrate_vx, ResλVx .- ResλVx0, diag_vx)
            λmin_vy = _stokes_λmin(α_vy, λrate_vy, ResλVy .- ResλVy0, diag_vy)
            λmin_P = _stokes_λmin(α_P, λrate_P, ResλP .- ResλP0, M_P)
            α_vx, β_vx = _stokes_cheb(Δτ_vx, λmin_vx, c_fact)
            α_vy, β_vy = _stokes_cheb(Δτ_vy, λmin_vy, c_fact)
            α_P, β_P = _stokes_cheb(Δτ_P, λmin_P, c_fact)

            collect_history && push!(history, (;
                iter, err, err_abs, err_rel, err_vx, err_vy, err_P,
                λmin_vx, λmin_vy, λmin_P,
                λmax_vx, λmax_vy, λmax_P,
            ))
            verbose && @printf(
                "experimental adj it=%06d err=%.3e abs=%.3e rel=%.3e R=[%.3e %.3e %.3e] λmin=[%.2e %.2e %.2e]\n",
                iter, err, err_abs, err_rel, err_vx, err_vy, err_P,
                λmin_vx, λmin_vy, λmin_P,
            )

            if err < adjoint_tol
                converged = true
                return (;
                    iter, err, err_abs, err_rel, err_vx, err_vy, err_P, converged,
                    λmin = (vx = λmin_vx, vy = λmin_vy, P = λmin_P),
                    λmax = (vx = λmax_vx, vy = λmax_vy, P = λmax_P),
                    bc_gradient_vx, bc_gradient_vy,
                    spectral_rowsums = (vx = rowsum_vx, vy = rowsum_vy, P = rowsum_P),
                    spectral_diagonals = (vx = diag_vx, vy = diag_vy, P = diag_P),
                    history,
                )
            end
        end
    end

    return (;
        iter = iterMax, err, err_abs = max(err_vx, err_vy, err_P),
        err_rel = max(
            relerr(err_vx, err_vx0), relerr(err_vy, err_vy0),
            relerr(err_P, err_P0),
        ),
        err_vx, err_vy, err_P, converged,
        λmin = (vx = λmin_vx, vy = λmin_vy, P = λmin_P),
        λmax = (vx = λmax_vx, vy = λmax_vy, P = λmax_P),
        bc_gradient_vx, bc_gradient_vy,
        spectral_rowsums = (vx = rowsum_vx, vy = rowsum_vy, P = rowsum_P),
        spectral_diagonals = (vx = diag_vx, vy = diag_vy, P = diag_P),
        history,
    )
end
