"""
    solve_stokes_adjoint_dyrel!(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                                phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                                objective_vx, objective_vy, λvx, λvy, λP,
                                backend, workgroup; kwargs...) -> NamedTuple

Solve the discrete Stokes adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` with the same
Powell-Hestenes / DYREL iteration used by [`solve_stokes_dyrel!`](@ref), and
store the adjoint fields in `λvx`, `λvy`, `λP` (modified in place).

The adjoint is assembled on the *same* T7/P1-disc spaces, quadrature, and element
operators as the forward problem and transposed exactly, so `λᵀ ∂R/∂m` is the
exact gradient of the discrete objective. The forward state in `dr` must already
be converged: the transpose Jacobian and its diagonal preconditioner are frozen
at that state, so the transposed element blocks are assembled once (with
[`assemble_momentum_adjoint_blocks`](@ref) and
[`assemble_pressure_adjoint_blocks`](@ref)) and every iteration applies them as
dense element mat-vecs. The initial λmax bound uses row sums of the transpose
(column sums of the forward Jacobian), while λmin and the Chebyshev pair are
re-estimated during the solve.

`objective_vx` and `objective_vy` carry the velocity part of `-∂J/∂u`
(the consistently assembled objective load); the pressure adjoint has no explicit
objective term. `M_P = dr.M_P` and the augmentation scaling `γP` must match the
forward solve. Homogeneous Dirichlet conditions are applied to the adjoint
velocity on `vx_nodes`/`vy_nodes`. The forward residual's Dirichlet overwrite
is included in the transpose operator through Enzyme's pullback: constrained
residual seeds are projected out before the interior Jacobian transpose is
applied, and derivatives with respect to the prescribed boundary values are
formed as part of that pullback.

Use `verbose` for outer Powell-Hestenes progress and `verbose_inner` for the
inner dynamic-relaxation trace. Returns a `NamedTuple` with `itPH`, `iter`,
`err`, `err_v`, `err_P`, `converged`, `bc_gradient_vx`, `bc_gradient_vy`, and
(when `collect_history`) `history`. The boundary gradients are aligned with
`vx_nodes` and `vy_nodes`.

All arrays read or written by kernels—including `mesh_stokes` connectivity,
`geo_v`, `geo_P`, phases, objective loads, adjoint fields, and boundary-node
arrays—must reside on `backend`. Construct unstructured meshes with
`Mesh(backend, coords, el2n)` and geometry with `MixedMeshCache` to maintain
that invariant. The one-time ForwardDiff block assembly and the per-iteration
block applies execute on the backend inferred from their output buffers.
"""
function solve_stokes_adjoint_dyrel!(
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
    ncheck = 50,
    adjoint_tol = 1.0e-6,
    rel_drop = 0.1,
    iterMax = 50_000,
    total_iterMax = 50_000,
    max_ph_iterations = 100,
    verbose = true,
    verbose_inner = false,
    collect_history = false,
)
    M_P = dr.M_P
    M_V = dr.M_V
    assemble_velocity_mass!(M_V, mesh_stokes, geo_v, element_v, backend, workgroup)
    total_mass_V = sum(M_V)
    free_mass_vx = total_mass_V - sum(M_V[vx_nodes])
    free_mass_vy = total_mass_V - sum(M_V[vy_nodes])
    total_mass_P = sum(M_P)

    adjoint_velocity_residual_norm() = max(
        _mass_weighted_rms(ResλVx, M_V, free_mass_vx),
        _mass_weighted_rms(ResλVy, M_V, free_mass_vy),
    ) / 2
    adjoint_pressure_residual_norm() =
        _mass_weighted_rms(ResλP, M_P, total_mass_P)

    # Scratch: adjoint residuals, DYREL rates, and the reverse-pass seeds/pullbacks.
    ResλVx = zero(dr.Rv_x)
    ResλVy = zero(dr.Rv_y)
    ResλP  = zero(dr.P)
    ResλVx0 = zero(dr.Rv_x)
    ResλVy0 = zero(dr.Rv_y)
    λrate_vx = zero(dr.vx)
    λrate_vy = zero(dr.vy)

    Rv_x_buf = zero(dr.Rv_x)
    Rv_y_buf = zero(dr.Rv_y)
    seed_Rv_x = zero(dr.Rv_x)
    seed_Rv_y = zero(dr.Rv_y)
    seed_RP   = zero(dr.RP)
    dvx = zero(dr.vx)
    dvy = zero(dr.vy)
    dP  = zero(dr.P)
    dP_scratch = zero(dr.P)
    # Real augmented-pressure array so its adjoint (∂Rv/∂Pnum)ᵀλv is obtained
    # directly rather than via the P == Pnum shortcut. Its value is irrelevant to
    # the transpose (Rv is linear in Pnum), so it stays zero.
    Pnum  = zero(dr.P)
    dPnum = zero(dr.P)

    zero_vx_bc = fill!(similar(λvx, length(vx_nodes)), 0)
    zero_vy_bc = fill!(similar(λvy, length(vy_nodes)), 0)
    residual_bc_gradient_vx = zero(zero_vx_bc)
    residual_bc_gradient_vy = zero(zero_vy_bc)
    bc_gradient_vx = zero(zero_vx_bc)
    bc_gradient_vy = zero(zero_vy_bc)

    # Freeze the transpose velocity block at the converged forward state. Its
    # diagonal matches the forward Jacobian, but its conservative λmax bound
    # needs column sums of the forward Jacobian (row sums of the transpose).
    assemble_augmented_momentum_jacobian_matrices_atomix!(
        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
        element_v, element_P, phases_v, phases_P,
        τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
        dr.ηb, Δt, γP, M_P, backend, workgroup;
        transpose_operator = true,
    )

    λmax_vx = _checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "adjoint vx")
    λmax_vy = _checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "adjoint vy")
    Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
    Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v
    α_vx, β_vx = _stokes_cheb(Δτ_vx, zero(λmax_vx), dr.c_fact)
    α_vy, β_vy = _stokes_cheb(Δτ_vy, zero(λmax_vy), dr.c_fact)
    verbose && @info "Initial adjoint momentum preconditioner" λmax_vx λmax_vy Δτ_vx Δτ_vy

    # The whole transpose operator is likewise frozen, so its element Jacobian
    # blocks are assembled once and every iteration applies them as small
    # dense mat-vecs. Differentiating inside the iteration would recompute
    # identical blocks thousands of times at roughly 40x the cost per apply.
    momentum_blocks = assemble_momentum_adjoint_blocks(
        dr.vx, dr.vy, dr.P, Pnum, dr.T,
        mesh_stokes, geo_v, element_v, element_P,
        phases_v, τ_old, plastic,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
        workgroup,
    )
    pressure_blocks = assemble_pressure_adjoint_blocks(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_P, Δt, workgroup,
    )

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

        # The forward momentum operator is boundary-conditioned as B(Rv), with
        # B overwriting constrained residual entries. Differentiate that same
        # operation with Enzyme before applying the interior residual
        # transpose: seed_Rv <- Bᵀλ. Besides projecting constrained state
        # seeds to zero, the pullback accumulates derivatives with respect to
        # the overwritten residual values in residual_bc_gradient_*.
        fill!(residual_bc_gradient_vx, 0)
        fill!(residual_bc_gradient_vy, 0)
        apply_dirichlet_pullback!(
            Rv_x_buf, seed_Rv_x, vx_nodes, zero_vx_bc,
            residual_bc_gradient_vx, workgroup)
        apply_dirichlet_pullback!(
            Rv_y_buf, seed_Rv_y, vy_nodes, zero_vy_bc,
            residual_bc_gradient_vy, workgroup)

        # Momentum transpose: (∂Rv/∂v)ᵀλv → dvx,dvy, (∂Rv/∂P)ᵀλv → dP,
        # and (∂Rv/∂Pnum)ᵀλv → dPnum.
        apply_momentum_adjoint_blocks!(
            dvx, dvy, dP, dPnum, seed_Rv_x, seed_Rv_y,
            momentum_blocks, mesh_stokes, element_v, element_P, workgroup,
        )

        # (∂Rv/∂P)ᵀλv is the adjoint pressure-constraint residual. Capture it before
        # the pressure pullback reuses dP.
        copyto!(ResλP, dP)

        # Pressure-residual transpose, applied to the sum of two seeds (it is
        # linear in the seed):
        #   * Powell-Hestenes augmented coupling. The forward momentum uses
        #     Pnum(v, P) = γP·RP(v, P)/M_P, so the chain through Pnum closes as
        #     (∂RP/∂u)ᵀ(γP·(∂Rv/∂Pnum)ᵀλv/M_P).
        #   * Saddle-point coupling to the pressure adjoint λP: (∂RP/∂u)ᵀλP.
        # For both, the velocity part (∂RP/∂v)ᵀ feeds the velocity adjoint and
        # the pressure part (∂RP/∂P)ᵀ feeds ResλP. The pressure part is nonzero
        # only for finite bulk modulus (`ηb`): RP stores pressure elastically
        # through the `-(P-P0)/(ηb·Δt)` term, so omitting it makes the adjoint
        # (and any gradient built from it) wrong by O(1/(ηb·Δt)) — exact only
        # in the incompressible limit.
        @. seed_RP = γP * dPnum / M_P + λP
        fill!(dP_scratch, 0)
        apply_pressure_adjoint_blocks!(
            dvx, dvy, dP_scratch, seed_RP,
            pressure_blocks, mesh_stokes, element_v, element_P, workgroup,
        )
        @. ResλP += dP_scratch

        @. ResλVx = objective_vx + dvx
        @. ResλVy = objective_vy + dvy

        # Pull back through the state overwrite v[dofs] = bc_vals. This both
        # projects the adjoint residual onto free state variables and retains
        # the cotangent of the prescribed velocity values for callers that use
        # boundary data as controls. The derivative is independent of the
        # primal boundary values, so the zero-valued scratch primal is valid.
        fill!(bc_gradient_vx, 0)
        fill!(bc_gradient_vy, 0)
        apply_dirichlet_pullback!(
            Rv_x_buf, ResλVx, vx_nodes, zero_vx_bc, bc_gradient_vx, workgroup)
        apply_dirichlet_pullback!(
            Rv_y_buf, ResλVy, vy_nodes, zero_vy_bc, bc_gradient_vy, workgroup)
        return nothing
    end

    iter = 0
    err = Inf
    err_v0 = 1.0
    err_P0 = 1.0
    err_v = Inf
    err_P = Inf
    itPH_done = 0
    converged = false
    history = NamedTuple[]

    verbose && @info "Starting adjoint PH/DYREL solve" adjoint_tol rel_drop

    for itPH in 1:max_ph_iterations
        itPH_done = itPH
        assemble_adjoint_residual!()

        err_v = adjoint_velocity_residual_norm()
        err_P = adjoint_pressure_residual_norm()
        if itPH == 1
            err_v0 = max(err_v, adjoint_tol) + eps(err_v)
            err_P0 = max(err_P, adjoint_tol) + eps(err_P)
        end
        err = max(min(err_v, err_v / err_v0), min(err_P, err_P / err_P0))
        collect_history && push!(history, (;
            iter, itPH, err, err_v, err_P,
            err_v_rel = err_v / err_v0,
            err_P_rel = err_P / err_P0,
        ))
        verbose && @printf("adj PH=%03d iter=%06d err=%.3e Rv=%.3e RP=%.3e\n",
            itPH, iter, err, err_v, err_P)

        if err < adjoint_tol
            converged = true
            break
        end

        target_v = max(err_v * rel_drop, adjoint_tol)
        inner = 0
        while err_v > target_v && inner < iterMax && iter < total_iterMax
            inner += 1
            iter += 1
            copyto!(ResλVx0, ResλVx)
            copyto!(ResλVy0, ResλVy)

            stokes_update_rate!(λrate_vx, ResλVx, dr.PC_vx, β_vx,
                mesh_stokes.nnodes, backend, workgroup)
            stokes_update_variable!(λvx, λrate_vx, -α_vx,
                mesh_stokes.nnodes, backend, workgroup)
            stokes_update_rate!(λrate_vy, ResλVy, dr.PC_vy, β_vy,
                mesh_stokes.nnodes, backend, workgroup)
            stokes_update_variable!(λvy, λrate_vy, -α_vy,
                mesh_stokes.nnodes, backend, workgroup)

            apply_dirichlet!(λvx, vx_nodes, zero_vx_bc, backend, workgroup)
            apply_dirichlet!(λvy, vy_nodes, zero_vy_bc, backend, workgroup)
            apply_dirichlet!(λrate_vx, vx_nodes, zero_vx_bc, backend, workgroup)
            apply_dirichlet!(λrate_vy, vy_nodes, zero_vy_bc, backend, workgroup)

            assemble_adjoint_residual!()

            if iszero(iter % ncheck)
                err_v = adjoint_velocity_residual_norm()

                # Re-estimate λmin and refresh the Chebyshev step. Δτ and λmax stay
                # fixed (the Jacobian depends only on the frozen forward state).
                λmin_vx = _stokes_λmin(α_vx, λrate_vx, ResλVx .- ResλVx0, dr.PC_vx)
                λmin_vy = _stokes_λmin(α_vy, λrate_vy, ResλVy .- ResλVy0, dr.PC_vy)
                α_vx, β_vx = _stokes_cheb(Δτ_vx, λmin_vx, dr.c_fact)
                α_vy, β_vy = _stokes_cheb(Δτ_vy, λmin_vy, dr.c_fact)

                verbose_inner && @printf(
                    "  adj inner it=%05d iter=%06d err_v=%.3e α=[%.2e %.2e] β=[%.2e %.2e]\n",
                    inner, iter, err_v, α_vx, α_vy, β_vx, β_vy)
            end
        end

        # Arrow-Hurwicz update for the pressure adjoint.
        @. λP += γP * ResλP / M_P

        iter >= total_iterMax && break
    end

    assemble_adjoint_residual!()
    err_v = adjoint_velocity_residual_norm()
    err_P = adjoint_pressure_residual_norm()
    err = max(min(err_v, err_v / err_v0), min(err_P, err_P / err_P0))

    return (;
        itPH = itPH_done,
        iter,
        err,
        err_v,
        err_P,
        converged = converged || err < adjoint_tol,
        bc_gradient_vx,
        bc_gradient_vy,
        history,
    )
end
