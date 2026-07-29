"""
    solve_stokes_adjoint_dyrel!(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                                phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                                objective_vx, objective_vy, λvx, λvy, λP,
                                backend, workgroup; kwargs...) -> NamedTuple

Solve the discrete Stokes adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` with the same
Powell-Hestenes / DYREL iteration used by [`solve_stokes_dyrel!`](@ref), and
store the adjoint fields in `λvx`, `λvy`, `λP` (modified in place).

The adjoint is assembled on the *same* T7/P1-disc spaces, quadrature, and element
operators as the forward problem and transposed exactly, so `-λᵀ ∂R/∂m` is the
exact gradient of the discrete objective. The forward state in `dr` must already
be converged: the transpose Jacobian, its diagonal preconditioner, and λmax are
frozen at that state, so only λmin (hence the Chebyshev pair) is re-estimated
during the solve.

`objective_vx` and `objective_vy` carry the velocity part of `-∂J/∂u`
(the consistently assembled objective load); the pressure adjoint has no explicit
objective term. `M_P = dr.M_P` and the augmentation scaling `γP` must match the
forward solve. Homogeneous Dirichlet conditions are applied to the adjoint
velocity on `vx_nodes`/`vy_nodes`.

Because the forward state is frozen, the adjoint residual is affine in `λ` with a
constant operator. With `frozen_operator` (the default) that operator is
assembled once as per-element blocks and then applied as a dense element product,
so no rheology is evaluated and no primal residual is recomputed during the
solve; see [`FrozenAdjointOperator`](@ref) for the blocks and their memory cost.
Setting `frozen_operator = false` selects the reverse-mode path instead, which
rebuilds the same products by automatic differentiation on every iteration: far
slower, but it stores nothing per element and so remains the option when the
block storage is too large.

Use `verbose` for outer Powell-Hestenes progress and `verbose_inner` for the
inner dynamic-relaxation trace. Returns a `NamedTuple` with `itPH`, `iter`,
`err`, `err_v`, `err_P`, `converged`, and (when `collect_history`) `history`.

All arrays read or written by kernels—including `mesh_stokes` connectivity,
`geo_v`, `geo_P`, phases, objective loads, adjoint fields, and boundary-node
arrays—must reside on `backend`. Construct unstructured meshes with
`Mesh(backend, coords, el2n)` and geometry with `MixedMeshCache` to maintain
that invariant. The Enzyme transpose assemblers execute on the backend inferred
from their output buffers.
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
    frozen_operator = true,
)
    M_P = dr.M_P

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

    # Freeze the transpose velocity block at the converged forward state: assemble
    # the augmented momentum Jacobian once to obtain the diagonal preconditioner
    # and λmax. Both stay constant through the adjoint solve.
    assemble_augmented_momentum_jacobian_matrices_atomix!(
        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
        element_v, element_P, phases_v, phases_P,
        τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
        dr.ηb, Δt, γP, M_P, backend, workgroup,
    )

    λmax_vx = _checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "adjoint vx")
    λmax_vy = _checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "adjoint vy")
    Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
    Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v
    α_vx, β_vx = _stokes_cheb(Δτ_vx, zero(λmax_vx), dr.c_fact)
    α_vy, β_vy = _stokes_cheb(Δτ_vy, zero(λmax_vy), dr.c_fact)

    # The operator is constant at the frozen forward state, so assembling it once
    # replaces the three reverse-mode sweeps every iteration would otherwise run.
    op = frozen_operator ?
        assemble_adjoint_operator(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup,
    ) : nothing

    function assemble_adjoint_residual_enzyme!()
        # Only the Enzyme shadows need zeroing: they are accumulated into by the
        # reverse passes. ResλVx, ResλVy, and ResλP are each fully overwritten below.
        fill!(dvx, 0)
        fill!(dvy, 0)
        fill!(dP, 0)
        fill!(dPnum, 0)

        copyto!(seed_Rv_x, λvx)
        copyto!(seed_Rv_y, λvy)

        # Momentum transpose: (∂Rv/∂v)ᵀλv → dvx,dvy, (∂Rv/∂P)ᵀλv → dP,
        # and (∂Rv/∂Pnum)ᵀλv → dPnum.
        assemble_momentum_residual_matrices_atomix_adj!(
            Rv_x_buf, seed_Rv_x, Rv_y_buf, seed_Rv_y,
            dr.vx, dvx, dr.vy, dvy, dr.P, dP, dr.T, Pnum, dPnum,
            mesh_stokes, geo_v, element_v, element_P,
            phases_v, τ_old, plastic,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            workgroup,
        )

        # (∂Rv/∂P)ᵀλv is the adjoint pressure-constraint residual. Capture it before
        # the pressure pullbacks reuse dP.
        copyto!(ResλP, dP)

        # Both couplings to the velocity adjoint run through the same transpose
        # (∂RP/∂v)ᵀ, which is linear in its seed, so one pass carries both:
        # the Powell-Hestenes augmented grad-div self-coupling — the forward
        # momentum uses Pnum(v) = γP·RP(v)/M_P, closing the chain through Pnum as
        # (∂RP/∂v)ᵀ(γP·(∂Rv/∂Pnum)ᵀλv/M_P) — and the saddle-point coupling
        # (∂RP/∂v)ᵀλP.
        @. seed_RP = λP + γP * dPnum / M_P
        fill!(dP_scratch, 0)
        assemble_pressure_residual_matrices_atomix_adj!(
            dr, seed_RP, dvx, dvy, dP_scratch,
            mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_P, Δt, workgroup,
        )

        return nothing
    end

    function assemble_adjoint_residual!()
        if op === nothing
            assemble_adjoint_residual_enzyme!()
        else
            apply_adjoint_operator!(
                dvx, dvy, ResλP, op, λvx, λvy, λP,
                mesh_stokes, element_v, element_P, backend, workgroup,
            )
        end
        @. ResλVx = objective_vx + dvx
        @. ResλVy = objective_vy + dvy
        apply_dirichlet!(ResλVx, vx_nodes, zero_vx_bc, backend, workgroup)
        apply_dirichlet!(ResλVy, vy_nodes, zero_vy_bc, backend, workgroup)
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

        err_v = max(norm(ResλVx), norm(ResλVy)) / sqrt(mesh_stokes.nnodes)
        err_P = norm(ResλP) / sqrt(mesh_stokes.nnodesP)
        if itPH == 1
            err_v0 = err_v + eps(err_v)
            err_P0 = err_P + eps(err_P)
        end
        err = max(min(err_v, err_v / err_v0), min(err_P, err_P / err_P0))
        collect_history && push!(history, (; iter, itPH, err, err_v, err_P))
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
                err_v = max(norm(ResλVx), norm(ResλVy)) / sqrt(mesh_stokes.nnodes)

                # Re-estimate λmin and refresh the Chebyshev step. Δτ and λmax stay
                # fixed (the Jacobian depends only on the frozen forward state).
                # The differences overwrite ResλVx0/ResλVy0, which the next
                # iteration refills from ResλVx/ResλVy before reading them again.
                @. ResλVx0 = ResλVx - ResλVx0
                @. ResλVy0 = ResλVy - ResλVy0
                λmin_vx = _stokes_λmin(α_vx, λrate_vx, ResλVx0, dr.PC_vx)
                λmin_vy = _stokes_λmin(α_vy, λrate_vy, ResλVy0, dr.PC_vy)
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

    # Breaking on the convergence test leaves the matching residual and errors in
    # hand; every other exit needs a fresh residual, since λP moved after the last
    # assembly.
    if !converged
        assemble_adjoint_residual!()
        err_v = max(norm(ResλVx), norm(ResλVy)) / sqrt(mesh_stokes.nnodes)
        err_P = norm(ResλP) / sqrt(mesh_stokes.nnodesP)
        err = max(min(err_v, err_v / err_v0), min(err_P, err_P / err_P0))
    end

    return (;
        itPH = itPH_done,
        iter,
        err,
        err_v,
        err_P,
        converged = converged || err < adjoint_tol,
        history,
    )
end
