function _stokes_λmin(step, rate, ΔR, PC)
    dV = step .* rate
    denom = sum(dV .^ 2)
    return iszero(denom) ? zero(denom) : abs(sum(dV .* (ΔR ./ PC))) / denom
end

function _stokes_cheb(Δτ, λmin, c_fact)
    c = 2 * sqrt(λmin) * c_fact
    return (2 * Δτ^2 / (2 + c * Δτ), (2 - c * Δτ) / (2 + c * Δτ))
end

@inline _normalize_stokes_verbose(verbose, verbose_inner, ::Nothing, ::Nothing) =
    (Bool(verbose), Bool(verbose_inner))

function _normalize_stokes_verbose(verbose, verbose_inner, verbose_PH, verbose_DR)
    if verbose_PH !== nothing
        Base.depwarn("`verbose_PH` is deprecated; use `verbose` instead", :solve_stokes_dyrel!)
        verbose = verbose_PH
    end
    if verbose_DR !== nothing
        Base.depwarn("`verbose_DR` is deprecated; use `verbose_inner` instead", :solve_stokes_dyrel!)
        verbose_inner = verbose_DR
    end
    return Bool(verbose), Bool(verbose_inner)
end

"""
    solve_stokes_dyrel!(dr, mesh, cache, bc_vx, bc_vy, Δt, γP;
                        plastic=nothing, workgroup=256, kwargs...)

Solve the Stokes system using geometry and elements from `cache`, material and
stress history from `dr`, and one Dirichlet boundary-condition object per
velocity component. Phase layouts and stress history may be overridden with
the `phases_v`, `phases_P`, and `τ_old` keywords.
"""
function solve_stokes_dyrel!(
    dr::StokesDR,
    mesh::MixedMesh,
    cache::MixedMeshCache,
    bc_vx::DirichletBoundaryCondition,
    bc_vy::DirichletBoundaryCondition,
    Δt,
    γP;
    plastic = nothing,
    phases_v = dr.phases_v,
    phases_P = dr.phases_P,
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old),
    workgroup = 256,
    kwargs...,
)
    isnothing(cache.element_v) && throw(ArgumentError("cache has no reference elements; construct it with MixedMeshCache(backend, workgroup, mesh, element_v, element_P)"))
    backend = KA.get_backend(mesh.coords)
    return solve_stokes_dyrel!(
        dr, mesh, cache, cache.element_v, cache.element_P,
        phases_v, phases_P, τ_old, plastic, dr.G, Δt, γP,
        bc_vx.DoFs, bc_vx.vals, bc_vy.vals, backend, workgroup;
        vx_nodes = bc_vx.DoFs, vy_nodes = bc_vy.DoFs, kwargs...,
    )
end

"""
    solve_stokes_dyrel!(dr, mesh_stokes, cache, element_v, element_P,
                        phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                        Γnodes, bc_vx_vals, bc_vy_vals, backend, workgroup;
                        kwargs...) -> NamedTuple
    solve_stokes_dyrel!(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                        ...)

Run the Powell-Hestenes / DYREL-style velocity-pressure iteration for a Stokes
state with Dirichlet velocity boundary conditions on `Γnodes`. `dr.M_P` must be
filled before calling. The two forms differ only in how the precomputed element
geometry is supplied: as a `MixedMeshCache` or as the raw `geo_v`, `geo_P`
arrays.

# Keyword arguments
- `ncheck = 50`: recompute spectral estimates and convergence every `ncheck`
  inner iterations.
- `ϵ_tol = 1e-6`: outer convergence tolerance on the combined residual.
- `iterMax = 50_000`: maximum inner (dynamic-relaxation) iterations per
  Powell-Hestenes step.
- `total_iterMax = 50_000`: maximum cumulative inner iterations across all
  Powell-Hestenes steps.
- `max_ph_iterations = 1000`: maximum outer Powell-Hestenes iterations.
- `rel_drop0 = 1e-2`: initial relative residual drop demanded of each inner
  solve; tightened automatically as the outer iteration proceeds.
- `vx_nodes = Γnodes`, `vy_nodes = Γnodes`: per-component Dirichlet node sets,
  for when the two velocity components are constrained on different boundaries.
- `verbose = true`: outer Powell-Hestenes progress; `verbose_inner = false`:
  inner dynamic-relaxation trace.
- `collect_history = false`: record `(iter, err_v, err_P)` at every check.
- `measure_λmax = false`: experimentally replace the Gershgorin bound with
  power iteration on the symmetrically Jacobi-scaled velocity operator.
- `freeze_jacobian = plastic === nothing`: reuse the constant linear momentum
  Jacobian instead of differentiating it at every convergence check.

# Return value
A `NamedTuple` with `itPH` (outer iterations), `iter` (cumulative inner
iterations), `err`, `err_abs`, `err_rel`, `err_v`, `err_P`,
`converged::Bool`, `reached_total_iter::Bool`, and `history` (empty unless
`collect_history`).
"""
function solve_stokes_dyrel!(
    dr,
    mesh_stokes,
    cache::MixedMeshCache,
    element_v,
    element_P,
    phases_v,
    phases_P,
    τ_old,
    plastic,
    G,
    Δt,
    γP,
    Γnodes,
    bc_vx_vals,
    bc_vy_vals,
    backend,
    workgroup;
    kwargs...,
)
    return solve_stokes_dyrel!(
        dr, mesh_stokes, cache.geo_v, cache.geo_P, element_v, element_P,
        phases_v, phases_P, τ_old, plastic, G, Δt, γP,
        Γnodes, bc_vx_vals, bc_vy_vals, backend, workgroup;
        kwargs...,
    )
end

function solve_stokes_dyrel!(
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
    Γnodes,
    bc_vx_vals,
    bc_vy_vals,
    backend,
    workgroup;
    ncheck = 50,
    ϵ_tol = 1.0e-6,
    iterMax = 50_000,
    total_iterMax = 50_000,
    max_ph_iterations = 1000,
    rel_drop0 = 1.0e-2,
    verbose = true,
    verbose_inner = false,
    verbose_PH = nothing,
    verbose_DR = nothing,
    vx_nodes = Γnodes,
    vy_nodes = Γnodes,
    collect_history = false,
    measure_λmax = false,
    λmax_power_iterations = 12,
    λmax_power_rtol = 1.0e-2,
    λmax_safety = 1.1,
    freeze_jacobian = plastic === nothing,
)
    verbose, verbose_inner = _normalize_stokes_verbose(verbose, verbose_inner, verbose_PH, verbose_DR)
    # Non-associated plastic tangents are non-normal, so the power estimate is
    # less predictive than for the symmetric viscous operator.
    spectral_safety = isnothing(plastic) ? λmax_safety : max(λmax_safety, 1.5)

    M_P = dr.M_P
    nout = ncheck
    zero_vx_bc = zero(bc_vx_vals)
    zero_vy_bc = zero(bc_vy_vals)

    fill!(dr.∂vx∂τ, 0)
    fill!(dr.∂vy∂τ, 0)
    fill!(dr.Rv_x0, 0)
    fill!(dr.Rv_y0, 0)

    velocity_op = if measure_λmax
        assemble_velocity_operator(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup)
    else
        assemble_augmented_momentum_jacobian_matrices_atomix!(
            dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P, phases_v, phases_P, τ_old, plastic,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb, Δt, γP, M_P,
            backend, workgroup)
        nothing
    end
    λmax_gershgorin = max(
        _checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "stokes vx"),
        _checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "stokes vy"))
    λmax_iterations = 0
    jacobian_assemblies = 1
    power_x = power_y = nothing
    if measure_λmax
        λmax_measured, λmax_iterations, power_x, power_y = estimate_velocity_λmax(
            velocity_op, mesh_stokes, element_v, dr.PC_vx, dr.PC_vy,
            vx_nodes, vy_nodes, backend, workgroup;
            max_iterations = λmax_power_iterations, rtol = λmax_power_rtol)
        λmax_vx = λmax_vy = min(λmax_gershgorin, spectral_safety * λmax_measured)
    else
        λmax_vx = _checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "stokes vx")
        λmax_vy = _checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "stokes vy")
    end
    Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
    Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v
    α_vx, β_vx = _stokes_cheb(Δτ_vx, zero(λmax_vx), dr.c_fact)
    α_vy, β_vy = _stokes_cheb(Δτ_vy, zero(λmax_vy), dr.c_fact)
    verbose && @info "Initial momentum preconditioner" λmax_vx λmax_vy Δτ_vx Δτ_vy

    err_min = Inf
    ϵ = eltype(dr.Rv_x)(ϵ_tol)
    err = 2 * ϵ
    err_abs = Inf
    err_rel = Inf
    err_v0 = 1.0
    err_P0 = 1.0
    err_v00 = 1.0
    err_v = Inf
    err_P = Inf
    err_v_rel = Inf
    err_P_rel = Inf
    iter = 0
    itPH_done = 0
    rel_drop = rel_drop0
    history = NamedTuple[]

    for itPH in 1:max_ph_iterations
        itPH_done = itPH

        assemble_pressure_residual_matrices_atomix!(
            dr.RP,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P,
            phases_P, dr.α, dr.ηb, Δt,
            backend, workgroup,
        )

        assemble_momentum_residual_matrices_atomix!(
            dr.Rv_x, dr.Rv_y,
            dr.vx, dr.vy, dr.P, dr.T, nothing,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
            element_v, element_P,
            phases_v, τ_old, plastic, nothing, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            backend, workgroup,
        )
        apply_dirichlet!(dr.Rv_x, vx_nodes, zero_vx_bc, backend, workgroup)
        apply_dirichlet!(dr.Rv_y, vy_nodes, zero_vy_bc, backend, workgroup)

        err_P = norm(dr.RP ./ M_P) / sqrt(mesh_stokes.nnodesP)
        err_v = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * sqrt(mesh_stokes.nnodes))
        if itPH == 1
            err_P0 = err_P + eps(err_P)
            err_v0 = err_v + eps(err_v)
        elseif itPH == 2
            err_P0 = err_P + eps(err_P)
        end
        err_v_rel = err_v / err_v0
        err_P_rel = err_P / err_P0
        err_abs = max(err_v, err_P)
        err_rel = max(err_v_rel, err_P_rel)
        err = min(err_abs, err_rel)

        isnan(err) && error("NaN detected in outer loop at PH=$itPH")
        err > 1e10 && error("Kaboom! Error > 1e10 in outer loop at PH=$itPH")

        if verbose
            @printf("itPH = %02d iter = %06d err = %.3e abs = %.3e rel = %.3e - norm[Rv=%.3e %.3e, Rp=%.3e %.3e]\n",
                itPH, iter, err, err_abs, err_rel, err_v, err_v_rel, err_P, err_P_rel)
        end
        err < ϵ && break

        if err > err_min * 1.05
            rel_drop = max(rel_drop * 0.1, 1e-3)
        end
        err_min = min(err_min, err)

        ϵ_vel = err * rel_drop
        itPT = 0

        while err > ϵ_vel && itPT ≤ iterMax
            itPT += 1
            iter += 1
            do_check = iszero(iter % nout)

            if do_check
                copyto!(dr.Rv_x0, dr.Rv_x)
                copyto!(dr.Rv_y0, dr.Rv_y)
            end

            assemble_pressure_residual_matrices_atomix!(
                dr.RP,
                dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                element_v, element_P,
                phases_P, dr.α, dr.ηb, Δt,
                backend, workgroup,
            )

            @. dr.Pnum = γP * dr.RP / M_P

            assemble_momentum_residual_matrices_atomix!(
                dr.Rv_x, dr.Rv_y,
                dr.vx, dr.vy, dr.P, dr.T, dr.Pnum,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                element_v, element_P,
                phases_v, τ_old, plastic, nothing, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                backend, workgroup,
            )

            apply_dirichlet!(dr.Rv_x, vx_nodes, zero_vx_bc, backend, workgroup)
            apply_dirichlet!(dr.∂vx∂τ, vx_nodes, zero_vx_bc, backend, workgroup)
            apply_dirichlet!(dr.Rv_y, vy_nodes, zero_vy_bc, backend, workgroup)
            apply_dirichlet!(dr.∂vy∂τ, vy_nodes, zero_vy_bc, backend, workgroup)

            update_stokes_velocity!(
                dr.∂vx∂τ, dr.∂vy∂τ, dr.vx, dr.vy,
                dr.Rv_x, dr.Rv_y, dr.PC_vx, dr.PC_vy,
                β_vx, β_vy, -α_vx, -α_vy,
                mesh_stokes.nnodes, backend, workgroup)

            apply_dirichlet!(dr.vx, vx_nodes, bc_vx_vals, backend, workgroup)
            apply_dirichlet!(dr.vy, vy_nodes, bc_vy_vals, backend, workgroup)

            if do_check
                err_v_inner = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * sqrt(mesh_stokes.nnodes))
                if iter == nout
                    err_v00 = err_v_inner + eps(err_v_inner)
                end
                err = max(err_v_inner / err_v00, err_v_inner)
                isnan(err) && error("NaN detected in inner loop PH=$itPH PT=$itPT")
                err > 1e10 && error("Kaboom! Error > 1e10 in inner loop PH=$itPH PT=$itPT")

                collect_history && push!(history, (; iter, err_v = err_v_inner, err_P))

                verbose_inner && @printf("  it = %d, iter = %d, err = %.3e\n", itPT, iter, err)

                λmin_vx = _stokes_λmin(α_vx, dr.∂vx∂τ, dr.Rv_x .- dr.Rv_x0, dr.PC_vx)
                λmin_vy = _stokes_λmin(α_vy, dr.∂vy∂τ, dr.Rv_y .- dr.Rv_y0, dr.PC_vy)

                if !freeze_jacobian
                    jacobian_assemblies += 1
                    velocity_op = if measure_λmax
                        assemble_velocity_operator(
                            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                            phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup)
                    else
                        assemble_augmented_momentum_jacobian_matrices_atomix!(
                            dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
                            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                            element_v, element_P, phases_v, phases_P, τ_old, plastic,
                            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                            dr.ηb, Δt, γP, M_P, backend, workgroup)
                        nothing
                    end

                    λmax_gershgorin = max(
                        _checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "stokes vx"),
                        _checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "stokes vy"))
                    if measure_λmax
                        λmax_measured, power_iterations, power_x, power_y =
                            estimate_velocity_λmax(
                            velocity_op, mesh_stokes, element_v, dr.PC_vx, dr.PC_vy,
                            vx_nodes, vy_nodes, backend, workgroup;
                            max_iterations = λmax_power_iterations, rtol = λmax_power_rtol,
                            x = power_x, y = power_y)
                        λmax_iterations += power_iterations
                        λmax_vx = λmax_vy =
                            min(λmax_gershgorin, spectral_safety * λmax_measured)
                    else
                        λmax_vx = _checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "stokes vx")
                        λmax_vy = _checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "stokes vy")
                    end
                end
                Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
                Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v

                α_vx, β_vx = _stokes_cheb(Δτ_vx, λmin_vx, dr.c_fact)
                α_vy, β_vy = _stokes_cheb(Δτ_vy, λmin_vy, dr.c_fact)
            end

            itPT == iterMax && @printf("  inner: max iters (%d) reached at PH=%d\n", iterMax, itPH)
            iter > total_iterMax && break
        end

        @. dr.P += γP * dr.RP / M_P
        # remove_pressure_mean!(dr.P, M_P)

        iter > total_iterMax && break
    end

    return (;
        itPH = itPH_done,
        iter,
        err,
        err_abs,
        err_rel,
        err_v,
        err_P,
        converged = err < ϵ,
        reached_total_iter = iter > total_iterMax,
        λmax = max(λmax_vx, λmax_vy),
        λmax_gershgorin,
        λmax_iterations,
        jacobian_assemblies,
        history,
    )
end

"""
    update_stokes_current_stress!(dr, mesh, cache, τ, Δt;
                                  plastic=nothing, workgroup=256)

Refresh integration-point stresses using cache-owned elements, solver-owned
material and stress history, and optional phase-layout overrides.
"""
function update_stokes_current_stress!(
    dr::StokesDR,
    mesh::MixedMesh,
    cache::MixedMeshCache,
    τ,
    Δt;
    plastic = nothing,
    phases_v = dr.phases_v,
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old),
    workgroup = 256,
)
    backend = KA.get_backend(mesh.coords)
    return update_stokes_current_stress!(
        dr, mesh, cache, cache.element_v, cache.element_P,
        phases_v, τ_old, plastic, τ, dr.G, Δt, backend, workgroup,
    )
end

"""
    update_stokes_current_stress!(dr, mesh_stokes, cache, element_v, element_P,
                                  phases_v, τ_old, plastic, τ, G, Δt,
                                  backend, workgroup)

Refresh the current integration-point stress tuple `τ` by running the momentum
residual assembly with stress storage enabled.
"""
function update_stokes_current_stress!(
    dr,
    mesh_stokes,
    cache::MixedMeshCache,
    element_v,
    element_P,
    phases_v,
    τ_old,
    plastic,
    τ,
    G,
    Δt,
    backend,
    workgroup,
)
    return update_stokes_current_stress!(
        dr, mesh_stokes, cache.geo_v, element_v, element_P,
        phases_v, τ_old, plastic, τ, G, Δt, backend, workgroup,
    )
end

function update_stokes_current_stress!(
    dr,
    mesh_stokes,
    geo_v,
    element_v,
    element_P,
    phases_v,
    τ_old,
    plastic,
    τ,
    G,
    Δt,
    backend,
    workgroup,
)
    assemble_momentum_residual_matrices_atomix!(
        dr.Rv_x, dr.Rv_y,
        dr.vx, dr.vy, dr.P, dr.T, nothing,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
        element_v, element_P,
        phases_v, τ_old, plastic, τ, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
        backend, workgroup,
    )
    return τ
end
