"""
    _stokes_λmin(step, rate, R, R0, PC) -> λmin

Estimate the smallest eigenvalue of the preconditioned operator from the change
in residual over one pseudo-transient step.

`R` and `R0` are the current residual and the snapshot taken at the previous
estimate; their difference is formed inside the reduction rather than as an
array, so a convergence check costs no temporaries.
"""
function _stokes_λmin(step, rate, R, R0, PC)
    denom = fused_sum(r -> (step * r)^2, rate)
    iszero(denom) && return zero(denom)
    num = fused_sum((r, x, x0, pc) -> (step * r) * ((x - x0) / pc), rate, R, R0, PC)
    return abs(num) / denom
end

function _stokes_cheb(Δτ, λmin, c_fact)
    c = 2 * sqrt(λmin) * c_fact
    return (2 * Δτ^2 / (2 + c * Δτ), (2 - c * Δτ) / (2 + c * Δτ))
end

@kernel function _transfer_temperature_kernel!(T_P, @Const(T), @Const(DoFsP), @Const(el2nP))
    i = @index(Global, Linear)
    T_P[DoFsP[i]] = T[el2nP[i]]
end

function _transfer_temperature!(T_P, T, mesh::MixedMesh, backend, workgroup)
    return launch!(
        _transfer_temperature_kernel!, backend, workgroup, length(mesh.DoFsP),
        T_P, T, mesh.DoFsP, mesh.el2nP,
    )
end

"""
    solve_coupled_dyrel!(thermal, stokes, thermal_mesh, stokes_mesh, cache,
                         bc_T, bc_vx, bc_vy, Δt, γP;
                         Tref=273, workgroup=256, kwargs...)

Solve one coupled thermal--Stokes time step. Each inner Stokes velocity
iteration advances one thermal dynamic-relaxation iteration, then gathers the
continuous thermal field at `stokes_mesh.el2nP` onto the discontinuous pressure
DoFs used by the Stokes residuals.

`thermal_mesh` must use the same node numbering and backend as the primary
field of `stokes_mesh`. The previous-time fields `thermal.T0`, `stokes.P0`, and
the Stokes stress history remain caller-owned. Remaining keywords are forwarded
to [`solve_stokes_dyrel!`](@ref); `ncheck` controls convergence checks for both
relaxation updates. Returns the Stokes statistics with additional `err_T` and
`thermal_iterations` fields.
"""
function solve_coupled_dyrel!(
    thermal::ThermalDiffusionDR,
    stokes::StokesDR,
    thermal_mesh::Mesh,
    stokes_mesh::MixedMesh,
    cache::MixedMeshCache,
    bc_T::DirichletBoundaryCondition,
    bc_vx::DirichletBoundaryCondition,
    bc_vy::DirichletBoundaryCondition,
    Δt,
    γP;
    Tref = eltype(thermal.T)(273),
    workgroup = 256,
    kwargs...,
)
    isnothing(thermal_mesh.geometry) && throw(ArgumentError(
        "thermal mesh has no geometry; construct it with Mesh(backend, coords, el2n, element)"))
    thermal_mesh.nnodes == stokes_mesh.nnodes || throw(DimensionMismatch(
        "thermal mesh nodes must match the Stokes velocity-node layout"))
    length(thermal.T) == thermal_mesh.nnodes || throw(DimensionMismatch(
        "thermal state size must match thermal mesh nodes"))
    length(stokes.T) == stokes_mesh.nnodesP || throw(DimensionMismatch(
        "Stokes temperature size must match pressure DoFs"))

    backend = KA.get_backend(stokes_mesh.coords)
    typeof(KA.get_backend(thermal_mesh.coords)) === typeof(backend) || throw(ArgumentError(
        "thermal and Stokes meshes must use the same backend"))
    _transfer_temperature!(stokes.T, thermal.T, stokes_mesh, backend, workgroup)
    _transfer_temperature!(stokes.T0, thermal.T0, stokes_mesh, backend, workgroup)

    coupled = (; dr = thermal, mesh = thermal_mesh, bc = bc_T, Tref)
    return solve_stokes_dyrel!(
        stokes, stokes_mesh, cache, bc_vx, bc_vy, Δt, γP;
        workgroup, _thermal = coupled, kwargs...,
    )
end

"""
    solve_stokes_dyrel!(dr, mesh, cache, bc_vx, bc_vy, Δt, γP;
                        plastic=nothing, workgroup=256, kwargs...)

Solve the Stokes system using geometry and elements from `cache`, material and
stress history from `dr`, and one Dirichlet boundary-condition object per
velocity component. Phase layouts and stress history may be overridden with
the `phases_v`, `phases_P`, and `τ_old` keywords.

This mixed-mesh method is two-dimensional and takes `StokesDR{<:Any, 2}`; a
three-dimensional state is a `MethodError` rather than a solve that ignores the
third component.
"""
function solve_stokes_dyrel!(
    dr::StokesDR{<:Any, 2},
    mesh::MixedMesh,
    cache::MixedMeshCache,
    bc_vx::DirichletBoundaryCondition,
    bc_vy::DirichletBoundaryCondition,
    Δt,
    γP;
    plastic = nothing,
    phases_v = dr.phases_v,
    phases_P = dr.phases_P,
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy),
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
    vx_nodes = Γnodes,
    vy_nodes = Γnodes,
    collect_history = false,
    measure_λmax = false,
    λmax_power_iterations = 12,
    λmax_power_rtol = 1.0e-2,
    λmax_safety = 1.1,
    freeze_jacobian = plastic === nothing,
    _thermal = nothing,
)
    verbose, verbose_inner = Bool(verbose), Bool(verbose_inner)
    # Non-associated plastic tangents are non-normal, so the power estimate is
    # less predictive than for the symmetric viscous operator.
    spectral_safety = isnothing(plastic) ? λmax_safety : max(λmax_safety, 1.5)

    M_P = dr.M_P
    nout = ncheck
    zero_vx_bc = zero(bc_vx_vals)
    zero_vy_bc = zero(bc_vy_vals)

    fill!(dr.∂v∂τ.x, 0)
    fill!(dr.∂v∂τ.y, 0)
    fill!(dr.Rv0.x, 0)
    fill!(dr.Rv0.y, 0)

    # Shared by every element, so the residual kernels read them from the backend
    # rather than carry them in the argument pack of each of their launches.
    valNV, valNP = local_nodes_val(element_v), local_nodes_val(element_P)
    Nq_v = quadrature_table(backend, shape_function_values(element_v))
    Nq_P = quadrature_table(backend, shape_function_values(element_P, element_v.integration_points))
    ∂N∂ξ_v = quadrature_table(backend, shape_function_gradients(element_v))

    velocity_op = if measure_λmax
        assemble_velocity_operator(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup)
    else
        assemble_augmented_momentum_jacobian_matrices_atomix!(
            dr.∂Rv∂v.x, dr.PC_v.x, dr.∂Rv∂v.y, dr.PC_v.y,
            dr.v.x, dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P, phases_v, phases_P,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb, Δt, γP, M_P,
            backend, workgroup; τ_old, plastic)
        nothing
    end
    λmax_gershgorin = max(
        _checked_λmax(dr.∂Rv∂v.x, dr.PC_v.x, "stokes vx"),
        _checked_λmax(dr.∂Rv∂v.y, dr.PC_v.y, "stokes vy"))
    λmax_iterations = 0
    jacobian_assemblies = 1
    power_x = power_y = nothing
    if measure_λmax
        λmax_measured, λmax_iterations, power_x, power_y = estimate_velocity_λmax(
            velocity_op, mesh_stokes, element_v, dr.PC_v.x, dr.PC_v.y,
            vx_nodes, vy_nodes, backend, workgroup;
            max_iterations = λmax_power_iterations, rtol = λmax_power_rtol)
        λmax_vx = λmax_vy = min(λmax_gershgorin, spectral_safety * λmax_measured)
    else
        λmax_vx = _checked_λmax(dr.∂Rv∂v.x, dr.PC_v.x, "stokes vx")
        λmax_vy = _checked_λmax(dr.∂Rv∂v.y, dr.PC_v.y, "stokes vy")
    end
    Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
    Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v
    α_vx, β_vx = _stokes_cheb(Δτ_vx, zero(λmax_vx), dr.c_fact)
    α_vy, β_vy = _stokes_cheb(Δτ_vy, zero(λmax_vy), dr.c_fact)
    verbose && @info "Initial momentum preconditioner" λmax_vx λmax_vy Δτ_vx Δτ_vy

    err_min = Inf
    ϵ = eltype(dr.Rv.x)(ϵ_tol)
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

    thermal_iter = 0
    err_T = isnothing(_thermal) ? nothing : Inf
    thermal_converged = isnothing(_thermal)
    α_T = β_T = λmax_T = thermal_nr0 = zero(eltype(dr.Rv.x))
    if !isnothing(_thermal)
        fill!(_thermal.dr.∂T∂τ, 0)
        fill!(_thermal.dr.R0, 0)
    end

    for itPH in 1:max_ph_iterations
        itPH_done = itPH

        assemble_pressure_residual_kernel!(
            dr.RP,
            dr.v.x, dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            phases_P, dr.α, dr.ηb, Δt, Nq_P, ∂N∂ξ_v,
            valNV, valNP, workgroup,
        )

        assemble_momentum_residual_kernel!(
            dr.Rv.x, dr.Rv.y,
            dr.v.x, dr.v.y, dr.P, dr.T, nothing,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
            phases_v, τ_old, plastic, nothing,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            Nq_v, Nq_P, ∂N∂ξ_v, valNV, valNP, workgroup,
        )
        apply_dirichlet!(dr.Rv.x, vx_nodes, zero_vx_bc, backend, workgroup)
        apply_dirichlet!(dr.Rv.y, vy_nodes, zero_vy_bc, backend, workgroup)

        err_P = norm(dr.RP ./ M_P) / sqrt(mesh_stokes.nnodesP)
        err_v = max(norm(dr.Rv.x), norm(dr.Rv.y)) / (2 * sqrt(mesh_stokes.nnodes))
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
        err < ϵ && thermal_converged && break

        if err > err_min * 1.05
            rel_drop = max(rel_drop * 0.1, 1e-3)
        end
        err_min = min(err_min, err)

        ϵ_vel = err * rel_drop
        itPT = 0

        while (err > ϵ_vel || !thermal_converged) && itPT ≤ iterMax
            itPT += 1
            iter += 1
            do_check = iszero(iter % nout)

            if do_check
                copyto!(dr.Rv0.x, dr.Rv.x)
                copyto!(dr.Rv0.y, dr.Rv.y)
            end

            if !isnothing(_thermal)
                thermal_iter += 1
                thermal_check = thermal_iter == 1 || iszero(thermal_iter % nout)
                thermal_check && copyto!(_thermal.dr.R0, _thermal.dr.R)
                _assemble_thermal!(
                    _thermal.dr, Δt, _thermal.mesh, _thermal.mesh.geometry,
                    _thermal.mesh.element, _thermal.Tref, backend, workgroup,
                    thermal_check,
                )
                apply_dirichlet!(
                    _thermal.dr.R, _thermal.bc.DoFs, _thermal.bc.zero_vals,
                    backend, workgroup,
                )
                apply_dirichlet!(
                    _thermal.dr.∂T∂τ, _thermal.bc.DoFs, _thermal.bc.zero_vals,
                    backend, workgroup,
                )
                thermal_check && (λmax_T = _checked_λmax(
                    _thermal.dr.∂R∂T, _thermal.dr.PC, "thermal diffusion"))
                update_rate_kernel!(backend, workgroup)(
                    _thermal.dr.∂T∂τ, _thermal.dr.R, _thermal.dr.PC, β_T;
                    ndrange = _thermal.mesh.nnodes,
                )
                update_variable_kernel!(backend, workgroup)(
                    _thermal.dr.T, _thermal.dr.∂T∂τ, α_T;
                    ndrange = _thermal.mesh.nnodes,
                )
                apply_dirichlet!(
                    _thermal.dr.T, _thermal.bc.DoFs, _thermal.bc.vals,
                    backend, workgroup,
                )
                _transfer_temperature!(dr.T, _thermal.dr.T, mesh_stokes, backend, workgroup)

                if thermal_check
                    nr_T = norm(_thermal.dr.R)
                    thermal_iter == 1 && (thermal_nr0 = max(nr_T, eps(nr_T)))
                    err_T = nr_T / thermal_nr0
                    isfinite(err_T) || error(
                        "non-finite thermal residual at coupled iteration $thermal_iter")
                    Δτ_T = 2 / sqrt(λmax_T) * _thermal.dr.CFL
                    denom_T = fused_sum(r -> (Δτ_T * r)^2, _thermal.dr.∂T∂τ)
                    λmin_T = (thermal_iter == 1 || iszero(denom_T)) ?
                        zero(eltype(_thermal.dr.R)) :
                        abs(fused_sum(
                            (r, x, x0, pc) -> (Δτ_T * r) * ((x - x0) / pc),
                            _thermal.dr.∂T∂τ, _thermal.dr.R, _thermal.dr.R0, _thermal.dr.PC,
                        )) / denom_T
                    α_T, β_T = _stokes_cheb(Δτ_T, λmin_T, _thermal.dr.c_fact)
                    thermal_converged = err_T < _thermal.dr.ϵ
                end
            end

            assemble_pressure_residual_kernel!(
                dr.RP,
                dr.v.x, dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                phases_P, dr.α, dr.ηb, Δt, Nq_P, ∂N∂ξ_v,
                valNV, valNP, workgroup,
            )

            @. dr.Pnum = γP * dr.RP / M_P

            assemble_momentum_residual_kernel!(
                dr.Rv.x, dr.Rv.y,
                dr.v.x, dr.v.y, dr.P, dr.T, dr.Pnum,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                phases_v, τ_old, plastic, nothing,
                dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                Nq_v, Nq_P, ∂N∂ξ_v, valNV, valNP, workgroup,
            )

            apply_dirichlet!(dr.Rv.x, vx_nodes, zero_vx_bc, backend, workgroup)
            apply_dirichlet!(dr.∂v∂τ.x, vx_nodes, zero_vx_bc, backend, workgroup)
            apply_dirichlet!(dr.Rv.y, vy_nodes, zero_vy_bc, backend, workgroup)
            apply_dirichlet!(dr.∂v∂τ.y, vy_nodes, zero_vy_bc, backend, workgroup)

            update_stokes_velocity!(
                dr.∂v∂τ.x, dr.∂v∂τ.y, dr.v.x, dr.v.y,
                dr.Rv.x, dr.Rv.y, dr.PC_v.x, dr.PC_v.y,
                β_vx, β_vy, -α_vx, -α_vy,
                mesh_stokes.nnodes, backend, workgroup)

            apply_dirichlet!(dr.v.x, vx_nodes, bc_vx_vals, backend, workgroup)
            apply_dirichlet!(dr.v.y, vy_nodes, bc_vy_vals, backend, workgroup)

            if do_check
                err_v_inner = max(norm(dr.Rv.x), norm(dr.Rv.y)) / (2 * sqrt(mesh_stokes.nnodes))
                if iter == nout
                    err_v00 = err_v_inner + eps(err_v_inner)
                end
                err = max(err_v_inner / err_v00, err_v_inner)
                isnan(err) && error("NaN detected in inner loop PH=$itPH PT=$itPT")
                err > 1e10 && error("Kaboom! Error > 1e10 in inner loop PH=$itPH PT=$itPT")

                collect_history && push!(history, (; iter, err_v = err_v_inner, err_P))

                verbose_inner && @printf("  it = %d, iter = %d, err = %.3e\n", itPT, iter, err)

                λmin_vx = _stokes_λmin(α_vx, dr.∂v∂τ.x, dr.Rv.x, dr.Rv0.x, dr.PC_v.x)
                λmin_vy = _stokes_λmin(α_vy, dr.∂v∂τ.y, dr.Rv.y, dr.Rv0.y, dr.PC_v.y)

                if !freeze_jacobian
                    jacobian_assemblies += 1
                    velocity_op = if measure_λmax
                        assemble_velocity_operator(
                            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                            phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup)
                    else
                        assemble_augmented_momentum_jacobian_matrices_atomix!(
                            dr.∂Rv∂v.x, dr.PC_v.x, dr.∂Rv∂v.y, dr.PC_v.y,
                            dr.v.x, dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
                            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                            element_v, element_P, phases_v, phases_P,
                            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                            dr.ηb, Δt, γP, M_P, backend, workgroup; τ_old, plastic)
                        nothing
                    end

                    λmax_gershgorin = max(
                        _checked_λmax(dr.∂Rv∂v.x, dr.PC_v.x, "stokes vx"),
                        _checked_λmax(dr.∂Rv∂v.y, dr.PC_v.y, "stokes vy"))
                    if measure_λmax
                        λmax_measured, power_iterations, power_x, power_y =
                            estimate_velocity_λmax(
                            velocity_op, mesh_stokes, element_v, dr.PC_v.x, dr.PC_v.y,
                            vx_nodes, vy_nodes, backend, workgroup;
                            max_iterations = λmax_power_iterations, rtol = λmax_power_rtol,
                            x = power_x, y = power_y)
                        λmax_iterations += power_iterations
                        λmax_vx = λmax_vy =
                            min(λmax_gershgorin, spectral_safety * λmax_measured)
                    else
                        λmax_vx = _checked_λmax(dr.∂Rv∂v.x, dr.PC_v.x, "stokes vx")
                        λmax_vy = _checked_λmax(dr.∂Rv∂v.y, dr.PC_v.y, "stokes vy")
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

    stats = (;
        itPH = itPH_done,
        iter,
        err,
        err_abs,
        err_rel,
        err_v,
        err_P,
        converged = err < ϵ && thermal_converged,
        reached_total_iter = iter > total_iterMax,
        λmax = max(λmax_vx, λmax_vy),
        λmax_gershgorin,
        λmax_iterations,
        jacobian_assemblies,
        history,
    )
    return isnothing(_thermal) ? stats : merge(stats, (; err_T, thermal_iterations = thermal_iter))
end

"""
    update_stokes_current_stress!(dr, mesh, cache, τ, Δt;
                                  plastic=nothing, workgroup=256)

Refresh integration-point stresses using cache-owned elements, solver-owned
material and stress history, and optional phase-layout overrides. `τ` receives
the in-plane components in Voigt order; the history defaults to the components
of `dr.τ_old`. Two-dimensional states only.
"""
function update_stokes_current_stress!(
    dr::StokesDR{<:Any, 2},
    mesh::MixedMesh,
    cache::MixedMeshCache,
    τ,
    Δt;
    plastic = nothing,
    phases_v = dr.phases_v,
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy),
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
        dr.Rv.x, dr.Rv.y,
        dr.v.x, dr.v.y, dr.P, dr.T, nothing,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
        element_v, element_P,
        phases_v, τ_old, plastic, τ, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
        backend, workgroup,
    )
    return τ
end

"""
    solve_stokes_dyrel!(velocity, pressure, mesh, cell_phase, η, ρ, g,
                        fixed_nodes; kwargs...)

Solve the viscous 3-D Hex27/Q2--P1 Stokes system through the same public
DYREL solver entry point used by the mixed 2-D solver. Multiple dispatch keeps
the cell-local four-mode pressure storage native to this discretization.
`fixed_nodes` contains constrained nodes for each velocity component;
velocity and pressure are updated in place.

`ncheck` controls residual checks, `ϵ_tol` the absolute combined tolerance,
and `total_iterMax` the iteration budget (`iterMax` supplies its default).
`velocity_step` and `γP` scale the velocity and pressure updates. `bc_values`
prescribes the constrained velocities, one value per entry of `fixed_nodes`;
`nothing` holds them all at zero. `load`, when provided, is an `NTuple{3}`
momentum right-hand side used by the adjoint method.
`workspace` supplies caller-owned scratch; pass a [`Stokes3DWorkspace`](@ref) to
reuse it across calls instead of allocating a mesh-sized set of arrays each time.
Returns convergence statistics including `iter`, `err`, `err_v`, `err_P`,
`converged`, and `reached_total_iter`.
"""
function solve_stokes_dyrel!(
    velocity::NTuple{3}, pressure::AbstractMatrix, mesh::Mesh, cell_phase,
    η, ρ, g::NTuple{3}, fixed_nodes::NTuple{3};
    ncheck = 100, ϵ_tol = 1e-5, iterMax = 3000, total_iterMax = iterMax,
    velocity_step = 0.6, γP = 0.2, bc_values = nothing, load = nothing,
    workgroup = 256, verbose = true,
    workspace = Stokes3DWorkspace(velocity, pressure, mesh, fixed_nodes),
)
    all(length(workspace.zero_bc[i]) == length(fixed_nodes[i]) for i in 1:3) ||
        throw(DimensionMismatch(
            "workspace was built for a different set of constrained nodes: " *
            "$(map(length, workspace.zero_bc)) vs $(map(length, fixed_nodes))"))
    residual_v, residual_p = workspace.residual_v, workspace.residual_p
    diagonal, pressure_mass = workspace.diagonal, workspace.pressure_mass
    zero_bc, tables = workspace.zero_bc, workspace.tables
    stokes_preconditioner_3d!(diagonal, pressure_mass, mesh, cell_phase, η; workgroup, tables)
    backend = KA.get_backend(first(velocity))
    # The residual is always zeroed on constrained nodes, whereas the velocity is
    # reset to its prescribed value there, which is zero only for `bc_values === nothing`.
    velocity_bc = if bc_values === nothing
        zero_bc
    else
        ntuple(3) do i
            length(bc_values[i]) == length(fixed_nodes[i]) ||
                throw(DimensionMismatch("bc_values[$i] has $(length(bc_values[i])) entries but fixed_nodes[$i] has $(length(fixed_nodes[i]))"))
            bc_values[i]
        end
    end
    # The first residual must see the prescribed boundary velocities too.
    for component in 1:3
        apply_dirichlet!(velocity[component], fixed_nodes[component], velocity_bc[component], backend, workgroup)
    end
    err_v = err_P = err = Inf
    iter = 0
    for iteration in 1:total_iterMax
        iter = iteration
        assemble_stokes_pressure_residual_3d!(residual_p, velocity, mesh; workgroup, tables)
        @. pressure += γP * residual_p / pressure_mass
        pmean = fused_sum(*, @view(pressure[1, :]), @view(pressure_mass[1, :])) /
                sum(@view pressure_mass[1, :])
        @views pressure[1, :] .-= pmean
        assemble_stokes_momentum_residual_3d!(
            residual_v, velocity, pressure, mesh, cell_phase, η, ρ, g; workgroup, tables,
        )
        if load !== nothing
            for component in 1:3
                @. residual_v[component] -= load[component]
            end
        end
        for component in 1:3
            apply_dirichlet!(residual_v[component], fixed_nodes[component], zero_bc[component], backend, workgroup)
            @. velocity[component] -= velocity_step * residual_v[component] / diagonal[component]
            apply_dirichlet!(velocity[component], fixed_nodes[component], velocity_bc[component], backend, workgroup)
        end
        (iszero(iter % ncheck) || iter == total_iterMax) || continue
        err_v = maximum(norm, residual_v)
        err_P = norm(residual_p)
        err = max(err_v, err_P)
        isfinite(err) || error("non-finite residual in 3D DYREL solve")
        verbose && @printf("iter = %06d err = %.3e - norm[Rv=%.3e, Rp=%.3e]\n",
            iter, err, err_v, err_P)
        err < ϵ_tol && break
    end
    return (; itPH = 1, iter, iterations = iter, err, err_v, err_P, err_p = err_P,
        converged = err < ϵ_tol, reached_total_iter = iter >= total_iterMax && err >= ϵ_tol)
end

"""
    solve_stokes_3d!(velocity, pressure, mesh, cell_phase, η, ρ, g,
                     fixed_nodes; maxiter=3000, tolerance=1e-5,
                     pressure_step=0.2, kwargs...)

Compatibility wrapper for the 3-D [`solve_stokes_dyrel!`](@ref) method.
`maxiter`, `tolerance`, and `pressure_step` map to `total_iterMax`, `ϵ_tol`,
and `γP`, respectively.
"""
function solve_stokes_3d!(
    velocity::NTuple{3}, pressure::AbstractMatrix, mesh::Mesh, cell_phase,
    η, ρ, g::NTuple{3}, fixed_nodes::NTuple{3};
    maxiter = 3000, tolerance = 1e-5, pressure_step = 0.2, kwargs...,
)
    return solve_stokes_dyrel!(
        velocity, pressure, mesh, cell_phase, η, ρ, g, fixed_nodes;
        iterMax = maxiter, total_iterMax = maxiter, ϵ_tol = tolerance,
        γP = pressure_step, verbose = false, kwargs...,
    )
end

"""
    solve_stokes_adjoint_3d!(velocity, pressure, objective_load, mesh,
                             cell_phase, η, fixed_nodes; maxiter=3000,
                             tolerance=1e-5, pressure_step=0.2, kwargs...)

Solve the transpose of the linear viscous 3-D Stokes operator. The operator is
symmetric, so this compatibility wrapper forwards to the 3-D
[`solve_stokes_adjoint_dyrel!`](@ref) method.
The compatibility keywords map to `total_iterMax`, `adjoint_tol`, and `γP`.
"""
function solve_stokes_adjoint_3d!(
    velocity::NTuple{3}, pressure::AbstractMatrix, objective_load::NTuple{3},
    mesh::Mesh, cell_phase, η, fixed_nodes::NTuple{3};
    maxiter = 3000, tolerance = 1e-5, pressure_step = 0.2, kwargs...,
)
    return solve_stokes_adjoint_dyrel!(
        velocity, pressure, objective_load, mesh, cell_phase, η, fixed_nodes;
        iterMax = maxiter, total_iterMax = maxiter, adjoint_tol = tolerance,
        γP = pressure_step, verbose = false, kwargs...,
    )
end
