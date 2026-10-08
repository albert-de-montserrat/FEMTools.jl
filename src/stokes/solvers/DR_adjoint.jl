"""
    solve_stokes_adjoint_dyrel!(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                                phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                                objective_v, λv, λP, backend, workgroup;
                                v_nodes, kwargs...) -> NamedTuple
    solve_stokes_adjoint_dyrel!(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                                phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                                objective_vx, objective_vy, λvx, λvy, λP,
                                backend, workgroup; vx_nodes, vy_nodes, kwargs...)

Solve the discrete Stokes adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` with the same
Powell-Hestenes / DYREL iteration used by [`solve_stokes_dyrel!`](@ref), and
store the adjoint fields in `λv` (one array per direction) and `λP`, modified in
place. The second form is the plane-strain spelling with the components passed
separately.

The adjoint is assembled on the *same* mixed velocity/pressure spaces, quadrature,
and element operators as the forward problem and transposed exactly. With this
method's sign convention, the total derivative is `∂J/∂m + λᵀ ∂R/∂m`, where `R`
is the residual the forward solve drives to zero with the Powell-Hestenes
augmentation folded into the momentum residual as `Pnum = γP·RP/M_P`. Its forward
state must already be converged: the transpose Jacobian, its diagonal
preconditioner, and λmax are frozen at that state, so only λmin (hence the
Chebyshev pair) is re-estimated during the solve.

`objective_v` carries the velocity part of `∂J/∂u`, one array per direction (the
consistently assembled objective load); the pressure adjoint has no explicit
objective term. `M_P = dr.M_P` and the augmentation scaling `γP` must match the
forward solve. Homogeneous Dirichlet conditions are applied to each adjoint
velocity component on the matching entry of `v_nodes`.

The input values of `λv` and `λP` are preserved as the initial iterate.
Pass zero-filled arrays for a cold solve, or fields from the previous design
iteration to warm-start an optimization loop.

Pass a caller-owned [`StokesAdjointWorkspace`](@ref) with the `workspace`
keyword to reuse mesh-sized scratch across solves. The default constructs a
fresh workspace for compatibility. A workspace used with `operator = :enzyme`
must have been constructed with `enzyme=true`.

Because the forward state is frozen, the adjoint residual is affine in `λ` with a
constant operator, and `operator` chooses how that operator is applied:

- `:blocks` (the default) assembles it once as per-element blocks and applies
  them as dense element products, so no rheology is evaluated and no primal
  residual is recomputed during the solve. See [`FrozenAdjointOperator`](@ref)
  for the blocks and their memory cost. It is the only operator in three
  dimensions, and the only one that supports a plastic model together with a
  nonzero fluid pressure `dr.Pf`.
- `:matrix_free` stores nothing per element and rebuilds the same products by
  forward-mode directional differentiation on every apply, at roughly three
  element residual evaluations apiece. It requires `plastic === nothing`, whose
  symmetric tangent is what lets a directional derivative stand in for a
  transposed product; see [`MatrixFreeAdjointOperator`](@ref).
- `:enzyme` also stores nothing per element and rebuilds the products by
  reverse-mode differentiation, three sweeps per apply. It is the slowest of the
  three and the only one that handles a plastic tangent without stored blocks.

With `measure_λmax` (the default, and unavailable with `operator = :enzyme`) the
largest eigenvalue of the preconditioned velocity block is measured by power
iteration rather than bounded by Gershgorin row sums. The bound is correct but
loose, and since `Δτ = 2/sqrt(λmax)·CFL_v` a loose bound shortens every step.

Use `verbose` for outer Powell-Hestenes progress and `verbose_inner` for the
inner dynamic-relaxation trace. Returns a `NamedTuple` with `itPH`, `iter`,
`err`, `err_v`, `err_P`, `converged`, the `λmax` actually used alongside the
`λmax_gershgorin` bound and the `λmax_iterations` spent measuring it, and (when
`collect_history`) `history`.

All arrays read or written by kernels—including `mesh_stokes` connectivity,
`geo_v`, `geo_P`, phases, objective loads, adjoint fields, and boundary-node
arrays—must reside on `backend`. Construct unstructured meshes with
`Mesh(backend, coords, el2n, element_v)`; a `MixedMesh` built from it computes
`mesh.geometry` on the same backend. The Enzyme transpose assemblers execute on the backend inferred
from their output buffers.
"""
function solve_stokes_adjoint_dyrel!(
        dr::StokesDR{<:Any, D},
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
        objective_v::NTuple{D, AbstractVector},
        λv::NTuple{D, AbstractVector},
        λP,
        backend,
        workgroup;
        v_nodes::NTuple{D},
        ncheck = 50,
        adjoint_tol = 1.0e-6,
        rel_drop = 0.1,
        iterMax = 50_000,
        total_iterMax = 50_000,
        max_ph_iterations = 100,
        verbose = true,
        verbose_inner = false,
        collect_history = false,
        operator = :blocks,
        measure_λmax = true,
        workspace = StokesAdjointWorkspace(dr, v_nodes...; enzyme = operator === :enzyme),
    ) where {D}
    operator in (:blocks, :matrix_free, :enzyme) || throw(
        ArgumentError(
            "operator must be :blocks, :matrix_free, or :enzyme, got $(repr(operator))"
        )
    )
    D == 2 || operator === :blocks || throw(
        ArgumentError("operator = $(repr(operator)) is implemented in two dimensions only; use :blocks")
    )
    # Only the stored blocks differentiate the yield model at the effective pressure.
    operator === :blocks || plastic === nothing || all(iszero, dr.Pf) || throw(
        ArgumentError(
            "the $(repr(operator)) plastic adjoint does not support a nonzero fluid pressure dr.Pf; use :blocks"
        )
    )
    M_P = dr.M_P
    labels = ("vx", "vy", "vz")
    PC_v = Tuple(getfield(dr, :PC_v))
    ∂Rv∂v = Tuple(getfield(dr, :∂Rv∂v))
    v = velocity(dr)
    Rv = Tuple(getfield(dr, :Rv))

    (; Resλv, Resλv0, ResλP, λrate, dv, zero_bc) = workspace.common
    for c in 1:D
        for (buffer, reference, label) in (
                (Resλv[c], Rv[c], "$(labels[c]) residual"),
                (Resλv0[c], Rv[c], "previous $(labels[c]) residual"),
                (λrate[c], v[c], "$(labels[c]) rate"),
                (dv[c], v[c], "$(labels[c]) pullback"),
            )
            axes(buffer) == axes(reference) || throw(
                DimensionMismatch(
                    "adjoint workspace $label axes $(axes(buffer)) do not match $(axes(reference))"
                )
            )
        end
        length(zero_bc[c]) == length(v_nodes[c]) || throw(
            DimensionMismatch(
                "adjoint workspace has $(length(zero_bc[c])) $(labels[c]) boundary values, " *
                    "but the $(labels[c]) nodes have $(length(v_nodes[c])) entries"
            )
        )
    end
    axes(ResλP) == axes(dr.P) || throw(
        DimensionMismatch(
            "adjoint workspace pressure residual axes $(axes(ResλP)) do not match $(axes(dr.P))"
        )
    )
    # The DYREL rates carry momentum between iterations, so a reused workspace
    # must start every solve from rest.
    foreach(a -> fill!(a, 0), λrate)

    # The operator is constant at the frozen forward state, so every path applies
    # the same thing; they differ in what they store to do it. Assembling the
    # blocks also fills the Jacobi diagonal and Gershgorin row sums from the
    # velocity blocks already in hand, which the other two must assemble for
    # themselves.
    op = if operator === :blocks
        assemble_adjoint_operator(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup,
        )
    elseif operator === :matrix_free
        matrix_free_adjoint_operator(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup,
        )
    else
        nothing
    end
    if operator !== :blocks
        assemble_augmented_momentum_jacobian_matrices_atomix!(
            ∂Rv∂v, PC_v, v, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P, phases_v, phases_P,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
            dr.ηb, Δt, γP, M_P, backend, workgroup; τ_old, plastic,
        )
    end

    # The assembled row sums give a Gershgorin bound on the preconditioned
    # spectral radius. Where an operator apply is available the eigenvalue itself
    # can be measured, which shortens the pseudo-time step correspondingly.
    λmax_components = ntuple(c -> _checked_λmax(∂Rv∂v[c], PC_v[c], "adjoint $(labels[c])"), Val(D))
    λmax_gershgorin = maximum(λmax_components)
    λmax_iterations = 0
    λmax_v = if measure_λmax && op !== nothing
        λmax_measured, λmax_iterations = estimate_adjoint_λmax(
            op, mesh_stokes, element_v, element_P, PC_v, v_nodes, backend, workgroup,
        )
        ntuple(_ -> λmax_measured, Val(D))
    else
        λmax_components
    end
    Δτ_v = map(λ -> 2 / sqrt(λ) * dr.CFL_v, λmax_v)
    cheb = map(Δτ -> _stokes_cheb(Δτ, zero(Δτ), dr.c_fact), Δτ_v)
    α_v, β_v = map(first, cheb), map(last, cheb)

    verbose && measure_λmax && op !== nothing &&
        @info "Adjoint λmax" λmax_measured = first(λmax_v) λmax_gershgorin ratio = λmax_gershgorin / first(λmax_v) λmax_iterations

    # Buffers, seeds and pullbacks that only the Enzyme reverse passes touch. The
    # other two paths replace those passes outright, so these nine mesh-sized
    # arrays are never allocated for them.
    enzyme_scratch = workspace.enzyme
    op === nothing && enzyme_scratch === nothing && throw(
        ArgumentError(
            "operator = :enzyme requires a StokesAdjointWorkspace constructed with enzyme=true"
        )
    )

    function assemble_adjoint_residual_enzyme!(scratch)
        (;
            Rv_x_buf, Rv_y_buf, seed_Rv_x, seed_Rv_y, seed_RP, dP, dP_scratch,
            Pnum, dPnum,
        ) = scratch
        dvx, dvy = dv
        # Only the Enzyme shadows need zeroing: they are accumulated into by the
        # reverse passes. Resλv and ResλP are each fully overwritten below.
        fill!(dvx, 0)
        fill!(dvy, 0)
        fill!(dP, 0)
        fill!(dPnum, 0)

        copyto!(seed_Rv_x, λv[1])
        copyto!(seed_Rv_y, λv[2])

        # Momentum transpose: (∂Rv/∂v)ᵀλv → dvx,dvy, (∂Rv/∂P)ᵀλv → dP,
        # and (∂Rv/∂Pnum)ᵀλv → dPnum.
        assemble_momentum_residual_matrices_atomix_adj!(
            Rv_x_buf, seed_Rv_x, Rv_y_buf, seed_Rv_y,
            dr.v.x, dvx, dr.v.y, dvy, dr.P, dP, dr.T, Pnum, dPnum,
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
        # The same pass also produced (∂RP/∂P)ᵀ of that seed, which is the storage
        # term of the pressure row: (∂RP/∂P)ᵀλP plus the augmentation's route from
        # λv through Pnum. It is zero only when the bulk is incompressible, so it
        # is added rather than dropped. See [`FrozenAdjointOperator`](@ref).
        @. ResλP += dP_scratch

        return nothing
    end

    function assemble_adjoint_residual!()
        if op === nothing
            assemble_adjoint_residual_enzyme!(enzyme_scratch)
        else
            apply_adjoint_operator!(
                dv, ResλP, op, λv, λP,
                mesh_stokes, element_v, element_P, backend, workgroup,
            )
        end
        for c in 1:D
            @. Resλv[c] = objective_v[c] + dv[c]
            apply_dirichlet!(Resλv[c], v_nodes[c], zero_bc[c], backend, workgroup)
        end
        return nothing
    end

    velocity_error() = maximum(norm, Resλv) / sqrt(mesh_stokes.nnodes)

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

        err_v = velocity_error()
        err_P = norm(ResλP) / sqrt(mesh_stokes.nnodesP)
        if itPH == 1
            err_v0 = err_v + eps(err_v)
            err_P0 = err_P + eps(err_P)
        end
        err = max(min(err_v, err_v / err_v0), min(err_P, err_P / err_P0))
        collect_history && push!(history, (; iter, itPH, err, err_v, err_P))
        verbose && @printf(
            "adj PH=%03d iter=%06d err=%.3e Rv=%.3e RP=%.3e\n",
            itPH, iter, err, err_v, err_P
        )

        if err < adjoint_tol
            converged = true
            break
        end

        target_v = max(err_v * rel_drop, adjoint_tol)
        inner = 0
        while err_v > target_v && inner < iterMax && iter < total_iterMax
            inner += 1
            iter += 1
            foreach(copyto!, Resλv0, Resλv)

            for c in 1:D
                stokes_update_rate!(
                    λrate[c], Resλv[c], PC_v[c], β_v[c],
                    mesh_stokes.nnodes, backend, workgroup
                )
                stokes_update_variable!(
                    λv[c], λrate[c], -α_v[c],
                    mesh_stokes.nnodes, backend, workgroup
                )
            end
            for c in 1:D
                apply_dirichlet!(λv[c], v_nodes[c], zero_bc[c], backend, workgroup)
                apply_dirichlet!(λrate[c], v_nodes[c], zero_bc[c], backend, workgroup)
            end

            assemble_adjoint_residual!()

            if iszero(iter % ncheck)
                err_v = velocity_error()

                # Re-estimate λmin and refresh the Chebyshev step. Δτ and λmax stay
                # fixed (the Jacobian depends only on the frozen forward state).
                cheb = ntuple(Val(D)) do c
                    λmin = _stokes_λmin(α_v[c], λrate[c], Resλv[c], Resλv0[c], PC_v[c])
                    _stokes_cheb(Δτ_v[c], λmin, dr.c_fact)
                end
                α_v, β_v = map(first, cheb), map(last, cheb)

                verbose_inner && @printf(
                    "  adj inner it=%05d iter=%06d err_v=%.3e α=%s β=%s\n",
                    inner, iter, err_v,
                    join((@sprintf("%.2e", a) for a in α_v), " "),
                    join((@sprintf("%.2e", b) for b in β_v), " "),
                )
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
        err_v = velocity_error()
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
        iterations = iter,
        residual = err,
        λmax = first(λmax_v),
        λmax_gershgorin,
        λmax_iterations,
        history,
    )
end

solve_stokes_adjoint_dyrel!(
    dr::StokesDR{<:Any, 2}, mesh_stokes, geo_v, geo_P, element_v, element_P,
    phases_v, phases_P, τ_old, plastic, G, Δt, γP,
    objective_vx::AbstractVector, objective_vy::AbstractVector,
    λvx::AbstractVector, λvy::AbstractVector, λP, backend, workgroup;
    vx_nodes, vy_nodes, kwargs...,
) = solve_stokes_adjoint_dyrel!(
    dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
    phases_v, phases_P, τ_old, plastic, G, Δt, γP,
    (objective_vx, objective_vy), (λvx, λvy), λP, backend, workgroup;
    v_nodes = (vx_nodes, vy_nodes), kwargs...,
)

"""
    solve_adjoint!(dr::StokesDR, mesh::MixedMesh, bc_v; dt, objective_v, λv, λP,
                   tolerance=1e-6, max_iterations=50_000, check_interval=50,
                   throw_on_failure=true, plastic=nothing, workgroup=256, kwargs...)
    solve_adjoint!(dr, mesh, bc_vx, bc_vy; dt, objective_vx, objective_vy, λvx, λvy, λP, kwargs...)

Solve the discrete Stokes adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` of a converged forward
[`solve!`](@ref) on the same state. Geometry and elements come from
`mesh.geometry`, and the pressure scale `dr.γP` and mass `dr.M_P` the forward solve
left in `dr` are reused, so the adjoint transposes the augmented residual that was
actually solved. `dt`, `plastic`, `phases_v`, `phases_P`, and `τ_old` must match the
forward solve. `bc_v` supplies the constrained velocity nodes; the adjoint
conditions are homogeneous, so its values are not used.

`objective_v` holds the velocity part of `∂J/∂u`, one array per direction, and
`λv` and `λP` are the adjoint outputs. Their input values are the initial iterate:
pass zeros for a cold solve or the previous design iteration's fields to warm-start.
`tolerance`, `max_iterations`, and `check_interval` map to `adjoint_tol`,
`total_iterMax`, and `ncheck` of [`solve_stokes_adjoint_dyrel!`](@ref), whose other
keywords (`operator`, `workspace`, `measure_λmax`, …) pass through. Returns its
statistics, which include `converged`, `iterations`, `residual`, and `history`; a
solve that does not converge throws unless `throw_on_failure=false`.

The adjoint is exact only in the incompressible gauge `K = Inf`; see the solver
notes for the finite-storage approximation.
"""
function solve_adjoint!(
        dr::StokesDR{<:Any, D},
        mesh::MixedMesh{D},
        bc_v::NTuple{D, DirichletBoundaryCondition};
        dt,
        objective_v::NTuple{D, AbstractVector},
        λv::NTuple{D, AbstractVector},
        λP,
        phases_v = dr.phases_v,
        phases_P = dr.phases_P,
        τ_old = stress_old(dr),
        plastic = nothing,
        tolerance = 1.0e-6,
        max_iterations = 50_000,
        check_interval = 50,
        throw_on_failure = true,
        workgroup = 256,
        kwargs...,
    ) where {D}
    all(iszero, dr.γP) && throw(
        ArgumentError(
            "dr.γP is unassembled; run the forward solve! on this state before solve_adjoint!"
        )
    )
    cache = _mesh_geometry(mesh)
    stats = solve_stokes_adjoint_dyrel!(
        dr, mesh, cache.geo_v, cache.geo_P, cache.element_v, cache.element_P,
        phases_v, phases_P, τ_old, plastic, dr.G, dt, dr.γP,
        objective_v, λv, λP, KA.get_backend(mesh.coords), workgroup;
        v_nodes = map(bc -> bc.DoFs, bc_v), adjoint_tol = tolerance,
        total_iterMax = max_iterations, ncheck = check_interval, kwargs...,
    )
    stats.converged || !throw_on_failure || error(
        "Stokes adjoint solve did not converge after $(stats.iter) iterations " *
        "(residual = $(stats.residual), tolerance = $tolerance)",
    )
    return stats
end

solve_adjoint!(
    dr::StokesDR{<:Any, 2}, mesh::MixedMesh,
    bc_vx::DirichletBoundaryCondition, bc_vy::DirichletBoundaryCondition;
    objective_vx, objective_vy, λvx, λvy, λP, kwargs...
) =
    solve_adjoint!(
    dr, mesh, (bc_vx, bc_vy);
    objective_v = (objective_vx, objective_vy), λv = (λvx, λvy), λP, kwargs...
)

"""
    solve_adjoint!(dr::CellPressureStokesDR, mesh::Mesh, bc_v; objective_v, λv, λP,
                   throw_on_failure=true, kwargs...)

Solve the discrete adjoint `(∂R/∂u)ᵀλ = ∂J/∂u` of the viscous 3-D Hex27 Stokes
operator. The operator is symmetric and linear, so this runs the forward
iteration of [`solve!`](@ref) with `objective_v` (one array per velocity
component) as momentum load, zero density and gravity, and homogeneous
conditions on the nodes of `bc_v`; its values are not used. It reads only the
viscosity, phases, and scratch of `dr` and leaves `dr.v` and `dr.P` unchanged.

`λv` (three nodal arrays) and the `4 × nels` `λP` are the adjoint outputs; their
input values are the initial iterate. The remaining keywords and the returned
statistics are those of `solve!`. A solve that does not converge throws unless
`throw_on_failure=false`. Pass `λv` to [`stokes_material_gradient_3d`](@ref) for
the material sensitivities.
"""
function solve_adjoint!(
        dr::CellPressureStokesDR, mesh::Mesh, bc_v::NTuple{3, DirichletBoundaryCondition};
        objective_v, λv, λP::AbstractMatrix, throw_on_failure = true,
        tolerance = 1.0e-5, kwargs...,
    )
    load = Tuple(objective_v)
    all(length(load[i]) == mesh.nnodes for i in 1:3) ||
        throw(DimensionMismatch("objective_v must have mesh.nnodes entries per component"))
    T = eltype(dr.P)
    stats = _relax_cell_pressure_stokes!(
        Tuple(λv), λP, dr, mesh, map(zero, dr.ρ), ntuple(_ -> zero(T), 3),
        map(bc -> bc.DoFs, bc_v), zero(T), load;
        tolerance, kwargs...,
    )
    stats.converged || !throw_on_failure || error(
        "3-D Stokes adjoint solve did not converge after $(stats.iter) iterations " *
        "(residual = $(stats.residual), tolerance = $tolerance)",
    )
    return stats
end

"""
    stokes_material_gradient_3d(forward_velocity, adjoint_velocity, mesh,
                                cell_phase, η, ρ, g; workgroup=256)
    stokes_material_gradient_3d(dr::CellPressureStokesDR, mesh, adjoint_velocity;
                                workgroup=256)

Contract the matrix-free 3-D adjoint with the density load derivative and
viscous operator derivative for every material phase at once. The returned
named tuple contains `density_gradient` and `viscosity_gradient`, each an
`NTuple` with one entry per phase of `η`/`ρ`. The state form reads the forward
velocity, phases, and material of `dr`.
"""
function stokes_material_gradient_3d(
        forward_velocity::NTuple{3}, adjoint_velocity::NTuple{3}, mesh::Mesh,
        cell_phase, η, ρ, g::NTuple{3}; workgroup = 256,
    )
    length(η) == length(ρ) || throw(ArgumentError("η and ρ must have the same length"))
    nphases = length(η)
    pressure = similar(first(forward_velocity), 4, mesh.nels)
    fill!(pressure, 0)
    residual = ntuple(i -> similar(forward_velocity[i]), 3)
    zero_velocity = ntuple(i -> fill!(similar(forward_velocity[i]), 0), 3)
    zero_g = ntuple(_ -> zero(first(g)), 3)
    zero_phase = map(zero, η)
    tables = stokes_tables_3d(KA.get_backend(first(forward_velocity)), mesh.element)

    density_gradient = ntuple(nphases) do phase
        density = ntuple(i -> i == phase ? one(ρ[i]) : zero(ρ[i]), nphases)
        assemble_stokes_momentum_residual_3d!(
            residual, zero_velocity, pressure, mesh, cell_phase, η, density, g; workgroup, tables,
        )
        -sum(dot(adjoint_velocity[i], residual[i]) for i in 1:3)
    end

    viscosity_gradient = ntuple(nphases) do phase
        viscosity = ntuple(i -> i == phase ? one(η[i]) : zero(η[i]), nphases)
        assemble_stokes_momentum_residual_3d!(
            residual, forward_velocity, pressure, mesh, cell_phase, viscosity, zero_phase, zero_g; workgroup, tables,
        )
        -sum(dot(adjoint_velocity[i], residual[i]) for i in 1:3)
    end

    return (; density_gradient, viscosity_gradient)
end

stokes_material_gradient_3d(dr::CellPressureStokesDR, mesh::Mesh, adjoint_velocity; workgroup = 256) =
    stokes_material_gradient_3d(
    Tuple(dr.v), Tuple(adjoint_velocity), mesh, dr.phases, dr.η, dr.ρ, dr.g; workgroup,
)
