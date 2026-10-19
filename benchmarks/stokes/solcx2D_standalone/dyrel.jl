module SolCxDYREL

# Extracted from the FEMTools 2-D Stokes path, specialized to this miniapp's
# purely viscous, incompressible T7/P1-discontinuous problem. All numerical
# functions used by the solve live here; the caller supplies prepared arrays.
using KernelAbstractions
import KernelAbstractions as KA
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Atomix
using Printf

@inline function launch!(kernel!, backend, workgroup, ndrange, args...)
    kernel!(backend, workgroup)(args...; ndrange)
    KA.synchronize(backend)
    return nothing
end

@inline local_nodes_of(nodes, e, ::Val{N}) where {N} =
    SVector{N}(ntuple(i -> nodes[i, e], Val(N)))
@inline _gather_local(a, nodes::SVector{N}) where {N} =
    SVector{N}(ntuple(i -> a[nodes[i]], Val(N)))

@generated function interp2ip_phase(N::SVector{M, T}, property, phase) where {M, T}
    return quote
        @inline
        value = zero(T)
        Base.@nexprs $M i -> value += N[i] * property[phase[i]]
        value
    end
end

# The driver supplies one phase per cell, shared by velocity and pressure.
@inline _cell_phase(phases, e) = Int(phases[1, e])
@inline _phase_nodes(phases, e) = SVector{7}(ntuple(_ -> _cell_phase(phases, e), Val(7)))

# Read the prepared inverse Jacobian directly, without invoking package methods.
@inline _geometry(geo, gradients, e, q) =
    (gradients[q] * geo[e][q].J⁻¹, geo[e][q].dΩ)

@generated function compute_velocity_divergence(
        v::NTuple{2, SVector{M}}, gradient,
    ) where {M}
    return quote
        @inline
        divergence = zero(gradient[1, 1] * v[1][1])
        Base.@nexprs 2 j -> begin
            vj = v[j]
            Base.@nexprs $M i -> divergence += gradient[i, j] * vj[i]
        end
        divergence
    end
end

@inline function integrate_pressure_residual(v, geo, geo_P, tables, e)
    T = promote_type(map(eltype, v)...)
    R = zero(SVector{3, T})
    for q in eachindex(geo_P[e])
        gradient, = _geometry(geo, tables.gradients, e, q)
        divergence = compute_velocity_divergence(v, gradient)
        Np, dΩ = tables.P[q], geo_P[e][q]
        R += SVector{3}(ntuple(i -> Np[i] * (-divergence) * dΩ, Val(3)))
    end
    return R
end

@inline function deviatoric_stress(v, gradient, Nv, η, phase)
    ∇vx, ∇vy = gradient' * v[1], gradient' * v[2]
    εxx, εyy = ∇vx[1], ∇vy[2]
    εxy = (∇vx[2] + ∇vy[1]) / 2
    tr = (εxx + εyy) / 3
    # Retain the package's Maxwell expression in its G=Inf limit.
    ηve = inv(inv(interp2ip_phase(Nv, η, phase)))
    return (2 * ηve * (εxx - tr), 2 * ηve * (εyy - tr), 2 * ηve * εxy)
end

@inline function integrate_momentum_residual(v, P, Pnum, geo, tables, e, η, phase)
    T = promote_type(map(eltype, v)...)
    R = ntuple(_ -> zero(SVector{7, T}), Val(2))
    for q in eachindex(geo[e])
        gradient, dΩ = _geometry(geo, tables.gradients, e, q)
        τxx, τyy, τxy = deviatoric_stress(v, gradient, tables.v[q], η, phase)
        τ = SMatrix{2, 2}(τxx, τxy, τxy, τyy)
        p = dot(tables.P[q], P) + (Pnum === nothing ? zero(eltype(P)) : dot(tables.P[q], Pnum))
        R = ntuple(Val(2)) do c
            pressure = SVector{2}(ntuple(j -> j == c ? p : zero(p), Val(2)))
            R[c] + gradient * (τ[:, c] - pressure) * dΩ
        end
    end
    return R
end

@kernel function pressure_residual_kernel!(RP, @Const(v), @Const(mesh), @Const(tables))
    e = @index(Global)
    nodes = local_nodes_of(mesh.el2n, e, Val(7))
    dofs = local_nodes_of(mesh.DoFsP, e, Val(3))
    vloc = map(a -> _gather_local(a, nodes), v)
    R = integrate_pressure_residual(vloc, mesh.geo_v, mesh.geo_P, tables, e)
    # Pressure DoFs are discontinuous and belong to exactly one element.
    for i in 1:3
        RP[dofs[i]] += R[i]
    end
end

function assemble_pressure_residual!(RP, v, mesh, tables, backend, workgroup)
    fill!(RP, 0)
    return launch!(pressure_residual_kernel!, backend, workgroup, mesh.nels, RP, v, mesh, tables)
end

@kernel function momentum_residual_kernel!(
        Rv, @Const(v), @Const(P), @Const(Pnum), @Const(mesh), @Const(tables),
        @Const(phases), η,
    )
    e = @index(Global)
    nodes = local_nodes_of(mesh.el2n, e, Val(7))
    dofs = local_nodes_of(mesh.DoFsP, e, Val(3))
    vloc = map(a -> _gather_local(a, nodes), v)
    Ploc = _gather_local(P, dofs)
    Pnumloc = Pnum === nothing ? nothing : _gather_local(Pnum, dofs)
    R = integrate_momentum_residual(
        vloc, Ploc, Pnumloc, mesh.geo_v, tables, e, η, _phase_nodes(phases, e),
    )
    for (i, node) in enumerate(nodes)
        ntuple(Val(2)) do c
            Atomix.@atomic :monotonic Rv[c][node] += R[c][i]
            nothing
        end
    end
end

function assemble_momentum_residual!(Rv, v, P, Pnum, mesh, tables, phases, η, backend, workgroup)
    foreach(a -> fill!(a, 0), Rv)
    return launch!(momentum_residual_kernel!, backend, workgroup, mesh.nels,
        Rv, v, P, Pnum, mesh, tables, phases, η)
end

@kernel function pressure_scaling_kernel!(MP, γP, @Const(mesh), @Const(tables), @Const(phases), η, factor)
    e = @index(Global)
    dofs = local_nodes_of(mesh.DoFsP, e, Val(3))
    phase = _phase_nodes(phases, e)
    for q in eachindex(mesh.geo_P[e])
        γnum = factor * interp2ip_phase(tables.v[q], η, phase)
        # The package's incompressible branch takes γphy=γnum.
        γq = γnum * γnum / (γnum + γnum)
        for a in 1:3
            weight = tables.P[q][a] * mesh.geo_P[e][q]
            MP[dofs[a]] += weight
            γP[dofs[a]] += weight * γq
        end
    end
end

@kernel function pressure_scaling_finalize_kernel!(γP, @Const(MP))
    i = @index(Global)
    γP[i] = MP[i] == 0 ? zero(γP[i]) : γP[i] / MP[i]
end

function assemble_pressure_scaling!(MP, γP, mesh, tables, phases, η, factor, backend, workgroup)
    fill!(MP, 0)
    fill!(γP, 0)
    launch!(pressure_scaling_kernel!, backend, workgroup, mesh.nels, MP, γP, mesh, tables, phases, η, factor)
    return launch!(pressure_scaling_finalize_kernel!, backend, workgroup, length(MP), γP, MP)
end

@inline _substitute_component(v::NTuple{D}, u, ::Val{J}) where {D, J} =
    ntuple(k -> k === J ? u : v[k], Val(D))
@inline _component(R, ::Val{I}) where {I} = R[I]
@inline function _velocity_jacobian_block(momentum, v, vi, vj::Val{J}) where {J}
    return ForwardDiff.jacobian(u -> _component(momentum(_substitute_component(v, u, vj)), vi), v[J])
end
@inline _abs_row_sums(block::SMatrix{M, N}) where {M, N} =
    SVector{M}(ntuple(i -> sum(ntuple(j -> abs(block[i, j]), Val(N))), Val(M)))
@inline _abs_diagonal(block::SMatrix{M}) where {M} =
    SVector{M}(ntuple(i -> abs(block[i, i]), Val(M)))
@inline _same_direction(::Val{I}, ::Val{J}) where {I, J} = I === J

@inline function _jacobian_row_diagnostics(momentum, v, vi)
    first_block = _velocity_jacobian_block(momentum, v, vi, Val(1))
    diagonal = _abs_diagonal(first_block)
    seed = (_abs_row_sums(first_block), _same_direction(vi, Val(1)) ? diagonal : zero(diagonal))
    second_block = _velocity_jacobian_block(momentum, v, vi, Val(2))
    return (seed[1] + _abs_row_sums(second_block),
        _same_direction(vi, Val(2)) ? _abs_diagonal(second_block) : seed[2])
end

@kernel function augmented_momentum_jacobian_kernel!(
        rowsums, PC, @Const(v), @Const(P), @Const(MP), @Const(γP),
        @Const(mesh), @Const(tables), @Const(phases), η,
    )
    e = @index(Global)
    nodes = local_nodes_of(mesh.el2n, e, Val(7))
    dofs = local_nodes_of(mesh.DoFsP, e, Val(3))
    vloc = map(a -> _gather_local(a, nodes), v)
    Ploc, Mloc, γloc = _gather_local(P, dofs), _gather_local(MP, dofs), _gather_local(γP, dofs)
    phase = _phase_nodes(phases, e)
    # The production preconditioner uses the maximum viscosity in an element.
    ηmax = maximum(η[phase[i]] for i in 1:7)
    ηpc = map(_ -> ηmax, η)
    momentum = trial -> begin
        RP = integrate_pressure_residual(trial, mesh.geo_v, mesh.geo_P, tables, e)
        Pnum = γloc .* RP ./ Mloc
        integrate_momentum_residual(trial, Ploc, Pnum, mesh.geo_v, tables, e, ηpc, phase)
    end
    map((Val(1), Val(2))) do vi
        row, diagonal = _jacobian_row_diagnostics(momentum, vloc, vi)
        _scatter_row_diagnostics!(rowsums, PC, nodes, row, diagonal, vi)
    end
end

@inline function _scatter_row_diagnostics!(rowsums, PC, nodes, row, diagonal, ::Val{C}) where {C}
    for (i, node) in enumerate(nodes)
        Atomix.@atomic :monotonic rowsums[C][node] += row[i]
        Atomix.@atomic :monotonic PC[C][node] += diagonal[i]
    end
    return nothing
end

function assemble_augmented_momentum_jacobian!(rowsums, PC, v, P, MP, γP, mesh, tables, phases, η, backend, workgroup)
    foreach(a -> fill!(a, 0), rowsums)
    foreach(a -> fill!(a, 0), PC)
    return launch!(augmented_momentum_jacobian_kernel!, backend, workgroup, mesh.nels,
        rowsums, PC, v, P, MP, γP, mesh, tables, phases, η)
end

@kernel function dirichlet_kernel!(v, @Const(nodes), @Const(values))
    i = @index(Global)
    v[nodes[i]] = values[i]
end

function apply_dirichlet_all!(fields, nodes, values, backend, workgroup)
    ntuple(Val(2)) do c
        isempty(nodes[c]) || launch!(dirichlet_kernel!, backend, workgroup, length(nodes[c]), fields[c], nodes[c], values[c])
        nothing
    end
    return nothing
end

@kernel function update_stokes_velocity_kernel!(rate, v, @Const(R), @Const(PC), β, α)
    i = @index(Global)
    ntuple(Val(2)) do c
        new_rate = β[c] * rate[c][i] + R[c][i] / PC[c][i]
        rate[c][i] = new_rate
        v[c][i] += α[c] * new_rate
        nothing
    end
end

function _subtract_body_force!(R, load)
    for c in eachindex(R)
        R[c] .-= load[c]
    end
    return nothing
end

function _checked_λmax(rowsums, PC)
    λ = maximum(rowsums ./ PC)
    isfinite(λ) && λ > zero(λ) || error("Stokes preconditioner produced invalid λmax = $λ")
    return λ
end

@inline fused_sum(f, arrays...) = sum(Base.Broadcast.instantiate(Base.Broadcast.broadcasted(f, arrays...)))
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

"""
    solve_stokes_dyrel!(state, mesh, tables, bc; phases, viscosity, body_force, ...)

Solve the miniapp's purely viscous incompressible T7/P1-disc system in place.
`state` is a NamedTuple of borrowed arrays (`v`, `rate`, `Rv`, `Rv0`, `rowsums`,
`PC`, `P`, `RP`, `Pnum`, `M_P`, `γP`) and controls (`CFL_v`, `c_fact`). `mesh`
contains `el2n`, `DoFsP`, `geo_v`, `geo_P`, `nels`, `nnodes`, `nnodesP`.
`tables` contains backend arrays `v`, `P`, `gradients`; `bc` contains two
NamedTuples with `DoFs` and `vals`. `phases` is a 1×nels cell-phase array.
All arrays must already share a backend and precision. No FEMTools functions
are called. The pressure scale and frozen augmented Jacobian are built here.
Statistics and convergence tests follow the package DYREL path. Failure to
converge throws unless `throw_on_failure=false`.
"""
function solve_stokes_dyrel!(
        state, mesh, tables, bc;
        phases, viscosity, body_force,
        pressure_factor = 50.0,
        tolerance = 1e-6,
        max_iterations = 50_000,
        check_interval = 50,
        iterMax = 50_000,
        max_ph_iterations = 1000,
        rel_drop0 = 1e-2,
        collect_history = false,
        verbose = true,
        verbose_inner = false,
        throw_on_failure = true,
        workgroup = 256,
    )
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    check_interval isa Integer && check_interval > 0 || throw(ArgumentError("check_interval must be a positive integer"))
    max_iterations isa Integer && max_iterations >= 0 || throw(ArgumentError("max_iterations must be a nonnegative integer"))
    max_ph_iterations isa Integer && max_ph_iterations > 0 || throw(ArgumentError("max_ph_iterations must be a positive integer"))
    iterMax isa Integer && iterMax >= 0 || throw(ArgumentError("iterMax must be a nonnegative integer"))
    isfinite(pressure_factor) && pressure_factor > 0 || throw(ArgumentError("pressure_factor must be finite and positive"))
    size(mesh.el2n) == (7, mesh.nels) || throw(DimensionMismatch("expected T7 connectivity"))
    size(mesh.DoFsP) == (3, mesh.nels) || throw(DimensionMismatch("expected P1-disc connectivity"))
    size(phases) == (1, mesh.nels) || throw(DimensionMismatch("phases must be 1 × nels"))
    all(x -> isfinite(x) && x > 0, viscosity) || throw(ArgumentError("viscosities must be finite and positive"))
    minimum(phases) >= 1 && maximum(phases) <= length(viscosity) || throw(ArgumentError("invalid cell phase"))
    backend = KA.get_backend(state.P)
    for f in body_force
        length(f) == mesh.nnodes || throw(DimensionMismatch("body_force must match velocity nodes"))
        eltype(f) === eltype(state.P) || throw(ArgumentError("body_force must match state precision"))
        typeof(KA.get_backend(f)) === typeof(backend) || throw(ArgumentError("body_force must match state backend"))
        all(isfinite, f) || throw(ArgumentError("body_force must be finite"))
    end

    v, rate, Rv, Rv0, rowsums, PC = state.v, state.rate, state.Rv, state.Rv0, state.rowsums, state.PC
    P, RP, Pnum, MP, γP = state.P, state.RP, state.Pnum, state.M_P, state.γP
    nodes, values = map(b -> b.DoFs, bc), map(b -> b.vals, bc)
    zero_bc = map(zero, values)
    foreach(a -> fill!(a, 0), rate)
    foreach(a -> fill!(a, 0), Rv0)
    assemble_pressure_scaling!(MP, γP, mesh, tables, phases, viscosity, pressure_factor, backend, workgroup)
    assemble_augmented_momentum_jacobian!(rowsums, PC, v, P, MP, γP, mesh, tables, phases, viscosity, backend, workgroup)
    λmax_v = map(_checked_λmax, rowsums, PC)
    Δτ_v = map(λ -> 2 / sqrt(λ) * state.CFL_v, λmax_v)
    cheb = map(Δτ -> _stokes_cheb(Δτ, zero(Δτ), state.c_fact), Δτ_v)
    α_v, β_v = map(first, cheb), map(last, cheb)

    ϵ = eltype(P)(tolerance)
    err_min, err_abs, err_rel, err_v, err_P = Inf, Inf, Inf, Inf, Inf
    err_v0 = err_P0 = err_v00 = 1.0
    err = 2 * ϵ
    iter = itPH_done = 0
    converged = false
    rel_drop = rel_drop0
    history = NamedTuple[]

    for itPH in 1:max_ph_iterations
        itPH_done = itPH
        assemble_pressure_residual!(RP, v, mesh, tables, backend, workgroup)
        assemble_momentum_residual!(Rv, v, P, nothing, mesh, tables, phases, viscosity, backend, workgroup)
        _subtract_body_force!(Rv, body_force)
        apply_dirichlet_all!(Rv, nodes, zero_bc, backend, workgroup)
        err_P = norm(RP ./ MP) / sqrt(mesh.nnodesP)
        err_v = maximum(norm, Rv) / (2 * sqrt(mesh.nnodes))
        if collect_history
            err_v_components = map(r -> norm(r) / (2 * sqrt(mesh.nnodes)), Rv)
            push!(history, (; iter, err_v, err_v_components, err_P))
        end
        if itPH == 1
            err_P0 = err_P + eps(err_P)
            err_v0 = err_v + eps(err_v)
        elseif itPH == 2
            err_P0 = err_P + eps(err_P)
        end
        err_v_rel, err_P_rel = err_v / err_v0, err_P / err_P0
        err_abs, err_rel = max(err_v, err_P), max(err_v_rel, err_P_rel)
        err = min(err_abs, err_rel)
        isfinite(err) || error("non-finite residual in outer loop PH=$itPH")
        err > 1e10 && error("residual > 1e10 in outer loop PH=$itPH")
        verbose && @printf("itPH = %02d iter = %06d abs = %.3e rel = %.3e\n", itPH, iter, err_abs, err_rel)
        if err < ϵ
            converged = true
            break
        end
        if err > err_min * 1.05
            rel_drop = max(rel_drop * 0.1, 1e-3)
        end
        err_min = min(err_min, err)
        ϵ_vel = err * rel_drop
        itPT = 0
        while err > ϵ_vel && itPT ≤ iterMax
            itPT += 1
            iter += 1
            do_check = iszero(iter % check_interval)
            do_check && foreach(copyto!, Rv0, Rv)
            assemble_pressure_residual!(RP, v, mesh, tables, backend, workgroup)
            @. Pnum = γP * RP / MP
            assemble_momentum_residual!(Rv, v, P, Pnum, mesh, tables, phases, viscosity, backend, workgroup)
            _subtract_body_force!(Rv, body_force)
            apply_dirichlet_all!(Rv, nodes, zero_bc, backend, workgroup)
            apply_dirichlet_all!(rate, nodes, zero_bc, backend, workgroup)
            launch!(update_stokes_velocity_kernel!, backend, workgroup, mesh.nnodes,
                rate, v, Rv, PC, β_v, map(-, α_v))
            apply_dirichlet_all!(v, nodes, values, backend, workgroup)
            if do_check
                err_v_inner = maximum(norm, Rv) / (2 * sqrt(mesh.nnodes))
                iter == check_interval && (err_v00 = err_v_inner + eps(err_v_inner))
                # Keep the absolute escape for an already satisfied momentum balance.
                err = err_v_inner ≤ ϵ ? err_v_inner : max(err_v_inner / err_v00, err_v_inner)
                isfinite(err) || error("non-finite residual in inner loop PH=$itPH PT=$itPT")
                err > 1e10 && error("residual > 1e10 in inner loop PH=$itPH PT=$itPT")
                if collect_history
                    err_v_components = map(r -> norm(r) / (2 * sqrt(mesh.nnodes)), Rv)
                    pressure_residual = norm(RP ./ MP) / sqrt(mesh.nnodesP)
                    push!(history, (; iter, err_v = err_v_inner, err_v_components, err_P = pressure_residual))
                end
                verbose_inner && @printf("  it = %d, iter = %d, err = %.3e\n", itPT, iter, err)
                λmin_v = map(_stokes_λmin, α_v, rate, Rv, Rv0, PC)
                cheb = map((Δτ, λmin) -> _stokes_cheb(Δτ, λmin, state.c_fact), Δτ_v, λmin_v)
                α_v, β_v = map(first, cheb), map(last, cheb)
            end
            iter > max_iterations && break
        end
        @. P += γP * RP / MP
        iter > max_iterations && break
    end

    stats = (; itPH = itPH_done, iter, err, err_abs, err_rel, err_v, err_P, converged,
        iterations = iter, residual = err_abs, reached_total_iter = iter > max_iterations,
        λmax = maximum(λmax_v), λmax_gershgorin = maximum(λmax_v),
        λmax_iterations = 0, jacobian_assemblies = 1, history)
    converged || !throw_on_failure || error("Stokes solve did not converge after $iter iterations (residual = $err_abs, tolerance = $tolerance)")
    return stats
end

end # module
