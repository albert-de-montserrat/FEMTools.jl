"""
    StokesDRND{D}

Dimension-parametric Stokes state used by the tuple-based 2D/3D forward solver.
Velocity, momentum residual, pseudo-time rate, Jacobian row-sum, and diagonal
preconditioner fields are stored as `NTuple{D}` arrays. The established 2D
`StokesDR` type remains available for source compatibility.
"""
struct StokesDRND{D, NPHS, A, AI, FP}
    v::NTuple{D, A}
    rate::NTuple{D, A}
    Rv::NTuple{D, A}
    Rv0::NTuple{D, A}
    rowsum::NTuple{D, A}
    PC::NTuple{D, A}
    P::A
    P0::A
    T::A
    T0::A
    RP::A
    MP::A
    Pnum::A
    phases_v::AI
    phases_P::AI
    η::NTuple{NPHS, FP}
    ηb::NTuple{NPHS, FP}
    α::NTuple{NPHS, FP}
    ρ0::NTuple{NPHS, FP}
    K::NTuple{NPHS, FP}
    g::NTuple{D, FP}
    Tref::FP
    CFL_v::FP
    c_fact::FP
    ϵ::FP
end

function StokesDRND(
    backend, ::Val{D}, nnodes_v, nnodes_P,
    η::NTuple{NPHS, FP}, ηb::NTuple{NPHS, FP}, α::NTuple{NPHS, FP};
    ρ0 = ntuple(_ -> one(FP), Val(NPHS)),
    K = ntuple(_ -> FP(Inf), Val(NPHS)),
    g = ntuple(_ -> zero(FP), Val(D)),
    Tref = zero(FP), CFL_v = FP(0.9), c_fact = FP(0.7), ϵ = FP(1e-6),
) where {D, NPHS, FP}
    newv() = KA.zeros(backend, FP, nnodes_v)
    newp() = KA.zeros(backend, FP, nnodes_P)
    newiv() = KA.ones(backend, Int, nnodes_v)
    newip() = KA.ones(backend, Int, nnodes_P)
    v = ntuple(_ -> newv(), Val(D))
    return StokesDRND{D, NPHS, typeof(v[1]), typeof(newiv()), FP}(
        v,
        ntuple(_ -> newv(), Val(D)),
        ntuple(_ -> newv(), Val(D)),
        ntuple(_ -> newv(), Val(D)),
        ntuple(_ -> newv(), Val(D)),
        ntuple(_ -> newv(), Val(D)),
        newp(), newp(), newp(), newp(), newp(), newp(), newp(),
        newiv(), newip(), η, ηb, α,
        NTuple{NPHS, FP}(ρ0), NTuple{NPHS, FP}(K), NTuple{D, FP}(g),
        FP(Tref), FP(CFL_v), FP(c_fact), FP(ϵ),
    )
end

@inline function _newtonian_deviatoric_stress_nd(v, ∂N∂x, ηq, ::Val{D}) where D
    # grad_v[a,b] = ∂v_a/∂x_b
    grad_v = SMatrix{D, D}(ntuple(k -> begin
        a = (k - 1) % D + 1
        b = (k - 1) ÷ D + 1
        dot(∂N∂x[:, b], v[a])
    end, Val(D * D)))
    ε = (grad_v + transpose(grad_v)) / 2
    εdev = ε - I * (tr(ε) / D)
    return 2ηq * εdev
end

"""Integrate a dimension-parametric Newtonian Stokes momentum element."""
@inline function integrate_momentum_residual_nd(
    # Each velocity component can carry a different scalar type while AD is
    # differentiating one block (Dual for that block, Float64 for the others).
    v::Tuple{Vararg{SVector{NV}, D}}, P_loc::SVector{NP}, Pnum_loc,
    T_loc::SVector{NP}, geo_el, phase_v, η, α, ρ0, K, g, Tref,
    Nq, NqP, ::Val{D},
) where {D, NV, NP}
    R = ntuple(_ -> zero(v[1]), Val(D))
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv, Np = Nq[q], NqP[q]
        ηq = interp2ip_phase(Nv, η, phase_v)
        τ = _newtonian_deviatoric_stress_nd(v, ∂N∂x, ηq, Val(D))
        Pq = dot(Np, P_loc)
        Pnumq = dot_or_zero(Np, Pnum_loc)
        Tq = dot(Np, T_loc)
        αq = interp2ip_phase(Nv, α, phase_v)
        ρ0q = interp2ip_phase(Nv, ρ0, phase_v)
        βq = interp2ip_phase(Nv, map(inv, K), phase_v)
        ρq = ρ0q * (1 - αq * (Tq - Tref) + βq * Pq)
        Ppress = Pq + Pnumq
        # Per node i and component a:
        #   incᵃᵢ = (Σ_b ∂Nᵢ/∂xᵦ·τ_aᵦ − ∂Nᵢ/∂x_a·P − Nᵢ·ρ·g_a) dΩ.
        # The stress divergence Σ_b ∂Nᵢ/∂xᵦ·τ_aᵦ over all nodes is the matvec
        # `∂N∂x * τ[a, :]`, which StaticArrays fully unrolls — faster than a
        # per-node `sum` generator, and much faster than it under ForwardDiff.
        #
        # `R` must not be referenced inside a closure: a variable both assigned
        # across loop iterations and captured by a closure is boxed as
        # `Core.Box`, making every `SVector{NV}` operation type-unstable and
        # heap-allocated. Build `inc` from local values only, then fold it in
        # with `map`, which takes `R` as an argument rather than capturing it.
        inc = ntuple(a -> (∂N∂x * τ[a, :] .- ∂N∂x[:, a] .* Ppress .- Nv .* (ρq * g[a])) .* dΩ, Val(D))
        R = map(+, R, inc)
    end
    return R
end

@inline function _gather_velocity_nd(v, nodes, ::Val{D}, ::Val{NV}) where {D, NV}
    return ntuple(a -> _gather_local(v[a], nodes, Val(NV)), Val(D))
end

# Place the ForwardDiff-seeded component `vb` at position `b` and promote every
# other component to `vb`'s element type. Differentiating one component seeds it
# with `Dual` numbers while the rest stay `Float64`; a mixed-eltype velocity
# tuple forces `_newtonian_deviatoric_stress_nd` to build mixed `SMatrix`
# intermediates that promote and heap-allocate at every quadrature point
# (~3.7 MB per 27x27 Hex27 Jacobian block). Promoting up front keeps the whole
# element evaluation on the stack (0 allocations).
@inline function _seed_trial_velocity(v::NTuple{D, <:SVector{NV}}, vb::SVector{NV, Tb}, b) where {D, NV, Tb}
    promoted = ntuple(a -> SVector{NV, Tb}(v[a]), Val(D))
    return Base.setindex(promoted, vb, b)
end

@inline function momentum_element_residual_nd(
    v, P, T, Pnum, el2n_v, dofsP, geo, phases_v,
    η, α, ρ0, K, g, Tref, Nq, NqP, iel,
    ::Val{D}, ::Val{NV}, ::Val{NP},
) where {D, NV, NP}
    nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    nodes_p = local_nodes_of(dofsP, iel, Val(NP))
    vloc = _gather_velocity_nd(v, nodes_v, Val(D), Val(NV))
    Ploc = _gather_local(P, nodes_p, Val(NP))
    Tloc = _gather_local(T, nodes_p, Val(NP))
    Pnumloc = _gather_or_nothing(Pnum, nodes_p, Val(NP))
    phase_v = _gather_phase(phases_v, nodes_v, iel, Val(NV))
    R = integrate_momentum_residual_nd(
        vloc, Ploc, Pnumloc, Tloc, geo[iel], phase_v,
        η, α, ρ0, K, g, Tref, Nq, NqP, Val(D),
    )
    return nodes_v, R
end

@kernel function momentum_residual_nd_kernel!(
    R, @Const(v), @Const(P), @Const(T), @Const(Pnum),
    @Const(el2n_v), @Const(dofsP), @Const(geo), @Const(phases_v),
    @Const(η), @Const(α), @Const(ρ0), @Const(K), @Const(g), Tref,
    @Const(Nq), @Const(NqP), ::Val{D}, ::Val{NV}, ::Val{NP},
) where {D, NV, NP}
    iel = @index(Global)
    nodes, Re = momentum_element_residual_nd(
        v, P, T, Pnum, el2n_v, dofsP, geo, phases_v,
        η, α, ρ0, K, g, Tref, Nq, NqP, iel, Val(D), Val(NV), Val(NP),
    )
    for a in 1:D, i in 1:NV
        Atomix.@atomic :monotonic R[a][nodes[i]] += Re[a][i]
    end
end

function assemble_momentum_residual_nd!(R, dr::StokesDRND{D}, mesh, geo, element_v, element_P, phases_v, Pnum, backend, workgroup) where D
    foreach(x -> fill!(x, 0), R)
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    momentum_residual_nd_kernel!(backend, workgroup)(
        R, dr.v, dr.P, dr.T, Pnum, mesh.el2n, mesh.DoFsP, geo, phases_v,
        dr.η, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Nq, NqP,
        Val(D), Val(length(element_v)), Val(length(element_P)); ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@inline function pressure_element_residual_nd(
    v, P, P0, T, T0, el2n_v, dofsP, geo_v, geo_P, phases_P,
    α, ηb, Δt, NqP, iel, ::Val{D}, ::Val{NV}, ::Val{NP},
) where {D, NV, NP}
    nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    nodes_p = local_nodes_of(dofsP, iel, Val(NP))
    vloc = _gather_velocity_nd(v, nodes_v, Val(D), Val(NV))
    Ploc, P0loc = _gather_local(P, nodes_p, Val(NP)), _gather_local(P0, nodes_p, Val(NP))
    Tloc, T0loc = _gather_local(T, nodes_p, Val(NP)), _gather_local(T0, nodes_p, Val(NP))
    phase_p = _gather_phase(phases_P, nodes_p, iel, Val(NP))
    Re = integrate_PH_pressure_residual(
        vloc, Ploc, P0loc, Tloc, T0loc, geo_v[iel], geo_P[iel], phase_p, α, ηb, Δt, NqP,
    )
    return nodes_p, Re
end

@kernel function pressure_residual_nd_kernel!(
    RP, @Const(v), @Const(P), @Const(P0), @Const(T), @Const(T0),
    @Const(el2n_v), @Const(dofsP), @Const(geo_v), @Const(geo_P), @Const(phases_P),
    @Const(α), @Const(ηb), Δt, @Const(NqP), ::Val{D}, ::Val{NV}, ::Val{NP},
) where {D, NV, NP}
    iel = @index(Global)
    nodes, Re = pressure_element_residual_nd(
        v, P, P0, T, T0, el2n_v, dofsP, geo_v, geo_P, phases_P,
        α, ηb, Δt, NqP, iel, Val(D), Val(NV), Val(NP),
    )
    _add_local!(RP, nodes, Re, Val(true))
end

function assemble_pressure_residual_nd!(dr::StokesDRND{D}, mesh, geo_v, geo_P, element_v, element_P, phases_P, Δt, backend, workgroup) where D
    fill!(dr.RP, 0)
    NqP = shape_function_values(element_P, element_v.integration_points)
    pressure_residual_nd_kernel!(backend, workgroup)(
        dr.RP, dr.v, dr.P, dr.P0, dr.T, dr.T0,
        mesh.el2n, mesh.DoFsP, geo_v, geo_P, phases_P,
        dr.α, dr.ηb, Δt, NqP, Val(D), Val(length(element_v)), Val(length(element_P));
        ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function pressure_mass_nd_kernel!(MP, @Const(dofsP), @Const(geo_P), @Const(NqP), ::Val{NP}) where NP
    iel = @index(Global)
    nodes = local_nodes_of(dofsP, iel, Val(NP))
    Me = zero(NqP[1])
    for q in eachindex(NqP)
        Me += NqP[q] * geo_P[iel][q][2]
    end
    _add_local!(MP, nodes, Me, Val(true))
end

function assemble_pressure_mass_nd!(MP, mesh, geo_P, element_v, element_P, backend, workgroup)
    fill!(MP, 0)
    NqP = shape_function_values(element_P, element_v.integration_points)
    pressure_mass_nd_kernel!(backend, workgroup)(MP, mesh.DoFsP, geo_P, NqP, Val(length(element_P)); ndrange = mesh.nels)
    KA.synchronize(backend)
    return nothing
end

@inline function augmented_component_residual_nd(
    component, v, P, P0, T, T0, geo_v, geo_P, phase_v, phase_p,
    η, ηb, α, ρ0, K, g, Tref, Δt, γP, MP, Nq, NqP, ::Val{D},
) where D
    RP = integrate_PH_pressure_residual(v, P, P0, T, T0, geo_v, geo_P, phase_p, α, ηb, Δt, NqP)
    Pnum = γP .* RP ./ MP
    return integrate_momentum_residual_nd(v, P, Pnum, T, geo_v, phase_v, η, α, ρ0, K, g, Tref, Nq, NqP, Val(D))[component]
end

@inline function element_augmented_diagnostics_nd(
    dr::StokesDRND{D}, mesh, geo_v, geo_P, phases_v, phases_P, Δt, γP, iel,
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {D, NV, NP}
    nodes_v = local_nodes_of(mesh.el2n, iel, Val(NV))
    nodes_p = local_nodes_of(mesh.DoFsP, iel, Val(NP))
    v = _gather_velocity_nd(dr.v, nodes_v, Val(D), Val(NV))
    P, P0 = _gather_local(dr.P, nodes_p, Val(NP)), _gather_local(dr.P0, nodes_p, Val(NP))
    T, T0 = _gather_local(dr.T, nodes_p, Val(NP)), _gather_local(dr.T0, nodes_p, Val(NP))
    MP = _gather_local(dr.MP, nodes_p, Val(NP))
    γ = _gather_local(γP, nodes_p, Val(NP))
    pv = _gather_phase(phases_v, nodes_v, iel, Val(NV))
    pp = _gather_phase(phases_P, nodes_p, iel, Val(NP))
    gv, gp = geo_v[iel], geo_P[iel]

    blocks = ntuple(a -> ntuple(b -> begin
        f = vb -> begin
            vtrial = _seed_trial_velocity(v, vb, b)
            augmented_component_residual_nd(a, vtrial, P, P0, T, T0, gv, gp, pv, pp,
                dr.η, dr.ηb, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt, γ, MP, Nq, NqP, Val(D))
        end
        ForwardDiff.jacobian(f, v[b])
    end, Val(D)), Val(D))
    rows = ntuple(a -> SVector{NV}(ntuple(i -> sum(abs(blocks[a][b][i, j]) for b in 1:D, j in 1:NV), Val(NV))), Val(D))
    diag = ntuple(a -> SVector{NV}(ntuple(i -> abs(blocks[a][a][i, i]), Val(NV))), Val(D))
    return nodes_v, rows, diag
end

@kernel function augmented_diagnostics_nd_kernel!(rowsum, PC, @Const(dr), @Const(mesh), @Const(geo_v), @Const(geo_P), @Const(phases_v), @Const(phases_P), Δt, @Const(γP), @Const(Nq), @Const(NqP), ::Val{NV}, ::Val{NP}) where {NV, NP}
    iel = @index(Global)
    nodes, rows, diag = element_augmented_diagnostics_nd(dr, mesh, geo_v, geo_P, phases_v, phases_P, Δt, γP, iel, Nq, NqP, Val(NV), Val(NP))
    for a in eachindex(rows), i in 1:NV
        Atomix.@atomic :monotonic rowsum[a][nodes[i]] += rows[a][i]
        Atomix.@atomic :monotonic PC[a][nodes[i]] += diag[a][i]
    end
end

function assemble_augmented_diagnostics_nd!(dr::StokesDRND, mesh, geo_v, geo_P, element_v, element_P, phases_v, phases_P, Δt, γP, backend, workgroup)
    foreach(x -> fill!(x, 0), dr.rowsum)
    foreach(x -> fill!(x, 0), dr.PC)
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    augmented_diagnostics_nd_kernel!(backend, workgroup)(
        dr.rowsum, dr.PC, dr, mesh, geo_v, geo_P, phases_v, phases_P, Δt, γP, Nq, NqP,
        Val(length(element_v)), Val(length(element_P)); ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return nothing
end

function _scaled_velocity_residual_nd(R, PC)
    return maximum(norm(R[a] ./ PC[a]) for a in eachindex(R)) / sqrt(length(R[1]))
end

"""
    solve_stokes_dyrel_nd!(dr, mesh, cache, element_v, element_P, phases_v,
                           phases_P, Δt, γP, velocity_nodes, velocity_values,
                           backend, workgroup; kwargs...)

Solve a dimension-parametric Newtonian Stokes system with the same nested
Powell--Hestenes/DYREL strategy as the established 2D solver.
"""
function solve_stokes_dyrel_nd!(
    dr::StokesDRND{D}, mesh, cache, element_v, element_P, phases_v, phases_P,
    Δt, γP, velocity_nodes::NTuple{D}, velocity_values::NTuple{D}, backend, workgroup;
    ncheck = 25, ϵ_tol = dr.ϵ, iterMax = 5_000, total_iterMax = 20_000,
    max_ph_iterations = 100, rel_drop = 0.1, verbose = true,
) where D
    assemble_pressure_mass_nd!(dr.MP, mesh, cache.geo_P, element_v, element_P, backend, workgroup)
    assemble_augmented_diagnostics_nd!(dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P, phases_v, phases_P, Δt, γP, backend, workgroup)
    λmax = ntuple(a -> _checked_λmax(dr.rowsum[a], dr.PC[a], "StokesND component $a"), Val(D))
    Δτ = ntuple(a -> 2 / sqrt(λmax[a]) * dr.CFL_v, Val(D))
    αβ = ntuple(a -> _stokes_cheb(Δτ[a], zero(Δτ[a]), dr.c_fact), Val(D))
    αdr, βdr = ntuple(a -> αβ[a][1], Val(D)), ntuple(a -> αβ[a][2], Val(D))
    zeros_bc = ntuple(a -> zero(velocity_values[a]), Val(D))
    history = NamedTuple[]
    iter = 0
    err0 = nothing
    err = Inf

    for iph in 1:max_ph_iterations
        assemble_pressure_residual_nd!(dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P, phases_P, Δt, backend, workgroup)
        assemble_momentum_residual_nd!(dr.Rv, dr, mesh, cache.geo_v, element_v, element_P, phases_v, nothing, backend, workgroup)
        for a in 1:D
            apply_dirichlet!(dr.Rv[a], velocity_nodes[a], zeros_bc[a], backend, workgroup)
        end
        errP = norm(dr.RP ./ dr.MP) / sqrt(length(dr.P))
        # Rv is an integrated nodal force and therefore shrinks with element
        # volume under mesh refinement.  Rv/PC is the corresponding velocity
        # correction and gives a mesh-independent convergence measure.
        errV = _scaled_velocity_residual_nd(dr.Rv, dr.PC)
        err0 === nothing && (err0 = max(errP, errV, eps(errV)))
        err = min(max(errP, errV), max(errP, errV) / err0)
        push!(history, (; iter, iph, err, errV, errP))
        verbose && @printf("%dD PH=%03d iter=%06d err=%.3e Rv=%.3e RP=%.3e\n", D, iph, iter, err, errV, errP)
        err < ϵ_tol && return (; converged = true, iter, iph, err, errV, errP, history)

        target = max(errV * rel_drop, ϵ_tol)
        inner = 0
        # A PH pressure update changes the momentum equation even when its
        # residual was already below the absolute tolerance. Always perform at
        # least one velocity relaxation so pressure cannot advance repeatedly
        # against an unchanged velocity field.
        while (inner == 0 || errV > target) && inner < iterMax && iter < total_iterMax
            inner += 1
            iter += 1
            foreach(a -> copyto!(dr.Rv0[a], dr.Rv[a]), 1:D)
            assemble_pressure_residual_nd!(dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P, phases_P, Δt, backend, workgroup)
            @. dr.Pnum = γP * dr.RP / dr.MP
            assemble_momentum_residual_nd!(dr.Rv, dr, mesh, cache.geo_v, element_v, element_P, phases_v, dr.Pnum, backend, workgroup)
            for a in 1:D
                apply_dirichlet!(dr.Rv[a], velocity_nodes[a], zeros_bc[a], backend, workgroup)
                stokes_update_rate!(dr.rate[a], dr.Rv[a], dr.PC[a], βdr[a], length(dr.v[a]), backend, workgroup)
                stokes_update_variable!(dr.v[a], dr.rate[a], -αdr[a], length(dr.v[a]), backend, workgroup)
                apply_dirichlet!(dr.v[a], velocity_nodes[a], velocity_values[a], backend, workgroup)
            end
            if iszero(iter % ncheck)
                errV = _scaled_velocity_residual_nd(dr.Rv, dr.PC)
                λmin = ntuple(a -> _stokes_λmin(αdr[a], dr.rate[a], dr.Rv[a] .- dr.Rv0[a], dr.PC[a]), Val(D))
                assemble_augmented_diagnostics_nd!(dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P, phases_v, phases_P, Δt, γP, backend, workgroup)
                λmax = ntuple(a -> _checked_λmax(dr.rowsum[a], dr.PC[a], "StokesND component $a"), Val(D))
                Δτ = ntuple(a -> 2 / sqrt(λmax[a]) * dr.CFL_v, Val(D))
                αβ = ntuple(a -> _stokes_cheb(Δτ[a], λmin[a], dr.c_fact), Val(D))
                αdr, βdr = ntuple(a -> αβ[a][1], Val(D)), ntuple(a -> αβ[a][2], Val(D))
            end
        end
        # Use the constraint residual of the updated velocity, not the stale
        # value assembled before the final pseudo-time update.
        assemble_pressure_residual_nd!(dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P, phases_P, Δt, backend, workgroup)
        @. dr.P += γP * dr.RP / dr.MP
        iter >= total_iterMax && break
    end
    return (; converged = false, iter, iph = max_ph_iterations, err, history)
end
