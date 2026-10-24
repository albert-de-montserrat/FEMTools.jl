# Time-dependent discrete adjoint of `dike_memory2D.jl`: the gradient of an
# objective J = Σₙ wₙ vy(station, step n) with respect to the injection rate of
# every step and dike segment, through the stress and pressure history.
#
# Step n solves R_n(u_n; h_{n-1}) = 0 for u = (v, P), then advances the
# integration-point history h = (τ_old, P_old) by h_n = H_n(u_n, h_{n-1}): the
# stress update with the return map, the Jaumann rotation, and the corrected
# pressure. Backward from the last step, with ν_n = dJ/dh_n and ν_N = 0,
#
#     (∂R_n/∂u_n)ᵀ λ_n = -(∂J/∂u_n + (∂H_n/∂u_n)ᵀ ν_n)
#     ν_{n-1} = (∂H_n/∂h_{n-1})ᵀ ν_n + (∂R_n/∂h_{n-1})ᵀ λ_n
#     dJ/dln Q_n = λ_nᵀ ∂R_n/∂ln Q_n.
#
# Temperature, pore and lithostatic pressure, and the mesh motion are held at
# their forward values: J does not see Q through them.

include(joinpath(@__DIR__, "dike_memory2D.jl"))

"""
    restore_step!(ctx, n) -> data

Load the converged state of step `n` into the solver and meshes of `ctx` and
return its element data in the layout of `element_state`.
"""
function restore_step!(ctx, n)
    cp = ctx.checkpoints[n]
    (; dr, mesh_stokes, element_v) = ctx
    for mesh in (ctx.mesh_v, mesh_stokes, ctx.mesh_T)
        copyto!(mesh.coords, cp.coords)
    end
    update_geometry!(mesh_stokes; workgroup)
    update_geometry!(ctx.mesh_T; workgroup)
    foreach(copyto!, (dr.v.x, dr.v.y), cp.v)
    foreach(copyto!, (dr.P, dr.P0, dr.T, dr.T0, dr.Pf, dr.Q, dr.M_P), (cp.P, cp.P0, cp.T, cp.T0, cp.Pf, cp.Q, cp.M_P))
    foreach(copyto!, (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy), cp.τ_old)
    return (;
        cp.v, cp.P, el2n = ctx.el2n_v, ctx.DoFsP,
        geo_v = Array(mesh_stokes.geometry.geo_v), geo_P = Array(mesh_stokes.geometry.geo_P),
        ∂N∂ξ = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_gradients(element_v)),
        Nq = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_values(element_v)),
        NqP = FEMTools.quadrature_table(CPU(), ctx.NqP),
        cp.T, cp.T0, cp.P_old, cp.Pf, cp.Q, cp.P_litho, cp.τ_old, ctx.phases_v, ctx.phases_P,
        dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.ηb, dr.g, dr.Tref, Δt = ctx.Δτ, ctx.plastic, cp.γP, cp.M_P,
    )
end

"""
    element_history(e, data, x, Val(NV), Val(NP), Val(NQ)) -> SVector{4NQ}

History map of one element: from `x = [vx; vy; P; τxx_old; τyy_old; τxy_old]`
(nodal velocity and pressure, old stress per integration point) to the next
`[τxx_old; τyy_old; τxy_old; P_old]` per integration point, as
`update_stokes_current_stress!` followed by `rotate_stress!` produce it.
"""
function element_history(e, data, x, ::Val{NV}, ::Val{NP}, ::Val{NQ}) where {NV, NP, NQ}
    vx = SVector{NV}(ntuple(i -> x[i], Val(NV)))
    vy = SVector{NV}(ntuple(i -> x[NV + i], Val(NV)))
    P = SVector{NP}(ntuple(i -> x[2NV + i], Val(NP)))
    old(c, q) = x[2NV + NP + (c - 1) * NQ + q]
    per_point = ntuple(Val(NQ)) do q
        ∂N∂x, _ = e.geo_v[q]
        Pfq = dot(data.NqP[q], e.Pf)
        (τxx, τyy, τxy), Pe = FEMTools.deviatoric_stress_and_pressure(
            (vx, vy), ∂N∂x, data.Nq[q], data.η, data.G, e.phase_v, data.Δt,
            (old(1, q), old(2, q), old(3, q)), dot(data.NqP[q], P) - Pfq, data.plastic, nothing, nothing,
        )
        ω = FEMTools._vorticity((∂N∂x' * vx, ∂N∂x' * vy))
        # `rotate_elastic_stress`'s 2D rotation, with a plain `sincos`: its
        # `@fastmath` form rejects dual numbers.
        rotated = FEMTools.GeoParams.tensor_rotation((τxx, τyy, τxy), sincos(ω * data.Δt)...)
        (rotated[1], rotated[2], rotated[3], Pe + Pfq)
    end
    return SVector{4NQ}(ntuple(i -> per_point[mod1(i, NQ)][cld(i, NQ)], Val(4NQ)))
end

"""
    element_residual_history(e, data, h, Val(NQ)) -> SVector

Plain element residual `[Rv_x; Rv_y; RP]` at the converged state of `e` with
the history replaced by `h = [τxx_old; τyy_old; τxy_old; P_old]`.
"""
function element_residual_history(e, data, h, ::Val{NQ}) where {NQ}
    τ_old = FEMTools.IntegrationPointStress(ntuple(c -> SVector{NQ}(ntuple(q -> h[(c - 1) * NQ + q], Val(NQ))), Val(3)))
    P_old = FEMTools.IntegrationPointPressure(SVector{NQ}(ntuple(q -> h[3NQ + q], Val(NQ))))
    e_h = merge(e, (; τ_old, P_old))
    return vcat(element_momentum(e_h, data, data.K, data.plastic, nothing), element_continuity(e_h, data, data.K, one(eltype(h))))
end

element_inputs(e) = [e.v[1]; e.v[2]; e.P; e.τ_old.τ[1]; e.τ_old.τ[2]; e.τ_old.τ[3]]
element_history_old(e) = [e.τ_old.τ[1]; e.τ_old.τ[2]; e.τ_old.τ[3]; e.P_old.P]

"""
    memory_adjoint(ctx, station_nodes, weights; adjoint_tol = 1e-8, verbose = true) -> NamedTuple

Gradients of `J_s = Σₙ weights[n] vy(station_nodes[s], step n)` (characteristic
units) for each station, from one backward sweep over the checkpoints of
`ctx = main(; checkpoint = true, ...).adjoint_context`. Returns
`dJ_dlnQ[s, n, segment]`, the derivative with respect to the log injection rate
of step `n` on each dike segment, and `dJ_dlneta[iel, s]`, with respect to a
time-constant log viscosity per element, both in the units of J.
"""
function memory_adjoint(ctx, station_nodes, weights; adjoint_tol = 1.0e-8, verbose = true)
    nsteps = length(ctx.checkpoints)
    length(weights) == nsteps || throw(DimensionMismatch("need $nsteps weights, got $(length(weights))"))
    (; dr, mesh_stokes, element_v, element_P) = ctx
    NV, NP = length(element_v), length(element_P)
    NQ = length(element_v.integration_points.ω)
    nels, nnodes, nP = mesh_stokes.nels, mesh_stokes.nnodes, mesh_stokes.nnodesP
    nu = 2NV + NP
    nstations = length(station_nodes)
    ν = [zeros(4NQ, nels) for _ in 1:nstations]
    λv = [(zeros(nnodes), zeros(nnodes)) for _ in 1:nstations]
    λP = [zeros(nP) for _ in 1:nstations]
    dJ_dlnQ = zeros(nstations, nsteps, ctx.n_segments)
    dJ_dlneta = zeros(nels, nstations)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)
    for n in nsteps:-1:1
        data = restore_step!(ctx, n)
        t_jac = @elapsed begin
            JH = Vector{Matrix{Float64}}(undef, nels)
            JR = Vector{Matrix{Float64}}(undef, nels)
            for iel in 1:nels
                e = element_state(data, iel, Val(NV), Val(NP))
                JH[iel] = ForwardDiff.jacobian(x -> element_history(e, data, x, Val(NV), Val(NP), Val(NQ)), element_inputs(e))
                JR[iel] = ForwardDiff.jacobian(h -> element_residual_history(e, data, h, Val(NQ)), element_history_old(e))
            end
        end
        t_adj = @elapsed for s in 1:nstations
            load_v = (zeros(nnodes), zeros(nnodes))
            load_P = zeros(nP)
            load_v[2][station_nodes[s]] += weights[n]
            for iel in 1:nels
                g = JH[iel]' * view(ν[s], :, iel)
                nodes_v = view(ctx.el2n_v, 1:NV, iel)
                nodes_P = view(ctx.DoFsP, 1:NP, iel)
                for a in 1:NV
                    load_v[1][nodes_v[a]] += g[a]
                    load_v[2][nodes_v[a]] += g[NV + a]
                end
                for a in 1:NP
                    load_P[nodes_P[a]] += g[2NV + a]
                end
            end
            stats = solve_stokes_adjoint_dyrel!(
                dr, mesh_stokes, mesh_stokes.geometry.geo_v, mesh_stokes.geometry.geo_P, element_v, element_P,
                ctx.phases_v, ctx.phases_P, τ_old, ctx.plastic, dr.G, ctx.Δτ, data.γP,
                load_v, λv[s], λP[s], backend, workgroup;
                v_nodes = (ctx.bc_vx.DoFs, ctx.bc_vy.DoFs), objective_P = load_P, workspace = ctx.workspace,
                measure_λmax = false, adjoint_tol, iterMax = 200_000, total_iterMax = 200_000,
                max_ph_iterations = 1000, verbose = false,
            )
            stats.converged || error("station $s adjoint did not converge at step $n: err = $(stats.err)")
            derivatives, λP_plain = log_sensitivities(data, λv[s], λP[s], Val(NV), Val(NP))
            for (iel, j) in zip(ctx.dike_cells, ctx.dike_segment)
                dJ_dlnQ[s, n, j] += derivatives.dlnJ_dlnQ[iel]
            end
            dJ_dlneta[:, s] .+= derivatives.dlnJ_dlneta
            ν_previous = similar(ν[s])
            for iel in 1:nels
                nodes_v = view(ctx.el2n_v, 1:NV, iel)
                nodes_P = view(ctx.DoFsP, 1:NP, iel)
                λ_e = [λv[s][1][nodes_v]; λv[s][2][nodes_v]; λP_plain[nodes_P]]
                # The old pressure does not enter the history map, only the residual.
                ν_previous[:, iel] = [view(JH[iel], :, nu + 1:nu + 3NQ)' * view(ν[s], :, iel); zeros(NQ)] +
                    JR[iel]' * λ_e
            end
            ν[s] = ν_previous
        end
        verbose && @info "Adjoint step" n t_jac t_adj dJ_dlnQ = round.(vec(sum(dJ_dlnQ[:, n, :]; dims = 2)); sigdigits = 4)
    end
    return (; dJ_dlnQ, dJ_dlneta)
end
