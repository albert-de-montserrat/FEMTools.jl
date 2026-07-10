using FEMTools

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Enzyme
using LinearAlgebra
using Printf
using Statistics
using StaticArrays
using KernelAbstractions
using Triangulate
using GLMakie: Figure, Axis, Colorbar, poly!, arrows2d!, lines!, Point2f, DataAspect, axislegend

const backend   = CPU()
const workgroup = 128

include("mesher.jl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@kernel inbounds = true function observationpoints_vy!(ResλVy, coords, cx, cy, half_width)
    i = @index(Global)
    x, y = coords[i]
    ResλVy[i] = abs(x - cx) ≤ half_width && abs(y - cy) ≤ half_width ? -1.0 : 0.0
end

function launch_observationpoints_vy!(ResλVy, coords, cx, cy, half_width, backend, workgroup)
    observationpoints_vy!(backend, workgroup)(
        ResλVy, coords, cx, cy, half_width;
        ndrange = length(ResλVy),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; max_area=1 / (1 * 64^2), Δt=1 / 6, show_plot=true) -> NamedTuple

Run one unstructured T7/P1-disc sinking-block Stokes solve.

The model builds a square domain with a rectangular inclusion, applies free-slip
boundary conditions, solves the Stokes system once, writes one VTK file, and
returns the stress diagnostics.
"""
function main(;
    max_area = 1 / 64^2,
    show_plot = true,
    ncheck = 50,
    ϵ_tol = 1.0e-6,
    iterMax = 50_000,
    total_iterMax = 50_000,
    adjoint_tol = 1.0e-6,
    adjoint_rel_drop = 5.0e-3,
    adjoint_iterMax = 50_000,
    adjoint_total_iterMax = 50_000,
    adjoint_max_ph_iterations = 100,
)
    Δt = 1
    # Domain
    Lx, Ly = 1.0, 1.0

    # Material (2 phases: matrix + inclusion)
    γfact = 20.0
    η     = (1.0,     1e2)   # shear viscosity
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     2e0)   # reference density
    K     = (Inf,     Inf)   # bulk modulus  (Inf → incompressible)
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (Inf,     Inf)   # Shear modulus
    G_stokes = G
    plastic = nothing
    g     = (0.0,     -1.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    # Inclusion geometry. The Triangle PSLG uses this rectangle as an internal
    # constrained boundary, so no element crosses the material interface.
    half_width = 0.1
    cx         = Lx / 2
    cy         = -Ly / 2

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, interface_nodes = build_triangle_t7_inclusion_mesh(;
        Lx, Ly,
        cx, cy, half_width,
        max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels half_width max_area n_interface_nodes=length(interface_nodes)

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)

    ξq_v    = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)
    ∂N∂ξq_P = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_v[q]), NQ_v)

    geo_v = Vector{NTuple{NQ_v, Tuple{SMatrix{NV, 2, Float64, 2NV}, Float64}}}(undef, mesh_stokes.nels)
    geo_P = Vector{NTuple{NQ_v, Tuple{SMatrix{NP, 2, Float64, 2NP}, Float64}}}(undef, mesh_stokes.nels)

    precompute_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n, ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels)
    precompute_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP, ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels)

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    dr = StokesDR(
        backend,
        mesh_stokes.nnodes,
        mesh_stokes.nnodesP,
        η, ηb, α;
        ρ0,
        K,
        g,
        Tref,
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
        # CFL_v = 0.03, CFL_P = 0.9, c_fact = 0.5,
    )
    M_P    = pressure_mass(dr)
    τ     = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — rectangular inclusion
    # ---------------------------------------------------------------------------

    in_incl(c) = abs(c[1] - cx) ≤ half_width && abs(c[2] - cy) ≤ half_width

    coords_v     = Array(mesh_stokes.coords)

    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    cell_phase = Int[
        in_incl((coords_v[el2nP_cpu[1, i]] + coords_v[el2nP_cpu[2, i]] + coords_v[el2nP_cpu[3, i]]) / 3) ? 2 : 1
        for i in 1:mesh_stokes.nels
    ]
    phases_solve = reshape(cell_phase, 1, :)


    corner_nodes = sort!(unique(vec(el2nP_cpu)))
    corner_id = Dict{Int32, Int32}(old => Int32(i) for (i, old) in enumerate(corner_nodes))
    coords_litho = coords_v[Int.(corner_nodes)]
    el2n_litho = Matrix{Int32}(undef, NP, mesh_stokes.nels)
    @inbounds for iel in 1:mesh_stokes.nels, a in 1:3
        el2n_litho[a, iel] = corner_id[Int32(el2nP_cpu[a, iel])]
    end
    mesh_litho = Mesh(backend, coords_litho, el2n_litho)
    ip_litho = element_P.integration_points
    NQ_litho = length(ip_litho.ω)
    ξq_litho = ntuple(q -> SVector(ip_litho.ξ[q], ip_litho.η[q]), NQ_litho)
    ∂N∂ξq_litho = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_litho[q]), NQ_litho)
    geo_litho = Vector{NTuple{NQ_litho, Tuple{SMatrix{NP, 2, Float64, 2NP}, Float64}}}(undef, mesh_litho.nels)
    precompute_geometry!(geo_litho, mesh_litho.coords, mesh_litho.el2n, ∂N∂ξq_litho, ip_litho.ω, Val(NP), mesh_litho.nels)

    lp_dr = LithostaticPressureDR(backend, mesh_litho.nnodes, ρ0, α, K; CFL = 0.9, ϵ = 1e-2)
    copyto!(lp_dr.phases, Int[in_incl(c) ? 2 : 1 for c in coords_litho])
    P0_litho = Float64[ρ0[1] * abs(g[2]) * (-c[2]) for c in coords_litho]
    copyto!(lp_dr.P, P0_litho)
    litho_tol = max(Lx, Ly) * eps(Float64) * 32
    top_nodes_litho = Int32[i for i in eachindex(coords_litho) if abs(coords_litho[i][2]) ≤ litho_tol]
    top_zero = zeros(Float64, length(top_nodes_litho))
    solver!(lp_dr, mesh_litho, geo_litho, element_P, top_nodes_litho, top_zero, top_zero,
        backend, workgroup; ncheck = 50, verbose = false, Tref = Tref, g = g)

    P_litho = Array(lp_dr.P)
    P_hydro = zeros(Float64, mesh_stokes.nnodesP)
    @inbounds for iel in 1:mesh_stokes.nels, a in 1:3
        P_hydro[DoFsP_cpu[a, iel]] = P_litho[corner_id[Int32(el2nP_cpu[a, iel])]]
    end
    copyto!(dr.P, P_hydro)
    copyto!(dr.P0, P_hydro)

    @info "Phases" n_incl=count(==(2), cell_phase)

    # ---------------------------------------------------------------------------
    # Boundary conditions — free slip on every wall:
    # normal velocity is zero, tangential velocity is unconstrained.
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] + Ly) ≤ tol]

    bc_vx_vals = zeros(Float64, length(vx_nodes))
    bc_vy_vals = zeros(Float64, length(vy_nodes))

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) max_vx = maximum(abs, bc_vx_vals) max_vy = maximum(abs, bc_vy_vals)

    # FEM pressure residuals are assembled in weak form:
    #
    #     RP_i = ∫ N_i (-∇⋅v) dΩ
    #
    # The Arrow-Hurwicz pressure step and numerical pressure
    # correction are calibrated for that pointwise residual.  If we feed the weak
    # residual directly into Pnum or P += γP*RP/M_P, the update is scaled by element
    # volume and pressure convergence stalls/refines incorrectly.
    #
    # Use the helper to assemble both:
    #   dr.M_P = ∫ N_i dΩ
    #   γP      = mean-viscosity pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction without
    # letting phase-local viscosity extremes set the pressure step.
    ηγP = ntuple(_ -> mean(η), Val(length(η)))
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup;
        phases_v = phases_solve, η = ηγP,
    )

    rel_drop0     = 1e-1     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" Δt iterMax total_iterMax ncheck ϵ_tol

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = joinpath(@__DIR__, "output_stokes")
    mkpath(out_dir)

    # ---------------------------------------------------------------------------
    # Forward solve
    # ---------------------------------------------------------------------------

    solve_stats = solve_stokes_dyrel!(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_solve, phases_solve, τ_old, plastic, G_stokes, Δt, γP,
        Γnodes, bc_vx_vals, bc_vy_vals, backend, workgroup;
        ncheck,
        ϵ_tol,
        iterMax,
        total_iterMax,
        rel_drop0,
        verbose = verbose_PH,
        verbose_inner = verbose_DR,
        vx_nodes,
        vy_nodes,
        collect_history = true,
    )
    solve_stats.converged || @warn "Forward solve did not reach tolerance" solve_stats

    # ---------------------------------------------------------------------------
    # Visualisation helper — per-element average of nodal fields on the P triangles
    # ---------------------------------------------------------------------------

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]
    xlo, xhi = cx - half_width, cx + half_width
    ylo, yhi = cy - half_width, cy + half_width

    function plot_fields(vx_dofs, vy_dofs, P_dofs, titles)
        el_vx = [mean(vx_dofs[el2nP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
        el_vy = [mean(vy_dofs[el2nP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
        el_P  = [mean(P_dofs[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
        fig = Figure(size = (1400, 440))
        for (col, title, values, colormap) in (
                (1, titles[1], el_vx, :vik),
                (3, titles[2], el_vy, :vik),
                (5, titles[3], el_P,  :glasgow),
            )
            limits = extrema(values)
            ax = Axis(fig[1, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
            poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
            Colorbar(fig[1, col + 1]; colormap, limits, width = 15, tellheight = false)
            lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo];
                color = :white, linewidth = 1.5, linestyle = :dash)
        end
        display(fig)
        return fig
    end

    # Combined figure matching the finite-difference reference: forward Vx/Vy/P,
    # adjoint λVx/λVy, and the forward + adjoint residual-evolution trace.
    function plot_summary(vx_dofs, vy_dofs, P_dofs, λvx_dofs, λvy_dofs, fwd_hist, adj_hist)
        el(dofs, conn) = [mean(dofs[conn[:, i]]) for i in 1:mesh_stokes.nels]
        fig = Figure(size = (1100, 1300))
        for (row, col, title, values, colormap) in (
                (1, 1, "Vx",  el(vx_dofs,  el2nP_cpu), :vik),
                (1, 3, "Vy",  el(vy_dofs,  el2nP_cpu), :vik),
                (2, 1, "P",   el(P_dofs,   DoFsP_cpu), :glasgow),
                (2, 3, "λVx", el(λvx_dofs, el2nP_cpu), :vik),
                (3, 1, "λVy", el(λvy_dofs, el2nP_cpu), :vik),
            )
            limits = extrema(values)
            ax = Axis(fig[row, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
            poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
            Colorbar(fig[row, col + 1]; colormap, limits, width = 15, tellheight = false)
            lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo];
                color = :white, linewidth = 1.5, linestyle = :dash)
        end

        ax = Axis(fig[3, 3:4]; xlabel = "iteration", ylabel = "log10 residual",
            title = "Residual evolution")
        logres(v) = log10.(max.(v, eps()))
        if !isempty(fwd_hist)
            it = Float64[h.iter for h in fwd_hist]
            lines!(ax, it, logres([h.err_v for h in fwd_hist]); label = "forward V")
            lines!(ax, it, logres([h.err_P for h in fwd_hist]); label = "forward P")
        end
        if !isempty(adj_hist)
            it = Float64[h.iter for h in adj_hist]
            lines!(ax, it, logres([h.err_v for h in adj_hist]); label = "adjoint V", linestyle = :dash)
            lines!(ax, it, logres([h.err_P for h in adj_hist]); label = "adjoint P", linestyle = :dash)
        end
        (isempty(fwd_hist) && isempty(adj_hist)) || axislegend(ax; position = :rt)

        display(fig)
        return fig
    end

    show_plot && plot_fields(Array(dr.vx), Array(dr.vy), Array(dr.P),
        ("Forward Vx", "Forward Vy", "Forward P"))

    # ---------------------------------------------------------------------------
    # Adjoint solve: (dR/du)' λ + dJ/du = 0
    # ---------------------------------------------------------------------------

    objective_vx = zero(dr.Rv_x)
    objective_vy = zero(dr.Rv_y)
    launch_observationpoints_vy!(
        objective_vy, mesh_stokes.coords, cx, cy, half_width, backend, workgroup,
    )

    λvx = zero(dr.vx)
    λvy = zero(dr.vy)
    λP  = zero(dr.P)
    λrate_vx = zero(dr.vx)
    λrate_vy = zero(dr.vy)

    ResλVx = zero(dr.Rv_x)
    ResλVy = zero(dr.Rv_y)
    ResλP  = zero(dr.P)
    ResλVx0 = zero(dr.Rv_x)
    ResλVy0 = zero(dr.Rv_y)

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

    zero_vx_bc = zero(bc_vx_vals)
    zero_vy_bc = zero(bc_vy_vals)

    # Reuse the forward augmented diagonal preconditioner for the transpose
    # velocity block. Its diagonal is identical to that of the transpose. The
    # jacobian depends only on the (now frozen) forward state, so λmax and the
    # pseudo-time step are constant through the adjoint solve; only λmin is
    # re-estimated below.
    FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
        element_v, element_P, phases_solve, phases_solve,
        τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
        dr.ηb, Δt, γP, M_P, backend, workgroup,
    )

    λmax_vx = FEMTools._checked_λmax(dr.∂Rv_x∂vx, dr.PC_vx, "adjoint vx")
    λmax_vy = FEMTools._checked_λmax(dr.∂Rv_y∂vy, dr.PC_vy, "adjoint vy")
    Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
    Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v
    α_vx, β_vx = FEMTools._stokes_cheb(Δτ_vx, zero(λmax_vx), dr.c_fact)
    α_vy, β_vy = FEMTools._stokes_cheb(Δτ_vy, zero(λmax_vy), dr.c_fact)

    function assemble_adjoint_residual!()
        fill!(dvx, 0)
        fill!(dvy, 0)
        fill!(dP, 0)
        fill!(dPnum, 0)
        copyto!(seed_Rv_x, λvx)
        copyto!(seed_Rv_y, λvy)

        # Momentum transpose: (∂Rv/∂v)ᵀλv → dvx,dvy, (∂Rv/∂P)ᵀλv → dP,
        # and (∂Rv/∂Pnum)ᵀλv → dPnum.
        FEMTools.assemble_momentum_residual_matrices_atomix_adj!(
            dr.Rv_x, seed_Rv_x, dr.Rv_y, seed_Rv_y,
            dr.vx, dvx, dr.vy, dvy, dr.P, dP, dr.T, Pnum, dPnum,
            mesh_stokes, geo_v, element_v, element_P,
            phases_solve, τ_old, plastic,
            dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            workgroup,
        )

        # (∂Rv/∂P)ᵀλv is the adjoint pressure-constraint residual (the analogue of
        # the finite-difference `ResλP .= dP`). Capture it before the pressure
        # pullbacks reuse dP.
        copyto!(ResλP, dP)

        # Powell–Hestenes augmented grad-div self-coupling of the velocity adjoint.
        # The forward momentum uses Pnum(v) = γP·RP(v)/M_P, so the chain through Pnum
        # closes as (∂RP/∂v)ᵀ(γP·(∂Rv/∂Pnum)ᵀλv/M_P). This is the FEM equivalent of
        # the FD `Schurx`/`Schury` term.
        @. seed_RP = γP * dPnum / M_P
        fill!(dP_scratch, 0)
        FEMTools.assemble_pressure_residual_matrices_atomix_adj!(
            dr, seed_RP, dvx, dvy, dP_scratch,
            mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_solve, Δt, workgroup,
        )

        # Saddle-point coupling to the pressure adjoint λP: (∂RP/∂v)ᵀλP.
        copyto!(seed_RP, λP)
        fill!(dP_scratch, 0)
        FEMTools.assemble_pressure_residual_matrices_atomix_adj!(
            dr, seed_RP, dvx, dvy, dP_scratch,
            mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_solve, Δt, workgroup,
        )

        @. ResλVx = objective_vx + dvx
        @. ResλVy = objective_vy + dvy
        FEMTools.apply_dirichlet!(ResλVx, vx_nodes, zero_vx_bc, backend, workgroup)
        FEMTools.apply_dirichlet!(ResλVy, vy_nodes, zero_vy_bc, backend, workgroup)
        return nothing
    end

    adjoint_ncheck = ncheck
    adjoint_iter = 0
    adjoint_err = Inf
    adjoint_err_v0 = 1.0
    adjoint_err_P0 = 1.0
    adjoint_history = NamedTuple[]
    adjoint_converged = false
    adjoint_itPH = 0

    @info "Starting adjoint PH/DYREL solve" adjoint_tol adjoint_rel_drop

    for itPH in 1:adjoint_max_ph_iterations
        adjoint_itPH = itPH
        assemble_adjoint_residual!()

        err_v = max(norm(ResλVx), norm(ResλVy)) / sqrt(mesh_stokes.nnodes)
        err_P = norm(ResλP) / sqrt(mesh_stokes.nnodesP)
        if itPH == 1
            adjoint_err_v0 = err_v + eps(err_v)
            adjoint_err_P0 = err_P + eps(err_P)
        end
        adjoint_err = max(
            min(err_v, err_v / adjoint_err_v0),
            min(err_P, err_P / adjoint_err_P0),
        )
        push!(adjoint_history, (; iter = adjoint_iter, itPH, err = adjoint_err, err_v, err_P))
        @printf("adj PH=%03d iter=%06d err=%.3e Rv=%.3e RP=%.3e\n",
            itPH, adjoint_iter, adjoint_err, err_v, err_P)

        if adjoint_err < adjoint_tol
            adjoint_converged = true
            break
        end

        target_v = max(err_v * adjoint_rel_drop, adjoint_tol)
        inner = 0
        while err_v > target_v && inner < adjoint_iterMax && adjoint_iter < adjoint_total_iterMax
            inner += 1
            adjoint_iter += 1
            copyto!(ResλVx0, ResλVx)
            copyto!(ResλVy0, ResλVy)

            FEMTools.stokes_update_rate!(λrate_vx, ResλVx, dr.PC_vx, β_vx,
                mesh_stokes.nnodes, backend, workgroup)
            FEMTools.stokes_update_variable!(λvx, λrate_vx, -α_vx,
                mesh_stokes.nnodes, backend, workgroup)
            FEMTools.stokes_update_rate!(λrate_vy, ResλVy, dr.PC_vy, β_vy,
                mesh_stokes.nnodes, backend, workgroup)
            FEMTools.stokes_update_variable!(λvy, λrate_vy, -α_vy,
                mesh_stokes.nnodes, backend, workgroup)

            FEMTools.apply_dirichlet!(λvx, vx_nodes, zero_vx_bc, backend, workgroup)
            FEMTools.apply_dirichlet!(λvy, vy_nodes, zero_vy_bc, backend, workgroup)
            FEMTools.apply_dirichlet!(λrate_vx, vx_nodes, zero_vx_bc, backend, workgroup)
            FEMTools.apply_dirichlet!(λrate_vy, vy_nodes, zero_vy_bc, backend, workgroup)

            assemble_adjoint_residual!()

            if iszero(adjoint_iter % adjoint_ncheck)
                err_v = max(norm(ResλVx), norm(ResλVy)) / sqrt(mesh_stokes.nnodes)

                # Re-estimate λmin from the rate/residual change and refresh the
                # Chebyshev step, exactly as the forward DYREL loop does. Δτ and
                # λmax stay fixed (the jacobian depends only on the frozen forward
                # state).
                λmin_vx = FEMTools._stokes_λmin(α_vx, λrate_vx, ResλVx .- ResλVx0, dr.PC_vx)
                λmin_vy = FEMTools._stokes_λmin(α_vy, λrate_vy, ResλVy .- ResλVy0, dr.PC_vy)
                α_vx, β_vx = FEMTools._stokes_cheb(Δτ_vx, λmin_vx, dr.c_fact)
                α_vy, β_vy = FEMTools._stokes_cheb(Δτ_vy, λmin_vy, dr.c_fact)
            end
        end

        # Arrow-Hurwicz update for the pressure adjoint, followed by gauge fixing.
        @. λP += γP * ResλP / M_P
        # FEMTools.remove_pressure_mean!(λP, M_P)

        adjoint_iter >= adjoint_total_iterMax && break
    end

    assemble_adjoint_residual!()
    final_err_v = max(norm(ResλVx), norm(ResλVy)) / sqrt(mesh_stokes.nnodes)
    final_err_P = norm(ResλP) / sqrt(mesh_stokes.nnodesP)
    final_err = max(
        min(final_err_v, final_err_v / adjoint_err_v0),
        min(final_err_P, final_err_P / adjoint_err_P0),
    )
    adjoint_stats = (;
        converged = adjoint_converged || final_err < adjoint_tol,
        itPH = adjoint_itPH,
        iter = adjoint_iter,
        err = final_err,
        err_v = final_err_v,
        err_P = final_err_P,
        history = adjoint_history,
    )
    @info "Adjoint solve complete" adjoint_stats

    show_plot && plot_summary(Array(dr.vx), Array(dr.vy), Array(dr.P),
        Array(λvx), Array(λvy), solve_stats.history, adjoint_history)

    return (; dr, mesh_stokes, solve_stats, adjoint_stats, λvx, λvy, λP, objective_vy)
end

# if abspath(PROGRAM_FILE) == @__FILE__
out = main();
nothing
# end
