import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Statistics
using StaticArrays
using KernelAbstractions
using Triangulate
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, arrows2d!, lines!, Point2f, DataAspect

const backend   = CPU()
const workgroup = 128

include("mesher.jl")
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
function main(; max_area = 1 / (1 * 64^2), show_plot = true)
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
    ncheck = 50         # convergence check interval
    ϵ_tol  = 1e-6        # relative residual tolerance

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
    P0_litho = Float64[ρ0[1] * abs(g[2]) * (Ly - c[2]) for c in coords_litho]
    copyto!(lp_dr.P, P0_litho)
    litho_tol = max(Lx, Ly) * eps(Float64) * 32
    top_nodes_litho = Int32[i for i in eachindex(coords_litho) if abs(coords_litho[i][2] - Ly) ≤ litho_tol]
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
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]

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

    iterMax       = 50_000   # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    rel_drop0     = 1e-1     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" Δt iterMax total_iterMax ncheck ϵ_tol

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = joinpath(@__DIR__, "output_stokes")
    mkpath(out_dir)

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
        vx_nodes = vx_nodes,
        vy_nodes = vy_nodes,
    )

    update_stokes_current_stress!(
        dr, mesh_stokes, geo_v, element_v, element_P,
        phases_solve, τ_old, plastic, τ, G_stokes, Δt, backend, workgroup,
    )

    P_cpu  = Array(dr.P)
    vx_cpu = Array(dr.vx)
    vy_cpu = Array(dr.vy)
    post = compute_strain_rate_stress_postprocess(
        vx_cpu, vy_cpu,
        el2n_v_cpu,
        Array(geo_v),
        τ,
        element_v,
    )
    mean_tauII = mean(post.tauII)

    vtk_path = joinpath(out_dir, "stokes_2D_sinking_block.vtk")
    write_stokes_vtk(
        vtk_path,
        mesh_stokes,
        coords_v,
        el2nP_cpu,
        DoFsP_cpu,
        P_cpu,
        vx_cpu,
        vy_cpu,
        post;
        title = "FEMTools Stokes 2D sinking block",
        cell_data = (; phase = cell_phase),
    )
    @info "Wrote VTK file" vtk_path mean_tauII iter=solve_stats.iter err=solve_stats.err
    show_plot || return (; mean_tauII, post)

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    # Per-element fields on the plotted pressure triangles.
    el_Vx = [mean(vx_cpu[el2nP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
    el_Vy = [mean(vy_cpu[el2nP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
    el_P  = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
            for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1400, 440))
    axes = Axis[]
    for (col, title, label, values, colormap) in (
        (1, "Horizontal velocity Vx", "Vx", el_Vx, :vik),
        (3, "Vertical velocity Vy", "Vy", el_Vy, :vik),
        (5, "Pressure P", "P", el_P, :glasgow),
    )
        limits = extrema(values)
        ax = Axis(fig[1, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
        poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
        Colorbar(fig[1, col + 1]; colormap, limits, label, width = 15, tellheight = false)
        push!(axes, ax)
    end

    # arrow_nodes = sort!(unique(vec(el2nP_cpu)))
    # arrow_step = max(1, length(arrow_nodes) ÷ 250)
    # arrow_nodes = arrow_nodes[1:arrow_step:end]
    # arrows2d!(
    #     axes[3],
    #     [coords_v[n][1] for n in arrow_nodes],
    #     [coords_v[n][2] for n in arrow_nodes],
    #     vx_cpu[arrow_nodes],
    #     vy_cpu[arrow_nodes];
    #     color = :white,
    #     lengthscale = 20,
    #     shaftwidth = 1,
    #     tipwidth = 8,
    #     tiplength = 8,
    # )

    xlo, xhi = cx - half_width, cx + half_width
    ylo, yhi = cy - half_width, cy + half_width
    for ax in axes
        lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo]; color = :white, linewidth = 1.5, linestyle = :dash)
    end

    show_plot && display(fig)
    # return (; mean_tauII, post)
    nothing
end

main()
