import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using DomainSets
using DomainSets: ×
using KernelAbstractions
using Triangulate
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend   = CPU()
const workgroup = 128

"""
    build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, n_circle=96, max_area=nothing) -> Tuple

Build an unstructured T7 velocity mesh around a circular inclusion.

Triangulate.jl generates a second-order T6 PSLG mesh with the circle as a
constrained internal boundary. The local midpoint ordering is remapped to
FEMTools' T6/T7 convention, then one centroid bubble node is appended per
element.
"""
function build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, n_circle = 96, max_area = nothing)
    rect_pts  = Cdouble[0.0 Lx  Lx 0.0;
                        0.0 0.0 Ly Ly]
    rect_segs = Cint[1 2; 2 3; 3 4; 4 1]'

    θ = range(0, 2π; length = n_circle + 1)[1:end-1]
    circ_pts = Matrix{Cdouble}(hcat(cx .+ r .* cos.(θ), cy .+ r .* sin.(θ))')
    circ_segs = Matrix{Cint}(hcat([
        [4 + i; 4 + mod1(i + 1, n_circle)] for i in 1:n_circle
    ]...))

    tio = TriangulateIO()
    tio.pointlist = hcat(rect_pts, circ_pts)
    tio.segmentlist = hcat(rect_segs, circ_segs)

    flags = isnothing(max_area) ? "pqo2Q" : "pq30o2a$(max_area)Q"
    result, _ = triangulate(flags, tio)

    pts = result.pointlist
    tris_t6 = Matrix{Int32}(result.trianglelist)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]

    # Triangle's second-order boundary nodes lie on straight constrained
    # segments. Project the circular-interface nodes back to the analytical
    # radius so the inclusion boundary is fitted by the high-order geometry.
    sagitta = r * (1 - cos(π / n_circle))
    circle_tol = max(2.5 * sagitta, 100eps(Float64) * max(Lx, Ly))
    for i in eachindex(coords)
        dx = coords[i][1] - cx
        dy = coords[i][2] - cy
        radius = hypot(dx, dy)
        if abs(radius - r) ≤ circle_tol && radius > 0
            coords[i] = SVector{2, Float64}(cx + r * dx / radius, cy + r * dy / radius)
        end
    end

    n_t6 = length(coords)
    nels = size(tris_t6, 2)
    el2n = Matrix{Int32}(undef, 7, nels)
    el2n[1:3, :] .= tris_t6[1:3, :]
    el2n[4, :] .= tris_t6[6, :] # FEMTools node 4 = mid(1, 2)
    el2n[5, :] .= tris_t6[4, :] # FEMTools node 5 = mid(2, 3)
    el2n[6, :] .= tris_t6[5, :] # FEMTools node 6 = mid(3, 1)
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        c1 = coords[tris_t6[1, iel]]
        c2 = coords[tris_t6[2, iel]]
        c3 = coords[tris_t6[3, iel]]
        push!(coords, (c1 + c2 + c3) / 3)
        el2n[7, iel] = Int32(n_t6 + iel)
    end

    tol = 100eps(Float64) * max(Lx, Ly)
    outer_nodes = Int32[
        i for i in 1:n_t6
        if abs(coords[i][1]) ≤ tol ||
           abs(coords[i][1] - Lx) ≤ tol ||
           abs(coords[i][2]) ≤ tol ||
           abs(coords[i][2] - Ly) ≤ tol
    ]
    circle_nodes = Int32[
        i for i in 1:n_t6
        if abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) ≤ circle_tol
    ]

    return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(circle_nodes))
end

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; nsteps=15, n_circle=96, max_area=1 / (1 * 64^2), Δt=1 / 6, show_plot=true) -> NamedTuple

Run the unstructured T7/P1-disc pure-shear Stokes example.

The model builds a square domain with a circular inclusion, applies pure-shear
boundary conditions, advances the viscoelastic-plastic Stokes solve, writes one
VTK file per physical step, and returns the stress-history diagnostics.
"""
function main(;
        nsteps = 15, n_circle = 96, max_area = 1 / (1 * 64^2), Δt = 1 / 6,
        show_plot = true, write_output = true, verbose = true,
        measure_λmax = false, λmax_safety = 1.1)
    # Domain
    Lx, Ly = 1.0, 1.0

    # Background pure-shear strain rate (non-dimensional)
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion)
    γfact = 20.0
    η     = (1.0,     1.0)   # shear viscosity
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     1.0)   # reference density
    K     = (4e0,     4e0)   # bulk modulus  (Inf → incompressible)
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (1e0,     0.5)   # Shear modulus
    G_stokes = G
    # Cohesion chosen so the yield stress C·cosϕ = 1.6 at zero pressure.
    # Background deviatoric stress in pure shear is 2η·ε̇_bg = 2, so the
    # inclusion (lower G) will enter the plastic regime after a few steps.
    τy      = 1.6 / cosd(30)                   # cohesion C; yield stress = C·cosϕ = 1.6
    plastic = DruckerPrager(
        (π/6, π/6),                            # friction angle ϕ = 30° [rad]
        (0.0, 0.0),                            # dilation angle Ψ = 0°  [rad] (non-associated)
        (τy, τy),                              # cohesion C [same for both phases]
        (8.0e-3,  8.0e-3),                     # plastic regularisation viscosity η_reg
        K,                                     # Kb (passed separately from elastic K)
    )
    g     = (0.0,     0.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    ncheck = 100          # convergence check interval
    ϵ_tol  = 1e-6        # relative residual tolerance

    # Inclusion geometry. The Triangle PSLG uses this circle as an internal
    # constrained boundary, so no element crosses the material interface.
    r_incl = 0.1
    cx     = Lx / 2
    cy     = Ly / 2

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_triangle_t7_inclusion_mesh(;
        Lx, Ly,
        cx, cy, r = r_incl,
        n_circle,
        max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_circle max_area n_interface_nodes=length(circle_nodes)

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)

    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    geo_v = cache.geo_v

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    material = StokesMaterial(; η, ηb, G = G_stokes, α, ρ0, K, g = Tuple(g), Tref)
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
    )
    M_P = pressure_mass(dr)
    τ = (dr.τ.xx, dr.τ.yy, dr.τ.xy)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)

    # ---------------------------------------------------------------------------
    # Phase assignment — circular inclusion
    # ---------------------------------------------------------------------------

    in_incl(c) = (c[1] - cx)^2 + (c[2] - cy)^2 ≤ r_incl^2

    coords_v     = Array(mesh_stokes.coords)
    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    cell_phase = Int[
        in_incl(sum(a -> coords_v[mesh_stokes.el2n[a, iel]], 1:NV) / NV) ? 2 : 1
        for iel in 1:mesh_stokes.nels
    ]
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)

    @info "Phases" n_incl_v=count(==(2), phases_v_cpu) n_incl_P=count(==(2), phases_P_cpu)

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear
    #   vx = +ε̇_bg * (x - Lx/2),   vy = -ε̇_bg * (y - Ly/2)
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]

    bc_vx_vals = [ ε̇_bg * (coords[n][1] - Lx / 2) for n in vx_nodes]
    bc_vy_vals = [-ε̇_bg * (coords[n][2] - Ly / 2) for n in vy_nodes]
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals)
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals)

    # Seed the full interior with the analytical pure-shear field so the
    # solver starts with a good initial guess (boundary nodes are overwritten
    # by apply_bc! below; the result is identical on those nodes).
    copyto!(dr.v.x, [ ε̇_bg * (c[1] - Lx / 2) for c in coords_v])
    copyto!(dr.v.y, [-ε̇_bg * (c[2] - Ly / 2) for c in coords_v])

    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)

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
    #   γP      = local viscosity-weighted pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction, but
    # adapts the pressure step to viscosity contrasts.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, cache, γfact, Δt; workgroup, phases_v = phases_v_cpu,
    )

    time_history = zeros(Float64, nsteps)
    mean_tauII_history = zeros(Float64, nsteps)

    iterMax       = 50_000   # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    rel_drop0     = 0.75     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt iterMax total_iterMax ncheck ϵ_tol

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = joinpath(@__DIR__, "output_stokes")
    mkpath(out_dir)
    post = nothing
    solve_stats_history = NamedTuple[]
    solve_time = 0.0

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        @info "Physical time step" istep nsteps t

        solve_stats = nothing
        solve_time += @elapsed solve_stats = solve_stokes_dyrel!(
                dr, mesh_stokes, cache, bc_vx, bc_vy, Δt, γP;
                phases_v = phases_v_cpu, phases_P = phases_P_cpu, τ_old, plastic, workgroup,
                ncheck, ϵ_tol, iterMax, total_iterMax, rel_drop0,
                verbose = verbose && verbose_PH,
                verbose_inner = verbose && verbose_DR,
                measure_λmax, λmax_safety)
        push!(solve_stats_history, solve_stats)

        update_stokes_current_stress!(
            dr, mesh_stokes, cache, τ, Δt;
            phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.v.x)
        vy_cpu = Array(dr.v.y)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(geo_v),
            τ,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        copyto!(dr.τ_old.xx, dr.τ.xx)
        copyto!(dr.τ_old.yy, dr.τ.yy)
        copyto!(dr.τ_old.xy, dr.τ.xy)

        if write_output
            vtk_path = joinpath(out_dir, @sprintf("stokes_2D_pure_shear_triangle_%04d.vtk", istep))
            write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
            @info "Wrote VTK file" vtk_path mean_tauII=mean_tauII_history[istep] iter=solve_stats.iter err=solve_stats.err
        end
    end  # physical time step loop

    P_cpu  = Array(dr.P)
    vx_cpu = Array(dr.v.x)
    vy_cpu = Array(dr.v.y)

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    # Per-element average pressure (mean of 3 pressure DoFs per element)
    el_P = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
            for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1200, 520))

    # Pressure
    clims_P = extrema(el_P)
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(),
            title = "Pressure  (T7/P1-disc, pure shear)", xlabel = "x", ylabel = "y")
    poly!(ax1, polys; color = el_P, colormap = :vik, colorrange = clims_P, strokewidth = 0)
    Colorbar(fig[1, 2]; colormap = :vik, limits = clims_P,
            label = "P", width = 15, tellheight = false)

    ax2 = Axis(fig[1, 3];
            title = "Mean τII history", xlabel = "time", ylabel = "mean(post.tauII)")
    scatterlines!(ax2, time_history, mean_tauII_history; color = :black, linewidth = 2)

    θ    = LinRange(0, 2π, 300)
    xs_c = cx .+ r_incl .* cos.(θ)
    ys_c = cy .+ r_incl .* sin.(θ)
    lines!(ax1, xs_c, ys_c; color = :white, linewidth = 1.5, linestyle = :dash)

    show_plot && display(fig)
    return (; time = time_history, mean_tauII = mean_tauII_history, post,
        solve_stats = solve_stats_history, solve_time)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
