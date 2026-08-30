import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using DomainSets
using DomainSets: ×
using KernelAbstractions
using Gmsh
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend   = CPU()
const workgroup = 128

include(joinpath(@__DIR__, "..", "..", "gmsh_meshing.jl"))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; nsteps=20, Δt=0.1, γfact=20.0, η_vp=1.0e-2, n_circle=64, max_area=1 / 48^2, show_plot=true) -> NamedTuple

Run the visco-elasto-plastic shear-localisation model on an unstructured
T7/P1-disc mesh.

A weak circular inclusion (radius `R = 0.1` m, shear modulus `G = 0.5` Pa) sits
at the centre of the unit domain `[-0.5, 0.5] × [-1, 0]` m, embedded in a matrix
with `G = 1.0` Pa.  Both phases share `η = 1.0` Pa·s and compressibility
`β = 0.01` Pa⁻¹, and yield according to a Drucker-Prager criterion with
cohesion `C = 1.6` Pa, friction angle `ϕ = 30°`, and dilation angle `Ψ = 3°`.
Background pure shear (`ε̇xx = -ε̇zz = 1.0` s⁻¹) is imposed through free-slip
walls; gravity is zero.

Elastic stress build-up reaches the yield stress `C·cos(ϕ) ≈ 1.386` Pa around
`t ≈ 1.2` s, after which conjugate shear bands nucleate at the inclusion and
propagate across the domain.

`η_vp` is the Perzyna regularisation viscosity.  It sets the shear-band width,
so the mesh must resolve it: values well below the default diverge the
dynamic-relaxation iteration once `max_area` is small.  `γfact` scales the
viscosity-weighted pressure update.

Returns the time series of mean inclusion pressure and mean τII, the last
post-processed field set, and per-step solver statistics.
"""
function main(;
        nsteps = 20, Δt = 0.1, γfact = 20.0, η_vp = 1.0e-2,
        n_circle = 64, max_area = 1 / 48^2,
        show_plot = true, write_output = true, verbose = true,
        measure_λmax = false, λmax_safety = 1.1)
    # Domain: unit square with the origin at the top centre, z increasing upward
    x0, y0 = -0.5, -1.0
    Lx, Ly = 1.0, 1.0

    # Background pure-shear strain rate: ε̇xx = -ε̇zz = 1 s⁻¹
    ε̇_bg = 1.0

    # Material (2 phases: matrix + weak inclusion)
    β     = 0.01
    η     = (1.0,     1.0)   # shear viscosity [Pa s]
    α     = (0.0,     0.0)   # thermal expansivity → isothermal
    ρ0    = (1.0,     1.0)   # reference density; gravity is zero so buoyancy is inert
    K     = (1 / β,   1 / β) # bulk modulus [Pa]
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (1.0,     0.5)   # shear modulus [Pa]; the inclusion is the weak phase
    G_stokes = G
    Ψ     = deg2rad(3)       # dilation angle [rad] (non-associated)
    ϕ     = deg2rad(30)      # friction angle [rad]
    plastic = DruckerPrager(
        (ϕ, ϕ),
        (Ψ, Ψ),
        (1.6, 1.6),                            # cohesion C [Pa]; yield stress = C·cosϕ
        (η_vp, η_vp),                          # Perzyna regularisation viscosity
        K,                                     # Kb (passed separately from elastic K)
    )
    g     = (0.0,     0.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    ncheck = 100          # convergence check interval
    ϵ_tol  = 1e-6         # relative residual tolerance

    # Gmsh fragments the domain at this material interface.
    r_incl = 0.1
    cx     = x0 + Lx / 2
    cy     = y0 + Ly / 2

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_gmsh_t7_circle_inclusion_mesh(;
        x0, y0, Lx, Ly,
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

    @info "Gmsh mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_circle max_area n_interface_nodes=length(circle_nodes)

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
    τ = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

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
    incl_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, findall(==(2), cell_phase)])))

    @info "Phases" n_incl_v=count(==(2), phases_v_cpu) n_incl_P=count(==(2), phases_P_cpu)

    # ---------------------------------------------------------------------------
    # Boundary conditions — free-slip walls driven by background pure shear
    #   vx = +ε̇_bg * (x - cx)   on the vertical walls
    #   vy = -ε̇_bg * (y - cy)   on the horizontal walls
    # The tangential component is left unconstrained on each wall, which is the
    # free-slip condition.
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    x1, y1 = x0 + Lx, y0 + Ly
    tol = max(abs(x0), abs(x1), abs(y0), abs(y1)) * eps(Float64) * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1] - x0) ≤ tol || abs(coords[n][1] - x1) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2] - y0) ≤ tol || abs(coords[n][2] - y1) ≤ tol]

    bc_vx_vals = [ ε̇_bg * (coords[n][1] - cx) for n in vx_nodes]
    bc_vy_vals = [-ε̇_bg * (coords[n][2] - cy) for n in vy_nodes]
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals)
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals)

    # Seed the full interior with the analytical pure-shear field so the
    # solver starts with a good initial guess (boundary nodes are overwritten
    # by apply_bc! below; the result is identical on those nodes).
    copyto!(dr.vx, [ ε̇_bg * (c[1] - cx) for c in coords_v])
    copyto!(dr.vy, [-ε̇_bg * (c[2] - cy) for c in coords_v])

    apply_bc!(dr.vx, bc_vx)
    apply_bc!(dr.vy, bc_vy)

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
    mean_P_incl_history = zeros(Float64, nsteps)

    iterMax       = 50_000   # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    rel_drop0     = 0.75     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt iterMax total_iterMax ncheck ϵ_tol

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = joinpath(@__DIR__, "output_shear_bands")
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
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(geo_v),
            τ,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        mean_P_incl_history[istep] = mean(@view P_cpu[incl_P_dofs])
        copyto!(dr.τxx_old, dr.τxx)
        copyto!(dr.τyy_old, dr.τyy)
        copyto!(dr.τxy_old, dr.τxy)

        if write_output
            vtk_path = joinpath(out_dir, @sprintf("stokes_2D_shear_bands_triangle_%04d.vtk", istep))
            write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
            @info "Wrote VTK file" vtk_path mean_P_incl=mean_P_incl_history[istep] mean_tauII=mean_tauII_history[istep] iter=solve_stats.iter err=solve_stats.err
        end
    end  # physical time step loop

    P_cpu  = Array(dr.P)
    vx_cpu = Array(dr.vx)
    vy_cpu = Array(dr.vy)

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    # Per-element average pressure (mean of 3 pressure DoFs per element)
    el_P = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
            for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1200, 520))

    clims_P = extrema(el_P)
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(),
            title = "Pressure  (T7/P1-disc, VEP shear bands)", xlabel = "x [m]", ylabel = "z [m]")
    poly!(ax1, polys; color = el_P, colormap = :vik, colorrange = clims_P, strokewidth = 0)
    Colorbar(fig[1, 2]; colormap = :vik, limits = clims_P,
            label = "P [Pa]", width = 15, tellheight = false)

    ax2 = Axis(fig[1, 3];
            title = "Mean pressure in weak inclusion", xlabel = "t [s]", ylabel = "⟨P⟩ [Pa]")
    scatterlines!(ax2, time_history, mean_P_incl_history; color = :black, linewidth = 2)

    θ    = LinRange(0, 2π, 300)
    xs_c = cx .+ r_incl .* cos.(θ)
    ys_c = cy .+ r_incl .* sin.(θ)
    lines!(ax1, xs_c, ys_c; color = :white, linewidth = 1.5, linestyle = :dash)

    show_plot && display(fig)
    return (; time = time_history, mean_P_incl = mean_P_incl_history,
        mean_tauII = mean_tauII_history, post,
        solve_stats = solve_stats_history, solve_time)
end

main()
