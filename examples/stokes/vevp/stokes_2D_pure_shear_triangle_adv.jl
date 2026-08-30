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

function corner_mesh_hmin(coords, el2n)
    hmin = Inf
    @inbounds for iel in axes(el2n, 2)
        n1, n2, n3 = Int(el2n[1, iel]), Int(el2n[2, iel]), Int(el2n[3, iel])
        hmin = min(hmin, norm(coords[n2] - coords[n1]))
        hmin = min(hmin, norm(coords[n3] - coords[n2]))
        hmin = min(hmin, norm(coords[n1] - coords[n3]))
    end
    return hmin
end

function corner_max_displacement(vx, vy, el2n, dt)
    return abs(dt) * corner_max_speed(vx, vy, el2n)
end

function corner_max_speed(vx, vy, el2n)
    visited = falses(length(vx))
    vmax = 0.0
    @inbounds for iel in axes(el2n, 2), a in 1:3
        n = Int(el2n[a, iel])
        if !visited[n]
            vmax = max(vmax, hypot(vx[n], vy[n]))
            visited[n] = true
        end
    end
    return vmax
end

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; nsteps=500, n_circle=96, max_area=1 / (1 * 64^2), Δt=1 / 6,
          show_plot=true, advect_mesh=true, mesh_cfl=0.99) -> NamedTuple

Run the unstructured T7/P1-disc pure-shear Stokes example.

The model builds a square domain with a circular inclusion, applies pure-shear
boundary conditions, advances the viscoelastic-plastic Stokes solve, writes one
VTK file per physical step, and returns the stress-history diagnostics. When
mesh advection is enabled, `Δt` is treated as the maximum step and the actual
step is limited by `mesh_cfl * hmin / vmax`.
"""
function main(;
    nsteps = 500,
    n_circle = 96,
    max_area = 1 / (1 * 64^2),
    Δt = 1 / 6,
    show_plot = true,
    advect_mesh = true,
    mesh_cfl = 0.99,
)
    # Domain
    Lx, Ly = 1.0, 1.0

    # Background pure-shear strain rate (non-dimensional)
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion)
    γfact = 20.0
    η     = (1e0,     1.0)   # shear viscosity
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     1.0)   # reference density
    K     = (4e0,     4e0)   # bulk modulus  (Inf → incompressible)
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (5e0,     0.5)   # Shear modulus
    G_stokes = G
    # Cohesion chosen so the yield stress C·cosϕ = 1.6 at zero pressure.
    # Background deviatoric stress in pure shear is 2η·ε̇_bg = 2, so the
    # inclusion (lower G) will enter the plastic regime after a few steps.
    τy      = 1.0 / cosd(30)                   # cohesion C; yield stress = C·cosϕ = 1.6
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
    ncheck = 50          # convergence check interval
    ϵ_tol  = 1e-6        # relative residual tolerance

    # Gmsh fragments the domain at this material interface.
    r_incl = 0.1
    cx     = Lx / 2
    cy     = Ly / 2

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_gmsh_t7_circle_inclusion_mesh(;
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

    @info "Gmsh mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_circle max_area n_interface_nodes=length(circle_nodes)

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)

    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    material = StokesMaterial(; η, ηb, G = G_stokes, α, ρ0, K, g = Tuple(g), Tref)
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
        # CFL_v = 0.03, CFL_P = 0.9, c_fact = 0.5,
    )
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

    @info "Phases" n_incl_v=count(==(2), phases_v_cpu) n_incl_P=count(==(2), phases_P_cpu)

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear
    #   vx = +ε̇_bg * (x - Lx/2),   vy = -ε̇_bg * (y - Ly/2)
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps() * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]

    bc_vx_vals = [ ε̇_bg * (coords[n][1] - Lx / 2) for n in vx_nodes]
    bc_vy_vals = [-ε̇_bg * (coords[n][2] - Ly / 2) for n in vy_nodes]
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals)
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals)

    # Seed the full interior with the analytical pure-shear field so the
    # solver starts with a good initial guess (boundary nodes are overwritten
    # by apply_bc! below; the result is identical on those nodes).
    copyto!(dr.vx, [ ε̇_bg * (c[1] - Lx / 2) for c in coords_v])
    copyto!(dr.vy, [-ε̇_bg * (c[2] - Ly / 2) for c in coords_v])

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
    #   γP      = local viscosity-weighted pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction, but
    # adapts the pressure step to viscosity contrasts.
    Δt = Δt === nothing ? 0.5 / max(abs(ε̇_bg), eps(Float64)) : Float64(Δt)
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    dt_history = zeros(Float64, nsteps)
    time_history = zeros(Float64, nsteps)
    mean_tauII_history = zeros(Float64, nsteps)

    iterMax       = 50_000   # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    rel_drop0     = 1e-2     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt iterMax total_iterMax ncheck ϵ_tol advect_mesh

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = if advect_mesh
        joinpath(@__DIR__, "output_stokes")
    else
        joinpath(@__DIR__, "output_stokes_no_adv")
    end
    mkpath(out_dir)
    post = nothing
    t = 0.0

    for istep in 1:nsteps
        # copyto!(dr.T0, dr.T)
        copyto!(dr.P0, dr.P)
        fill!(dr.RP, 0.0)
        # fill!(dr.P, 0.0)


        xmin = minimum(n -> coords_v[n][1], Γnodes)
        xmax = maximum(n -> coords_v[n][1], Γnodes)
        ymin = minimum(n -> coords_v[n][2], Γnodes)
        ymax = maximum(n -> coords_v[n][2], Γnodes)
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        @inbounds for (i, n) in enumerate(vx_nodes)
            c = coords_v[n]
            bc_vx_vals[i] = ε̇_bg * (c[1] - xmid)
        end
        @inbounds for (i, n) in enumerate(vy_nodes)
            c = coords_v[n]
            bc_vy_vals[i] = -ε̇_bg * (c[2] - ymid)
        end
        apply_bc!(dr.vx, bc_vx)
        apply_bc!(dr.vy, bc_vy)
        dt_step = Δt
        if advect_mesh
            vmax = corner_max_speed(Array(dr.vx), Array(dr.vy), el2n_v_cpu)
            hmin = corner_mesh_hmin(coords_v, el2n_v_cpu)
            dt_step = iszero(vmax) ? Δt : min(Δt, mesh_cfl * hmin / vmax)
        end
        t += dt_step
        dt_history[istep] = dt_step
        time_history[istep] = t
        assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh_stokes, cache, γfact, dt_step; workgroup, phases_v = phases_v_cpu,
        )
        @info "Physical time step" istep nsteps t Δt=dt_step Δt_max=Δt

        solve_stats = solve_stokes_dyrel!(
            dr, mesh_stokes, cache, bc_vx, bc_vy, dt_step, γP;
            phases_v = phases_v_cpu, phases_P = phases_P_cpu, τ_old, plastic, workgroup,
            ncheck,
            ϵ_tol,
            iterMax,
            total_iterMax,
            rel_drop0,
            verbose = verbose_PH,
            verbose_inner = verbose_DR,
        )

        update_stokes_current_stress!(
            dr, mesh_stokes, cache, τ, dt_step;
            phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(cache.geo_v),
            τ,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        if advect_mesh
            rotate_stress!(dr, mesh_stokes, cache, element_v, dt_step)
        else
            copyto!(dr.τxx_old, dr.τxx)
            copyto!(dr.τyy_old, dr.τyy)
            copyto!(dr.τxy_old, dr.τxy)
        end
        vtk_path = joinpath(out_dir, @sprintf("stokes_2D_pure_shear_advected_triangle_%04d.vtk", istep))
        write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
        @info "Wrote VTK file" vtk_path mean_tauII=mean_tauII_history[istep] iter=solve_stats.iter err=solve_stats.err

        if advect_mesh
            hmin = corner_mesh_hmin(coords_v, el2n_v_cpu)
            max_disp = corner_max_displacement(vx_cpu, vy_cpu, el2n_v_cpu, dt_step)
            if max_disp > mesh_cfl * hmin * (1 + √(eps(Float64)))
                error("Mesh advection step is too large: max displacement $(max_disp) exceeds $(mesh_cfl) * minimum edge length $(hmin). Reduce Δt or set advect_mesh=false.")
            end
            visited = falses(length(coords_v))
            @inbounds for iel in axes(el2n_v_cpu, 2), a in 1:3
                n = Int(el2n_v_cpu[a, iel])
                if !visited[n]
                    coords_v[n] += dt_step * SVector(vx_cpu[n], vy_cpu[n])
                    visited[n] = true
                end
            end
            FEMTools.straighten_t7_geometry!(coords_v, el2n_v_cpu)
            @inbounds for iel in axes(el2n_v_cpu, 2)
                n1, n2, n3 = Int(el2n_v_cpu[1, iel]), Int(el2n_v_cpu[2, iel]), Int(el2n_v_cpu[3, iel])
                @assert coords_v[el2n_v_cpu[4, iel]] == (coords_v[n1] + coords_v[n2]) / 2
                @assert coords_v[el2n_v_cpu[5, iel]] == (coords_v[n2] + coords_v[n3]) / 2
                @assert coords_v[el2n_v_cpu[6, iel]] == (coords_v[n3] + coords_v[n1]) / 2
                @assert coords_v[el2n_v_cpu[7, iel]] == (coords_v[n1] + coords_v[n2] + coords_v[n3]) / 3
            end
            copyto!(mesh_v.coords, coords_v)
            copyto!(mesh_stokes.coords, coords_v)
            # The advected coordinates invalidate every Jacobian, so rebuild the
            # geometry cache from the updated mesh.
            cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
        end
        # fill!(dr.vx, 0.0)
        # fill!(dr.vy, 0.0)
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
    return (; time = time_history, Δt = dt_history, mean_tauII = mean_tauII_history, post)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(; advect_mesh = true)
