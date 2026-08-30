# Heat diffusion on a rectangle with circular holes, meshed with Gmsh.jl.

using Gmsh
using StaticArrays
using KernelAbstractions
using Printf
using WriteVTK
using GLMakie
using GeometryBasics
using TimerOutputs
using FEMTools

include(joinpath(@__DIR__, "..", "gmsh_meshing.jl"))

const backend   = CPU()
const workgroup = 128

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(; max_area=1e5)
    TDev = FEMTools.TA(backend)

    # Domain geometry
    Lx, Ly = 15e3, 30e3
    holes  = [(-7e3, -10e3, 2.5e3),   # (cx, cy, r), T = 1173 K
              (  0e0, -11.5e3, 2e3)] # (cx, cy, r), T = 1273 K
    T_holes = Float64[873, 1173]

    # Generate mesh
    coords_cpu, el2n_cpu, outer_nodes, hole_nodes_per_hole = build_gmsh_hole_mesh(;
        Lx, Ly, holes,
        n_circle = 64,
        max_area = max_area,
        order = 1,
    )
    element = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = Mesh(backend, coords_cpu, el2n_cpu, element; workgroup)
    @printf("mesh: %d nodes, %d elements\n", mesh.nnodes, mesh.nels)

    # Dirichlet BCs --------------------------------------------------------
    T_top    = 273.0   # cold top wall  [K]
    T_bottom = 873.0   # hot bottom wall [K]

    top_nodes    = filter(i -> coords_cpu[i][2] ≈  0.0, outer_nodes)
    bottom_nodes = filter(i -> coords_cpu[i][2] ≈ -Ly,  outer_nodes)

    hole_dof_vecs = [fill(T_holes[h], length(hole_nodes_per_hole[h]))
                     for h in eachindex(holes)]

    Γ_dofs = TDev(vcat(top_nodes, bottom_nodes, hole_nodes_per_hole...))
    Γ_vals = TDev(vcat(
        fill(T_top,    length(top_nodes)),
        fill(T_bottom, length(bottom_nodes)),
        hole_dof_vecs...,
    ))
    bc_T = DirichletBoundaryCondition(nothing, Γ_dofs, Γ_vals)

    @printf("Dirichlet: %d top, %d bottom", length(top_nodes), length(bottom_nodes))
    for (h, hn) in enumerate(hole_nodes_per_hole)
        @printf(", %d hole-%d nodes (T=%.0f K)", length(hn), h, T_holes[h])
    end
    @printf("\n")

    # Material properties (single homogeneous phase) ----------------------
    material = ThermalMaterial(; k = (3.0,), Cp = (1200.0,), ρ0 = (3300.0,), α = (3e-5,), K = (1e11,))
    Tref = 273.0
    g    = SA[0.0, -9.81]
    Δt  = 20e3 * 365.25 * 24 * 3600   # 20 kyr time step [s]

    # Solver state --------------------------------------------------------
    dr = ThermalDiffusionDR(backend, mesh.nnodes, material;
                            CFL = 0.9, ϵ = 1e-8)

    T_init = Float64[T_top + (T_bottom - T_top) * (-coords_cpu[i][2] / Ly) for i in eachindex(coords_cpu)]
    copyto!(dr.T, T_init)
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # Lithostatic pressure: solve ∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ on the initial T.
    # BC: P = 0 on the free surface (top), Neumann elsewhere.
    # Warm-start from the analytical P = ρ₀ g depth so the initial residual is small;
    # this prevents β=1 undamped accumulation in the DR solver for pure Poisson.
    lp_dr = LithostaticPressureDR(backend, mesh.nnodes, material; CFL = 0.9, ϵ = 1e-2)
    copyto!(lp_dr.T, dr.T)
    P0_litho = Float64[material.ρ0[1] * (-g[2]) * (-coords_cpu[i][2]) for i in eachindex(coords_cpu)]
    copyto!(lp_dr.P, P0_litho)
    bc_P = DirichletBoundaryCondition(nothing, TDev(top_nodes), zero(dr.P[top_nodes]))
    @printf("solving initial lithostatic pressure …\n")
    to = TimerOutput()
    @timeit to "litho P init" solver!(lp_dr, mesh, bc_P; workgroup, ncheck = 50, Tref = Tref, g = g)
    copyto!(dr.P, lp_dr.P)

    # VTK time-series setup -----------------------------------------------
    vtk_pts = zeros(3, length(coords_cpu))
    for (i, p) in enumerate(coords_cpu)
        vtk_pts[1, i] = p[1]
        vtk_pts[2, i] = p[2]
    end
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, el2n_cpu[:, i]) for i in axes(el2n_cpu, 2)]

    out_dir = "output"
    mkpath(out_dir)
    t_phys = 0.0   # physical time [s]

    # Build figure once so each time step can update it in-place -----------
    pts2d  = [Point2f(p[1], p[2]) for p in coords_cpu]
    faces  = [GeometryBasics.TriangleFace(Int(el2n_cpu[1,i]), Int(el2n_cpu[2,i]), Int(el2n_cpu[3,i]))
              for i in axes(el2n_cpu, 2)]
    T_obs  = Observable(Array(dr.T))
    P_obs  = Observable(Array(lp_dr.P))
    t_obs  = Observable(0.0)

    P_max = material.ρ0[1] * (-g[2]) * Ly   # analytical pressure at max depth

    fig = Figure(size = (1300, 640))
    θ_c = range(0, 2π; length = 300)

    ax_T = Axis(fig[1, 1];
                title   = @lift("T  t = $(round($t_obs / (1e3*365.25*24*3600); digits=1)) kyr"),
                aspect  = DataAspect(), xlabel = "x [m]", ylabel = "y [m]")
    mT = mesh!(ax_T, pts2d, faces;
               color      = T_obs,
               colormap   = :thermal,
               colorrange = (T_top, maximum(T_holes)),
               shading    = NoShading)
    for (cx, cy, r) in holes
        lines!(ax_T, cx .+ r .* cos.(θ_c), cy .+ r .* sin.(θ_c); color = :white, linewidth = 2)
    end
    Colorbar(fig[1, 2], mT; label = "T [K]")

    ax_P = Axis(fig[1, 3];
                title   = "Lithostatic P",
                aspect  = DataAspect(), xlabel = "x [m]", ylabel = "y [m]")
    mP = mesh!(ax_P, pts2d, faces;
               color      = P_obs,
               colormap   = :viridis,
               colorrange = (0.0, P_max),
               shading    = NoShading)
    for (cx, cy, r) in holes
        lines!(ax_P, cx .+ r .* cos.(θ_c), cy .+ r .* sin.(θ_c); color = :white, linewidth = 2)
    end
    Colorbar(fig[1, 4], mP; label = "P [Pa]")

    gif_path = joinpath(out_dir, "heat_diffusion_unstructured.gif")

    # Time loop -----------------------------------------------------------
    nsteps = 75
    to = TimerOutput()
    pvd = paraview_collection(joinpath(out_dir, "heat_diffusion_unstructured"))
    record(fig, gif_path, 1:nsteps; framerate = 15) do step
        @printf("─── time step %2d / %d ───\n", step, nsteps)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)
        @timeit to "solver" solver!(dr, Δt, mesh, bc_T; workgroup, ncheck = 100, Tref = Tref)
        t_phys += Δt

        @timeit to "update obs" begin
            T_obs[] = Array(dr.T)
            P_obs[] = Array(dr.P)
            t_obs[] = t_phys
        end

        @timeit to "vtk" vtk_grid(joinpath(out_dir, "heat_diffusion_unstructured_$step"), vtk_pts, cells) do vtk
            vtk["T"] = Array(dr.T)
            vtk["P"] = Array(dr.P)
            pvd[t_phys] = vtk
        end
    end
    vtk_save(pvd)
    @printf("saved %s (%d steps)\n", joinpath(out_dir, "heat_diffusion_unstructured.pvd"), nsteps)
    @printf("saved %s\n", gif_path)
    print_timer(to; title = "heat diffusion — unstructured", sortby = :firstexec)

    display(fig)

    @printf("mesh: %d nodes, %d elements\n", mesh.nnodes, mesh.nels)

    return nothing
end

main(; max_area=1e5/2)
