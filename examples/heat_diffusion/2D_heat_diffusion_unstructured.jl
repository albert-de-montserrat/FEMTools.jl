# Heat diffusion on a rectangle with a circular hole, meshed with Triangulate.jl
# (wrapper around J.R. Shewchuk's Triangle library).

using Triangulate
using StaticArrays
using KernelAbstractions
using Printf
using WriteVTK
using GLMakie
using GeometryBasics
using TimerOutputs
using FEMTools

const backend   = CPU()
const workgroup = 128

# ---------------------------------------------------------------------------
# Geometry precomputation
# ---------------------------------------------------------------------------

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T <: AbstractElement{2, N} where N
    ip    = element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KernelAbstractions.synchronize(backend)
    return geo
end

# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

"""
    build_mesh(; Lx, Ly, holes, n_circle=64, max_area=nothing)

Triangulate the rectangle [-Lx,Lx]×[-Ly,0] with circular holes using
Triangulate.jl (Triangle library).

`holes` is a vector of `(cx, cy, r)` tuples, one per hole. `n_circle`
is the number of boundary segments per circle. `max_area` bounds element
area; `nothing` skips quality refinement.

Returns `(coords, el2n, outer_nodes, hole_nodes_per_hole)` where
`hole_nodes_per_hole[h]` lists the boundary nodes of hole `h`.
"""
function build_mesh(; Lx, Ly, holes, n_circle=64, max_area=nothing)
    # ---- outer rectangle (counter-clockwise) ----
    rect_pts  = Cdouble[-Lx  Lx  Lx -Lx;
                        -Ly -Ly  0.0  0.0]
    rect_segs = Cint[1 2; 2 3; 3 4; 4 1]'   # 2×4

    all_pts  = Matrix{Cdouble}(rect_pts)
    all_segs = Matrix{Cint}(rect_segs)
    hole_xy  = Matrix{Cdouble}(undef, 2, length(holes))

    for (h, (cx, cy, r)) in enumerate(holes)
        θ        = range(0, 2π; length = n_circle + 1)[1:end-1]
        circ_pts = Matrix{Cdouble}(hcat(cx .+ r .* cos.(θ), cy .+ r .* sin.(θ))')
        n_so_far = size(all_pts, 2)
        circ_segs = Matrix{Cint}(hcat([
            [n_so_far + i; n_so_far + mod1(i + 1, n_circle)] for i in 1:n_circle
        ]...))
        all_pts  = hcat(all_pts,  circ_pts)
        all_segs = hcat(all_segs, circ_segs)
        hole_xy[:, h] = [cx; cy]
    end

    # ---- assemble TriangulateIO ----
    tio = TriangulateIO()
    tio.pointlist   = all_pts
    tio.segmentlist = all_segs
    tio.holelist    = hole_xy

    flags = isnothing(max_area) ? "pqQ" : "pq30a$(max_area)Q"
    result, _ = triangulate(flags, tio)

    pts  = result.pointlist    # 2 × nnodes
    tris = result.trianglelist # 3 × nels  (1-based)

    nnodes = size(pts, 2)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in 1:nnodes]
    el2n   = Matrix{Int32}(tris)

    tol_lin = max(Lx, Ly) * 1e-8
    outer_nodes = Int32[i for i in 1:nnodes if
        abs(coords[i][2] + Ly) < tol_lin ||
        abs(coords[i][2])      < tol_lin ||
        abs(coords[i][1] + Lx) < tol_lin ||
        abs(coords[i][1] - Lx) < tol_lin]

    hole_nodes_per_hole = map(holes) do (cx, cy, r)
        tol_circ = r * 1e-4
        Int32[i for i in 1:nnodes if
            abs(sqrt((coords[i][1] - cx)^2 + (coords[i][2] - cy)^2) - r) < tol_circ]
    end

    return coords, el2n, outer_nodes, hole_nodes_per_hole
end

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
    coords_cpu, el2n_cpu, outer_nodes, hole_nodes_per_hole = build_mesh(;
        Lx, Ly, holes,
        n_circle = 64,
        max_area = max_area,
    )
    mesh = Mesh(backend, coords_cpu, el2n_cpu)
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
    Γ_zero = zero(Γ_vals)

    @printf("Dirichlet: %d top, %d bottom", length(top_nodes), length(bottom_nodes))
    for (h, hn) in enumerate(hole_nodes_per_hole)
        @printf(", %d hole-%d nodes (T=%.0f K)", length(hn), h, T_holes[h])
    end
    @printf("\n")

    # Element and geometry ------------------------------------------------
    element = ReferenceElement(LinearElement{2, 3, Float64})
    geo     = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)

    # Material properties (single homogeneous phase) ----------------------
    k   = (3.0,)
    Cp  = (1200.0,)
    ρ0  = (3300.0,)
    α   = (3e-5,)
    K   = (1e11,)
    Tref = 273.0
    g    = SA[0.0, -9.81]
    Δt  = 20e3 * 365.25 * 24 * 3600   # 20 kyr time step [s]

    # Solver state --------------------------------------------------------
    dr = ThermalDiffusionDR(backend, mesh.nnodes, k, Cp, ρ0, α, K;
                            CFL = 0.9, ϵ = 1e-8)

    T_init = Float64[T_top + (T_bottom - T_top) * (-coords_cpu[i][2] / Ly) for i in eachindex(coords_cpu)]
    copyto!(dr.T, T_init)
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # Lithostatic pressure: solve ∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ on the initial T.
    # BC: P = 0 on the free surface (top), Neumann elsewhere.
    # Warm-start from the analytical P = ρ₀ g depth so the initial residual is small;
    # this prevents β=1 undamped accumulation in the DR solver for pure Poisson.
    lp_dr = LithostaticPressureDR(backend, mesh.nnodes, ρ0, α, K; CFL = 0.9, ϵ = 1e-2)
    copyto!(lp_dr.T, dr.T)
    P0_litho = Float64[ρ0[1] * (-g[2]) * (-coords_cpu[i][2]) for i in eachindex(coords_cpu)]
    copyto!(lp_dr.P, P0_litho)
    Γ_P_dofs = TDev(top_nodes)
    Γ_P_zero_vals = zero(dr.P[top_nodes])  # P = 0 at free surface
    @printf("solving initial lithostatic pressure …\n")
    to = TimerOutput()
    @timeit to "litho P init" solver!(lp_dr, mesh, geo, element, Γ_P_dofs, Γ_P_zero_vals, Γ_P_zero_vals, backend, workgroup; ncheck = 50, Tref = Tref, g = g)
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

    P_max = ρ0[1] * (-g[2]) * Ly   # analytical pressure at max depth

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
        @timeit to "solver" solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; ncheck = 100, Tref = Tref)
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
