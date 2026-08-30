import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelAbstractions: CPU
using FEMTools
using StaticArrays
using Gmsh
using GLMakie: Axis, Colorbar, DataAspect, Figure, Point2f, poly!

include(joinpath(@__DIR__, "..", "stokes", "sinking_block", "mesher.jl"))

function main(; max_area = 1 / 32^2, show_plot = true, CFL = 0.9,
              c_fact = 0.9, ϵ = 1e-6, ncheck = 25, verbose = true)
    backend, workgroup = CPU(), 128
    Lx, Ly = 1.0, 1.0
    cx, cy, half_width = Lx / 2, -Ly / 2, 0.1
    ρ0, α, K = (1.0, 2.0), (0.0, 0.0), (Inf, Inf)
    g, Tref = (0.0, -1.0), 0.0

    all_coords, el2n_t7, _, _ = build_gmsh_t7_rectangle_inclusion_mesh(;
        Lx, Ly, cx, cy, half_width, max_area,
    )
    # The first three T7 nodes are the pressure triangle used by sinking_block.jl.
    corner_nodes = sort!(unique(vec(el2n_t7[1:3, :])))
    corner_id = Dict(node => Int32(i) for (i, node) in enumerate(corner_nodes))
    coords = all_coords[corner_nodes]
    el2n = Matrix{Int32}(undef, 3, size(el2n_t7, 2))
    @inbounds for i in axes(el2n, 2), a in axes(el2n, 1)
        el2n[a, i] = corner_id[el2n_t7[a, i]]
    end

    element = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = Mesh(backend, coords, el2n, element; workgroup)
    in_block(c) = abs(c[1] - cx) ≤ half_width && abs(c[2] - cy) ≤ half_width

    material = ThermalMaterial(; k = one.(ρ0), Cp = one.(ρ0), ρ0, α, K)
    dr = LithostaticPressureDR(backend, mesh.nnodes, material; CFL, c_fact, ϵ)
    copyto!(dr.phases, Int[in_block(c) ? 2 : 1 for c in coords])
    copyto!(dr.P, [ρ0[1] * abs(g[2]) * (0.0 - c[2]) for c in coords])

    tol = max(Lx, Ly) * eps(Float64) * 32
    top_nodes = Int32[i for i in eachindex(coords) if abs(coords[i][2]) ≤ tol]
    bc = DirichletBoundaryCondition(nothing, top_nodes, zeros(length(top_nodes)))
    solver!(dr, mesh, bc; workgroup, ncheck, verbose, Tref, g)

    P = Array(dr.P)
    show_plot || return (; coords, el2n, P, phases = Array(dr.phases))

    points = Point2f.(coords)
    triangles = [[points[el2n[a, i]] for a in 1:3] for i in axes(el2n, 2)]
    cell_pressure = [sum(P[el2n[:, i]]) / 3 for i in axes(el2n, 2)]
    fig = Figure(size = (650, 600))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y",
              title = "Lithostatic pressure")
    limits = extrema(cell_pressure)
    poly!(ax, triangles; color = cell_pressure, colormap = :glasgow,
          colorrange = limits, strokewidth = 0)
    Colorbar(fig[1, 2]; colormap = :glasgow, limits, label = "P")
    return fig
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
