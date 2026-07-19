import Pkg
Pkg.activate(@__DIR__)

using FEMTools
using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using KernelAbstractions
using Gmsh
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend   = CPU()
const workgroup = 128

function straighten_t7_geometry!(coords, el2n)
    @inbounds for iel in axes(el2n, 2)
        n1, n2, n3 = Int(el2n[1, iel]), Int(el2n[2, iel]), Int(el2n[3, iel])
        coords[Int(el2n[4, iel])] = (coords[n1] + coords[n2]) / 2
        coords[Int(el2n[5, iel])] = (coords[n2] + coords[n3]) / 2
        coords[Int(el2n[6, iel])] = (coords[n3] + coords[n1]) / 2
        coords[Int(el2n[7, iel])] = (coords[n1] + coords[n2] + coords[n3]) / 3
    end
    return coords
end

"""
    build_gmsh_t7_hole_mesh(; Lx, Ly, cx, cy, r, max_area, hole_refine_factor)

Build an unstructured T7 velocity mesh of `[0,Lx] x [0,Ly]` with an empty
circular hole using Gmsh. The returned `outer_nodes` are the square boundary
nodes used for pure-shear Dirichlet BCs; `hole_nodes` are left unconstrained.
"""
function build_gmsh_t7_hole_mesh(; Lx, Ly, cx, cy, r, max_area = 1 / 64^2, hole_refine_factor = 4.0)
    mesh_size = sqrt(2 * max_area)
    hole_mesh_size = min(mesh_size / hole_refine_factor, r / 16)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("pure_shear_hole")
    try
        rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        surfaces, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        hole_curves = Int32[]
        for (dim, tag) in gmsh.model.getBoundary(surfaces, false, false)
            dim == 1 || continue
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            hypot((xmin + xmax) / 2 - cx, (ymin + ymax) / 2 - cy) < r / 2 && push!(hole_curves, tag)
        end
        isempty(hole_curves) && error("Could not identify hole boundary curve for mesh refinement")
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", hole_curves)
        gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", hole_mesh_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", mesh_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2r)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        nodetags, coords_flat, _ = gmsh.model.mesh.getNodes()
        tag2idx = Dict{Int, Int32}(Int(tag) => Int32(i) for (i, tag) in enumerate(nodetags))
        coords = [SVector{2, Float64}(coords_flat[3(i - 1) + 1], coords_flat[3(i - 1) + 2])
                  for i in eachindex(nodetags)]

        elemtypes, _, elemnodetags = gmsh.model.mesh.getElements(2)
        tri_idx = findfirst(==(9), elemtypes) # 6-node second-order triangle
        isnothing(tri_idx) && error("No Gmsh T6 triangle elements found")
        tri_flat = elemnodetags[tri_idx]
        nels = length(tri_flat) ÷ 6

        n_t6 = length(coords)
        el2n = Matrix{Int32}(undef, 7, nels)
        for iel in 1:nels
            base = 6 * (iel - 1)
            nodes = Int32[tag2idx[Int(tri_flat[base + a])] for a in 1:6]
            p1, p2, p3 = coords[nodes[1]], coords[nodes[2]], coords[nodes[3]]
            if det(hcat(p2 - p1, p3 - p1)) < 0
                nodes = nodes[[1, 3, 2, 6, 5, 4]]
            end
            el2n[1:6, iel] .= nodes
        end

        inside = count(iel -> begin
            c = (coords[el2n[1, iel]] + coords[el2n[2, iel]] + coords[el2n[3, iel]]) / 3
            hypot(c[1] - cx, c[2] - cy) < r
        end, 1:nels)
        inside == 0 || error("Gmsh returned $inside elements inside the hole")

        sizehint!(coords, n_t6 + nels)
        for iel in 1:nels
            c1 = coords[el2n[1, iel]]
            c2 = coords[el2n[2, iel]]
            c3 = coords[el2n[3, iel]]
            push!(coords, (c1 + c2 + c3) / 3)
            el2n[7, iel] = Int32(n_t6 + iel)
        end
        straighten_t7_geometry!(coords, el2n)

        tol = 1e-8 * max(Lx, Ly)
        outer_nodes = Int32[
            i for i in 1:n_t6
            if abs(coords[i][1]) ≤ tol ||
               abs(coords[i][1] - Lx) ≤ tol ||
               abs(coords[i][2]) ≤ tol ||
               abs(coords[i][2] - Ly) ≤ tol
        ]
        hole_nodes = Int32[
            i for i in 1:n_t6
            if abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) ≤ tol
        ]

        return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(hole_nodes))
    finally
        gmsh.finalize()
    end
end

function main(;
    nsteps = 15,
    max_area = 1 / 64^2,
    Δt = 1 / 6,
    show_plot = true,
    iterMax = 50_000,
    total_iterMax = 50_000,
    verbose_PH = true,
    verbose_DR = false,
    advect_mesh = true,
)
    Lx, Ly = 1.0, 1.0
    ε̇_bg = 1.0 / 100

    # Single matrix phase; the circular feature is an empty hole.
    η     = (1.0,)
    γfact = 20.0
    α     = (0.0,)
    ρ0    = (1.0,)
    K     = (4.0,)
    ηb    = K
    G     = (Inf,)
    τy    = Inf * 1.6 / cosd(30)
    plastic = DruckerPrager(
        (π/6,),
        (0.0,),
        (τy,),
        (8.0e-3,),
        K,
    )
    g = (0.0, 0.0)
    Tref = 0.0

    ncheck = 50
    ϵ_tol  = 1e-6

    r_hole = 0.1
    cx = Lx / 2
    cy = Ly / 2

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})

    coords_v_cpu, el2n_v_cpu, outer_nodes, hole_nodes = build_gmsh_t7_hole_mesh(;
        Lx, Ly, cx, cy, r = r_hole, max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)
    @info "Gmsh mixed mesh with empty hole (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_outer=length(outer_nodes) n_hole=length(hole_nodes)

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)
    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)

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
    )
    τ = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    coords_v = Array(mesh_stokes.coords)
    phases_v_cpu = ones(Int, NV, mesh_stokes.nels)
    phases_P_cpu = ones(Int, NP, mesh_stokes.nels)

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]
    bc_vx_vals = Float64[ ε̇_bg * (coords[n][1] - Lx / 2) for n in vx_nodes]
    bc_vy_vals = Float64[-ε̇_bg * (coords[n][2] - Ly / 2) for n in vy_nodes]

    copyto!(dr.vx, Float64[ ε̇_bg * (c[1] - Lx / 2) for c in coords_v])
    copyto!(dr.vy, Float64[-ε̇_bg * (c[2] - Ly / 2) for c in coords_v])
    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))

    @info "Pure-shear outer BCs; hole rim is traction-free" n_vx=length(vx_nodes) n_vy=length(vy_nodes) max_vx=maximum(abs, bc_vx_vals) max_vy=maximum(abs, bc_vy_vals)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)

    time_history = zeros(Float64, nsteps)
    mean_tauII_history = zeros(Float64, nsteps)
    rel_drop0 = 1e-2

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt iterMax total_iterMax ncheck ϵ_tol advect_mesh

    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu  = Array(mesh_stokes.el2nP)
    DoFsP_cpu  = Array(mesh_stokes.DoFsP)
    out_dir = joinpath(@__DIR__, "output_stokes_hole")
    mkpath(out_dir)
    post = nothing

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        xmin = minimum(n -> coords_v[n][1], Γnodes)
        xmax = maximum(n -> coords_v[n][1], Γnodes)
        ymin = minimum(n -> coords_v[n][2], Γnodes)
        ymax = maximum(n -> coords_v[n][2], Γnodes)
        Lx_t = xmax - xmin
        Ly_t = ymax - ymin
        xmid = xmin + Lx_t / 2
        ymid = ymin + Ly_t / 2
        @inbounds for (i, n) in enumerate(vx_nodes)
            c = coords_v[n]
            bc_vx_vals[i] = ε̇_bg * (c[1] - xmid)
        end
        @inbounds for (i, n) in enumerate(vy_nodes)
            c = coords_v[n]
            bc_vy_vals[i] = -ε̇_bg * (c[2] - ymid)
        end
        apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
        apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))
        assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh_stokes, cache.geo_P, element_v, element_P,
            γfact, Float64(Δt), backend, workgroup; phases_v = phases_v_cpu,
        )
        @info "Physical time step" istep nsteps t

        solve_stats = solve_stokes_dyrel!(
            dr, mesh_stokes, cache, element_v, element_P,
            phases_v_cpu, phases_P_cpu, τ_old, plastic, G, Float64(Δt), γP,
            Γnodes, bc_vx_vals, bc_vy_vals, backend, workgroup;
            ncheck,
            ϵ_tol,
            iterMax,
            total_iterMax,
            rel_drop0,
            verbose_PH,
            verbose_DR,
            vx_nodes = vx_nodes,
            vy_nodes = vy_nodes,
        )

        update_stokes_current_stress!(
            dr, mesh_stokes, cache, element_v, element_P, phases_v_cpu,
            τ_old, plastic, τ, G, Float64(Δt), backend, workgroup,
        )

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = FEMTools.compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(cache.geo_v),
            τ,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        # advance stress history with corotational (Jaumann) rotation: τ_old = rotate(τ)
        rotate_stress!(dr, mesh_stokes, cache, element_v, Float64(Δt))

        if advect_mesh
            @inbounds for i in eachindex(coords_v)
                coords_v[i] += Δt * SVector(vx_cpu[i], vy_cpu[i])
            end
            straighten_t7_geometry!(coords_v, el2n_v_cpu)
            copyto!(mesh_stokes.coords, coords_v)
            copyto!(mesh_v.coords, coords_v)
            cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
        end

        vtk_path = joinpath(out_dir, @sprintf("stokes_2D_pure_shear_triangle_hole_%04d.vtk", istep))
        write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post; title = "FEMTools Stokes 2D pure shear hole")
        @info "Wrote VTK file" vtk_path mean_tauII=mean_tauII_history[istep] iter=solve_stats.iter err=solve_stats.err
    end

    P_cpu = Array(dr.P)
    el_P = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]

    pts = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1200, 520))
    clims_P = extrema(el_P)
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(),
        title = "Pressure  (T7/P1-disc, pure shear, empty hole)", xlabel = "x", ylabel = "y")
    poly!(ax1, polys; color = el_P, colormap = :vik, colorrange = clims_P, strokewidth = 0)
    hole_center = sum(coords_v[Int(n)] for n in hole_nodes) / length(hole_nodes)
    hole_loop = sort(hole_nodes, by = n -> atan(coords_v[Int(n)][2] - hole_center[2], coords_v[Int(n)][1] - hole_center[1]))
    hole_ring = [Point2f(coords_v[Int(n)]) for n in hole_loop]
    push!(hole_ring, first(hole_ring))
    poly!(ax1, hole_ring; color = :white, strokewidth = 0)
    lines!(ax1, getindex.(hole_ring, 1), getindex.(hole_ring, 2); color = :black, linewidth = 2)
    Colorbar(fig[1, 2]; colormap = :vik, limits = clims_P,
        label = "P", width = 15, tellheight = false)

    ax2 = Axis(fig[1, 3];
        title = "Mean tauII history", xlabel = "time", ylabel = "mean(post.tauII)")
    scatterlines!(ax2, time_history, mean_tauII_history; color = :black, linewidth = 2)

    show_plot && display(fig)
    return (; time = time_history, mean_tauII = mean_tauII_history, post)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
