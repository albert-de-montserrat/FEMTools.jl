import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra
using Printf
using DomainSets
using DomainSets: ×
using KernelAbstractions
using StaticArrays
using Triangulate
using FEMTools

const backend = CPU()
const workgroup = 128
const KYR = 1.0

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
    build_arch_bridge_mesh(; Lx, Ly, radius, n_arch, max_area)

Generate a quadratic triangular mesh for a rectangular ice body with a
semicircular opening cut into its bottom boundary.  The opening is part of the
outer PSLG boundary (rather than a closed interior hole), so the result has two
solid end piers and a genuine arch profile.
"""
function build_arch_bridge_mesh(; Lx, Ly, radius, n_arch, max_area)
    cx = Lx / 2
    x_left, x_right = cx - radius, cx + radius

    # Counter-clockwise boundary of the ice.  The bottom edge is replaced by
    # the upper half-circle, traversed from its left spring to its right spring.
    arc_angles = range(π, 0; length = n_arch + 1)[1:end-1]
    arch_points = [SVector{2, Float64}(cx + radius * cos(θ), radius * sin(θ))
                   for θ in arc_angles]
    boundary_points = vcat(
        [SVector{2, Float64}(0.0, 0.0), SVector{2, Float64}(x_left, 0.0)],
        arch_points[2:end],
        [SVector{2, Float64}(x_right, 0.0), SVector{2, Float64}(Lx, 0.0),
         SVector{2, Float64}(Lx, Ly), SVector{2, Float64}(0.0, Ly)],
    )
    points = Cdouble[hcat(getindex.(boundary_points, 1)...);
                     hcat(getindex.(boundary_points, 2)...)]
    n_points = length(boundary_points)
    segments = Cint.(hcat(collect(1:n_points), vcat(collect(2:n_points), 1))')

    tio = TriangulateIO()
    tio.pointlist = points
    tio.segmentlist = segments
    flags = "pq30o2a$(max_area)Q"
    result, _ = triangulate(flags, tio)

    pts = result.pointlist
    tris_t6 = Matrix{Int32}(result.trianglelist)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]
    n_t6 = length(coords)
    nels = size(tris_t6, 2)
    el2n = Matrix{Int32}(undef, 7, nels)
    el2n[1:3, :] .= tris_t6[1:3, :]
    el2n[4, :] .= tris_t6[6, :]
    el2n[5, :] .= tris_t6[4, :]
    el2n[6, :] .= tris_t6[5, :]
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        c1, c2, c3 = coords[tris_t6[1, iel]], coords[tris_t6[2, iel]], coords[tris_t6[3, iel]]
        push!(coords, (c1 + c2 + c3) / 3)
        el2n[7, iel] = Int32(n_t6 + iel)
    end

    tol = 100eps(Float64) * max(Lx, Ly)
    circle_tol = max(2.5 * radius * (1 - cos(π / n_arch)), tol)
    outer_nodes = Int32[i for i in 1:n_t6 if
        abs(coords[i][1]) ≤ tol || abs(coords[i][1] - Lx) ≤ tol ||
        abs(coords[i][2] - Ly) ≤ tol || abs(coords[i][2]) ≤ tol ||
        (abs(hypot(coords[i][1] - cx, coords[i][2]) - radius) ≤ circle_tol &&
         coords[i][2] ≥ -tol)]
    return coords, el2n, sort!(unique!(outer_nodes))
end

"""Generate the mixed T7/P1-disc mesh before any material or solver setup."""
function build_ice_bridge_mesh(; Lx, Ly, radius, n_arch, max_area)
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    coords, el2n, outer_nodes = build_arch_bridge_mesh(; Lx, Ly, radius, n_arch, max_area)
    dofs = Int32.(1:length(coords))
    mesh_v = Mesh(element_v, nothing, nothing, coords, dofs, el2n, outer_nodes)
    mesh_stokes = MixedMesh(mesh_v, element_P)
    return (; element_v, element_P, mesh_v, mesh_stokes)
end

"""
    main(; kwargs...) -> NamedTuple

Solve a gravity-loaded, linear Maxwell visco-elastic ice bridge.

The bridge is 20 km long and 6 km high, with a 4 km-radius semicircular
opening cut into its bottom.  Its lower surface is supported over the left and
right 4 km piers; the arch and the remainder of the boundary are traction free.
The mesh is generated and written to `mesh_ice_bridge_2D.vtk` before the first
Stokes solve.  Material values are representative of temperate glacier ice:
the viscosity and shear modulus are intentionally exposed as keywords so the
example can also be used for sensitivity experiments.  The solve uses
lengths in km, time in kyr, and scaled stress/density/gravity units; this keeps
the physical ratios while avoiding poorly conditioned SI magnitudes.  With
`advect_mesh=true`, each converged velocity field updates the Lagrangian mesh;
the timestep is limited by `mesh_cfl` times the shortest corner edge.
"""
function main(; Lx = 20.0, Ly = 6.0, arch_radius = 4.0,
        n_arch = 48, max_area = 0.15,
        η0 = 1.0, G0 = 0.1, ρice = 1.0, gravity = 1.0e-4,
        bulk_modulus = 10.0,
        endtime_kyr = 3.0, Δt_kyr = 0.25,
        advect_mesh = true, mesh_cfl = 0.5,
        output_every = 4, output_dir = joinpath(@__DIR__, "output_ice_bridge"),
        verbose = true)

    # -----------------------------------------------------------------------
    # 1. Mesh generation and mesh-only output.  This is deliberately first.
    # -----------------------------------------------------------------------
    bridge = build_ice_bridge_mesh(; Lx, Ly, radius = arch_radius, n_arch, max_area)
    (; element_v, element_P, mesh_v, mesh_stokes) = bridge
    mkpath(output_dir)
    mesh_path = joinpath(output_dir, "mesh_ice_bridge_2D.vtk")
    write_vtk(mesh_path, mesh_stokes; title = "2-D ice bridge mesh (T7/P1-disc)")
    @info "Generated ice-bridge mesh" (; mesh_path,
        nnodes = mesh_stokes.nnodes,
        pressure_nodes = mesh_stokes.nnodesP,
        elements = mesh_stokes.nels)

    # -----------------------------------------------------------------------
    # 2. Geometry, material and visco-elastic solver state.
    # -----------------------------------------------------------------------
    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    η = (Float64(η0),)
    ηb = (Float64(bulk_modulus),)
    G = (Float64(G0),)
    K = (Float64(bulk_modulus),)
    α = (0.0,)
    ρ0 = (Float64(ρice),)
    g = (0.0, -Float64(gravity))
    material = StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref = 0.0)
    dr = StokesDR(backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9)

    phases_v = ones(Int, length(element_v), mesh_stokes.nels)
    phases_P = ones(Int, length(element_P), mesh_stokes.nels)
    Δt_max = Float64(Δt_kyr) * KYR
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)

    # End supports: vertical velocity is zero on two bottom pier patches.  One
    # left-support node also fixes horizontal translation without overconstraining
    # the bridge.  The top and side boundaries remain traction free.
    coords = Array(mesh_v.coords)
    # Γnodes contains the linear boundary vertices; use all velocity nodes so
    # quadratic edge/mid-edge nodes are supported as well.
    boundary = collect(eachindex(coords))
    tol = max(Lx, Ly) * eps(Float64) * 64
    support_length = 4.0
    left_support = Int32[n for n in boundary if coords[n][2] ≤ tol && coords[n][1] ≤ support_length]
    right_support = Int32[n for n in boundary if coords[n][2] ≤ tol && coords[n][1] ≥ Lx - support_length]
    vy_nodes = sort!(unique(vcat(left_support, right_support)))
    vx_nodes = Int32[minimum(left_support)]
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, zeros(Float64, length(vx_nodes)))
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, zeros(Float64, length(vy_nodes)))
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)
    @info "Ice-bridge supports" (; horizontal_anchor = length(vx_nodes),
        vertical_support = length(vy_nodes))

    # -----------------------------------------------------------------------
    # 3. Time stepping with linear Maxwell visco-elastic stress memory.
    # -----------------------------------------------------------------------
    el2n_v = Array(mesh_stokes.el2n)
    el2nP = Array(mesh_stokes.el2nP)
    DoFsP = Array(mesh_stokes.DoFsP)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)
    t_end = Float64(endtime_kyr) * KYR
    t = 0.0
    istep = 0
    time_kyr = Float64[]
    max_speed = Float64[]
    max_stress = Float64[]
    post = nothing

    while t < t_end - eps(t_end)
        istep += 1
        vx_before = Array(dr.v.x)
        vy_before = Array(dr.v.y)
        dt_step = Δt_max
        if advect_mesh
            vmax = corner_max_speed(vx_before, vy_before, el2n_v)
            hmin = corner_mesh_hmin(coords, el2n_v)
            dt_step = iszero(vmax) ? Δt_max : min(Δt_max, mesh_cfl * hmin / vmax)
        end
        dt_step = min(dt_step, t_end - t)
        t += dt_step
        assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh_stokes, cache, 20.0, dt_step;
            workgroup, phases_v,
        )
        stats = solve_stokes_dyrel!(
            dr, mesh_stokes, cache, bc_vx, bc_vy, dt_step, γP;
            phases_v, phases_P, τ_old,
            iterMax = 5_000, total_iterMax = 100_000,
            max_ph_iterations = 20, ϵ_tol = 1.0e-6,
            rel_drop0 = 0.75, verbose, verbose_inner = false,
        )
        stats.converged || @warn "Ice-bridge step did not fully converge" istep stats

        vx = Array(dr.v.x)
        vy = Array(dr.v.y)
        post = compute_strain_rate_stress_postprocess(
            vx, vy, el2n_v, Array(cache.geo_v), phases_v, τ_old, η, G, dt_step, element_v,
        )
        update_old_stress_from_cells!(τ_old, post, el2n_v, mesh_stokes.nnodes)

        t_kyr = t / KYR
        push!(time_kyr, t_kyr)
        push!(max_speed, maximum(hypot.(vx, vy)))
        push!(max_stress, maximum(abs, post.tauII))
        @info "Ice-bridge step" (; istep, time_kyr = t_kyr,
            converged = stats.converged, iterations = stats.iter,
            max_speed = max_speed[end], max_stress_Pa = max_stress[end])

        if output_every > 0 && (mod(istep, output_every) == 0 || t ≥ t_end - eps(t_end))
            vtk_path = joinpath(output_dir, @sprintf("ice_bridge_2D_%04d.vtk", istep))
            write_stokes_vtk(vtk_path, mesh_stokes, coords, el2nP, DoFsP,
                Array(dr.P), vx, vy, post; title = "2-D visco-elastic ice bridge")
        end

        if advect_mesh
            hmin = corner_mesh_hmin(coords, el2n_v)
            max_disp = dt_step * corner_max_speed(vx, vy, el2n_v)
            max_disp ≤ mesh_cfl * hmin * (1 + √eps(Float64)) ||
                error("Mesh advection step is too large; reduce Δt_kyr or mesh_cfl")
            visited = falses(length(coords))
            @inbounds for iel in axes(el2n_v, 2), a in 1:3
                n = Int(el2n_v[a, iel])
                if !visited[n]
                    coords[n] += dt_step * SVector(vx[n], vy[n])
                    visited[n] = true
                end
            end
            straighten_t7_geometry!(coords, el2n_v)
            copyto!(mesh_v.coords, coords)
            copyto!(mesh_stokes.coords, coords)
            cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
        end
    end

    return (; mesh_stokes, time_kyr, max_speed, max_stress, post)
end

Lx = 20.0
Ly = 6.0
arch_radius = 4.0
n_arch = 48
max_area = 0.15
η0 = 1.0
G0 = 0.1
ρice = 1.0
gravity = 1.0e-4
bulk_modulus = 10.0
endtime_kyr = 3.0
Δt_kyr = 0.25
advect_mesh = true
mesh_cfl = 0.5
output_every = 4
output_dir = joinpath(@__DIR__, "output_ice_bridge")
verbose = true

main()


function foo(mesh_stokes, mesh_v, dr, dt, backend, workgroup, element_v, element_P)

    for inode in 1:length(mesh_stokes.coords)
        mesh_stokes.coords[inode] += dt * SVector(dr.v.x[inode], dr.v.y[inode])
    end

    straighten_t7_geometry!(mesh_stokes.coords, mesh_stokes.el2n)
    copyto!(mesh_v.coords, mesh_stokes.coords)
    copyto!(mesh_stokes.coords, mesh_stokes.coords)
    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    return cache
end

function move_mesh!(mesh_stokes::MixedMesh{2}, dr, dt)
    for inode in 1:length(mesh_stokes.coords)
        mesh_stokes.coords[inode] += dt * SVector(dr.v.x[inode], dr.v.y[inode])
    end
end

function move_mesh!(mesh_stokes::MixedMesh{3}, dr, dt)
    for inode in 1:length(mesh_stokes.coords)
        mesh_stokes.coords[inode] += dt * SVector(dr.v.x[inode], dr.v.y[inode], dr.vz[inode])
    end
end

foo(mesh_stokes, mesh_v, dr, dt, backend, workgroup, element_v, element_P)
@code_warntype foo(mesh_stokes, mesh_v, dr, dt, backend, workgroup, element_v, element_P)

move_mesh!(mesh_stokes, dr, dt)

struct SymmetricTensor{T, N} where {T, N}
    data::SVector{N * (N + 1) ÷ 2, T}
end