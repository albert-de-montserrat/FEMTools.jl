# Function-only 3-D counterpart of chamber_memory_setup2D.jl for the Svartsengi
# geometry (PLAN.md Stage F): a compressible magma chamber in a Maxwell shell,
# fed by a uniform continuity source, on the saved T11/P1-discontinuous mesh of
# svartsengi_mesh3D.jl. Runs on any KernelAbstractions backend; host copies
# are kept only for station sampling, budgets, and output.
module ChamberMemory3D

using FEMTools
using KernelAbstractions
using StaticArrays
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "..", "..", "examples", "reykjanes", "injection_source.jl"))

const day = 86_400.0

"""
    load_chamber_memory_mesh(path; backend, workgroup=128) -> NamedTuple

Build the solver mesh, phases, source weights, and boundary conditions from the
`(; coords, el2n, groups, geometry)` file written by svartsengi_mesh3D.jl.
Coordinates are SI in the file and scaled by `L_c` here. Phase 1 is host, 2
magma, 3 shell. The top is traction-free; the x walls carry no x velocity, the
y walls no y velocity, and the base no z velocity.
"""
function load_chamber_memory_mesh(path; backend = CPU(), workgroup = 128)
    (; coords, el2n, groups, geometry) = deserialize(path)
    # Fixed scales, as in 2-D, keep absolute residuals comparable across runs.
    L_c, t_c, σ_c = 1.0e3, day, 1.0e7
    element_v = ReferenceElement(QuadraticElement{3, 11, Float64})
    element_P = ReferenceElement(LinearElement{3, 4, Float64})
    coords_nd = [SVector{3, Float64}(c ./ L_c) for c in coords]
    mesh_v = Mesh(backend, coords_nd, el2n, element_v; workgroup)
    mesh = MixedMesh(mesh_v, element_P; workgroup)
    TDev = FEMTools.TA(backend)
    cell_phase = collect(groups.phase)
    phases_v = TDev(repeat(reshape(cell_phase, 1, :), length(element_v), 1))
    phases_P = TDev(repeat(reshape(cell_phase, 1, :), length(element_P), 1))
    dirichlet(nodes) = DirichletBoundaryCondition(nothing, TDev(Int32.(nodes)), TDev(zeros(length(nodes))))
    bc_v = (dirichlet(sort!(union(groups.left, groups.right))), dirichlet(sort!(union(groups.front, groups.back))),
        dirichlet(groups.bottom))
    DoFsP = Array(mesh.DoFsP)
    geo_P = Array(mesh.geometry.geo_P)
    NqP = shape_function_values(element_P, element_v.integration_points)
    chamber_cells = findall(==(2), cell_phase)
    chamber_volume = sum(sum(geo_P[e]) for e in chamber_cells) * L_c^3
    source_weights = TDev(uniform_pressure_source(1.0, DoFsP, geo_P, chamber_cells))
    stations = Int.(groups.stations)
    station_xy = [(coords[n][1], coords[n][2]) for n in stations]
    return (; mesh, mesh_v, element_v, element_P, phases_v, phases_P, cell_phase, chamber_cells, DoFsP,
        geo_P, NqP, chamber_volume, source_weights, bc_v, stations, station_ids = collect(groups.station_ids),
        station_xy, surface = Int.(groups.surface), coords_SI = coords, el2n, L_c, t_c, σ_c, workgroup,
        backend, geometry = merge(geometry, (; mesh_file = abspath(path), nels = size(el2n, 2), nnodes = length(coords))))
end

"""Lithostatic pressure of `material`'s density at the pressure DoFs of `model`, in solver units; zero at the surface."""
function chamber_lithostatic_pressure(model, material; ϵ = 1.0e-10)
    (; mesh_v, cell_phase, el2n, workgroup, backend) = model
    TDev = FEMTools.TA(backend)
    # A node takes the magma phase if any of its elements is magma, else the shell's.
    node_phase = ones(Int, mesh_v.nnodes)
    for phase in (3, 2), e in findall(==(phase), cell_phase), a in axes(el2n, 1)
        node_phase[el2n[a, e]] = phase
    end
    litho = LithostaticPressureDR(backend, mesh_v.nnodes, material.ρ0, material.α, material.K; ϵ)
    copyto!(litho.phases, TDev(node_phase))
    fill!(litho.T, material.Tref)
    bc = DirichletBoundaryCondition(nothing, TDev(Int32.(model.surface)), TDev(zeros(length(model.surface))))
    solver!(litho, mesh_v, bc; workgroup, material.Tref, material.g, verbose = false)
    P_nodal = Array(litho.P)
    el2nP = Array(model.mesh.el2nP)
    P = zeros(model.mesh.nnodesP)
    for e in axes(el2nP, 2), a in axes(el2nP, 1)
        P[model.DoFsP[a, e]] = P_nodal[el2nP[a, e]]
    end
    return P
end

"""
    run_chamber_memory(model, history; η_shell=3e17, gravity=false, advection=false, ...) -> NamedTuple

Advance the 3-D Maxwell chamber with finite elastic bulk storage, as the 2-D
`ChamberMemory2D.run_chamber_memory` without spin-up or feeding. History has
`time_edges` [s], interval-average `recharge_rate` [m³/s], and positive
`withdrawal_volume` [m³] removed in each interval. Returns station
displacements [m] (stations × epochs, zero first sample), the final cumulative
displacement at every node, the chamber-mean pressure perturbation [Pa], per-step
chamber expansion and compression storage [m³], the continuity-budget residual
[m³], and solver statistics. `gravity` and `advection` act as in 2-D: the body
force of `ρ0 (1 + P/K)` from a lithostatic start, and a Lagrangian mesh (every
node moves with the velocity) with a corotated stress history; the mesh is
restored after the run.
"""
function run_chamber_memory(model, history; η_shell = 3.0e17, η_host = 1.0e23,
        η_magma = 1.0e14, G_host = 3.0e10, G_magma = 1.0e8, K_host = 5.0e10, K_magma = 1.0e10,
        gravity = false, advection = false, ρ_host = 2700.0, g_SI = 9.81,
        ϵ_tol = 1.0e-10, iterMax = 100_000, verbose = false)
    (; time_edges, recharge_rate, withdrawal_volume) = history
    nsteps = length(time_edges) - 1
    length(recharge_rate) == length(withdrawal_volume) == nsteps ||
        throw(DimensionMismatch("one recharge rate and withdrawal volume are required per interval"))
    all(x -> isfinite(x) && x > 0, (η_shell, η_host, η_magma, G_host, G_magma, K_host, K_magma, ϵ_tol)) ||
        throw(ArgumentError("material parameters and tolerance must be finite and positive"))
    (; mesh, phases_v, phases_P, L_c, t_c, σ_c, workgroup, chamber_cells, backend) = model
    η_c = σ_c * t_c
    ρ_c, g_c = σ_c * t_c^2 / L_c^2, L_c / t_c^2
    center_pressure = ρ_host * g_SI * -model.geometry.chamber_center[3]
    ρ_magma = ρ_host * (1 + center_pressure / K_host) / (1 + center_pressure / K_magma)
    material = StokesMaterial(;
        η = (η_host, η_magma, η_shell) ./ η_c,
        ηb = (K_host, K_magma, K_host) ./ σ_c,
        G = (G_host, G_magma, G_host) ./ σ_c,
        K = (K_host, K_magma, K_host) ./ σ_c,
        α = (0.0, 0.0, 0.0), ρ0 = gravity ? (ρ_host, ρ_magma, ρ_host) ./ ρ_c : (1.0, 1.0, 1.0),
        g = gravity ? (0.0, 0.0, -g_SI / g_c) : (0.0, 0.0, 0.0), Tref = 0.0,
    )
    nq = length(model.element_v.integration_points.ω)
    P_initial = gravity ? chamber_lithostatic_pressure(model, material) : zeros(mesh.nnodesP)
    dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, material;
        stress_size = (nq, mesh.nels), CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9)
    copyto!(dr.P, P_initial)
    copyto!(dr.P0, P_initial)
    reference_coords = copy(mesh.coords)
    γP = KernelAbstractions.zeros(backend, Float64, mesh.nnodesP)
    ns = length(model.stations)
    ux, uy, uz = zeros(ns, nsteps + 1), zeros(ns, nsteps + 1), zeros(ns, nsteps + 1)
    u_field = ntuple(_ -> zeros(mesh.nnodes), 3)
    pressure = zeros(nsteps + 1)
    expansion, storage, budget = zeros(nsteps), zeros(nsteps), zeros(nsteps)
    stats = NamedTuple[]
    ∂N∂ξ = shape_function_gradients(model.element_v)
    volume_nd = model.chamber_volume / L_c^3
    integrate_P(P) = pressure_source_integral(P, model.DoFsP, model.geo_P, model.NqP, chamber_cells)
    τ = FEMTools.stress(dr)
    τ_old = FEMTools.stress_old(dr)
    try
        for n in 1:nsteps
            Δτ = (time_edges[n + 1] - time_edges[n]) / t_c
            prescribed = recharge_rate[n] * (time_edges[n + 1] - time_edges[n]) - withdrawal_volume[n]
            assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh, 100.0, Δτ; phases_v, workgroup)
            step_stats = solve_stokes_dyrel!(dr, mesh, model.bc_v, Δτ, γP;
                phases_v, phases_P, finite_K = true, Qq = model.source_weights,
                Q2D = prescribed / L_c^3 / Δτ, ncheck = 50, ϵ_tol,
                iterMax, total_iterMax = iterMax, rel_drop0 = 0.1, verbose, workgroup)
            step_stats.converged || error("chamber step $n did not converge: $(step_stats.err_abs), $(step_stats.err_rel)")
            push!(stats, step_stats)
            v = map(Array, (dr.v.x, dr.v.y, dr.v.z))
            P, P0 = Array(dr.P), Array(dr.P0)
            geo_v = Array(mesh.geometry.geo_v)
            divergence_integral = sum(chamber_cells) do e
                nodes = view(model.el2n, :, e)
                vx, vy, vz = (SVector{11}(c[nodes]) for c in v)
                sum(element_geometry(geo_v, e, ∂N∂ξ)) do (B, dΩ)
                    ((B' * vx)[1] + (B' * vy)[2] + (B' * vz)[3]) * dΩ
                end
            end
            expansion[n] = divergence_integral * Δτ * L_c^3
            storage[n] = integrate_P(P .- P0) / (K_magma / σ_c) * L_c^3
            budget[n] = prescribed - expansion[n] - storage[n]
            pressure[n + 1] = integrate_P(P .- P_initial) / volume_nd * σ_c
            for (u, c) in zip((ux, uy, uz), v)
                u[:, n + 1] .= u[:, n] .+ c[model.stations] .* (L_c * Δτ)
            end
            foreach((f, c) -> f .+= c .* (L_c * Δτ), u_field, v)
            update_stokes_current_stress!(dr, mesh, τ, Δτ; phases_v, workgroup)
            copyto!(dr.P0, dr.P)
            if advection
                rotate_stress!(dr, mesh, Δτ)
                moved = Array(mesh.coords)
                for node in eachindex(moved)
                    moved[node] += Δτ * SVector(v[1][node], v[2][node], v[3][node])
                end
                copyto!(mesh.coords, moved)
                update_geometry!(mesh; workgroup)
            else
                foreach(copyto!, τ_old, τ)
            end
        end
    finally
        if advection
            copyto!(mesh.coords, reference_coords)
            update_geometry!(mesh; workgroup)
        end
    end
    return (; time = copy(time_edges), station_ids = model.station_ids, station_xy = model.station_xy,
        ux, uy, uz, u_field, pressure, expansion, storage, budget, stats,
        chamber_volume = model.chamber_volume, geometry = model.geometry,
        material = (; η_shell, η_host, η_magma, G_host, G_magma, K_host, K_magma),
        physics = (; gravity, advection, ρ_host, ρ_magma = gravity ? ρ_magma : nothing), numerics = (; ϵ_tol, iterMax))
end

end
