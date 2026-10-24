# julia --project=examples examples/stokes/volcano/chamber_memory_convergence2D.jl
# S0/S1 mesh- and domain-size convergence for the chamber-memory pilot, at
# fixed physical geometry, material, and history. Reuses
# build_chamber_memory_mesh's existing mesh/domain keywords; no new
# forward-model code. Mesh refinement (max_area) directly controls how well
# the circular chamber/shell boundary is resolved, which is S0 item 1's
# "source footprint."
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
using .ChamberMemory2D
using LinearAlgebra
using Printf
using Serialization

"""Run one chamber-memory case and return cell/node counts, near-center final
uz (for display only), the full uz time series (for the convergence metric),
and the discrete budget-closure ratio."""
function convergence_case(; mesh_options = (;), run_options = (;))
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    history = ChamberMemory2D.chamber_cycle_history()
    result = ChamberMemory2D.run_chamber_memory(model, history; run_options...)
    station = argmin(abs.(model.station_x))
    budget_error = maximum(abs, result.budget) / maximum(abs, result.net_area)
    return (; nels = model.mesh.nels, nnodes = model.mesh.nnodes,
        uz_final = result.uz[station, end], uz = result.uz, budget_error)
end

"""Print a refinement table and report whether the two finest levels agree
within `gate` in an L2 norm over all stations and epochs (not a single
station's nearest-node value, which is noisy under unstructured remeshing);
returns that pass/fail."""
function report_sweep(label, cases, values; gate = 0.02)
    println(label)
    previous = nothing
    for (value, case) in zip(values, cases)
        change = isnothing(previous) ? NaN : norm(case.uz - previous) / norm(previous)
        @printf("  %-10s cells %5d  nodes %5d  uz(x≈0,final) %7.3f mm  budget/source %.2e  L2 change %s\n",
            string(value), case.nels, case.nnodes, 1.0e3case.uz_final, case.budget_error,
            isnan(change) ? "n/a" : @sprintf("%.3f%%", 100change))
        previous = case.uz
    end
    finest_change = norm(cases[end].uz - cases[end - 1].uz) / norm(cases[end - 1].uz)
    passed = finest_change < gate
    @printf("  finest two levels agree to %.3f%% (L2, all stations/epochs); proposed %.0f%% gate: %s\n",
        100finest_change, 100gate, passed ? "PASS" : "NOT YET")
    return passed
end

function chamber_convergence_demo(; out_dir = joinpath(@__DIR__, "output_chamber_convergence"),
        station_x = range(-8.0e3, 8.0e3; length = 9))
    # A single nearest-surface-node station is noisy across meshes rebuilt from
    # scratch per level, because the nearest node to a requested x shifts by up
    # to about one surface element. 17 stations average that snapping error out
    # in the L2 comparison below instead of exposing one point's snap directly.
    mkpath(out_dir)
    # build_gmsh_t7_concentric_disk_mesh derives mesh_size = √(2 max_area) and
    # grades the near-chamber mesh from it, but surface_size (the station
    # region's local size) is an independent absolute length. Scaling it with
    # √max_area keeps surface_size/mesh_size fixed, so the station region is
    # actually nested-refined rather than drifting relative to the background.
    max_areas = (1.0e7, 4.0e6, 1.6e6, 8.0e5)
    default_max_area, default_surface_size = 4.0e6, 1.0e3
    mesh_cases = [convergence_case(; mesh_options = (; max_area, station_x,
            surface_size = default_surface_size * sqrt(max_area / default_max_area)))
        for max_area in max_areas]
    mesh_pass = report_sweep("Mesh resolution (fixed 40×20 km domain, max_area in m², surface_size scaled with it):",
        mesh_cases, max_areas)

    widths = (20.0e3, 40.0e3, 80.0e3, 160.0e3)
    domain_cases = [convergence_case(; mesh_options = (; domain = (width, width / 2), station_x))
        for width in widths]
    domain_pass = report_sweep("Domain size (fixed mesh density, Lx × Lx/2 in m):", domain_cases, widths)

    serialize(joinpath(out_dir, "mesh_sweep.jls"), (; max_areas, mesh_cases))
    serialize(joinpath(out_dir, "domain_sweep.jls"), (; widths, domain_cases))
    println(mesh_pass && domain_pass ? "Both sweeps pass the proposed 2% gate." :
        "At least one sweep has not converged at the finest level tested: widen before trusting S1 interpretation.")
    return (; mesh_cases, domain_cases)
end

chamber_convergence_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_convergence") : first(ARGS))
