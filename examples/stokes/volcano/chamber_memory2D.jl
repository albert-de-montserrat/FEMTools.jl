# Run from the repository root:
# julia --project=examples examples/stokes/volcano/chamber_memory2D.jl
# No graphics or thermal solver; outputs are SI-valued CSVs and Julia snapshots.
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
using .ChamberMemory2D
using LinearAlgebra
using Printf

function chamber_memory_demo(; out_dir = joinpath(@__DIR__, "output_chamber_memory"),
        Δt = 10ChamberMemory2D.day, mesh_options = (;), run_options = (;))
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    @info "Chamber prototype mesh" cells = model.mesh.nels nodes = model.mesh.nnodes
    history = ChamberMemory2D.chamber_cycle_history(; Δt)
    fine_history = ChamberMemory2D.chamber_cycle_history(; Δt = Δt / 2)
    baseline = ChamberMemory2D.run_chamber_memory(model, history; run_options...)
    refined = ChamberMemory2D.run_chamber_memory(model, fine_history; run_options...)
    elastic_options = merge(run_options, (; η_host = 1.0e25, η_shell = 1.0e25))
    elastic = ChamberMemory2D.run_chamber_memory(model, history; elastic_options...)
    common = [only(findall(==(t), refined.time)) for t in baseline.time]
    timestep_change = norm(baseline.uz - refined.uz[:, common]) / norm(refined.uz[:, common])
    for (name, result) in (("viscoelastic", baseline), ("half_timestep", refined), ("elastic_host", elastic))
        budget_error = maximum(abs, result.budget) / maximum(abs, result.net_area)
        @assert budget_error < 1.0e-4 "discrete chamber continuity budget did not close"
        ChamberMemory2D.write_chamber_result(out_dir, name, result)
        @printf("%-14s %5d cells  %2d steps  max budget/source %.2e  final P %.3f MPa\n",
            name, result.nels, length(result.stats), budget_error, last(result.pressure) / 1.0e6)
    end
    elastic_reference = ChamberMemory2D.chamber_elastic_reference(elastic)
    @printf("Elastic-host displacement misfit to the same-area Mogi line source: %.3f%%\n",
        100elastic_reference.relative_error)
    @printf("Vertical displacement change on halving Δt: %.3f%%; proposed 2%% gate: %s\n",
        100timestep_change, timestep_change < 0.02 ? "PASS" : "NOT YET")
    println("Plane-strain synthetic controls only; mesh/domain and spherical/3D qualification remain separate.")
    return (; baseline, refined, elastic, elastic_reference, timestep_change)
end

chamber_memory_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_memory") : first(ARGS))
