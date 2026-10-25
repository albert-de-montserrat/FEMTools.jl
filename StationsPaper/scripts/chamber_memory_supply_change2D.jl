# julia --project=examples examples/stokes/volcano/chamber_memory_supply_change2D.jl
# S2: is a *relative* supply change (recent 30 days versus before) identifiable
# where the absolute recharge amplitude is not? One five-parameter pilot
# (recharge amplitude, shell viscosity, withdrawal amplitude, magma bulk
# modulus, recent-supply ratio) over three 40-day cycles. Each Jacobian column
# is a parameter's own perturbation, independent of which other parameters are
# estimated, so σ for any nuisance subset is recomputed from the one Jacobian.
# The 12 independent forward runs go to single-threaded worker processes:
# threading inside one solve is slower on this mesh.
using Distributed
const nworkers_requested = parse(Int, get(ENV, "CHAMBER_WORKERS", "4"))
addprocs(nworkers_requested; exeflags = "--project=$(Base.active_project())")
@everywhere include(joinpath($(@__DIR__), "chamber_memory_setup2D.jl"))
@everywhere using .ChamberMemory2D
using LinearAlgebra
using Printf
using Serialization

"""Marginal σ of parameter `target` when only `keep` (which must include it) are estimated."""
function marginal_sigma(J, prior, noise, keep, target)
    Js = J[:, keep]
    precision = Js' * Js / noise^2 + Diagonal(1 ./ prior[keep] .^ 2)
    return sqrt(inv(Symmetric(precision))[findfirst(==(target), keep), findfirst(==(target), keep)])
end

function supply_change_demo(; out_dir = joinpath(@__DIR__, "output_chamber_supply_change"),
        mesh_options = (;), cycle_lengths = ntuple(_ -> 40ChamberMemory2D.day, 3), pilot_options = (;))
    mkpath(out_dir)
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    history = ChamberMemory2D.chamber_cycle_history(; cycle_lengths)
    elapsed = @elapsed pilot = ChamberMemory2D.chamber_identifiability_pilot(model, history;
        parameters = ChamberMemory2D.IDENTIFIABILITY_PARAMETERS[1:5], target = :log_recent_supply_ratio,
        mapper = pmap, pilot_options...)
    names = pilot.parameters
    amplitude = findfirst(==(:log_recharge_amplitude), names)
    ratio = findfirst(==(:log_recent_supply_ratio), names)
    nuisance = [k for k in eachindex(names) if k ∉ (amplitude, ratio)]
    # Nuisance sets: none, each one alone, all.
    sets = [("none", Int[]); [(string(names[k]), [k]) for k in nuisance]; ("all", nuisance)]
    (; prior_std, noise) = pilot.settings
    open(joinpath(out_dir, "supply_change.csv"), "w") do io
        println(io, "experiment,nuisance,sigma_ratio,sigma_amplitude")
        for r in pilot.rows, (label, extra) in sets
            keep = [amplitude; ratio; extra]
            println(io, join((r.name, label, marginal_sigma(r.J, prior_std, noise, keep, ratio),
                marginal_sigma(r.J, prior_std, noise, keep, amplitude)), ','))
        end
    end
    dense = first(pilot.rows)
    @printf("%d forward runs on %d workers in %.0f s; dense sampling, prior σ 0.5 on both targets\n",
        2length(names) + 2, nworkers(), elapsed)
    for (label, extra) in sets
        keep = [amplitude; ratio; extra]
        @printf("  nuisance %-26s σ(log recent ratio) %.4f   σ(log amplitude) %.4f\n", label,
            marginal_sigma(dense.J, prior_std, noise, keep, ratio),
            marginal_sigma(dense.J, prior_std, noise, keep, amplitude))
    end
    @printf("All five estimated: ratio RMSE %.4f, coverage %.2f, tangent error %.1f%%\n",
        dense.rmse_joint, dense.coverage_joint, 100dense.tangent_error)
    serialize(joinpath(out_dir, "pilot.jls"), pilot)
    return pilot
end

supply_change_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_supply_change") : first(ARGS))
