# julia --project=examples examples/stokes/volcano/chamber_memory_cycle_sweep2D.jl
# S2: does repeating cycles under one shared rheology break the recharge/nuisance
# degeneracy found by chamber_memory_identifiability2D.jl? Sweeps cycle count
# for two designs at equal total withdrawal: "identical" (every cycle withdraws
# 0.75 of its recharge) and "varied" (fraction 0.75 + 0.25 cos(2πj/k), mean 0.75
# for k ≥ 2). Repeated 40-day cycles share one η_shell. Reuses
# chamber_identifiability_pilot unchanged.
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
using .ChamberMemory2D
using Printf
using Serialization

varied_fractions(k) = ntuple(j -> 0.75 + 0.25cospi(2(j - 1) / k), k)

function cycle_sweep_demo(; out_dir = joinpath(@__DIR__, "output_chamber_cycle_sweep"),
        mesh_options = (;), cycle_counts = (1, 2, 3, 5), cycle_length = 40ChamberMemory2D.day,
        pilot_options = (;))
    mkpath(out_dir)
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    # A single cycle has no diversity to vary.
    designs = [[(; cycles = k, design = "identical", fractions = 0.75) for k in cycle_counts];
        [(; cycles = k, design = "varied", fractions = varied_fractions(k)) for k in cycle_counts if k > 1]]
    pilots = map(designs) do d
        history = ChamberMemory2D.chamber_cycle_history(;
            cycle_lengths = ntuple(_ -> cycle_length, d.cycles), withdrawal_fraction = d.fractions)
        elapsed = @elapsed pilot = ChamberMemory2D.chamber_identifiability_pilot(model, history; pilot_options...)
        dense = first(pilot.rows)
        @printf("%-9s %d cycles  %.0f s  σ(logQ) fixed %.4f / marginal %.4f  correlations %s\n",
            d.design, d.cycles, elapsed, dense.σ_fixed, dense.σ_marginal,
            join((@sprintf("%s %+.3f", n, c) for (n, c) in zip(pilot.nuisance_parameters, dense.correlations)), "  "))
        flush(stdout)
        pilot
    end
    nuisance_names = first(pilots).nuisance_parameters
    open(joinpath(out_dir, "cycle_sweep.csv"), "w") do io
        println(io, "design,cycles,experiment,sigma_logQ_fixed,sigma_logQ_marginal," *
            join(string.("correlation_", nuisance_names), ','))
        for (d, pilot) in zip(designs, pilots), r in pilot.rows
            println(io, join((d.design, d.cycles, r.name, r.σ_fixed, r.σ_marginal,
                r.correlations...), ','))
        end
    end
    serialize(joinpath(out_dir, "pilots.jls"), (; designs, pilots))
    println("Julia threads: $(Threads.nthreads()). Shared rheology: one η_shell per run.")
    return pilots
end

cycle_sweep_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_cycle_sweep") : first(ARGS))
