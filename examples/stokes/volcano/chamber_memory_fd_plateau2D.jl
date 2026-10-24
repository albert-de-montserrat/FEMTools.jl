# julia --project=examples examples/stokes/volcano/chamber_memory_fd_plateau2D.jl
# S0 item 3 of DIKE_MEMORY_PAPER_PLAN.md: converge the FD step size before
# trusting chamber_identifiability_pilot's sensitivities. Reuses the pilot's
# own fd_step keyword; adds no new forward-model code.
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
using .ChamberMemory2D
using Printf

function chamber_fd_plateau_demo(;
        out_dir = joinpath(@__DIR__, "output_chamber_fd_plateau"), mesh_options = (;),
        fd_steps = (0.20, 0.05, 0.02, 0.01), plateau_tol = 0.02, pilot_options = (;))
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    history = ChamberMemory2D.chamber_cycle_history()
    pilots = [ChamberMemory2D.chamber_identifiability_pilot(model, history; fd_step, pilot_options...)
        for fd_step in fd_steps]
    names = map(r -> r.name, first(pilots).rows)
    mkpath(out_dir)
    open(joinpath(out_dir, "fd_plateau.csv"), "w") do io
        println(io, "experiment,fd_step,sigma_logQ_marginal,tangent_error")
        for (i, name) in enumerate(names)
            println("Experiment: ", name)
            σ_prev = nothing
            for (fd_step, pilot) in zip(fd_steps, pilots)
                σ = pilot.rows[i].σ_marginal
                println(io, join((name, fd_step, σ, pilot.rows[i].tangent_error), ','))
                change = isnothing(σ_prev) ? NaN : abs(σ - σ_prev) / σ_prev
                @printf("  fd_step %.3f  σ(logQ) marginal %.4f  tangent error %.3f%%  change from previous step %s\n",
                    fd_step, σ, 100pilot.rows[i].tangent_error,
                    isnan(change) ? "n/a" : @sprintf("%.3f%%", 100change))
                σ_prev = σ
            end
        end
    end
    # The two finest steps bound the plateau: their difference is the achieved FD-convergence limit.
    finest, finer = pilots[end - 1], pilots[end]
    achieved = [abs(finest.rows[i].σ_marginal - finer.rows[i].σ_marginal) /
        finer.rows[i].σ_marginal for i in eachindex(names)]
    @printf("Finest two FD steps (%.3f, %.3f) agree to %.3f%% (worst experiment); proposed %.0f%% gate: %s\n",
        fd_steps[end - 1], fd_steps[end], 100maximum(achieved), 100plateau_tol,
        maximum(achieved) < plateau_tol ? "PASS" : "NOT YET")
    return pilots
end

chamber_fd_plateau_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_fd_plateau") : first(ARGS))
