# julia --project=examples examples/stokes/volcano/chamber_memory_gauss_newton2D.jl
# Is the recent-supply ratio's under-coverage (0.88 with the one-step estimate
# in chamber_memory_supply_change2D.jl) a linearization artifact? Gauss–Newton
# MAP fit of all five parameters to the noise-free truth, then noisy recovery
# at the one-step and converged linearization points with the same noise.
# Forward runs go to single-threaded worker processes (CHAMBER_WORKERS, default 4).
using Distributed
addprocs(parse(Int, get(ENV, "CHAMBER_WORKERS", "4")); exeflags = "--project=$(Base.active_project())")
@everywhere include(joinpath($(@__DIR__), "chamber_memory_setup2D.jl"))
@everywhere using .ChamberMemory2D
using Printf
using Serialization

function gauss_newton_demo(; out_dir = joinpath(@__DIR__, "output_chamber_gauss_newton"),
        mesh_options = (;), cycle_lengths = ntuple(_ -> 40ChamberMemory2D.day, 3), fit_options = (;))
    mkpath(out_dir)
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    history = ChamberMemory2D.chamber_cycle_history(; cycle_lengths)
    elapsed = @elapsed fit = ChamberMemory2D.chamber_gauss_newton(model, history; mapper = pmap, fit_options...)
    t = findfirst(==(fit.target), fit.parameters)
    @printf("%s after %d iterations (%.0f s, %d workers); truth %s = %+.4f\n",
        fit.converged ? "Converged" : "NOT converged", length(fit.steps), elapsed, nworkers(),
        fit.target, fit.θ_true[t])
    open(joinpath(out_dir, "iterations.csv"), "w") do io
        println(io, "iteration,max_step,misfit_over_noise," * join(string.(fit.parameters), ','))
        for s in fit.steps
            println(io, join((s.iteration, s.max_step, s.misfit, s.θ...), ','))
            @printf("  iteration %d  max step %.2e  rms misfit / noise %.3f  %s %+.4f\n",
                s.iteration, s.max_step, s.misfit, fit.target, s.θ[t])
        end
    end
    for (label, r) in (("one-step (nominal)", fit.one_step), ("converged", fit.converged_fit))
        @printf("  %-19s bias %+.4f  σ %.4f  RMSE %.4f  95%% coverage %.2f\n",
            label, r.bias, r.σ, r.rmse, r.coverage)
    end
    @printf("  final estimate vs truth: %s\n", join((@sprintf("%s %+.3f/%+.3f", p, a, b)
        for (p, a, b) in zip(fit.parameters, fit.θ, fit.θ_true)), "  "))
    serialize(joinpath(out_dir, "fit.jls"), fit)
    return fit
end

gauss_newton_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_gauss_newton") : first(ARGS))
