#
# Pressurised elliptical cavity benchmark: accuracy and solver cost in the elastic-dominated regime.
#
# A soft compressible inclusion is injected through the volumetric source `Q` inside a Maxwell host
# and solved for one step (`solve_elliptical_cavity` in
# `examples/reykjanes/elliptical_cavity_setup.jl`), then compared with the closed-form plane-strain
# solution of a pressurised elliptical hole. Each sweep changes one setting
# from the defaults (2 563 elements, `Δt` of one day, an inclusion shear modulus of 1e-3 of the
# host's, `ϵ_tol = 1e-6`) and reports, per run:
#
#   - accuracy: cavity pressure, area change and opening against the closed form, and the
#     displacement error outside the cavity;
#   - cost: inner iterations and wall time, which are reported separately because a change that
#     cheapens an iteration and a change that removes iterations are different improvements;
#   - stopping: the relative residual reduction actually reached, because the stopping test takes
#     the smaller of an absolute and a relative error and so depends on the units.
#
# `converged` must be true for a row to mean anything. A run that stops on the iteration cap
# reports the cost of the budget, not of solving the problem.
#
# Run from a warm session: the numbers of interest are steady-state solver throughput, so
# compilation is amortised by a warm-up run instead of being folded into the first row.
#
#     julia --project=examples
#     include("examples/benchmarks/stokes/elliptical_cavity/elliptical_cavity.jl")
#     rows = run_cavity_benchmark()
#     rows = run_cavity_benchmark(; sweeps = (:mesh, :tolerance))
#

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "reykjanes", "elliptical_cavity_setup.jl"))

"""
    cavity_sweeps() -> NamedTuple

The settings of each sweep, as keyword arguments of `solve_elliptical_cavity`: `mesh`
(element size and refinement), `contrast` (inclusion shear modulus over the host's), `time_step`
(from an hour to a millennium, against a host Maxwell time of 106 yr), `tolerance` (`ϵ_tol`),
`inner_solve` (`rel_drop0` and the check interval `ncheck`) and `scale` (the stress scale `σ_c` of
the characteristic units, for a fixed physical problem).
"""
function cavity_sweeps()
    year = 365.25 * 86_400
    return (;
        mesh = [
            (; max_area, refinement) for (max_area, refinement) in (
                (4.0e6, 2), (4.0e6, 4), (1.0e6, 4), (1.0e6, 8), (1.0e6, 16), (2.5e5, 8), (2.5e5, 16),
            )
        ],
        contrast = [(; G_cavity_ratio) for G_cavity_ratio in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)],
        time_step = [
            (; Δt) for Δt in (3600.0, 86_400.0, 30 * 86_400.0, year, 10year, 100year, 1000year)
        ],
        tolerance = [(; ϵ_tol) for ϵ_tol in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)],
        inner_solve = [
            (; rel_drop0, ncheck) for (rel_drop0, ncheck) in (
                (0.1, 100), (0.01, 100), (0.001, 100), (0.1, 50), (0.01, 50), (0.01, 200), (0.01, 400),
            )
        ],
        scale = [(; σ_c) for σ_c in (1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11)],
    )
end

"""
    cavity_benchmark_case(; kwargs...) -> NamedTuple

Run one cavity solve with the keyword arguments of `solve_elliptical_cavity` and reduce it to the
scalars the benchmark tracks, plus the wall time in seconds.

A case that throws, for example when the relaxation diverges, yields a row with
`converged = false`, `missing` numbers and the message in `failure`, so a sweep records the failure
and continues instead of aborting on it.
"""
function cavity_benchmark_case(; kwargs...)
    return try
        time = @elapsed result = solve_elliptical_cavity(; kwargs...)
        (;
            nels = result.nels, P_ratio = result.P_cavity / result.P_reference,
            area_ratio = result.ΔA / result.ΔA_reference, opening_ratio = result.opening_ratio,
            error_L2 = result.error_L2, iter = result.iter, err_abs = result.err_abs,
            err_rel = result.err_rel, converged = result.converged, time, failure = "",
        )
    catch err
        err isa InterruptException && rethrow()
        (;
            nels = missing, P_ratio = missing, area_ratio = missing, opening_ratio = missing,
            error_L2 = missing, iter = missing, err_abs = missing, err_rel = missing,
            converged = false, time = missing, failure = sprint(showerror, err),
        )
    end
end

function print_cavity_row(sweep, setting, row)
    label = join(("$k = $v" for (k, v) in pairs(setting)), ", ")
    if ismissing(row.iter)
        @printf("%-12s %-34s FAILED: %s\n", sweep, label, first(split(row.failure, '\n')))
        return
    end
    @printf(
        "%-12s %-34s %6d els  P %.5f  ΔA %.5f  opening %.4f  L2 %.1e  iter %6d  rel %.1e  %6.1f s  %s\n",
        sweep, label, row.nels, row.P_ratio, row.area_ratio, row.opening_ratio, row.error_L2,
        row.iter, row.err_rel, row.time, row.converged ? "" : "NOT CONVERGED",
    )
    return
end

"""
    run_cavity_benchmark(; sweeps = keys(cavity_sweeps()), repeats = 5, warmup = true) -> Vector{NamedTuple}

Run the requested sweeps, print one line per run and return the rows, each tagged with its `sweep`
and `setting`. `P`, `ΔA` and `opening` in the printout are ratios to the closed form and `rel` is the
relative residual reduction at the stop. A final block repeats the default case `repeats` times and
prints the spread of the cavity pressure and the iteration counts, which is the run-to-run
reproducibility of the atomic momentum assembly; pass `repeats = 0` to skip it.
"""
function run_cavity_benchmark(; sweeps = keys(cavity_sweeps()), repeats = 5, warmup = true)
    warmup && cavity_benchmark_case(; max_area = 4.0e6, refinement = 4)
    settings = cavity_sweeps()
    rows = NamedTuple[]
    for sweep in sweeps
        for setting in settings[sweep]
            row = cavity_benchmark_case(; setting...)
            print_cavity_row(sweep, setting, row)
            push!(rows, merge((; sweep, setting), row))
        end
    end
    if repeats > 0
        runs = [solve_elliptical_cavity() for _ in 1:repeats]
        pressures = [run.P_cavity for run in runs]
        spread = (maximum(pressures) - minimum(pressures)) / (sum(pressures) / repeats)
        @printf(
            "%-12s %d identical runs: iterations %s, cavity pressure spread %.1e (relative)\n",
            "repeatability", repeats, string([run.iter for run in runs]), spread,
        )
    end
    return rows
end
