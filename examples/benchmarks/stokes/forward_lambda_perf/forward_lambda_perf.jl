using Printf

include(joinpath(@__DIR__, "..", "..", "..", "miniapps", "stokes", "sinking_block_adj", "sinking_block_adj.jl"))

function forward_lambda_case(;
        measure_λmax, max_area = 1 / 32^2, η_incl = 1.0,
        power_iterations = 12, power_rtol = 1.0e-2, safety = 1.1)
    out = main(;
        max_area, η_incl, show_plot = false, verbose = false,
        forward_measure_λmax = measure_λmax,
        forward_λmax_power_iterations = power_iterations,
        forward_λmax_power_rtol = power_rtol,
        forward_λmax_safety = safety,
        adjoint_verbose = false, adjoint_verbose_inner = false,
    )
    s = out.solve_stats
    return (; measure_λmax, max_area, η_incl, iter = s.iter, time = out.t_forward,
        converged = s.converged, λmax = s.λmax,
        λmax_gershgorin = s.λmax_gershgorin,
        λmax_iterations = s.λmax_iterations)
end

function run_forward_lambda_benchmark(;
        max_areas = (1 / 32^2,), η_contrasts = (1.0, 10.0), warmup = true)
    if warmup
        forward_lambda_case(; measure_λmax = false, max_area = 1 / 16^2)
        forward_lambda_case(; measure_λmax = true, max_area = 1 / 16^2)
    end
    rows = NamedTuple[]
    for max_area in max_areas, η_incl in η_contrasts
        baseline = forward_lambda_case(; measure_λmax = false, max_area, η_incl)
        measured = forward_lambda_case(; measure_λmax = true, max_area, η_incl)
        append!(rows, (baseline, measured))
        @printf("area=%8.2e η=%5g  iter %6d -> %6d (%5.2fx)  time %.3f -> %.3f s (%5.2fx)  λG/λP=%5.2f\n",
            max_area, η_incl, baseline.iter, measured.iter,
            baseline.iter / measured.iter, baseline.time, measured.time,
            baseline.time / measured.time,
            measured.λmax_gershgorin / measured.λmax)
    end
    return rows
end
