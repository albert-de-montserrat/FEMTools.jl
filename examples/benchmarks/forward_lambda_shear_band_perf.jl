using Printf

include(joinpath(@__DIR__, "..", "stokes", "vevp", "stokes_2D_pure_shear_triangle.jl"))

function shear_band_lambda_case(;
        measure_λmax, nsteps = 6, max_area = 1 / 32^2, λmax_safety = 1.1)
    out = main(;
        nsteps, n_circle = 48, max_area, show_plot = false, write_output = false,
        verbose = false, measure_λmax, λmax_safety)
    stats = out.solve_stats
    return (; measure_λmax, nsteps, max_area,
        iterations = sum(s.iter for s in stats),
        time = out.solve_time,
        converged = all(s.converged for s in stats),
        λmax_iterations = sum(s.λmax_iterations for s in stats),
        λmax = [s.λmax for s in stats],
        λmax_gershgorin = [s.λmax_gershgorin for s in stats],
        stress = out.mean_tauII)
end

function run_shear_band_lambda_benchmark(; nsteps = 15, max_area = 1 / 32^2)
    shear_band_lambda_case(; measure_λmax = false, nsteps = 1, max_area = 1 / 16^2)
    shear_band_lambda_case(; measure_λmax = true, nsteps = 1, max_area = 1 / 16^2)
    baseline = shear_band_lambda_case(; measure_λmax = false, nsteps, max_area)
    measured = shear_band_lambda_case(; measure_λmax = true, nsteps, max_area)
    @printf("shear band (%d steps): iter %d -> %d (%4.2fx), time %.3f -> %.3f s (%4.2fx), converged=%s/%s\n",
        nsteps, baseline.iterations, measured.iterations,
        baseline.iterations / measured.iterations,
        baseline.time, measured.time, baseline.time / measured.time,
        baseline.converged, measured.converged)
    return baseline, measured
end
