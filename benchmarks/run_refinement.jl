function plot_refinement(rows, output_dir; show_plot, write_plots)
    fig = GLMakie.Figure(size = (1000, 800))
    for (panel, (case, sweep)) in enumerate((("SolKz", "space"), ("SolCx", "space"),
                                           ("ThermalDiffusion", "space"), ("ThermalDiffusion", "time")))
        selected = filter(r -> r.case == case && r.sweep == sweep, rows)
        ax = GLMakie.Axis(fig[cld(panel, 2), isodd(panel) ? 1 : 2];
            title = "$case: $sweep", xlabel = sweep == "space" ? "1 / cells per side" : "1 / time steps",
            ylabel = "Relative L² error", xscale = log10, yscale = log10)
        xs = [1 / (sweep == "space" ? r.resolution : r.nsteps) for r in selected]
        for field in (case == "ThermalDiffusion" ? (:temperature,) : (:velocity, :pressure))
            GLMakie.scatterlines!(ax, xs, [getproperty(r, field) for r in selected]; label = string(field))
        end
        GLMakie.axislegend(ax)
    end
    write_plots && GLMakie.save(joinpath(output_dir, "refinement.png"), fig)
    show_plot && display(fig)
    return fig
end
using TOML
module SolKzBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "stokes", "solkz2D", "SolKz2D_triangle.jl"))
end
end
module SolCxBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "stokes", "solcx2D", "SolCx2D_triangle.jl"))
end
end
module ThermalDiffusionBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "thermal", "thermal_diffusion2D", "ThermalDiffusion2D.jl"))
end
end

"""
    main(; resolutions=(4, 8, 16), timesteps=(4, 8, 16), backend=CPU(), ...)

Run spatial Stokes refinement and separate spatial/temporal diffusion sweeps.
Use a small fixed physical timestep for the thermal spatial sweep. Record
observed rates, solver residuals, iterations, and parameters in a CSV file.
Timing includes compilation and is not a performance comparison.
"""
function main(; resolutions = (4, 8, 16), timesteps = (4, 8, 16),
              backend = SolKzBenchmark.CPU(), stokes_contrasts = (10.0, 2.0),
              thermal_spatial_steps = 2000, show_plot = false, write_plots = false,
              output_dir = joinpath(@__DIR__, "output"))
    length(stokes_contrasts) == 2 || throw(ArgumentError("supply one contrast for each Stokes case"))
    all(n -> n isa Integer && n > 0, resolutions) && issorted(resolutions) && allunique(resolutions) ||
        throw(ArgumentError("resolutions must be distinct increasing positive integers"))
    isempty(resolutions) && throw(ArgumentError("resolutions must not be empty"))
    all(iseven, resolutions) || throw(ArgumentError("SolCx requires even resolutions"))
    all(n -> n isa Integer && n > 0, timesteps) && issorted(timesteps) && allunique(timesteps) ||
        throw(ArgumentError("timesteps must be distinct increasing positive integers"))
    rows = NamedTuple[]
    for (case, contrast) in zip((:SolKz, :SolCx), stokes_contrasts)
        for resolution in resolutions
            benchmark = case == :SolKz ? SolKzBenchmark : SolCxBenchmark
            r = benchmark.main(; save_history = false, resolution, contrast, backend)
            push!(rows, (; case = string(case), sweep = "space", resolution, nsteps = 0,
                contrast, velocity = r.errors.velocity.relative, pressure = r.errors.pressure.relative,
                temperature = NaN, residual = r.stats.err_abs, iterations = r.stats.iter))
        end
    end
    for (sweep, cases) in (("space", [(n, thermal_spatial_steps) for n in resolutions]),
                           ("time", [(last(resolutions), n) for n in timesteps]))
        for (resolution, nsteps) in cases
            r = ThermalDiffusionBenchmark.main(; save_history = false, resolution, nsteps, backend)
            push!(rows, (; case = "ThermalDiffusion", sweep, resolution, nsteps,
                contrast = NaN, velocity = NaN, pressure = NaN, temperature = r.errors.relative,
                residual = r.errors.residual, iterations = missing))
        end
    end
    mkpath(output_dir)
    open(joinpath(output_dir, "refinement.csv"), "w") do io
        println(io, "case,sweep,resolution,nsteps,contrast,velocity,pressure,temperature,residual,iterations")
        for row in rows
            println(io, join(values(row), ','))
        end
    end
    rates = NamedTuple[]
    for case in ("SolKz", "SolCx", "ThermalDiffusion"), sweep in ("space", "time")
        selected = filter(r -> r.case == case && r.sweep == sweep, rows)
        for i in 2:length(selected)
            previous, current = selected[i - 1], selected[i]
            ratio = sweep == "space" ? current.resolution / previous.resolution : current.nsteps / previous.nsteps
            rate(a, b) = log(a / b) / log(ratio)
            push!(rates, (; case, sweep, resolution = current.resolution, nsteps = current.nsteps,
                velocity = rate(previous.velocity, current.velocity),
                pressure = rate(previous.pressure, current.pressure),
                temperature = rate(previous.temperature, current.temperature)))
        end
    end
    open(joinpath(output_dir, "rates.csv"), "w") do io
        println(io, "case,sweep,resolution,nsteps,velocity,pressure,temperature")
        for row in rates
            println(io, join(values(row), ','))
        end
    end
    open(joinpath(output_dir, "metadata.toml"), "w") do io
        TOML.print(io, Dict("julia" => string(VERSION), "backend" => string(typeof(backend)),
            "FEMTools" => string(pkgversion(SolKzBenchmark.FEMTools)),
            "ExactFieldSolutions" => string(pkgversion(SolKzBenchmark.ExactFieldSolutions)),
            "thermal_spatial_steps" => thermal_spatial_steps))
    end
    if show_plot || write_plots
        @eval import GLMakie
        Base.invokelatest(plot_refinement, rows, output_dir; show_plot, write_plots)
    end
    return (; rows, rates)
end

result = main()
@info "Exact-field refinement rates" result.rates
