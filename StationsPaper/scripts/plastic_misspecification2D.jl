# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/plastic_misspecification2D.jl [forward_dir] [out_dir]
# Does tensile plasticity, with dry, hydrostatic, or lithostatic pore pressure,
# change what the inversions report? Truth: forward runs of plastic_forward2D.jl
# (gravity + Lagrangian mesh, Maxwell shell, Drucker-Prager with a tensile cap),
# each minus the source-free reference run. Inversions, both blind to
# plasticity:
#   - the chamber model with the reference impulse response
#     (impulse_response2D_3yr), set up as in synthetic_recovery2D.jl: 7-day
#     window bins, 720 days of unknown 30-day pre-window bins, prior mean and σ
#     Q_ref, empirical noise (3 mm + 3.5 mm common);
#   - elastic Mogi (mogi_comparison2D.jl) by weighted least squares, depth from a
#     grid search, supply = (1 + G/K_magma) × source-area rate.
# Noise-free, so the numbers are expected values. Each plastic truth is compared
# with the same inversion of the elastic (non-plastic) truth of the same
# history, predicted by the impulse response; the "reference" forward run checks
# that prediction. Δχ² is the increase of the chamber-model MAP objective; biases
# are in Q_ref. Writes plastic_misspecification.csv and plastic_misspecification.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "stokes", "volcano", "elastic_dike2D.jl"))

function plastic_misspecification_demo(; forward_dir = joinpath(@__DIR__, "..", "output", "plastic_forward2D"),
        out_dir = joinpath(@__DIR__, "..", "output", "plastic_misspecification2D"),
        impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        cases = ("reference", "tensile_dry", "tensile_hydrostatic", "tensile_lithostatic"), histories = ("H0", "H1", "H6"),
        layouts = ("6 near" => (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3),
            "6 near + 6 far" => (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3, 8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)),
        width = 7, prewidth = 30, σ = 3.0e-3, σ_common = 3.5e-3, quarter = 13, depths = 2.0e3:100.0:8.0e3)
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    baseline = deserialize(joinpath(forward_dir, "reference_none.jls"))
    (; prewindow, window, Q_ref) = baseline
    (; G_host, K_host, K_magma) = response.material
    ν = (3K_host - 2G_host) / (2 * (3K_host + G_host))
    compressibility = 1 + G_host / K_magma
    rows = NamedTuple[]
    for (layout, station_x) in layouts
        ns = length(station_x)
        stations = [findfirst(==(x), response.station_x) for x in station_x]
        forward_stations = [findfirst(==(x), baseline.station_x) for x in station_x]
        noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
        whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
        H = response.H[stations, :]
        jacobian = window_jacobian(H, 1:ns; window, width, prewindow, prewidth)
        (; nwindow, bins) = jacobian
        Kw = whiten(reshape(jacobian.K, ns * window, length(bins)))
        prior_mean = fill(Q_ref, length(bins))
        covariance = inv(Symmetric(Kw' * Kw + I / Q_ref^2))
        chamber_fit(data) = (m = prior_mean + covariance * (Kw' * (data - Kw * prior_mean));
            (; estimate = m, objective = sum(abs2, data - Kw * m) + sum(abs2, (m - prior_mean) ./ Q_ref)))
        mogi_kernels = map(depths) do depth
            g = [last(line_source_surface(x; area_change = response.Δt, depth, ν)) for x in station_x]
            whiten(reshape(window_jacobian(repeat(g, 1, window), 1:ns; window, width).K, ns * window, nwindow))
        end
        function mogi_supply(data)
            fits = [(r = Kw_m \ data; (; rates = r, misfit = sum(abs2, data - Kw_m * r))) for Kw_m in mogi_kernels]
            best = argmin(getfield.(fits, :misfit))
            return (; depth = depths[best], ratio = compressibility * mean(fits[best].rates) / Q_ref)
        end
        for history in histories
            # The elastic prediction of the same history, from the impulse response.
            truth = deserialize(joinpath(forward_dir, "$(cases[1])_$(history).jls"))
            rates = Q_ref .* truth.q
            elastic_data = whiten(window_displacement(H, rates, prewindow))
            elastic = chamber_fit(elastic_data)
            elastic_mogi = mogi_supply(elastic_data)
            for case in cases
                file = joinpath(forward_dir, "$(case)_$(history).jls")
                isfile(file) || (println("skip ", file); continue)
                run = deserialize(file)
                u = (run.uz .- baseline.uz)[forward_stations, :]
                # Window records relative to the window start, as window_displacement orders them.
                data = whiten(vec(u[:, (prewindow + 2):end] .- u[:, prewindow + 1]))
                fit = chamber_fit(data)
                mogi = mogi_supply(data)
                bias = (fit.estimate .- elastic.estimate) ./ Q_ref
                window_bias = bias[1:nwindow]
                first_yield = findfirst(>(0), run.plastic_γ)
                push!(rows, (; layout, case, history, Δχ2 = fit.objective - elastic.objective,
                    data_difference = norm(data - elastic_data) / norm(elastic_data),
                    window_bias = mean(window_bias), last_quarter = mean(window_bias[(end - quarter + 1):end]),
                    apparent_trend = mean(window_bias[(end - quarter + 1):end]) - mean(window_bias[1:quarter]),
                    prewindow_bias = mean(bias[(nwindow + 1):end]),
                    mogi_ratio = mogi.ratio, mogi_ratio_elastic = elastic_mogi.ratio, mogi_depth = mogi.depth,
                    max_plastic_γ = maximum(run.plastic_γ; init = 0.0),
                    first_yield_day = isnothing(first_yield) ? -1 : first_yield,
                    max_pressure_MPa = maximum(run.pressure) / 1.0e6))
            end
        end
    end
    @printf("Q_ref %.3e m²/s; chamber-model biases in Q_ref against the same inversion of the elastic truth\n", Q_ref)
    for r in rows
        @printf("  %-15s %-20s %-3s data Δ %.1e  Δχ² %8.2f  window %+.3f  last 13 wk %+.3f  trend %+.3f  pre %+.3f  Mogi %.2f (elastic %.2f)  γmax %.1e  yield day %d  Pmax %.1f MPa\n",
            r.layout, r.case, r.history, r.data_difference, r.Δχ2, r.window_bias, r.last_quarter, r.apparent_trend,
            r.prewindow_bias, r.mogi_ratio, r.mogi_ratio_elastic, r.max_plastic_γ, r.first_yield_day, r.max_pressure_MPa)
    end
    open(joinpath(out_dir, "plastic_misspecification.csv"), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        foreach(r -> println(io, join((v isa String ? "\"" * v * "\"" : v for v in values(r)), ',')), rows)
    end
    result = (; rows, Q_ref, window, prewindow, width, σ, σ_common, forward_dir, impulse_dir)
    serialize(joinpath(out_dir, "plastic_misspecification.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

plastic_misspecification_demo(;
    forward_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "plastic_forward2D"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "plastic_misspecification2D"))
