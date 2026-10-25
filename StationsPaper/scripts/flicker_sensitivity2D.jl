# julia --project=. StationsPaper/scripts/flicker_sensitivity2D.jl [impulse_dir] [out_dir]
# Time-correlated noise test of the Q-history inversion (PLAN.md next work 3).
# Noise models, each a station-specific and a common-mode part:
#   "empirical"  white 3 mm + common white 3.5 mm (station_information2D.jl);
#   "IMO flicker" white plus flicker as fitted by imo_flicker.jl;
#   "2× flicker"  the same with both flicker amplitudes doubled, because the fit
#                 only constrains periods shorter than about a month.
# For every pair of true and assumed noise model the estimate stays linear, so
# its error covariance is exact: G C_true Gᵀ + (I - G K) C_prior (I - G K)ᵀ,
# with G the gain under the assumed model. Reported (posterior) and actual
# standard deviations are compared for the window-mean rate, the trend (mean of
# the last minus the first 13 weeks), single 7-day bins, and the pre-window
# mean. Setup as in synthetic_recovery2D.jl. Writes flicker_sensitivity.csv.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_kernels.jl"))
include(joinpath(@__DIR__, "imo_flicker.jl"))

"""Covariance of `ns` stations × `window` daily epochs (station fastest): specific plus common white and flicker parts."""
function noise_covariance(ns, F; σ, σ_flicker = 0.0, σ_common, σ_common_flicker = 0.0)
    window = size(F, 1)
    time_station = σ^2 * I(window) + σ_flicker^2 * F
    time_common = σ_common^2 * I(window) + σ_common_flicker^2 * F
    return Symmetric(kron(time_station, Matrix(1.0I, ns, ns)) + kron(time_common, ones(ns, ns)))
end

function flicker_sensitivity_demo(; impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        out_dir = joinpath(@__DIR__, "..", "output", "flicker_sensitivity2D"),
        window = 365, prewindow = 720, width = 7, prewidth = 30, uplift = 0.24, quarter = 13,
        imo_fit = (; σ = 1.93e-3, σ_flicker = 0.92e-3, σ_common = 3.35e-3, σ_common_flicker = 1.71e-3))
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    centre = findfirst(==(0.0), response.station_x)
    Q_ref = uplift / sum(response.H[centre, 1:window])
    T = power_law_factor(window)
    F = T * T'
    doubled = merge(imo_fit, (; σ_flicker = 2imo_fit.σ_flicker, σ_common_flicker = 2imo_fit.σ_common_flicker))
    models = ["empirical" => (; σ = 3.0e-3, σ_common = 3.5e-3), "IMO flicker" => imo_fit, "2× flicker" => doubled]
    pairs = [("IMO flicker", "IMO flicker"), ("IMO flicker", "empirical"),
        ("2× flicker", "2× flicker"), ("2× flicker", "empirical"), ("2× flicker", "IMO flicker")]
    near = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3)
    far = (8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)
    layouts = ["x=0 km" => (0.0,), "6 near" => near, "6 near + 6 far" => (near..., far...)]

    rows = NamedTuple[]
    @printf("Q_ref %.3e m²/s; %d-day bins; σ in Q_ref; reported / actual\n", Q_ref, width)
    for (layout, station_x) in layouts
        stations = [findfirst(==(x), response.station_x) for x in station_x]
        ns = length(stations)
        jacobian = window_jacobian(response.H, stations; window, width, prewindow, prewidth)
        (; nwindow, bins) = jacobian
        nbins = length(bins)
        K = reshape(jacobian.K, ns * window, nbins)
        lengths = length.(bins)
        functionals = ["window mean" => [j <= nwindow ? lengths[j] / window : 0.0 for j in 1:nbins],
            "trend" => [j <= quarter ? -1 / quarter : nwindow - quarter < j <= nwindow ? 1 / quarter : 0.0 for j in 1:nbins],
            "pre-window mean" => [j > nwindow ? lengths[j] / prewindow : 0.0 for j in 1:nbins]]
        factors = Dict(name => cholesky(noise_covariance(ns, F; model...)) for (name, model) in models)
        C_prior = Q_ref^2 * I(nbins)
        for (truth, assumed) in pairs
            Y = factors[assumed] \ K
            posterior = inv(Symmetric(K' * Y + inv(C_prior)))
            G = posterior * Y'
            C_true = Matrix(factors[truth])
            error_covariance = G * C_true * G' + (I - G * K) * C_prior * (I - G * K)'
            ratio(g) = sqrt(g' * error_covariance * g) / sqrt(g' * posterior * g)
            reported(g) = sqrt(g' * posterior * g) / Q_ref
            weekly_reported = median(sqrt.(diag(posterior)[1:nwindow])) / Q_ref
            weekly_actual = median(sqrt.(diag(error_covariance)[1:nwindow])) / Q_ref
            row = (; layout, truth, assumed, dof = tr(G * K),
                weekly_reported, weekly_actual,
                (Symbol(replace(name, " " => "_", "-" => "_") * "_reported") => reported(g) for (name, g) in functionals)...,
                (Symbol(replace(name, " " => "_", "-" => "_") * "_actual") => reported(g) * ratio(g) for (name, g) in functionals)...)
            push!(rows, row)
            @printf("  %-15s true %-11s assumed %-11s dof %5.1f  week %.2f/%.2f  mean %.3f/%.3f  trend %.3f/%.3f  pre-window %.3f/%.3f\n",
                layout, truth, assumed, row.dof, weekly_reported, weekly_actual, row.window_mean_reported,
                row.window_mean_actual, row.trend_reported, row.trend_actual, row.pre_window_mean_reported,
                row.pre_window_mean_actual)
        end
    end
    open(joinpath(out_dir, "flicker_sensitivity.csv"), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        foreach(r -> println(io, join((v isa String ? "\"" * v * "\"" : v for v in values(r)), ',')), rows)
    end
    serialize(joinpath(out_dir, "flicker_sensitivity.jls"), (; rows, Q_ref, models, window, prewindow, width))
    println("output: ", abspath(out_dir))
    return rows
end

flicker_sensitivity_demo(;
    impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "flicker_sensitivity2D"))
