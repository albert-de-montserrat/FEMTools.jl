# julia --project=. StationsPaper/scripts/gravity_misspecification2D.jl [output_root] [out_dir]
# Does ignoring gravity and finite deformation bias the Q-history inversion?
# Data come from one impulse response and are inverted with another:
#   "linear model, gravity truth": the inversion uses the linear chamber on a
#   fixed mesh without gravity (linear/impulse_response2D_3yr), the data the
#   reference chamber with gravity and a Lagrangian mesh (impulse_response2D_3yr);
#   "gravity model, linear truth": the reverse.
# Noise-free, so the numbers are expected values (as viscosity_ambiguity2D.jl):
# Δχ² is the minimum MAP objective (zero for the right mechanics with a history
# the prior mean matches), and biases are in Q_ref. Inversion setup as in
# synthetic_recovery2D.jl: 7-day window bins, 720 days of unknown 30-day
# pre-window bins, prior mean and σ Q_ref, empirical noise (3 mm + 3.5 mm
# common). Histories H0, H1, H3 of synthetic_recovery2D.jl. Writes
# gravity_misspecification.csv and gravity_misspecification.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "synthetic_recovery2D.jl"))

function gravity_misspecification_demo(; output_root = joinpath(@__DIR__, "..", "output"),
        out_dir = joinpath(output_root, "gravity_misspecification2D"),
        linear_dir = joinpath("linear", "impulse_response2D_3yr"), gravity_dir = "impulse_response2D_3yr",
        window = 365, prewindow = 720, width = 7, prewidth = 30, uplift = 0.24,
        σ = 3.0e-3, σ_common = 3.5e-3, quarter = 13)
    mkpath(out_dir)
    linear = deserialize(joinpath(output_root, linear_dir, "impulse_response.jls"))
    gravity = deserialize(joinpath(output_root, gravity_dir, "impulse_response.jls"))
    linear.station_x == gravity.station_x || throw(ArgumentError("the two responses have different stations"))
    at(xs) = [findfirst(==(x), linear.station_x) for x in xs]
    near = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3)
    far = (8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)
    layouts = ["x=0 km" => at([0.0]), "6 near (0-4.5 km)" => at(near),
        "6 near + 6 far" => at([near..., far...]), "all 29" => collect(eachindex(linear.station_x))]
    centre = findfirst(==(0.0), linear.station_x)
    # Q_ref of the linear model, so both directions use the same rates.
    Q_ref = uplift / sum(linear.H[centre, 1:window])
    histories = filter(h -> startswith(first(h), r"H[013]"), synthetic_histories(prewindow, window))
    cases = ["linear model, gravity truth" => (linear, gravity), "gravity model, linear truth" => (gravity, linear)]

    rows = NamedTuple[]
    for (layout, stations) in layouts, (case, (model, truth)) in cases
        ns = length(stations)
        jacobian = window_jacobian(model.H, stations; window, width, prewindow, prewidth)
        (; nwindow, bins) = jacobian
        noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
        whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
        Kw = whiten(reshape(jacobian.K, ns * window, length(bins)))
        prior_mean = fill(Q_ref, length(bins))
        covariance = inv(Symmetric(Kw' * Kw + I / Q_ref^2))
        for (history, q) in histories
            rates = Q_ref .* q
            exact = [sum(rates[b]) / length(b) for b in bins]
            data = whiten(window_displacement(truth.H[stations, :], rates, prewindow))
            estimate = prior_mean + covariance * (Kw' * (data - Kw * prior_mean))
            # The same fit with the true mechanics separates model error from prior smoothing.
            truth_K = whiten(reshape(window_jacobian(truth.H, stations; window, width, prewindow, prewidth).K,
                ns * window, length(bins)))
            reference = prior_mean + inv(Symmetric(truth_K' * truth_K + I / Q_ref^2)) * (truth_K' * (data - truth_K * prior_mean))
            objective(K, m) = sum(abs2, data - K * m) + sum(abs2, (m - prior_mean) ./ Q_ref)
            Δχ2 = objective(Kw, estimate) - objective(truth_K, reference)
            bias = (estimate .- reference) ./ Q_ref
            window_bias = bias[1:nwindow]
            push!(rows, (; layout, case, history, Δχ2,
                window_bias = mean(window_bias), last_quarter = mean(window_bias[(end - quarter + 1):end]),
                apparent_trend = mean(window_bias[(end - quarter + 1):end]) - mean(window_bias[1:quarter]),
                prewindow_bias = mean(bias[(nwindow + 1):end]),
                window_total = sum((estimate[1:nwindow] - exact[1:nwindow]) .* length.(bins[1:nwindow])) /
                    sum(rates[(prewindow + 1):end])))
        end
    end
    open(joinpath(out_dir, "gravity_misspecification.csv"), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        foreach(r -> println(io, join((v isa String ? "\"" * v * "\"" : v for v in values(r)), ',')), rows)
    end
    @printf("Q_ref %.3e m²/s; biases in Q_ref against the fit with the true mechanics; window total against the truth\n", Q_ref)
    for r in rows
        @printf("  %-18s %-28s %-32s Δχ² %8.2f  window %+.3f  trend %+.3f  pre-window %+.3f  window total %+.3f\n",
            r.layout, r.case, r.history, r.Δχ2, r.window_bias, r.apparent_trend, r.prewindow_bias, r.window_total)
    end
    result = (; rows, Q_ref, window, prewindow, width, σ, σ_common, linear_dir, gravity_dir)
    serialize(joinpath(out_dir, "gravity_misspecification.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    gravity_misspecification_demo(;
        output_root = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"),
        out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"), "gravity_misspecification2D"))
end
