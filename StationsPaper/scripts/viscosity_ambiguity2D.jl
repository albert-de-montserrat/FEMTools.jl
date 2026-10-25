# julia --project=. StationsPaper/scripts/viscosity_ambiguity2D.jl [output_root] [out_dir]
# Stage D ambiguity (RESULTS.md result 10) as a sweep: data generated with shell
# viscosity η_true are inverted for Q(t) with the reference η_ref = 3e17 Pa s.
# Everything is noise-free and linear, so the numbers are expected values:
# Δχ², the minimum MAP objective of the wrong-mechanics fit (the correct fit of
# a constant history has zero), measures whether the error is detectable;
# rate biases measure what it does to the recovered history. Inversion setup as
# in synthetic_recovery2D.jl: 7-day window bins, 720 days of unknown 30-day
# pre-window bins, prior mean and σ Q_ref, empirical noise (3 mm + 3.5 mm
# common). Needs 1095-step impulse responses of impulse_response2D.jl (cases
# "eta_<value>" and the reference) under output_root. Writes
# viscosity_ambiguity.csv and viscosity_ambiguity.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_kernels.jl"))

function viscosity_ambiguity_demo(; output_root = joinpath(@__DIR__, "..", "output"),
        out_dir = joinpath(output_root, "viscosity_ambiguity2D"), η_ref = 3.0e17,
        truths = (1.0e17 => "impulse_response2D_soft_shell_3yr", 1.5e17 => "impulse_response2D_eta_1.5e17",
            2.0e17 => "impulse_response2D_eta_2.0e17", 3.0e17 => "impulse_response2D_3yr",
            4.5e17 => "impulse_response2D_eta_4.5e17", 9.0e17 => "impulse_response2D_eta_9.0e17"),
        model_dir = "impulse_response2D_3yr", window = 365, prewindow = 720, width = 7, prewidth = 30,
        uplift = 0.24, σ = 3.0e-3, σ_common = 3.5e-3, quarter = 13)
    mkpath(out_dir)
    model = deserialize(joinpath(output_root, model_dir, "impulse_response.jls"))
    at(xs) = [findfirst(==(x), model.station_x) for x in xs]
    near = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3)
    far = (8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)
    layouts = ["x=0 km" => at([0.0]), "6 near (0-4.5 km)" => at(near),
        "6 near + 6 far" => at([near..., far...]), "all 29" => collect(eachindex(model.station_x))]
    centre = findfirst(==(0.0), model.station_x)
    Q_ref = uplift / sum(model.H[centre, 1:window])
    rates = fill(Q_ref, prewindow + window)
    available = [(η, dir) for (η, dir) in truths if isfile(joinpath(output_root, dir, "impulse_response.jls"))]
    length(available) < length(truths) && @warn "missing truth responses" setdiff(last.(truths), last.(available))

    rows = NamedTuple[]
    for (layout, stations) in layouts
        ns = length(stations)
        jacobian = window_jacobian(model.H, stations; window, width, prewindow, prewidth)
        (; nwindow, centres, bins) = jacobian
        noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
        whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
        Kw = whiten(reshape(jacobian.K, ns * window, length(bins)))
        prior_mean = fill(Q_ref, length(bins))
        covariance = inv(Symmetric(Kw' * Kw + I / Q_ref^2))
        for (η, dir) in available
            truth = deserialize(joinpath(output_root, dir, "impulse_response.jls"))
            truth.station_x == model.station_x || throw(ArgumentError("$dir stations differ from the model"))
            data = whiten(window_displacement(truth.H[stations, :], rates, prewindow))
            estimate = prior_mean + covariance * (Kw' * (data - Kw * prior_mean))
            Δχ2 = sum(abs2, data - Kw * estimate) + sum(abs2, (estimate - prior_mean) ./ Q_ref)
            bias = (estimate .- Q_ref) ./ Q_ref
            window_bias = bias[1:nwindow]
            push!(rows, (; layout, η_true = η, η_ratio = η / η_ref, Δχ2,
                window_bias = mean(window_bias), first_quarter = mean(window_bias[1:quarter]),
                last_quarter = mean(window_bias[(end - quarter + 1):end]),
                apparent_trend = mean(window_bias[(end - quarter + 1):end]) - mean(window_bias[1:quarter]),
                prewindow_bias = mean(bias[(nwindow + 1):end]), centres, bias))
        end
    end
    open(joinpath(out_dir, "viscosity_ambiguity.csv"), "w") do io
        println(io, "layout,eta_true_Pa_s,eta_ratio,delta_chi2,window_bias,first_quarter_bias,last_quarter_bias,apparent_trend,prewindow_bias")
        for r in rows
            println(io, join(("\"" * r.layout * "\"", r.η_true, r.η_ratio, r.Δχ2, r.window_bias, r.first_quarter,
                r.last_quarter, r.apparent_trend, r.prewindow_bias), ','))
        end
    end
    @printf("Q_ref %.3e m²/s, model η %.1e Pa s; biases in Q_ref, trend = last minus first %d weeks\n", Q_ref, η_ref, quarter)
    for r in rows
        @printf("  %-18s η_true/η_ref %5.2f  Δχ² %8.2f  trend %+.3f  window %+.3f  pre-window %+.3f\n",
            r.layout, r.η_ratio, r.Δχ2, r.apparent_trend, r.window_bias, r.prewindow_bias)
    end
    result = (; rows, Q_ref, η_ref, window, prewindow, width, σ, σ_common, layouts = first.(layouts))
    serialize(joinpath(out_dir, "viscosity_ambiguity.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

viscosity_ambiguity_demo(;
    output_root = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"), "viscosity_ambiguity2D"))
