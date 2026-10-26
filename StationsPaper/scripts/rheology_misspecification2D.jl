# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/rheology_misspecification2D.jl [output_root] [out_dir]
# PLAN.md task 7: does a single Maxwell time suffice? Data from the thermal
# aureole (viscosity rising log-linearly from 1e17 Pa s at the wall to 1e20 Pa s
# at 3 km; impulse_response2D.jl case "aureole") are inverted with every
# single-viscosity shell of the viscosity sweep (shell to 2 km), as
# viscosity_ambiguity2D.jl does for a wrong viscosity. Noise-free, so Δχ² is
# the expected misfit increase (the fit with the true mechanics has none for a
# history the prior mean matches) and biases, in Q_ref, are against the fit
# with the true mechanics. Inversion setup as in synthetic_recovery2D.jl. All
# responses have gravity and a Lagrangian mesh. Writes
# rheology_misspecification.csv and rheology_misspecification.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "synthetic_recovery2D.jl"))

function rheology_misspecification_demo(; output_root = joinpath(@__DIR__, "..", "output"),
        out_dir = joinpath(output_root, "rheology_misspecification2D"),
        truth_dir = "impulse_response2D_aureole_3yr",
        models = ("η_shell 1e17" => "impulse_response2D_soft_shell_3yr", "η_shell 1.5e17" => "impulse_response2D_eta_1.5e17",
            "η_shell 2e17" => "impulse_response2D_eta_2.0e17", "η_shell 3e17" => "impulse_response2D_3yr",
            "η_shell 4.5e17" => "impulse_response2D_eta_4.5e17", "η_shell 9e17" => "impulse_response2D_eta_9.0e17"),
        window = 365, prewindow = 720, width = 7, prewidth = 30, uplift = 0.24,
        σ = 3.0e-3, σ_common = 3.5e-3, quarter = 13)
    mkpath(out_dir)
    truth = deserialize(joinpath(output_root, truth_dir, "impulse_response.jls"))
    reference = deserialize(joinpath(output_root, "impulse_response2D_3yr", "impulse_response.jls"))
    at(xs) = [findfirst(==(x), truth.station_x) for x in xs]
    near = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3)
    far = (8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)
    layouts = ["6 near (0-4.5 km)" => at(near), "6 near + 6 far" => at([near..., far...])]
    Q_ref = uplift / sum(reference.H[findfirst(==(0.0), reference.station_x), 1:window])
    histories = filter(h -> startswith(first(h), r"H[013]"), synthetic_histories(prewindow, window))
    rows = NamedTuple[]
    for (layout, stations) in layouts
        ns = length(stations)
        noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
        whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
        kernel(H) = (J = window_jacobian(H, stations; window, width, prewindow, prewidth);
            (; J.nwindow, J.bins, Kw = whiten(reshape(J.K, ns * window, length(J.bins)))))
        true_kernel = kernel(truth.H)
        (; nwindow, bins) = true_kernel
        prior_mean = fill(Q_ref, length(bins))
        fit(Kw, data) = (m = prior_mean + inv(Symmetric(Kw' * Kw + I / Q_ref^2)) * (Kw' * (data - Kw * prior_mean));
            (; m, objective = sum(abs2, data - Kw * m) + sum(abs2, (m - prior_mean) ./ Q_ref)))
        for (model_name, dir) in models
            model_kernel = kernel(deserialize(joinpath(output_root, dir, "impulse_response.jls")).H).Kw
            for (history, q) in histories
                rates = Q_ref .* q
                data = whiten(window_displacement(truth.H[stations, :], rates, prewindow))
                correct, wrong = fit(true_kernel.Kw, data), fit(model_kernel, data)
                bias = (wrong.m .- correct.m) ./ Q_ref
                window_bias = bias[1:nwindow]
                push!(rows, (; layout, model = model_name, history, Δχ2 = wrong.objective - correct.objective,
                    window_bias = mean(window_bias), last_quarter = mean(window_bias[(end - quarter + 1):end]),
                    apparent_trend = mean(window_bias[(end - quarter + 1):end]) - mean(window_bias[1:quarter]),
                    prewindow_bias = mean(bias[(nwindow + 1):end])))
            end
        end
    end
    @printf("aureole truth inverted with single-viscosity shells; Q_ref %.3e m²/s; biases in Q_ref\n", Q_ref)
    for r in rows
        @printf("  %-18s %-15s %-32s Δχ² %8.2f  window %+.3f  last 13 wk %+.3f  trend %+.3f  pre-window %+.3f\n",
            r.layout, r.model, r.history, r.Δχ2, r.window_bias, r.last_quarter, r.apparent_trend, r.prewindow_bias)
    end
    open(joinpath(out_dir, "rheology_misspecification.csv"), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        foreach(r -> println(io, join((v isa String ? "\"" * v * "\"" : v for v in values(r)), ',')), rows)
    end
    result = (; rows, Q_ref, window, prewindow, width, σ, σ_common, truth_dir)
    serialize(joinpath(out_dir, "rheology_misspecification.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

rheology_misspecification_demo(;
    output_root = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"), "rheology_misspecification2D"))
