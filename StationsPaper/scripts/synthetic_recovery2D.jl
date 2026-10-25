# julia --project=. StationsPaper/scripts/synthetic_recovery2D.jl [impulse_dir] [out_dir] [truth_dir]
# Stage D of StationsPaper/PLAN.md: recover known synthetic Q(t) histories from
# noisy vertical station records, with the exact linear map built from the
# impulse response (impulse_response2D.jl, at least 1095 steps). No new forward
# runs. Truth is daily; the recovery uses `width`-day bins aligned to the window
# end plus unknown 30-day pre-window bins, a Gaussian prior with mean and σ
# Q_ref, and the empirical noise model (per-station plus common-mode σ).
# Histories follow OBSERVATION_MODEL.md: H0 constant, H1 20% step decline,
# H2 ramp, H3 one-week doubling, H4 drainage cycles, and H5, a pair in which
# extra supply during the year before the window is offset by the smooth window
# trend that best mimics its relaxation. Writes histories.csv, summary.csv, and
# synthetic_recovery.jls.
using LinearAlgebra
using Printf
using Random
using Serialization
include(joinpath(@__DIR__, "station_kernels.jl"))

"""Daily synthetic net rates over pre-window + window, in units of Q_ref (OBSERVATION_MODEL.md)."""
function synthetic_histories(prewindow, window; cycle = 60, drained_fraction = 0.6, recharge_time = 20.0)
    t = 1:window
    base = ones(window)
    drainage = map(t) do day
        since = mod(day - 1, cycle)
        # Recharge decays after each drainage; the drainage day removes most of what the cycle brought.
        recharge = 0.4 + 1.6 * exp(-since / recharge_time)
        since == cycle - 1 ? -drained_fraction * sum(0.4 + 1.6 * exp(-s / recharge_time) for s in 0:(cycle - 2)) : recharge
    end
    window_histories = [
        "H0 constant" => base,
        "H1 20% decline at day 240" => [day < 240 ? 1.0 : 0.8 for day in t],
        "H2 ramp 0.5 to 1.5" => collect(range(0.5, 1.5; length = window)),
        "H3 one-week doubling at day 180" => [180 <= day < 187 ? 2.0 : 1.0 for day in t],
        "H4 drainage every $cycle days" => drainage,
    ]
    return [name => [ones(prewindow); q] for (name, q) in window_histories]
end

function synthetic_recovery_demo(; impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        out_dir = joinpath(@__DIR__, "..", "output", "synthetic_recovery2D"),
        window = 365, prewindow = 720, width = 7, prewidth = 30, uplift = 0.24,
        σ = 3.0e-3, σ_common = 3.5e-3, station_x = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3),
        ambiguity_amplitude = 0.5, seed = 2026, truth_dir = impulse_dir)
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    stations = [findfirst(==(x), response.station_x) for x in station_x]
    H = response.H[stations, :]
    # Data come from `truth_dir`'s mechanics; a different one tests a mis-specified model.
    truth_response = deserialize(joinpath(truth_dir, "impulse_response.jls"))
    truth_response.station_x == response.station_x || throw(ArgumentError("truth and model stations differ"))
    H_truth = truth_response.H[stations, :]
    size(H_truth, 2) >= prewindow + window || throw(ArgumentError("truth impulse response shorter than pre-window + window"))
    size(H, 2) >= prewindow + window || throw(ArgumentError("impulse response shorter than pre-window + window"))
    centre = findfirst(==(0.0), response.station_x)
    Q_ref = uplift / sum(response.H[centre, 1:window])
    ns, nsteps = length(stations), prewindow + window

    jacobian = window_jacobian(H, 1:ns; window, width, prewindow, prewidth)
    (; nwindow, centres, bins) = jacobian
    K = reshape(jacobian.K, ns * window, length(bins))
    noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
    whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
    Kw = whiten(K)
    prior_mean, prior_σ = fill(Q_ref, length(bins)), Q_ref
    covariance = inv(Symmetric(Kw' * Kw + I / prior_σ^2))
    resolution = covariance * (Kw' * Kw)
    bin_average(q) = [sum(q[b]) / length(b) for b in bins]

    histories = synthetic_histories(prewindow, window)
    # H5: more supply in the year before the window, against the quadratic window
    # trend whose displacement best mimics its relaxation (whitened least squares).
    window_change(rates) = window_displacement(H, rates, prewindow)
    earlier = [prewindow - 360 < day <= prewindow ? ambiguity_amplitude : 0.0 for day in 1:nsteps]
    τ = [day > prewindow ? (day - prewindow - 0.5) / window : 0.0 for day in 1:nsteps]
    trend_basis = [Float64.(τ .> 0), τ, τ .^ 2]
    trend_data = reduce(hcat, [window_change(b) for b in trend_basis])
    coefficients = -(whiten(trend_data) \ whiten(window_change(earlier)))
    mimic = earlier .+ sum(coefficients .* trend_basis)
    push!(histories, "H5a ambiguity pair, base" => ones(nsteps))
    push!(histories, "H5b ambiguity pair, earlier supply" => 1 .+ mimic)

    rng = MersenneTwister(seed)
    noise = [noise_factor * randn(rng, ns) for _ in 1:window]
    results = map(histories) do (name, q)
        rates = Q_ref .* q
        clean = window_displacement(H_truth, rates, prewindow)
        observed = clean .+ reduce(vcat, noise)
        estimate = prior_mean + covariance * (Kw' * (whiten(observed) - Kw * prior_mean))
        truth = bin_average(rates)
        expected = prior_mean + resolution * (truth - prior_mean)
        residual = whiten(observed - K * estimate)
        posterior_σ = sqrt.(diag(covariance))
        z = (estimate[1:nwindow] - truth[1:nwindow]) ./ posterior_σ[1:nwindow]
        (; name, rates, truth, estimate, expected, posterior_σ, clean, observed,
            chi2_per_datum = sum(abs2, residual) / length(residual),
            coverage = count(abs.(z) .<= 1.96) / nwindow,
            rms_error = sqrt(sum(abs2, estimate[1:nwindow] - truth[1:nwindow]) / nwindow) / Q_ref)
    end
    data_difference = norm(whiten(results[end].clean - results[end - 1].clean))
    @printf("Q_ref %.3e m²/s; %d stations; %d-day bins; σ %.1f mm + common %.1f mm\n",
        Q_ref, ns, width, 1.0e3σ, 1.0e3σ_common)
    # χ² separation of the noise-free pair is data_difference²; below ~4 the pair is not distinguishable at 2σ.
    @printf("H5 pair: +%.1f Q_ref for 360 days before the window, mimicked by a window trend from %+.2f to %+.2f Q_ref; χ² separation %.2f\n",
        ambiguity_amplitude, mimic[prewindow + 1], mimic[end], data_difference^2)
    for r in results
        @printf("  %-34s χ²/datum %.2f  95%% coverage %.2f  rms error %.2f Q_ref\n",
            r.name, r.chi2_per_datum, r.coverage, r.rms_error)
    end

    open(joinpath(out_dir, "histories.csv"), "w") do io
        println(io, "history,bin,bin_centre_day,truth_m2_per_s,estimate_m2_per_s,posterior_sigma_m2_per_s,expected_m2_per_s")
        for r in results, j in 1:nwindow
            println(io, join(("\"" * r.name * "\"", j, centres[j], r.truth[j], r.estimate[j], r.posterior_σ[j], r.expected[j]), ','))
        end
    end
    open(joinpath(out_dir, "summary.csv"), "w") do io
        println(io, "history,chi2_per_datum,coverage95,rms_error_over_Qref")
        foreach(r -> println(io, join(("\"" * r.name * "\"", r.chi2_per_datum, r.coverage, r.rms_error), ',')), results)
    end
    result = (; results, centres, nwindow, window, prewindow, width, Q_ref, σ, σ_common,
        station_x = collect(station_x), data_difference, seed, impulse_dir = abspath(impulse_dir),
        truth_dir = abspath(truth_dir))
    serialize(joinpath(out_dir, "synthetic_recovery.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

# Near-field network alone, and with six far stations that mainly measure the
# common-mode error; then, with an optional truth_dir argument, data from other
# mechanics (e.g. a softer shell) inverted with the reference model. Skipped
# when the file is included for its functions.
if abspath(PROGRAM_FILE) == @__FILE__
    let impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
            out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "synthetic_recovery2D")
        near = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3)
        far = (8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)
        if length(ARGS) > 2
            synthetic_recovery_demo(; impulse_dir, truth_dir = ARGS[3], station_x = (near..., far...),
                out_dir = joinpath(out_dir, "near6_far6_truth_" * basename(normpath(ARGS[3]))))
        else
            for (label, station_x) in (("near6", near), ("near6_far6", (near..., far...)))
                synthetic_recovery_demo(; impulse_dir, out_dir = joinpath(out_dir, label), station_x)
            end
        end
    end
end
