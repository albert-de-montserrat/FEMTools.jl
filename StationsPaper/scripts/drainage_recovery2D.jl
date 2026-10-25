# julia --project=. StationsPaper/scripts/drainage_recovery2D.jl [impulse_dir] [out_dir]
# Follow-up to H4 in synthetic_recovery2D.jl (RESULTS.md result 9): drainage
# cycles every 60 days, recovered with
#   1. the H0–H4 setup (prior mean and σ Q_ref), where drainage weeks fail;
#   2. a wide prior (σ = 5 Q_ref) that allows withdrawal;
#   3. known drainage days as separate one-day unknowns (prior mean 0, σ
#      `event_σ`), taken out of their 7-day bins; drainage onsets are sharp in
#      daily GNSS records, so their days are taken as observed;
#   4. the same with every drainage day one day late, to test that assumption;
#   5. drainage days found from the data, by moving each day of case 4 to the
#      lowest MAP objective within ±`search` days.
# Same mechanics, noise model, noise realization, and window as
# synthetic_recovery2D.jl. Recharge error is measured on the 7-day bins (truth:
# bin-average rate over the bin's own days); drainage error on the event volumes.
# Writes summary.csv, histories.csv, and drainage_recovery.jls.
include(joinpath(@__DIR__, "synthetic_recovery2D.jl"))

function drainage_case(H, rates, noise_factor, observed; window, prewindow, width, prewidth, events,
        prior_σ, event_σ, Q_ref, true_volumes)
    ns = size(H, 1)
    jacobian = window_jacobian(H, 1:ns; window, width, prewindow, prewidth, events)
    (; nwindow, bins) = jacobian
    whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
    K = reshape(jacobian.K, ns * window, length(bins))
    Kw = whiten(K)
    nevents = length(events)
    prior_mean = [fill(Q_ref, length(bins) - nevents); zeros(nevents)]
    prior_sd = [fill(prior_σ, length(bins) - nevents); fill(event_σ, nevents)]
    covariance = inv(Symmetric(Kw' * Kw + Diagonal(1 ./ prior_sd .^ 2)))
    estimate = prior_mean + covariance * (Kw' * (whiten(observed) - Kw * prior_mean))
    posterior_σ = sqrt.(diag(covariance))
    truth = [sum(rates[b]) / length(b) for b in bins]
    residual = whiten(observed - K * estimate)
    z = (estimate[1:nwindow] - truth[1:nwindow]) ./ posterior_σ[1:nwindow]
    # One-day bins carry the drainage volume (rate × 1 day); compare with the true events in order.
    event_volume = estimate[(end - nevents + 1):end]
    event_volume_σ = posterior_σ[(end - nevents + 1):end]
    objective = sum(abs2, residual) + sum(abs2, (estimate - prior_mean) ./ prior_sd)
    return (; estimate, posterior_σ, truth, nwindow, bins, events, event_volume, event_volume_σ, objective,
        chi2_per_datum = sum(abs2, residual) / length(residual),
        coverage = count(abs.(z) .<= 1.96) / nwindow,
        rms_error = sqrt(sum(abs2, estimate[1:nwindow] - truth[1:nwindow]) / nwindow) / Q_ref,
        volume_error = nevents == length(true_volumes) ? sqrt(sum(abs2, event_volume - true_volumes) / nevents) / sqrt(sum(abs2, true_volumes) / nevents) : NaN,
        volume_coverage = nevents == length(true_volumes) ? count(abs.(event_volume - true_volumes) .<= 1.96 .* event_volume_σ) / nevents : NaN)
end

function drainage_recovery_demo(; impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        out_dir = joinpath(@__DIR__, "..", "output", "drainage_recovery2D"),
        window = 365, prewindow = 720, width = 7, prewidth = 30, uplift = 0.24, cycle = 60,
        σ = 3.0e-3, σ_common = 3.5e-3, wide_σ = 5.0, event_σ = 100.0, search = 3, seed = 2026,
        layouts = ("6 near" => (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3),
            "6 near + 6 far" => (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3, 8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3)))
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    centre = findfirst(==(0.0), response.station_x)
    Q_ref = uplift / sum(response.H[centre, 1:window])
    history = last(only(filter(h -> startswith(first(h), "H4"), synthetic_histories(prewindow, window; cycle))))
    rates = Q_ref .* history
    true_events = collect(cycle:cycle:window)
    true_volumes = rates[prewindow .+ true_events]
    cases = ["σ = Q_ref" => (; events = Int[], prior_σ = Q_ref),
        "σ = $(wide_σ) Q_ref" => (; events = Int[], prior_σ = wide_σ * Q_ref),
        "known drainage days" => (; events = true_events, prior_σ = Q_ref),
        "drainage days 1 day late" => (; events = true_events .+ 1, prior_σ = Q_ref)]

    rows = NamedTuple[]
    @printf("Q_ref %.3e m²/s; drainage every %d days removes %.1f Q_ref·day; %d-day bins; σ %.1f mm + common %.1f mm\n",
        Q_ref, cycle, -true_volumes[1] / Q_ref, width, 1.0e3σ, 1.0e3σ_common)
    for (layout, station_x) in layouts
        stations = [findfirst(==(x), response.station_x) for x in station_x]
        H = response.H[stations, :]
        ns = length(stations)
        noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
        rng = MersenneTwister(seed)
        observed = window_displacement(H, rates, prewindow) .+ reduce(vcat, [noise_factor * randn(rng, ns) for _ in 1:window])
        solve(options) = drainage_case(H, rates, noise_factor, observed; window, prewindow, width, prewidth,
            event_σ = event_σ * Q_ref, Q_ref, true_volumes, options...)
        # Drainage days picked from the data: start one day late, then move each day
        # within ±search days to the lowest MAP objective until no day moves.
        events = true_events .+ 1
        moved = true
        while moved
            moved = false
            for i in eachindex(events)
                trials = [setindex!(copy(events), events[i] + shift, i) for shift in -search:search]
                best = trials[argmin([solve((; events = t, prior_σ = Q_ref)).objective for t in trials])]
                moved |= best != events
                events = best
            end
        end
        println("  ", layout, ": scanned drainage days minus true days ", events .- true_events)
        scanned = "drainage days from objective scan" => (; events, prior_σ = Q_ref)
        for (case, options) in [cases; scanned]
            r = solve(options)
            push!(rows, (; layout, case, r...))
            @printf("  %-15s %-34s χ²/datum %.2f  coverage %.2f  recharge rms %.2f Q_ref  drainage volume error %s\n",
                layout, case, r.chi2_per_datum, r.coverage, r.rms_error,
                isnan(r.volume_error) ? "-" : @sprintf("%.3f (coverage %.2f)", r.volume_error, r.volume_coverage))
        end
    end

    open(joinpath(out_dir, "summary.csv"), "w") do io
        println(io, "layout,case,chi2_per_datum,coverage95,rms_error_over_Qref,volume_rms_relative_error,volume_coverage95")
        for r in rows
            println(io, join(("\"" * r.layout * "\"", "\"" * r.case * "\"", r.chi2_per_datum, r.coverage, r.rms_error,
                r.volume_error, r.volume_coverage), ','))
        end
    end
    open(joinpath(out_dir, "histories.csv"), "w") do io
        println(io, "layout,case,bin,first_day,last_day,truth_m2_per_s,estimate_m2_per_s,posterior_sigma_m2_per_s")
        for r in rows, j in eachindex(r.bins)
            # Window days count from the window start; pre-window days are negative.
            days = extrema(r.bins[j]) .- prewindow
            println(io, join(("\"" * r.layout * "\"", "\"" * r.case * "\"", j, days..., r.truth[j], r.estimate[j], r.posterior_σ[j]), ','))
        end
    end
    result = (; rows, Q_ref, rates, true_events, true_volumes, window, prewindow, width, σ, σ_common, seed, impulse_dir = abspath(impulse_dir))
    serialize(joinpath(out_dir, "drainage_recovery.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

drainage_recovery_demo(;
    impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "drainage_recovery2D"))
