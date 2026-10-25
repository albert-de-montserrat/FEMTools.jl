# julia --project=. StationsPaper/scripts/mogi_comparison2D.jl [output_root] [out_dir]
# Stage E of StationsPaper/PLAN.md: the conventional elastic workflow against
# the chamber-model inversion, on the same synthetic records. Truth comes from
# the chamber model with an elastic shell (`impulse_response2D_eta_1.0e23`,
# control), with the reference Maxwell shell (`impulse_response2D_3yr`,
# η_shell = 3e17 Pa s), and with shell viscosities 1e17, 2e17, and 9e17 Pa s
# from the viscosity sweep, and with the source centre at 3 km (`shallow`), all
# with gravity and a Lagrangian mesh; and the reference chamber without them
# (`linear/impulse_response2D_3yr`).
# Both inversions use the 6 near + 6 far stations, the empirical noise model
# and noise realization of synthetic_recovery2D.jl, 7-day window bins, and the
# prior mean and σ Q_ref.
#   Chamber model: the true mechanics, with 720 days of unknown 30-day
#   pre-window bins (as in Stage D).
#   Mogi: a plane-strain line source in an elastic half-space (2-D Mogi) at
#   x = 0, depth from a grid search, no memory and no pre-window. Its unknown is
#   the rate of cavity-area change in each bin, i.e. the fitted source area
#   history differenced over the same bins; it becomes a supply estimate through
#   the compressibility factor 1 + G/K_magma of a cylindrical cavity (exact
#   material values, the most favourable case). Without that factor the Mogi
#   rate is the source-area rate, not the influx.
# History "H6 supply stops at day 180" is added to H0, H1, H3: after the stop,
# viscous relaxation keeps the stations rising. Writes summary.csv,
# histories.csv, and mogi_comparison.jls.
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics
include(joinpath(@__DIR__, "synthetic_recovery2D.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "stokes", "volcano", "elastic_dike2D.jl"))

"""MAP estimate, posterior covariance and σ for whitened Jacobian `Kw` and data `dw`, prior mean `m` and σ `s`."""
function gaussian_map(Kw, dw, m, s)
    covariance = inv(Symmetric(Kw' * Kw + I / s^2))
    estimate = m + covariance * (Kw' * (dw - Kw * m))
    objective = sum(abs2, dw - Kw * estimate) + sum(abs2, (estimate - m) ./ s)
    return (; estimate, covariance, σ = sqrt.(diag(covariance)), objective)
end

function mogi_comparison_demo(; output_root = joinpath(@__DIR__, "..", "output"),
        out_dir = joinpath(output_root, "mogi_comparison2D"),
        truths = ("elastic shell" => "impulse_response2D_eta_1.0e23", "Maxwell shell" => "impulse_response2D_3yr",
            "η_shell 1e17" => "impulse_response2D_soft_shell_3yr", "η_shell 2e17" => "impulse_response2D_eta_2.0e17",
            "η_shell 9e17" => "impulse_response2D_eta_9.0e17", "source at 3 km" => "impulse_response2D_shallow_3yr",
            "no gravity, fixed mesh" => joinpath("linear", "impulse_response2D_3yr")),
        station_x = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3, 8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3),
        window = 365, prewindow = 720, width = 7, prewidth = 30, uplift = 0.24, stop_day = 180,
        σ = 3.0e-3, σ_common = 3.5e-3, depths = 2.0e3:100.0:8.0e3, seed = 2026, quarter = 13)
    mkpath(out_dir)
    ns = length(station_x)
    noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
    whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
    rng = MersenneTwister(seed)
    noise = reduce(vcat, [noise_factor * randn(rng, ns) for _ in 1:window])
    reference = deserialize(joinpath(output_root, "impulse_response2D_3yr", "impulse_response.jls"))
    centre = findfirst(==(0.0), reference.station_x)
    Q_ref = uplift / sum(reference.H[centre, 1:window])
    histories = filter(h -> startswith(first(h), r"H[013]"), synthetic_histories(prewindow, window))
    push!(histories, "H6 supply stops at day $stop_day" => [day <= prewindow + stop_day ? 1.0 : 0.0 for day in 1:(prewindow + window)])

    rows = NamedTuple[]
    for (truth_name, dir) in truths
        response = deserialize(joinpath(output_root, dir, "impulse_response.jls"))
        stations = [findfirst(==(x), response.station_x) for x in station_x]
        H = response.H[stations, :]
        (; G_host, K_host, K_magma) = response.material
        ν = (3K_host - 2G_host) / (2 * (3K_host + G_host))
        compressibility = 1 + G_host / K_magma
        chamber = window_jacobian(H, 1:ns; window, width, prewindow, prewidth)
        (; nwindow, centres, bins) = chamber
        Kw_chamber = whiten(reshape(chamber.K, ns * window, length(bins)))
        # Mogi response to a unit supply rate during one step: instantaneous and permanent.
        mogi_jacobian(depth) = whiten(reshape(window_jacobian(
            repeat([last(line_source_surface(x; area_change = response.Δt, depth, ν)) / compressibility for x in station_x], 1, window),
            1:ns; window, width).K, ns * window, nwindow))
        mogi_kernels = [mogi_jacobian(d) for d in depths]
        for (name, q) in histories
            rates = Q_ref .* q
            truth = [sum(rates[b]) / length(b) for b in bins[1:nwindow]]
            data = whiten(window_displacement(H, rates, prewindow) .+ noise)
            model = gaussian_map(Kw_chamber, data, fill(Q_ref, length(bins)), Q_ref)
            fits = [gaussian_map(Kw, data, fill(Q_ref, nwindow), Q_ref) for Kw in mogi_kernels]
            best = argmin(getfield.(fits, :objective))
            mogi = fits[best]
            summary(estimate) = (; rms = sqrt(mean(abs2, estimate[1:nwindow] - truth)) / Q_ref,
                last_quarter = mean(estimate[(nwindow - quarter + 1):nwindow] - truth[(end - quarter + 1):end]) / Q_ref,
                window_total = (sum(estimate[1:nwindow] .* length.(bins[1:nwindow])) - sum(rates[(prewindow + 1):end])) / sum(rates[(prewindow + 1):end]))
            chamber_summary, mogi_summary = summary(model.estimate), summary(mogi.estimate)
            # Cumulative supply [m²] at each window-bin end; its σ needs the full posterior
            # covariance, because neighbouring bin rates are anti-correlated.
            durations = length.(bins[1:nwindow]) .* response.Δt
            accumulate_bins = [k <= j ? durations[k] : 0.0 for j in 1:nwindow, k in 1:nwindow]
            cumulative(estimate, covariance) = (accumulate_bins * estimate[1:nwindow],
                sqrt.(diag(accumulate_bins * covariance[1:nwindow, 1:nwindow] * accumulate_bins')))
            chamber_cumulative, chamber_cumulative_σ = cumulative(model.estimate, model.covariance)
            mogi_cumulative, mogi_cumulative_σ = cumulative(mogi.estimate, mogi.covariance)
            push!(rows, (; truth_name, history = name, mogi_depth = depths[best],
                chi2_chamber = model.objective / (ns * window), chi2_mogi = mogi.objective / (ns * window),
                chamber_rms = chamber_summary.rms, mogi_rms = mogi_summary.rms,
                chamber_last_quarter = chamber_summary.last_quarter, mogi_last_quarter = mogi_summary.last_quarter,
                chamber_window_total = chamber_summary.window_total, mogi_window_total = mogi_summary.window_total,
                compressibility, centres, truth, chamber = model.estimate[1:nwindow], chamber_σ = model.σ[1:nwindow],
                mogi = mogi.estimate, mogi_σ = mogi.σ, bin_ends = [last(b) - prewindow for b in bins[1:nwindow]],
                truth_cumulative = accumulate_bins * truth, chamber_cumulative, chamber_cumulative_σ,
                mogi_cumulative, mogi_cumulative_σ))
        end
    end

    @printf("Q_ref %.3e m²/s; %d stations; %d-day bins; Mogi supply = (1 + G/K_magma) × source-area rate\n", Q_ref, ns, width)
    @printf("  %-14s %-30s %9s  %-21s  %-21s  %-21s  %s\n", "truth", "history", "Mogi d", "objective/datum c | M",
        "rms/Q_ref c | M", "last 13 wk c | M", "window total c | M")
    for r in rows
        @printf("  %-14s %-30s %6.1f km  %8.2f | %8.2f  %8.2f | %8.2f  %+8.2f | %+8.2f  %+7.2f | %+7.2f\n",
            r.truth_name, r.history, r.mogi_depth / 1.0e3, r.chi2_chamber, r.chi2_mogi, r.chamber_rms, r.mogi_rms,
            r.chamber_last_quarter, r.mogi_last_quarter, r.chamber_window_total, r.mogi_window_total)
    end
    open(joinpath(out_dir, "summary.csv"), "w") do io
        println(io, "truth,history,mogi_depth_m,objective_per_datum_chamber,objective_per_datum_mogi,rms_chamber,rms_mogi,last_quarter_bias_chamber,last_quarter_bias_mogi,window_total_error_chamber,window_total_error_mogi")
        for r in rows
            println(io, join(("\"" * r.truth_name * "\"", "\"" * r.history * "\"", r.mogi_depth, r.chi2_chamber, r.chi2_mogi,
                r.chamber_rms, r.mogi_rms, r.chamber_last_quarter, r.mogi_last_quarter, r.chamber_window_total, r.mogi_window_total), ','))
        end
    end
    open(joinpath(out_dir, "histories.csv"), "w") do io
        println(io, "truth,history,bin,bin_centre_day,truth_m2_per_s,chamber_m2_per_s,chamber_sigma,mogi_m2_per_s,mogi_sigma,",
            "bin_end_day,truth_cumulative_m2,chamber_cumulative_m2,chamber_cumulative_sigma,mogi_cumulative_m2,mogi_cumulative_sigma")
        for r in rows, j in eachindex(r.truth)
            println(io, join(("\"" * r.truth_name * "\"", "\"" * r.history * "\"", j, r.centres[j], r.truth[j],
                r.chamber[j], r.chamber_σ[j], r.mogi[j], r.mogi_σ[j], r.bin_ends[j], r.truth_cumulative[j],
                r.chamber_cumulative[j], r.chamber_cumulative_σ[j], r.mogi_cumulative[j], r.mogi_cumulative_σ[j]), ','))
        end
    end
    result = (; rows, Q_ref, window, prewindow, width, σ, σ_common, station_x = collect(station_x), seed)
    serialize(joinpath(out_dir, "mogi_comparison.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

mogi_comparison_demo(;
    output_root = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"), "mogi_comparison2D"))
