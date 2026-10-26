# julia --project=. StationsPaper/scripts/station_information2D.jl [impulse_dir] [out_dir]
# Stage D start of StationsPaper/PLAN.md: what vertical station records resolve
# about Q(t), from the impulse response alone (no new forward runs). Reads
# impulse_response.jls written by impulse_response2D.jl and, following
# OBSERVATION_MODEL.md, builds the exact linear Jacobian for each source-bin
# width, then the Gaussian posterior, averaging kernels, and degrees of freedom
# for station subsets and noise models. Writes resolution.csv, separability.csv,
# kernels.csv, and station_information.jls.
#
# Assumptions, all declared in OBSERVATION_MODEL.md: known mechanics, 365-day
# window of daily vertical epochs, independent N(0, Q_ref²) prior on every bin
# rate, Q_ref set so that constant supply lifts x = 0 by 0.24 m in the window.
# With a long enough impulse response, the source history before the window is
# unknown too (30-day bins), observed only through its relaxation. Noise models
# are (name, per-station σ, common-mode σ) per daily vertical solution;
# "empirical" is the high-pass scatter of the IMO Svartsengi series (RESULTS.md).
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_kernels.jl"))

"""
    posterior_summary(K, nwindow; σ, σ_common, σ_prior) -> NamedTuple

Gaussian posterior for independent N(0, σ_prior²) bin rates. Each epoch's noise
has `σ` per station plus a common-mode part `σ_common` shared by all stations.
Statistics refer to the window bins; pre-window bins are marginalized.
"""
function posterior_summary(K, nwindow; σ = 7.0e-3, σ_common = 0.0, σ_prior)
    ns, nt, nb = size(K)
    noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
    whitened = reshape(noise_factor \ reshape(K, ns, nt * nb), ns * nt, nb)
    information = Symmetric(whitened' * whitened)
    covariance = inv(Symmetric(information + I / σ_prior^2))
    resolution = (covariance * information)[1:nwindow, 1:nwindow]
    rdiag = diag(resolution)
    centre = cld(nwindow, 2)
    row = resolution[centre, :]
    # Contiguous bins at or above half the peak; the row sum is the fraction of a
    # change much longer than the kernel that the centre estimate recovers.
    half = maximum(row) / 2
    peak = argmax(row)
    first_bin = something(findprev(<(half), row, peak), 0) + 1
    last_bin = something(findnext(<(half), row, peak), nwindow + 1) - 1
    return (; dof = tr(resolution), median_rdiag = median(rdiag), last_rdiag = rdiag[end],
        median_σ_ratio = median(sqrt.(diag(covariance)[1:nwindow])) / σ_prior,
        last_σ_ratio = sqrt(covariance[nwindow, nwindow]) / σ_prior,
        centre_fwhm_bins = last_bin - first_bin + 1, centre_sum = sum(row), centre_row = row, rdiag)
end

function station_information_demo(; impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        out_dir = joinpath(@__DIR__, "..", "output", "station_information2D"),
        window = 365, prewindow = 730, widths = (1, 3, 7, 14, 30),
        noise_models = (("white", 7.0e-3, 0.0), ("white + common", 7.0e-3, 3.0e-3), ("empirical", 3.0e-3, 3.5e-3)),
        uplift = 0.24)
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    H, station_x = response.H, response.station_x
    prewindow = min(prewindow, 30 * fld(size(H, 2) - window, 30))
    centre = findfirst(==(0.0), station_x)
    σ_prior = uplift / sum(H[centre, 1:window])
    at(xs) = [findfirst(==(x), station_x) for x in xs]
    subsets = [[@sprintf("x=%g km", x / 1.0e3) => at([x]) for x in (0.0, 2.0e3, 4.0e3, 8.0e3, 12.0e3, 20.0e3)];
        "6 near (0-4.5 km)" => at([0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3]);
        "6 far (8-18 km)" => at([8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3]);
        # Far stations see little source signal but the same common-mode error.
        "x=0 + 20 km" => at([0.0, 20.0e3]);
        "6 near + 6 far" => at([0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3, 8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3]);
        "all 29" => collect(eachindex(station_x))]
    @printf("Q_ref = %.3e m²/s (%.2f m uplift at x = 0 in %d days); pre-window %d days\n",
        σ_prior, uplift, window, prewindow)

    # Space-time separability of the station responses over the window.
    singular_values = svdvals(H[:, 1:window])
    factors = svd(H[:, 1:window])
    spatial_mode = factors.U[:, 1] .* sign(factors.U[centre, 1])
    open(joinpath(out_dir, "separability.csv"), "w") do io
        println(io, "mode,singular_value_ratio")
        foreach(k -> println(io, k, ',', singular_values[k] / singular_values[1]), 1:min(10, length(singular_values)))
    end
    @printf("singular values of H / first: %s\n", join((@sprintf("%.2e", s / singular_values[1]) for s in singular_values[2:4]), ", "))

    rows = NamedTuple[]
    kernels = NamedTuple[]
    for (memory, pre) in (("known start", 0), ("unknown pre-window", prewindow)), width in widths,
            (noise, σ, σc) in noise_models, (name, stations) in subsets
        pre == 0 && memory == "unknown pre-window" && continue
        (; K, nwindow, centres) = window_jacobian(H, stations; window, width, prewindow = pre)
        summary = posterior_summary(K, nwindow; σ, σ_common = σc, σ_prior)
        gain = sum(abs2, H[stations, 1]) / H[centre, 1]^2
        # If station responses differ only in scale, the subset equals one x = 0 station with σ/√gain.
        equivalent = posterior_summary(window_jacobian(H, [centre]; window, width, prewindow = pre).K, nwindow;
            σ = σ / sqrt(gain), σ_prior)
        push!(rows, (; memory, width, noise, subset = name, nstations = length(stations), gain, nwindow,
            summary.dof, dof_equivalent = equivalent.dof, summary.median_rdiag, summary.last_rdiag, summary.median_σ_ratio,
            summary.last_σ_ratio, centre_fwhm_days = width * summary.centre_fwhm_bins, summary.centre_sum))
        name in ("x=0 km", "6 near (0-4.5 km)") && noise == "white" && width in (1, 7, 30) &&
            push!(kernels, (; memory, width, subset = name, centres, row = summary.centre_row, summary.rdiag))
    end
    open(joinpath(out_dir, "resolution.csv"), "w") do io
        println(io, "memory,bin_days,noise,subset,nstations,elastic_gain_vs_x0,nbins,dof,dof_equivalent_single_station,median_rdiag,last_rdiag,median_sigma_over_prior,last_sigma_over_prior,centre_kernel_fwhm_days,centre_kernel_sum")
        for r in rows
            println(io, join((r.memory, r.width, r.noise, "\"" * r.subset * "\"", r.nstations, r.gain, r.nwindow,
                r.dof, r.dof_equivalent, r.median_rdiag, r.last_rdiag, r.median_σ_ratio, r.last_σ_ratio, r.centre_fwhm_days, r.centre_sum), ','))
        end
    end
    open(joinpath(out_dir, "kernels.csv"), "w") do io
        println(io, "memory,bin_days,subset,bin,bin_centre_day,resolution")
        for k in kernels, (j, value) in enumerate(k.row)
            println(io, join((k.memory, k.width, "\"" * k.subset * "\"", j, k.centres[j], value), ','))
        end
    end
    for r in rows
        r.width in (1, 7, 30) || continue
        @printf("%-18s %-14s %2d d  %-18s gain %6.2f  dof %5.1f/%3d (one station %5.1f)  median R %.2f  last R %.2f  σ/prior %.2f  FWHM %3d d  sum %.2f\n",
            r.memory, r.noise, r.width, r.subset, r.gain, r.dof, r.nwindow, r.dof_equivalent, r.median_rdiag, r.last_rdiag,
            r.median_σ_ratio, r.centre_fwhm_days, r.centre_sum)
    end
    result = (; rows, kernels, singular_values, spatial_mode, station_x, σ_prior, noise_models, window,
        prewindow, widths, uplift, impulse_dir = abspath(impulse_dir))
    serialize(joinpath(out_dir, "station_information.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

# Run only as a script, so other drivers can include posterior_summary.
if abspath(PROGRAM_FILE) == @__FILE__
    station_information_demo(;
        impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "station_information2D"))
end
