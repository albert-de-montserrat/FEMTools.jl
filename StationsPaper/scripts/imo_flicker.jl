# julia --project=. StationsPaper/scripts/imo_flicker.jl [out_dir]
# White plus flicker noise amplitudes of the IMO daily vertical series around
# Svartsengi, for the time-correlated noise test of the Q-history inversion.
# Uses the high-passed residuals of imo_noise.jl, split into the common mode
# and each station's specific part. Periodograms are averaged in log-spaced
# period bands (`periods`) and fitted, with relative weights, by a
# non-negative combination of two templates: the mean periodograms of
# simulated unit white and unit flicker noise passed through the same gaps and
# the same running-median high-pass. The templates carry the filter, so the
# fitted variances describe the noise before filtering. Flicker is the
# fractional-difference process of index κ = -1 (Williams, 2003); amplitudes are
# given per day^(1/4) and in the usual mm/yr^(1/4). Writes imo_flicker.csv.
include(joinpath(@__DIR__, "imo_noise.jl"))
using LinearAlgebra
using Random

"""Lower-triangular Toeplitz factor `T` of unit power-law noise of index `κ`: `x = T * w` for unit white `w`."""
function power_law_factor(n; κ = -1.0)
    ψ = ones(n)
    for m in 2:n
        ψ[m] = ψ[m - 1] * (m - 2 - κ / 2) / (m - 1)
    end
    return [i >= j ? ψ[i - j + 1] : 0.0 for i in 1:n, j in 1:n]
end

"""Mean periodogram of the non-NaN samples of `x` in each period band `[periods[b], periods[b + 1])` (days)."""
function band_power(x, periods)
    valid = .!isnan.(x)
    y = ifelse.(valid, x .- mean(x[valid]), 0.0)
    n = length(y)
    power = [abs2(sum(y[t] * cis(-2π * j * (t - 1) / n) for t in 1:n)) / count(valid) for j in 1:(n ÷ 2)]
    period = n ./ (1:(n ÷ 2))
    return [mean(power[periods[b] .<= period .< periods[b + 1]]) for b in 1:(length(periods) - 1)]
end

"""Non-negative weighted least squares for two columns: minimise ‖(y - A c) ./ y‖ with c ≥ 0."""
function fit_two(A, y)
    Aw, yw = A ./ y, ones(length(y))
    c = Aw \ yw
    all(>=(0), c) && return c
    single = [max(Aw[:, j] \ yw, 0.0) for j in 1:2]
    misfit = [norm(yw - Aw[:, j] * single[j]) for j in 1:2]
    j = argmin(misfit)
    return [i == j ? single[j] : 0.0 for i in 1:2]
end

function fit_series(x, T, periods, rng; nsim = 200)
    mask = isnan.(x)
    filtered(w) = (y = copy(w); y[mask] .= NaN; high_pass(y))
    white = mean(band_power(filtered(randn(rng, length(x))), periods) for _ in 1:nsim)
    flicker = mean(band_power(filtered(T * randn(rng, length(x))), periods) for _ in 1:nsim)
    observed = band_power(x, periods)
    variances = fit_two([white flicker], observed)
    return (; σ_white = sqrt(variances[1]), σ_flicker = sqrt(variances[2]), observed, white, flicker,
        relative_misfit = norm((observed - [white flicker] * variances) ./ observed) / sqrt(length(observed)))
end

function imo_flicker_demo(; out_dir = joinpath(@__DIR__, "..", "output", "imo_noise"),
        periods = exp.(range(log(2.0), log(60.0); length = 11)), seed = 2026)
    noise = imo_noise_demo(; out_dir)
    n = length(noise.days)
    T = power_law_factor(n)
    rng = MersenneTwister(seed)
    to_year = (365.25)^0.25
    series = ["common mode" => noise.common;
        [noise.ids[i] => noise.residual[i] .- noise.common for i in noise.used]]
    fits = [name => fit_series(x, T, periods, rng) for (name, x) in series]
    # Flicker std of the displacement change across the whole record, per unit amplitude per day^(1/4).
    year_change = norm(T[end, :] - T[1, :])
    @printf("white and flicker fit, periods %.0f-%.0f days; flicker in mm/day^(1/4) and mm/yr^(1/4)\n", extrema(periods)...)
    open(joinpath(out_dir, "imo_flicker.csv"), "w") do io
        println(io, "series,sigma_white_mm,sigma_flicker_mm_per_day4,flicker_mm_per_yr4,relative_misfit")
        for (name, f) in fits
            println(io, join((name, f.σ_white, f.σ_flicker, f.σ_flicker * to_year, f.relative_misfit), ','))
            @printf("  %-12s white %5.2f mm  flicker %5.2f (%5.2f)  misfit %.2f\n", name, f.σ_white, f.σ_flicker,
                f.σ_flicker * to_year, f.relative_misfit)
        end
    end
    specific = last.(fits[2:end])
    @printf("station-specific median: white %.2f mm, flicker %.2f mm/day^(1/4) (%.1f mm/yr^(1/4))\n",
        median(getfield.(specific, :σ_white)), median(getfield.(specific, :σ_flicker)),
        median(getfield.(specific, :σ_flicker)) * to_year)
    @printf("unit flicker: std of the %d-day displacement change %.2f per mm/day^(1/4)\n", n - 1, year_change)
    println("output: ", abspath(out_dir))
    return (; fits, periods, year_change)
end

if abspath(PROGRAM_FILE) == @__FILE__
    imo_flicker_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "..", "output", "imo_noise") : first(ARGS))
end
