# Step response of the surface velocity to a change of the dike injection rate,
# from a baseline run of `dike_memory2D.jl` and one whose rate steps by
# `δ` at `t_on`, against the instantaneous response of the 2D Okada dislocation.
# The baseline surface velocity is also inverted, step by step, with the
# dislocation of the true geometry for an apparent injection rate.
#
#   julia --project=examples examples/stokes/volcano/dike_memory_kernels2D.jl

using Serialization

include(joinpath(@__DIR__, "dike_memory2D.jl"))
include(joinpath(@__DIR__, "elastic_dike2D.jl"))
using GLMakie: hlines!

"""
    fit_line_source(x, vy; depths = 0.5e3:0.25e3:20e3, ν = 0.25) -> (; area_rate, depth, misfit)

Best-fitting 2D Mogi line source under `x = 0` for the surface vertical
velocity `vy` [mm/yr] at `x` [m]: depth by grid search over `depths`, strength
by least squares at each depth. `area_rate` is in m²/s, `misfit` relative.
"""
function fit_line_source(x, vy; depths = 0.5e3:0.25e3:20.0e3, ν = 0.25)
    axes(x) == axes(vy) || throw(DimensionMismatch("x and vy must match: $(axes(x)) vs $(axes(vy))"))
    norm(vy) > 0 || throw(ArgumentError("cannot fit a line source to zero velocity"))
    best = (; area_rate = NaN, depth = NaN, misfit = Inf)
    for depth in depths
        g = [line_source_surface(xi; area_change = 1.0, depth, ν)[2] for xi in x] .* (1.0e3 * yr)
        a = dot(vy, g) / dot(g, g)
        misfit = norm(vy .- a .* g) / norm(vy)
        misfit < best.misfit && (best = (; area_rate = a, depth, misfit))
    end
    return best
end

"""
    memory_kernels(; t_on = 10yr, δ = 0.2, run...) -> NamedTuple

Run `main(; run...)` with constant injection and with injection
`Q_dike (1 + δ)` from `t_on` on. Returns the surface coordinates, the time,
both velocity histories, the step response `∂vy/∂ln Q` [mm/yr] per surface node
and step (zero before `t_on` up to solver error), the Okada step response of
the same dislocation (one row, constant in time), and the apparent injection
rate of the baseline from a least-squares fit of the Okada shape within
`fit_half_width` of the dike. `mogi_baseline` and `mogi_step` hold the
best line source (`fit_line_source`) of the baseline velocity and of the step
response at every step: the depth and strength a Mogi inversion would report.
"""
function memory_kernels(; t_on = 10yr, δ = 0.2, fit_half_width = 20.0e3,
        Q_dike = 1.0e-4, dip = 90.0, dike_center_depth = -5.0e3, dike_length = 6.0e3, run...)
    δ > -1 || throw(ArgumentError("δ must exceed −1, got $δ"))
    common = (; Q_dike, dip, dike_center_depth, dike_length, run...)
    baseline = main(; common...)
    stepped  = main(; common..., Q_factor = t -> t >= t_on ? 1 + δ : 1.0)
    x = baseline.surface_x_km .* 1.0e3
    dvy_dlnQ = (stepped.surface_vy_mm_yr .- baseline.surface_vy_mm_yr) ./ log1p(δ)

    # Okada: vy is linear in the opening rate Q/L, so ∂vy/∂ln Q = vy.
    okada = [tensile_dislocation_surface(xi; opening = Q_dike / dike_length,
        center_depth = -dike_center_depth, length = dike_length, dip)[2] * 1.0e3 * yr for xi in x]
    near = abs.(x) .<= fit_half_width
    Q_apparent = [dot(baseline.surface_vy_mm_yr[near, n], okada[near]) / dot(okada[near], okada[near]) * Q_dike
        for n in axes(baseline.surface_vy_mm_yr, 2)]
    fit_steps = axes(baseline.surface_vy_mm_yr, 2)
    mogi_baseline = [fit_line_source(x[near], baseline.surface_vy_mm_yr[near, n]) for n in fit_steps]
    on = findfirst(t -> t > t_on + baseline.time[1] / 2, baseline.time)
    mogi_step = [n >= on ? fit_line_source(x[near], dvy_dlnQ[near, n]) : missing for n in fit_steps]
    return (; x_km = x ./ 1.0e3, time_yr = baseline.time ./ yr, t_on_yr = t_on / yr, δ,
        vy_baseline = baseline.surface_vy_mm_yr, vy_stepped = stepped.surface_vy_mm_yr,
        dvy_dlnQ, okada_dvy_dlnQ = okada, Q_apparent, Q_dike,
        center_depth_km = -dike_center_depth / 1.0e3, mogi_baseline, mogi_step, baseline, stepped)
end

"""
    plot_memory_kernels(k; stations_km = (0, 2, 4, 8, 16)) -> Figure

Step response against time at the surface nodes nearest `stations_km`, with
the Okada value dashed; the step response over x and time; the apparent
injection rate of the baseline.
"""
function plot_memory_kernels(k; stations_km = (0.0, 2.0, 4.0, 8.0, 16.0))
    fig = Figure(size = (1400, 1000))
    ax = Axis(fig[1, 1]; xlabel = "time [yr]", ylabel = "∂vy/∂lnQ [mm/yr]",
        title = @sprintf("Step response to δ = %+g %% at t = %g yr (dashed: Okada)", 100k.δ, k.t_on_yr))
    for xs in stations_km
        i = argmin(abs.(k.x_km .- xs))
        l = lines!(ax, k.time_yr, k.dvy_dlnQ[i, :]; label = @sprintf("x = %.1f km", k.x_km[i]))
        lines!(ax, [k.t_on_yr, last(k.time_yr)], fill(k.okada_dvy_dlnQ[i], 2); color = l.color, linestyle = :dash)
    end
    axislegend(ax; position = :lt)

    near = abs.(k.x_km) .<= 30
    limit = maximum(abs, k.dvy_dlnQ[near, :])
    ax = Axis(fig[1, 2]; xlabel = "x [km]", ylabel = "time [yr]", title = "∂vy/∂lnQ [mm/yr]")
    heatmap!(ax, k.x_km[near], k.time_yr, k.dvy_dlnQ[near, :]; colormap = :vik, colorrange = (-limit, limit))
    Colorbar(fig[1, 3]; colormap = :vik, limits = (-limit, limit), width = 12)

    ax = Axis(fig[2, 1]; xlabel = "x [km]", ylabel = "∂vy/∂lnQ [mm/yr]", title = "Profiles after the step (black dashed: Okada)")
    for n in unique(round.(Int, range(findfirst(>(k.t_on_yr), k.time_yr), length(k.time_yr); length = 4)))
        lines!(ax, k.x_km[near], k.dvy_dlnQ[near, n]; label = @sprintf("t = %.0f yr", k.time_yr[n]))
    end
    lines!(ax, k.x_km[near], k.okada_dvy_dlnQ[near]; color = :black, linestyle = :dash)
    axislegend(ax; position = :rt)

    ax = Axis(fig[2, 2]; xlabel = "time [yr]", ylabel = "Q / Q_true", title = "Apparent injection rate: Okada (solid), Mogi (dashed)")
    lines!(ax, k.time_yr, k.Q_apparent ./ k.Q_dike)
    lines!(ax, k.time_yr, getfield.(k.mogi_baseline, :area_rate) ./ k.Q_dike; linestyle = :dash)
    ax = Axis(fig[3, 1:2]; xlabel = "time [yr]", ylabel = "depth [km]", yreversed = true,
        title = "Apparent Mogi depth: baseline (solid), step response (dashed); true center dotted")
    lines!(ax, k.time_yr, getfield.(k.mogi_baseline, :depth) ./ 1.0e3)
    stepped = findall(!ismissing, k.mogi_step)
    lines!(ax, k.time_yr[stepped], [m.depth for m in k.mogi_step[stepped]] ./ 1.0e3; linestyle = :dash)
    hlines!(ax, [k.center_depth_km]; color = :black, linestyle = :dot)
    return fig
end

"""
    marginal_information(step, trend) -> Real

Fisher information (unit noise) on the amplitude of `step` when the amplitude
of `trend` is also unknown: |step|² − (step·trend)²/|trend|², summed over all
entries (stations × epochs). Zero when the two have the same shape; |step|²
when the trend vanishes on the network and so does not enter the data.
"""
function marginal_information(step, trend)
    axes(step) == axes(trend) || throw(DimensionMismatch("step and trend must match: $(axes(step)) vs $(axes(trend))"))
    tt = dot(trend, trend)
    tt == 0 && return dot(step, step)
    return dot(step, step) - dot(step, trend)^2 / tt
end

"""
    greedy_design(step, trend, candidates, n_stations) -> Vector

Rows of `step` and `trend` (stations × epochs), chosen one at a time from
`candidates` to maximize `marginal_information` of the network.
"""
function greedy_design(step, trend, candidates, n_stations)
    chosen = eltype(candidates)[]
    for _ in 1:n_stations
        score(c) = c in chosen ? -Inf : marginal_information(step[[chosen; c], :], trend[[chosen; c], :])
        push!(chosen, argmax(score, candidates))
    end
    return chosen
end

"""
    compare_designs(k; n_stations = 4, half_width_km = 18, σ_uz_mm = 3, snr = 3, lead_yr = 3) -> NamedTuple

Networks of `n_stations` surface nodes within `half_width_km` of the dike,
from the epochs up to `lead_yr` after the onset. The data are vertical
displacements; a supply change must be told apart from an unknown scaling of
the ongoing deformation (the trend). Design models:

- `signal`: largest baseline |vy| before the onset;
- `elastic`: Okada; step and trend share one spatial shape, so the step is
  separable only in time;
- `frozen`: step shape from the first step after the onset, trend from the
  baseline velocity before it, both extrapolated linearly in time;
- `memory`: the FEM displacement change of the stepped run and the FEM
  baseline displacement.

Each network detects the change when its marginal signal-to-noise ratio,
√`marginal_information` of the FEM displacement change against the FEM
baseline over epochs `1:m`, over `σ_uz_mm`, first exceeds `snr`; detection
times are in years after the onset (`Inf` if never).
"""
function compare_designs(k; n_stations = 4, half_width_km = 18.0, σ_uz_mm = 3.0, snr = 3.0, lead_yr = 3.0)
    candidates = findall(x -> abs(x) <= half_width_km, k.x_km)
    n_stations <= length(candidates) ||
        throw(ArgumentError("only $(length(candidates)) candidate nodes within $half_width_km km"))
    Δt_yr = k.time_yr[2] - k.time_yr[1]
    # First stepped step: the rate is taken at the start of each step, `time_yr` is its end.
    on = findfirst(t -> t > k.t_on_yr + Δt_yr / 2, k.time_yr)
    isnothing(on) && throw(ArgumentError("no step after the onset"))
    # Displacements [mm], per node and step.
    Δuz = cumsum((k.vy_stepped .- k.vy_baseline) .* Δt_yr; dims = 2)
    uz = cumsum(k.vy_baseline .* Δt_yr; dims = 2)
    window = firstindex(k.time_yr):min(on + round(Int, lead_yr / Δt_yr) - 1, lastindex(k.time_yr))
    t = k.time_yr[window]
    ramp = max.(t .- (k.time_yr[on] - Δt_yr), 0)
    design(step, trend) = greedy_design(step, trend, candidates, n_stations)
    networks = (;
        signal = candidates[partialsortperm(abs.(k.vy_baseline[candidates, on - 1]), 1:n_stations; rev = true)],
        elastic = design(k.okada_dvy_dlnQ * ramp', k.okada_dvy_dlnQ * t'),
        frozen = design(k.dvy_dlnQ[:, on] * ramp', k.vy_baseline[:, on - 1] * t'),
        memory = design(Δuz[:, window], uz[:, window]),
    )
    function detection_time(nodes)
        n = findfirst(on:lastindex(k.time_yr)) do m
            sqrt(marginal_information(Δuz[nodes, 1:m], uz[nodes, 1:m])) / σ_uz_mm > snr
        end
        return isnothing(n) ? Inf : n * Δt_yr
    end
    return (; networks = map(nodes -> k.x_km[nodes], networks), detection_yr = map(detection_time, networks))
end

if abspath(PROGRAM_FILE) == @__FILE__
    out_dir = joinpath(@__DIR__, "output_dike_memory_kernels")
    mkpath(out_dir)
    k = memory_kernels(; nsteps = 30, t_on = 10yr, δ = 0.2, ε̇_bg = 1.0e-30, λ_fluid = 1,
        refinement = 100, surface_size = 500.0, ϵ_tol = 1.0e-8, iterMax = 500_000,
        write_output = false, out_dir)
    serialize(joinpath(out_dir, "kernels.jls"), k)
    save(joinpath(out_dir, "memory_kernels.png"), plot_memory_kernels(k))
    @info "Network designs" compare_designs(k)...
end
