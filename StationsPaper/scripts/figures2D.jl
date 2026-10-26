# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/figures2D.jl [impulse_dir] [information_dir] [recovery_dir] [viscosity_dir] [mogi_dir] [drainage_dir] [bias_dir] [reference_dir]
# Draft paper figures from saved output only (no simulation): run
# impulse_response2D.jl, station_information2D.jl, synthetic_recovery2D.jl,
# viscosity_ambiguity2D.jl, mogi_comparison2D.jl, drainage_recovery2D.jl, bias_sign2D.jl, and
# reference_station2D.jl
# first (figures 6–10 are skipped without their output). Writes PNG and PDF
# into each input directory. All panels are synthetic 2-D plane-strain results.
using CairoMakie
CairoMakie.activate!()
using Printf
using Serialization

include(joinpath(@__DIR__, "figures_style.jl"))

"""Impulse response by station distance: memory growth relative to the elastic step, and amplitude versus distance."""
function impulse_figure(response; out_dir, distances = (0.0, 4.0e3, 8.0e3, 20.0e3))
    (; H, station_x, Δt) = response
    days = (1:size(H, 2)) .* (Δt / 86_400.0)
    fig = Figure(size = (720, 300))
    ax = Axis(fig[1, 1]; xlabel = "time after one-day unit pulse [days]",
        ylabel = "u_z(t) / u_z(day 1)", title = "(a) memory relative to the elastic step")
    for (c, x) in zip(DISTANCE_RAMP, distances)
        i = findfirst(==(x), station_x)
        lines!(ax, days, H[i, :] ./ H[i, 1]; color = c)
        text!(ax, days[end], H[i, end] / H[i, 1]; text = @sprintf(" %g km", x / 1.0e3),
            color = INK, align = (:left, :center), fontsize = 10)
    end
    xlims!(ax, 0, days[end] * 1.12)
    bx = Axis(fig[1, 2]; xlabel = "station distance x [km]", ylabel = "u_z per unit rate [m/(m²/s)]",
        title = "(b) elastic step and response after one year", yscale = log10,
        yticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5], ytickformat = values -> [@sprintf("%g", v) for v in values])
    x_km = station_x ./ 1.0e3
    year = min(size(H, 2), round(Int, 365 * 86_400.0 / Δt))
    for (c, n, label) in ((SERIES[1], 1, "day 1"), (SERIES[2], year, "day $year"))
        lines!(bx, x_km, H[:, n]; color = c, label)
    end
    axislegend(bx; position = :rt, framevisible = false)
    return save_both(out_dir, "fig2_impulse_response", fig)
end

"""Resolution versus bin width for station subsets, information versus elastic gain, and averaging kernels."""
function resolution_figure(info; out_dir, memory = "known start")
    rows = filter(r -> r.memory == memory, info.rows)
    fig = Figure(size = (900, 300))
    ax = Axis(fig[1, 1]; xlabel = "source bin width [days]", ylabel = "median averaging-kernel diagonal",
        title = "(a) resolution of interior bins", xscale = log10, xticks = collect(info.widths))
    for (c, subset) in zip(SERIES, ("x=0 km", "6 near (0-4.5 km)", "6 far (8-18 km)", "all 29"))
        r = sort(filter(r -> r.subset == subset && r.noise == "white", rows); by = r -> r.width)
        scatterlines!(ax, [r.width for r in r], [r.median_rdiag for r in r]; color = c, label = subset,
            markersize = 7, strokecolor = :white, strokewidth = 1)
    end
    ylims!(ax, 0, 1.05)
    axislegend(ax; position = :lt, framevisible = false, labelsize = 9)

    bx = Axis(fig[1, 2]; xlabel = "elastic gain Σ H_i(1)² / H_x=0(1)²", ylabel = "degrees of freedom resolved",
        title = "(b) 7-day bins: information vs gain", xscale = log10)
    for (c, noise, marker) in ((SERIES[1], "white", :circle), (SERIES[2], "white + common", :utriangle),
            (SERIES[3], "empirical", :diamond))
        r = sort(filter(r -> r.width == 7 && r.noise == noise, rows); by = r -> r.gain)
        scatter!(bx, [r.gain for r in r], [r.dof for r in r]; color = c, marker, label = noise,
            strokecolor = :white, strokewidth = 1)
    end
    r = sort(filter(r -> r.width == 7 && r.noise == "white", rows); by = r -> r.gain)
    lines!(bx, [r.gain for r in r], [r.dof_equivalent for r in r]; color = MUTED, linewidth = 1,
        linestyle = :dash, label = "one station, σ/√gain (white)")
    axislegend(bx; position = :lt, framevisible = false, labelsize = 9)

    cx = Axis(fig[1, 3]; xlabel = "source time [days]", ylabel = "averaging kernel per day (centre bin)",
        title = "(c) averaging kernels, 6 near stations")
    for (c, width) in zip(SERIES, (1, 7))
        k = only(filter(k -> k.width == width && k.subset == "6 near (0-4.5 km)" && k.memory == memory, info.kernels))
        # Per-day density, so kernels of different bin widths share one scale.
        stairs!(cx, k.centres, k.row ./ width; color = c, step = :center, label = "$width-day bins")
    end
    xlims!(cx, 120, 245)
    axislegend(cx; position = :rt, framevisible = false, labelsize = 9)
    return save_both(out_dir, "fig3_resolution", fig)
end

"""Averaging-kernel diagonal of each window bin against its age at the end of the window."""
function latency_figure(info; out_dir, subset = "6 near (0-4.5 km)")
    fig = Figure(size = (560, 300))
    ax = Axis(fig[1, 1]; xlabel = "bin age at end of window [days]", ylabel = "averaging-kernel diagonal",
        title = "Resolution by bin age, $(first(split(subset, " (")))", xreversed = true)
    for (c, width) in zip(SERIES, (1, 7, 30)), (style, memory) in ((:solid, "known start"), (:dash, "unknown pre-window"))
        matches = filter(k -> k.width == width && k.subset == subset && k.memory == memory, info.kernels)
        isempty(matches) && continue
        k = only(matches)
        scatterlines!(ax, info.window .- k.centres, k.rdiag; color = c, linestyle = style,
            markersize = width == 1 ? 0 : 6, label = memory == "known start" ? "$width-day bins" : "$width-day, unknown pre-window")
    end
    xlims!(ax, info.window, 0)
    ylims!(ax, 0, 1.05)
    Legend(fig[1, 2], ax; framevisible = false, labelsize = 9)
    return save_both(out_dir, "fig4_latency", fig)
end

"""True and recovered Q(t) for each synthetic history; the ambiguity pair shares one panel."""
function recovery_figure(recovery; out_dir)
    (; results, centres, nwindow, window, prewindow, width, Q_ref) = recovery
    fig = Figure(size = (900, 480))
    panels = [[r] for r in results if !startswith(r.name, "H5")]
    push!(panels, [r for r in results if startswith(r.name, "H5")])
    days = (1:window) .- 0.5
    for (p, group) in enumerate(panels)
        ax = Axis(fig[fld1(p, 3), mod1(p, 3)]; title = startswith(first(group).name, "H5") ?
            "H5 pair: earlier supply vs trend (χ² sep. $(@sprintf("%.1f", recovery.data_difference^2)))" : first(group).name,
            xlabel = fld1(p, 3) == 2 ? "day of window" : "", ylabel = mod1(p, 3) == 1 ? "Q / Q_ref" : "")
        for (c, r) in zip(SERIES, group)
            lines!(ax, days, r.rates[(prewindow + 1):end] ./ Q_ref; color = (MUTED, 0.5), linewidth = 0.8)
            σ = r.posterior_σ[1:nwindow] ./ Q_ref
            estimate = r.estimate[1:nwindow] ./ Q_ref
            band!(ax, centres, estimate .- 2σ, estimate .+ 2σ; color = (c, 0.18))
            stairs!(ax, centres, r.truth[1:nwindow] ./ Q_ref; step = :center, color = INK, linewidth = 1.2)
            stairs!(ax, centres, estimate; step = :center, color = c, linewidth = 1.6)
        end
        xlims!(ax, 0, window)
        # Drainage days reach tens of Q_ref; the binned history is the comparison.
        startswith(first(group).name, "H4") && ylims!(ax, -6, 4)
    end
    Legend(fig[3, 1:3], [LineElement(color = (MUTED, 0.5), linewidth = 0.8), LineElement(color = INK),
            [PolyElement(color = (SERIES[1], 0.18)), LineElement(color = SERIES[1])], LineElement(color = SERIES[2])],
        ["daily truth", "$width-day truth", "recovered ± 2σ", "H5b recovered"];
        orientation = :horizontal, framevisible = false, labelsize = 9)
    return save_both(out_dir, "fig5_recovery", fig)
end

"""Shell-viscosity error: its detectability, the trend it fakes, and the recovered-rate bias over time."""
function viscosity_figure(sweep; out_dir, layout = "6 near + 6 far")
    rows = sweep.rows
    fig = Figure(size = (900, 340))
    ratios = ([1 / 3, 1 / 2, 2 / 3, 1, 1.5, 3], ["1/3", "1/2", "2/3", "1", "1.5", "3"])
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "η_true / η_model", xticks = ratios,
        yticks = ([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"]),
        ylabel = "noise-free Δχ² of wrong-η fit", title = "(a) detectability")
    bx = Axis(fig[1, 2]; xscale = log10, xlabel = "η_true / η_model", xticks = ratios,
        ylabel = "apparent trend [Q_ref per year]", title = "(b) trend faked by the η error")
    for (c, name) in zip(SERIES, sweep.layouts)
        r = sort(filter(r -> r.layout == name && r.η_ratio != 1, rows); by = r -> r.η_ratio)
        scatterlines!(ax, [r.η_ratio for r in r], [r.Δχ2 for r in r]; color = c, label = name, markersize = 7)
        r = sort(filter(r -> r.layout == name, rows); by = r -> r.η_ratio)
        scatterlines!(bx, [r.η_ratio for r in r], [r.apparent_trend for r in r]; color = c, markersize = 7)
    end
    hlines!(ax, [4.0]; color = MUTED, linestyle = :dash, linewidth = 1)
    text!(ax, 1.0, 4.0; text = "Δχ² = 4", color = MUTED, align = (:center, :bottom), fontsize = 9)
    hlines!(bx, [0.0]; color = MUTED, linewidth = 0.8)
    Legend(fig[2, 1:2], ax; orientation = :horizontal, framevisible = false, labelsize = 9)

    cx = Axis(fig[1, 3]; xlabel = "day of window", ylabel = "recovered − true [Q_ref]",
        title = "(c) bias over time, $layout")
    softer = sort(filter(r -> r.layout == layout && r.η_ratio < 1, rows); by = r -> r.η_ratio)
    stiffer = sort(filter(r -> r.layout == layout && r.η_ratio > 1, rows); by = r -> -r.η_ratio)
    # Diverging pair: blue for a softer true shell, red for a stiffer one; darker is further from the model.
    blues, reds = ("#104281", "#3987e5", "#86b6ef"), ("#8f1d1d", "#e34948", "#f0a3a2")
    for (palette, group) in ((blues, softer), (reds, stiffer)), (c, r) in zip(palette, group)
        stairs!(cx, r.centres, r.bias[1:length(r.centres)]; step = :center, color = c,
            label = @sprintf("η ×%.2g", r.η_ratio))
    end
    hlines!(cx, [0.0]; color = MUTED, linewidth = 0.8)
    xlims!(cx, 0, sweep.window)
    axislegend(cx; position = :rt, framevisible = false, labelsize = 8)
    return save_both(out_dir, "fig6_viscosity_ambiguity", fig)
end

"""Cumulative supply recovered by the chamber model and by elastic Mogi from the same records, for elastic and Maxwell-shell truth."""
function mogi_figure(comparison; out_dir, histories = ("H1", "H6"), truths = ("elastic shell", "Maxwell shell"))
    (; rows, Q_ref, window) = comparison
    # One unit is a year of supply at the reference rate.
    year_supply = Q_ref * 365 * 86_400.0
    fig = Figure(size = (720, 480))
    for (i, truth) in enumerate(truths), (j, prefix) in enumerate(histories)
        r = only(filter(r -> r.truth_name == truth && startswith(r.history, prefix), rows))
        ax = Axis(fig[i, j]; title = "$(truth): $(r.history)", xlabel = i == length(truths) ? "day of window" : "",
            ylabel = j == 1 ? "cumulative supply [Q_ref · yr]" : "")
        days = [0; r.bin_ends]
        for (c, estimate, σ) in ((SERIES[1], r.chamber_cumulative, r.chamber_cumulative_σ),
                (SERIES[2], r.mogi_cumulative, r.mogi_cumulative_σ))
            band!(ax, days, [0; estimate .- 2σ] ./ year_supply, [0; estimate .+ 2σ] ./ year_supply; color = (c, 0.2))
            lines!(ax, days, [0; estimate] ./ year_supply; color = c, linewidth = 1.6)
        end
        lines!(ax, days, [0; r.truth_cumulative] ./ year_supply; color = INK, linewidth = 1.2, linestyle = :dash)
        xlims!(ax, 0, window)
        ylims!(ax, 0, 2)
    end
    Legend(fig[length(truths) + 1, 1:length(histories)], [LineElement(color = INK, linestyle = :dash),
            [PolyElement(color = (SERIES[1], 0.2)), LineElement(color = SERIES[1])],
            [PolyElement(color = (SERIES[2], 0.2)), LineElement(color = SERIES[2])]],
        ["truth", "chamber model ± 2σ", "elastic Mogi, (1 + G/K) × source volume ± 2σ"];
        orientation = :horizontal, framevisible = false, labelsize = 9)
    return save_both(out_dir, "fig7_mogi_comparison", fig)
end

"""Drainage cycles (H4): recharge recovered without and with drainage days as unknowns, and the drained volumes."""
function drainage_figure(drainage; out_dir, layout = "6 near + 6 far",
        cases = ("σ = Q_ref", "drainage days from objective scan"))
    (; rows, Q_ref, rates, prewindow, window, true_events, true_volumes) = drainage
    pick(case) = only(filter(r -> r.layout == layout && r.case == case, rows))
    centres(r) = [(first(b) + last(b) - 1) / 2 - prewindow for b in r.bins[1:r.nwindow]]
    fig = Figure(size = (900, 320))
    ax = Axis(fig[1, 1]; xlabel = "day of window", ylabel = "Q / Q_ref",
        title = "(a) recharge, $layout")
    lines!(ax, (1:window) .- 0.5, rates[(prewindow + 1):end] ./ Q_ref; color = (MUTED, 0.5), linewidth = 0.8,
        label = "daily truth")
    events = pick(last(cases))
    # Truth for the event case: bin-average recharge over each bin's non-drainage days.
    stairs!(ax, centres(events), events.truth[1:events.nwindow] ./ Q_ref; step = :center, color = INK,
        linewidth = 1.2, label = "7-day recharge truth")
    for (c, case, label) in ((SERIES[2], first(cases), "drainage inside 7-day bins"),
            (SERIES[1], last(cases), "drainage days as unknowns"))
        r = pick(case)
        x, estimate, σ = centres(r), r.estimate[1:r.nwindow] ./ Q_ref, r.posterior_σ[1:r.nwindow] ./ Q_ref
        band!(ax, x, estimate .- 2σ, estimate .+ 2σ; color = (c, 0.18))
        stairs!(ax, x, estimate; step = :center, color = c, linewidth = 1.6, label)
    end
    xlims!(ax, 0, window)
    ylims!(ax, -3, 4)
    Legend(fig[1, 2], ax; framevisible = false, labelsize = 9)

    bx = Axis(fig[1, 3]; xlabel = "drainage day", ylabel = "drained volume [Q_ref · day]",
        title = "(b) drained volume per event", xticks = collect(true_events))
    scatter!(bx, true_events, -true_volumes ./ Q_ref; color = INK, marker = :hline, markersize = 18,
        label = "truth")
    errorbars!(bx, events.events, -events.event_volume ./ Q_ref, 2events.event_volume_σ ./ Q_ref;
        color = SERIES[1], whiskerwidth = 6)
    scatter!(bx, events.events, -events.event_volume ./ Q_ref; color = SERIES[1], label = "recovered ± 2σ")
    axislegend(bx; position = :rb, framevisible = false, labelsize = 9)
    colsize!(fig.layout, 1, Relative(0.5))
    return save_both(out_dir, "fig8_drainage", fig)
end

"""Sign of the elastic-Mogi supply bias: apparent/true with the exact compressibility factor, and the factor where it is unbiased."""
function bias_sign_figure(sweep; out_dir)
    (; rows) = sweep
    # Shell viscosity in the distance ramp (soft to stiff); other mechanics in categorical slots.
    styles = ("elastic shell" => (INK, :dash), "η_shell 1e17" => (DISTANCE_RAMP[4], :solid),
        "η_shell 2e17" => (DISTANCE_RAMP[3], :solid), "η_shell 3e17 (reference)" => (DISTANCE_RAMP[2], :solid),
        "η_shell 9e17" => (DISTANCE_RAMP[1], :solid), "shell to 3 km" => (SERIES[3], :solid),
        "source at 3 km" => (SERIES[4], :solid), "aureole 1e17-1e20" => (SERIES[2], :solid))
    fig = Figure(size = (760, 320))
    ax = Axis(fig[1, 1]; xlabel = "supply already running at window start [days]",
        ylabel = "Mogi / true window supply", title = "(a) exact factor 1 + G/K_magma")
    bx = Axis(fig[1, 2]; xlabel = "supply already running at window start [days]",
        ylabel = "assumed factor c for no bias", title = "(b) unbiased compressibility factor c*")
    hlines!(ax, [1.0]; color = MUTED, linewidth = 0.8)
    for (name, (color, linestyle)) in styles
        r = filter(r -> r.truth_name == name, rows)
        isempty(r) && continue
        lead = getfield.(r, :lead)
        scatterlines!(ax, lead, getfield.(r, :ratio_exact); color, linestyle, markersize = 6, label = name)
        scatterlines!(bx, lead, getfield.(r, :c_crossover); color, linestyle, markersize = 6)
    end
    c_true = first(rows).c_true
    hlines!(bx, [1.0, c_true]; color = MUTED, linewidth = 0.8, linestyle = :dot)
    text!(bx, 0, 1.0; text = "c = 1: cavity change read as injected volume", fontsize = 8, color = MUTED,
        align = (:left, :bottom))
    text!(bx, 0, c_true; text = "exact c = $(round(c_true; digits = 2))", fontsize = 8, color = MUTED,
        align = (:left, :top))
    ylims!(bx, 0, 1.1c_true)
    Legend(fig[1, 3], ax; framevisible = false, labelsize = 9)
    return save_both(out_dir, "fig9_bias_sign", fig)
end

"""Reference stations: information added by one far station, the bias of differencing against it, and the 3-D Svartsengi distances."""
function reference_figure(reference; out_dir)
    (; profile, baseline_dof, svartsengi, depths) = reference
    x = getfield.(profile, :x_ref) ./ 1.0e3
    fig = Figure(size = (760, 320))
    ax = Axis(fig[1, 1]; xlabel = "reference distance [km]", ylabel = "added degrees of freedom",
        title = @sprintf("(a) joint estimation, 6 near (%.1f dof) + 1", baseline_dof))
    scatterlines!(ax, x, getfield.(profile, :dof_gain); color = SERIES[1], markersize = 6)
    bx = Axis(fig[1, 2]; xlabel = "distance from source [km]", ylabel = "fraction of epicentral signal",
        yscale = log10, yticks = ([1.0, 0.3, 0.1, 0.03, 0.01], ["1", "0.3", "0.1", "0.03", "0.01"]),
        title = "(b) signal at the reference = bias of differencing")
    scatterlines!(bx, x, 1 .- getfield.(profile, :differencing_ratio); color = SERIES[1], markersize = 6,
        label = "2-D: supply bias of differenced records")
    r = range(0.0, 20.0; length = 200)
    lines!(bx, r, (1 .+ (r ./ (first(depths) / 1.0e3)) .^ 2) .^ -1.5; color = SERIES[2],
        label = @sprintf("3-D Mogi, %.1f km deep", first(depths) / 1.0e3))
    scatter!(bx, [s.r for s in svartsengi] ./ 1.0e3, [first(s.fractions) for s in svartsengi];
        color = SERIES[2], marker = :diamond, markersize = 8, label = "Svartsengi stations")
    for s in svartsengi
        s.r > 8.0e3 && text!(bx, s.r / 1.0e3, first(s.fractions); text = s.id, fontsize = 8, color = MUTED,
            align = (:right, :top), offset = (-3, -3))
    end
    hlines!(bx, [0.05, 0.01]; color = MUTED, linewidth = 0.8, linestyle = :dot)
    ylims!(bx, 5.0e-3, 1.2)
    axislegend(bx; position = :lb, framevisible = false, labelsize = 8)
    return save_both(out_dir, "fig10_reference_station", fig)
end

function figures_demo(; impulse_dir, information_dir, recovery_dir = nothing, viscosity_dir = nothing, mogi_dir = nothing,
        drainage_dir = nothing, bias_dir = nothing, reference_dir = nothing)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    info = deserialize(joinpath(information_dir, "station_information.jls"))
    paths = with_theme(paper_theme()) do
        figures = [impulse_figure(response; out_dir = impulse_dir), resolution_figure(info; out_dir = information_dir),
            latency_figure(info; out_dir = information_dir)]
        isnothing(recovery_dir) || push!(figures,
            recovery_figure(deserialize(joinpath(recovery_dir, "synthetic_recovery.jls")); out_dir = recovery_dir))
        isnothing(viscosity_dir) || !isfile(joinpath(viscosity_dir, "viscosity_ambiguity.jls")) || push!(figures,
            viscosity_figure(deserialize(joinpath(viscosity_dir, "viscosity_ambiguity.jls")); out_dir = viscosity_dir))
        isnothing(mogi_dir) || !isfile(joinpath(mogi_dir, "mogi_comparison.jls")) || push!(figures,
            mogi_figure(deserialize(joinpath(mogi_dir, "mogi_comparison.jls")); out_dir = mogi_dir))
        isnothing(drainage_dir) || !isfile(joinpath(drainage_dir, "drainage_recovery.jls")) || push!(figures,
            drainage_figure(deserialize(joinpath(drainage_dir, "drainage_recovery.jls")); out_dir = drainage_dir))
        isnothing(bias_dir) || !isfile(joinpath(bias_dir, "bias_sign.jls")) || push!(figures,
            bias_sign_figure(deserialize(joinpath(bias_dir, "bias_sign.jls")); out_dir = bias_dir))
        isnothing(reference_dir) || !isfile(joinpath(reference_dir, "reference_station.jls")) || push!(figures,
            reference_figure(deserialize(joinpath(reference_dir, "reference_station.jls")); out_dir = reference_dir))
        figures
    end
    foreach(println, paths)
    return paths
end

figures_demo(;
    impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
    information_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "station_information2D"),
    recovery_dir = length(ARGS) > 2 ? ARGS[3] : joinpath(@__DIR__, "..", "output", "synthetic_recovery2D", "near6_far6"),
    viscosity_dir = length(ARGS) > 3 ? ARGS[4] : joinpath(@__DIR__, "..", "output", "viscosity_ambiguity2D"),
    mogi_dir = length(ARGS) > 4 ? ARGS[5] : joinpath(@__DIR__, "..", "output", "mogi_comparison2D"),
    drainage_dir = length(ARGS) > 5 ? ARGS[6] : joinpath(@__DIR__, "..", "output", "drainage_recovery2D"),
    bias_dir = length(ARGS) > 6 ? ARGS[7] : joinpath(@__DIR__, "..", "output", "bias_sign2D"),
    reference_dir = length(ARGS) > 7 ? ARGS[8] : joinpath(@__DIR__, "..", "output", "reference_station2D"))
