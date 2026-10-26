# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=StationsPaper StationsPaper/scripts/figures3D.jl [impulse_dir] [analysis_dir]
# Draft figure 11 from saved output only (no simulation): run
# impulse_response3D.jl and station_analysis3D.jl first. Station signal at the
# 17 Svartsengi GNSS stations, memory relative to the elastic step, and degrees
# of freedom by station subset. Writes PNG and PDF into `analysis_dir`. All
# panels are synthetic 3-D results at the real station positions.
using CairoMakie
CairoMakie.activate!()
using Printf
using Serialization

include(joinpath(@__DIR__, "figures_style.jl"))

"""Signal fraction on the station map, memory relative to the elastic step, and degrees of freedom by subset."""
function svartsengi_figure(response, analysis; out_dir, memory = "known start")
    (; H, station_ids, station_xy, Δt, geometry) = response
    (; distance, window) = analysis
    centre = geometry.chamber_center[1:2]
    fraction = [sum(H[s, 1:window]) for s in eachindex(station_ids)]
    fraction ./= maximum(fraction)
    fig = Figure(size = (940, 320))

    ax = Axis(fig[1, 1]; xlabel = "east of SENG [km]", ylabel = "north of SENG [km]", aspect = DataAspect(),
        title = "(a) window signal / largest")
    for radius in (5.0, 10.0, 15.0)
        θ = range(0, 2π; length = 181)
        lines!(ax, centre[1] / 1.0e3 .+ radius .* cos.(θ), centre[2] / 1.0e3 .+ radius .* sin.(θ);
            color = (MUTED, 0.4), linewidth = 0.8, linestyle = :dot)
    end
    east, north = first.(station_xy) ./ 1.0e3, last.(station_xy) ./ 1.0e3
    plot = scatter!(ax, east, north; color = log10.(fraction), colormap = reverse(collect(DISTANCE_RAMP)),
        colorrange = (-2.2, 0), strokecolor = INK, strokewidth = 0.6, markersize = 10)
    scatter!(ax, [centre[1] / 1.0e3], [centre[2] / 1.0e3]; marker = :xcross, color = SERIES[2], markersize = 12)
    # Label only the stations outside the near cluster, where labels do not overlap.
    for s in eachindex(station_ids)
        distance[s] > 5.0e3 && text!(ax, east[s], north[s]; text = "  " * station_ids[s], fontsize = 9, color = INK,
            align = (:left, :center))
    end
    Colorbar(fig[1, 2], plot; label = "log₁₀ signal fraction", width = 8, height = Relative(0.7), ticklabelsize = 9)

    bx = Axis(fig[1, 3]; xlabel = "time after one-day unit pulse [days]", ylabel = "u_z(t) / u_z(day 1)",
        title = "(b) memory relative to the elastic step")
    days = (1:size(H, 2)) .* (Δt / 86_400.0)
    order = sortperm(distance; rev = true)
    bins = (5.0e3, 10.0e3, 15.0e3)
    for s in order
        lines!(bx, days, H[s, :] ./ H[s, 1]; color = DISTANCE_RAMP[1 + count(<(distance[s]), bins)], linewidth = 1.2)
    end
    for (c, label) in zip(DISTANCE_RAMP, ("< 5 km", "5–10 km", "10–15 km", "> 15 km"))
        lines!(bx, [NaN], [NaN]; color = c, label)
    end
    axislegend(bx; position = :lt, framevisible = false, labelsize = 9)

    cx = Axis(fig[1, 4]; xlabel = "degrees of freedom, 7-day bins", title = "(c) information by station subset",
        yreversed = true)
    subsets = unique(r.subset for r in analysis.subset_rows)
    for (k, (c, noise)) in enumerate(zip(SERIES, ("white", "empirical")))
        rows = [only(filter(r -> r.subset == name && r.noise == noise && r.memory == memory, analysis.subset_rows))
                for name in subsets]
        barplot!(cx, eachindex(subsets) .+ (k - 1.5) * 0.38, [r.dof for r in rows]; direction = :x, width = 0.36,
            color = c, label = noise)
    end
    cx.yticks = (eachindex(subsets), subsets)
    xlims!(cx, 0, nothing)
    Legend(fig[2, 4], cx; orientation = :horizontal, framevisible = false, labelsize = 9, tellheight = true)
    colsize!(fig.layout, 1, Relative(0.3))
    return save_both(out_dir, "fig11_svartsengi", fig)
end

function figures3D_demo(; impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response3D_365"),
        analysis_dir = joinpath(@__DIR__, "..", "output", "station_analysis3D_365"))
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    analysis = deserialize(joinpath(analysis_dir, "station_analysis3D.jls"))
    path = with_theme(() -> svartsengi_figure(response, analysis; out_dir = analysis_dir), paper_theme())
    println(path)
    return path
end

figures3D_demo(;
    impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response3D_365"),
    analysis_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "station_analysis3D_365"))
