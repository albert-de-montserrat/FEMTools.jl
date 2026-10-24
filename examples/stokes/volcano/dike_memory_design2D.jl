# Station design from the time-dependent adjoint of `dike_memory_adjoint2D.jl`:
# the sensitivity of the surface displacement uz(x_s, t_m) to the injection
# rate of every year and dike segment, and greedy D-optimal networks from it and
# from the elastic (Okada) Green's functions of the same dike.
#
#   julia --project=examples examples/stokes/volcano/dike_memory_design2D.jl [dip, default 90]

using Serialization

include(joinpath(@__DIR__, "dike_memory_adjoint2D.jl"))
include(joinpath(@__DIR__, "elastic_dike2D.jl"))

"""
    greedy_network(G, n_select; noise, prior) -> (selected, logdets)

Greedy D-optimal choice of `n_select` stations from `G[s, m, p]` (station,
epoch, parameter): each pick adds all epochs of one station and maximizes
`logdet(I + (prior/noise)² GₛᵀGₛ)`.
"""
function greedy_network(G, n_select; noise, prior)
    rows(stations) = reshape(G[stations, :, :], :, size(G, 3))
    information(stations) = logdet(I + (prior / noise)^2 * rows(stations)' * rows(stations))
    selected, logdets = Int[], Float64[]
    for _ in 1:n_select
        candidates = setdiff(axes(G, 1), selected)
        gains = [information(vcat(selected, s)) for s in candidates]
        push!(selected, candidates[argmax(gains)])
        push!(logdets, maximum(gains))
    end
    return selected, logdets
end

network_information(G, stations; noise, prior) =
    logdet(I + (prior / noise)^2 * reshape(G[stations, :, :], :, size(G, 3))' * reshape(G[stations, :, :], :, size(G, 3)))

"""
    target_std(G, stations, c; noise, prior) -> Real

Posterior standard deviation of the target `cᵀp` from the data of `stations`,
with `G[s, m, p]` the data Jacobian, independent data noise `noise` and an
independent prior `prior` on every parameter.
"""
function target_std(G, stations, c; noise, prior)
    A = reshape(G[stations, :, :], :, size(G, 3))
    posterior = Symmetric(A' * A ./ noise^2 + I / prior^2)
    return sqrt(dot(c, posterior \ c))
end

# All `k`-subsets of `items`, as vectors.
subsets(items, k) = k == 0 ? [eltype(items)[]] :
    [[items[i]; rest] for i in eachindex(items) for rest in subsets(items[(i + 1):end], k - 1)]

"""
    best_network(score, candidates, k) -> (network, value)

Exhaustive minimizer of `score(network)` over all `k`-subsets of `candidates`.
"""
function best_network(score, candidates, k)
    networks = subsets(collect(candidates), k)
    values = map(score, networks)
    i = argmin(values)
    return networks[i], values[i]
end

"""
    goal_designs(res; targets, n_max = 5) -> NamedTuple

Per target (a coefficient vector over the parameters of `res.G_design`) and
network size 1:`n_max`, the network that minimizes the target's posterior
standard deviation under the FEM kernels and under the Okada kernels, both
evaluated with the FEM kernels, plus the standard deviation Okada claims for its
own network.
"""
function goal_designs(res; targets, n_max = 5)
    G_okada = reshape(res.G_okada, size(res.G_okada, 1), size(res.G_okada, 2), :)
    candidates = axes(res.G_design, 1)
    std_fem(net, c) = target_std(res.G_design, net, c; res.noise, res.prior)
    return map(targets) do c
        rows = map(1:n_max) do k
            fem, σ_fem = best_network(net -> std_fem(net, c), candidates, k)
            elastic, σ_claimed = best_network(net -> target_std(G_okada, net, c; res.noise, res.prior), candidates, k)
            (; fem, elastic, σ_fem, σ_elastic = std_fem(elastic, c), σ_claimed)
        end
        (; rows, σ_prior = res.prior * norm(c))
    end
end

"""
    plot_goal_designs(res, goals) -> Figure

Posterior standard deviation of each target against network size for the FEM-
and Okada-optimal networks (both judged by the FEM kernels; dotted: what Okada
claims for its own network), and the chosen station positions.
"""
function plot_goal_designs(res, goals)
    fig = Figure(size = (1500, 450 * length(goals)))
    for (row, (name, goal)) in enumerate(pairs(goals))
        n = eachindex(goal.rows)
        ax = Axis(fig[row, 1]; xlabel = "stations", ylabel = "posterior std of target",
            title = "$name (prior std $(round(goal.σ_prior; sigdigits = 2)))", yscale = log10)
        scatterlines!(ax, n, [r.σ_fem for r in goal.rows]; label = "FEM-optimal network")
        scatterlines!(ax, n, [r.σ_elastic for r in goal.rows]; label = "Okada-optimal network, true std")
        scatterlines!(ax, n, [r.σ_claimed for r in goal.rows]; linestyle = :dot, label = "Okada-optimal network, Okada's claim")
        axislegend(ax; position = :rt)
        ax = Axis(fig[row, 2]; xlabel = "station x [km]", ylabel = "stations in network",
            title = "$name: chosen stations (circles: FEM, crosses: Okada)", yticks = n)
        vlines!(ax, res.station_x_km; color = (:gray, 0.3))
        for (k, r) in pairs(goal.rows)
            scatter!(ax, res.station_x_km[r.fem], fill(k + 0.1, k); markersize = 14)
            scatter!(ax, res.station_x_km[r.elastic], fill(k - 0.1, k); marker = :xcross, markersize = 14)
        end
    end
    return fig
end

"""
    plot_design(res) -> Figure

Forward surface velocity, the adjoint kernels ∂uz(t_m)/∂ln Q(year) per
station (summed over segments) for every epoch done, the per-segment kernel of
the last epoch, the Okada kernel, and the networks when available.
"""
function plot_design(res)
    fig = Figure(size = (1500, 1100))
    near = abs.(res.surface_x_km) .<= 30
    ax = Axis(fig[1, 1]; xlabel = "x [km]", ylabel = "time [yr]", title = "Forward surface vy [mm/yr]")
    hm = heatmap!(ax, res.surface_x_km[near], res.time_yr, res.surface_vy_mm_yr[near, :]; colormap = :viridis)
    Colorbar(fig[1, 2], hm; width = 12)
    ax = Axis(fig[1, 3]; xlabel = "x [km]", ylabel = "vy [mm/yr]", title = "Surface vy profiles")
    for n in unique(round.(Int, range(1, length(res.time_yr); length = 5)))
        lines!(ax, res.surface_x_km[near], res.surface_vy_mm_yr[near, n]; label = @sprintf("t = %.0f yr", res.time_yr[n]))
    end
    axislegend(ax; position = :rt)
    isempty(res.epochs_done) && return fig

    ax = Axis(fig[2, 1]; xlabel = "injection year", ylabel = "∂uz/∂ln Q(year) [mm]",
        title = "Adjoint kernels, summed over segments (solid: FEM, dashed: Okada)")
    for (k, m) in enumerate(res.epochs_done), s in get(res, :kernel_stations, (1, 4, length(res.station_x_km)))
        total = vec(sum(res.G_fem[s, k, :, :]; dims = 2))
        l = lines!(ax, 1:m, total[1:m]; label = @sprintf("x = %.1f km, t = %d yr", res.station_x_km[s], m))
        lines!(ax, 1:m, vec(sum(res.G_okada[s, k, 1:m, :]; dims = 2)); color = l.color, linestyle = :dash)
    end
    axislegend(ax; position = :lt, labelsize = 10, nbanks = 2)

    k = length(res.epochs_done)
    m = res.epochs_done[k]
    ax = Axis(fig[2, 3]; xlabel = "station x [km]", ylabel = "injection year",
        title = @sprintf("∂uz(t = %d yr)/∂ln Q(year), all segments [mm]", m))
    total = dropdims(sum(res.G_fem[:, k, 1:m, :]; dims = 3); dims = 3)
    hm = heatmap!(ax, res.station_x_km, 1:m, total; colormap = :viridis)
    Colorbar(fig[2, 4], hm; width = 12)

    ax = Axis(fig[3, 1]; xlabel = "station x [km]", ylabel = "∂uz/∂ln Q(segment) [mm]",
        title = @sprintf("Per-segment kernel at t = %d yr, summed over years (solid: FEM, dashed: Okada); 1 = bottom", m))
    for j in axes(res.G_fem, 4)
        l = scatterlines!(ax, res.station_x_km, vec(sum(res.G_fem[:, k, 1:m, j]; dims = 2)); label = "segment $j")
        lines!(ax, res.station_x_km, vec(sum(res.G_okada[:, k, 1:m, j]; dims = 2)); color = l.color, linestyle = :dash)
    end
    axislegend(ax; position = :rt, labelsize = 10, nbanks = 2)

    if haskey(res, :networks)
        ax = Axis(fig[3, 3]; xlabel = "stations", ylabel = "information under the FEM kernels [nats]",
            title = "Greedy networks evaluated with the FEM kernels")
        for (name, net) in pairs(res.networks)
            info = [network_information(res.G_design, net[1:i]; res.noise, res.prior) for i in eachindex(net)]
            scatterlines!(ax, eachindex(net), info; label = @sprintf("%s: %s km", name,
                join(round.(res.station_x_km[net]; digits = 1), ", ")))
        end
        axislegend(ax; position = :rb, labelsize = 10)
    end
    return fig
end

"""
    recent_targets(nsteps, n_segments) -> NamedTuple

Coefficient vectors over the parameters `(year, segment)` of `G_design`: mean
ln Q over the last and the first five years, and over the last five years of
the top segment.
"""
function recent_targets(nsteps, n_segments)
    index = LinearIndices((nsteps, n_segments))
    function target(years, segments)
        c = zeros(nsteps * n_segments)
        c[vec(index[years, segments])] .= 1 / (length(years) * length(segments))
        return c
    end
    return (; recent_supply = target((nsteps - 4):nsteps, 1:n_segments),
        recent_top_segment = target((nsteps - 4):nsteps, n_segments:n_segments),
        old_supply = target(1:5, 1:n_segments))
end

"""
    run_design(; dip, station_x, kernel_station_x, out_dir, ...) -> NamedTuple

Forward run with checkpoints, adjoint sweeps of uz at `epochs` for the surface
nodes nearest `station_x` [m], greedy and goal-oriented networks; figures and
`design.jls` go to `out_dir` after every stage. `kernel_station_x` picks the
stations of the kernel-versus-year panel.
"""
function run_design(; dip, station_x, kernel_station_x, out_dir, nsteps = 30, epochs = (10, 20, 30),
        Q_dike = 1.0e-4, dike_length = 6.0e3, dike_center_depth = -5.0e3, n_segments = 6,
        noise = 3.0, prior = 1.0, n_select = 4)
    mkpath(out_dir)
    r = main(; nsteps, Q_dike, dike_length, dike_center_depth, dip, n_segments,
        ε̇_bg = 1.0e-30, λ_fluid = 1, refinement = 50, surface_size = 1000.0,
        ϵ_tol = 1.0e-10, iterMax = 500_000, write_output = false, verbose = false, checkpoint = true)
    ctx = r.adjoint_context
    idx = [argmin(abs.(r.surface_x_km .* 1.0e3 .- x)) for x in station_x]
    allunique(idx) || error("two stations map to the same surface node")
    nodes = ctx.surface_nodes[idx]
    station_x_km = r.surface_x_km[idx]
    kernel_stations = [argmin(abs.(station_x_km .* 1.0e3 .- x)) for x in kernel_station_x]
    Δt_yr = r.time[1] / yr

    # Okada: uz(t_m) gains Δt · opening rate · Gz per segment from every year up to t_m.
    _, Gz = dike_segment_greens(station_x_km .* 1.0e3; center_depth = -dike_center_depth,
        length = dike_length, dip, n_segments)
    G_okada = zeros(length(nodes), length(epochs), nsteps, n_segments)
    for (k, m) in enumerate(epochs), n in 1:m
        G_okada[:, k, n, :] .= Gz .* (Q_dike / dike_length * Δt_yr * yr * 1.0e3)
    end
    G_fem = zeros(length(nodes), length(epochs), nsteps, n_segments)
    res = (; dip, surface_x_km = r.surface_x_km, time_yr = r.time ./ yr, surface_vy_mm_yr = r.surface_vy_mm_yr,
        station_x_km, kernel_stations, epochs, epochs_done = Int[], G_fem, G_okada)
    serialize(joinpath(out_dir, "design.jls"), res)
    save(joinpath(out_dir, "design_forward.png"), plot_design(res))

    for (k, m) in enumerate(epochs)
        sweep = merge(ctx, (; checkpoints = ctx.checkpoints[1:m]))
        t = @elapsed g = memory_adjoint(sweep, nodes, fill(ctx.Δτ, m); verbose = false)
        G_fem[:, k, 1:m, :] .= g.dJ_dlnQ .* (ctx.L_c * 1.0e3)
        push!(res.epochs_done, m)
        @info "Adjoint sweep" m t
        serialize(joinpath(out_dir, "design.jls"), res)
        save(joinpath(out_dir, @sprintf("design_epoch_%02d.png", m)), plot_design(res))
    end

    G_design = reshape(G_fem, size(G_fem, 1), size(G_fem, 2), :)
    fem, _ = greedy_network(G_design, n_select; noise, prior)
    elastic, _ = greedy_network(reshape(G_okada, size(G_okada, 1), size(G_okada, 2), :), n_select; noise, prior)
    even = round.(Int, range(1, length(nodes); length = n_select))
    res = merge(res, (; G_design, noise, prior, networks = (; fem, elastic, even)))
    serialize(joinpath(out_dir, "design.jls"), res)
    save(joinpath(out_dir, "design.png"), plot_design(res))
    @info "Greedy networks [km]" map(net -> station_x_km[net], res.networks)

    goals = goal_designs(res; targets = recent_targets(nsteps, n_segments))
    serialize(joinpath(out_dir, "goals.jls"), goals)
    save(joinpath(out_dir, "goal_designs.png"), plot_goal_designs(res, goals))
    return res, goals
end

if abspath(PROGRAM_FILE) == @__FILE__
    dip = parse(Float64, get(ARGS, 1, "90"))
    if dip == 90
        station_x = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0) .* 1.0e3
        kernel_station_x = (0.0, 3.0e3, 16.0e3)
    else
        station_x = (-16.0, -10.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 10.0, 16.0) .* 1.0e3
        kernel_station_x = (-2.0e3, 2.0e3, 10.0e3)
    end
    out_dir = joinpath(@__DIR__, dip == 90 ? "output_dike_memory_design" : @sprintf("output_dike_memory_design_dip%02d", dip))
    run_design(; dip, station_x, kernel_station_x, out_dir)
end
