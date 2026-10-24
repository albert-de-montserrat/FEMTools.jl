# julia --project=examples examples/stokes/volcano/chamber_memory_figures2D.jl
# Headless CairoMakie figures from chamber_memory2D.jl and
# chamber_memory_fd_plateau2D.jl output. Run those drivers first; this script
# only reads their CSV/JLS results and adds no new physics.
using CairoMakie
using GeometryBasics: Polygon
CairoMakie.activate!()
using Serialization
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
using .ChamberMemory2D

const day = 86_400.0

# Reference palette (dataviz skill, references/palette.md): fixed categorical
# order for phase identity, a two-hue + neutral-gray diverging pair for signed
# displacement.
const PHASE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # host, magma, shell
const DIVERGING_CMAP = cgrad(["#2a78d6", "#f0efec", "#e34948"])

"""Corner-node (T7 → 3-node) triangle vertices for element `e`, in the mesh's
own km-scaled coordinates."""
triangle_points(coords, el2n, e) = [Point2f(coords[el2n[k, e]][1], coords[el2n[k, e]][2]) for k in 1:3]

"""Mesh colored by material phase (host/magma/shell), rebuilt from `result.geometry`
(a deterministic rebuild — no solve)."""
function chamber_phase_figure(geometry; out_dir)
    model = ChamberMemory2D.build_chamber_memory_mesh(; geometry...)
    coords, el2n, cell_phase = model.mesh.coords, model.mesh.el2n, model.cell_phase
    phase_names = ("host", "magma chamber", "Maxwell shell")
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; xlabel = "x [km]", ylabel = "depth [km]",
        title = "Chamber-memory mesh — $(model.mesh.nels) elements, 3 phases",
        aspect = DataAspect())
    for phase in 1:3
        elements = findall(==(phase), cell_phase)
        isempty(elements) && continue
        polys = [Polygon(triangle_points(coords, el2n, e)) for e in elements]
        poly!(ax, polys; color = PHASE_COLORS[phase], strokewidth = 0.15,
            strokecolor = (:black, 0.3), label = phase_names[phase])
    end
    xlims!(ax, -15, 15)
    axislegend(ax; position = :rb, framevisible = false)
    path = joinpath(out_dir, "chamber_mesh_phases.png")
    save(path, fig)
    return path
end

"""Final-time nodal displacement field (`result.ux_field`/`uz_field`), flat-shaded
per element (mean of its 3 corner nodes), on a mesh rebuilt from `result.geometry`."""
function chamber_field_figure(result, field_name, label; out_dir)
    model = ChamberMemory2D.build_chamber_memory_mesh(; result.geometry...)
    coords, el2n = model.mesh.coords, model.mesh.el2n
    nodal = getfield(result, field_name) .* 1.0e3  # m -> mm
    values = [sum(nodal[el2n[k, e]] for k in 1:3) / 3 for e in axes(el2n, 2)]
    vmax = maximum(abs, values)
    fig = Figure(size = (950, 500))
    ax = Axis(fig[1, 1]; xlabel = "x [km]", ylabel = "depth [km]",
        title = "Final-time $label", aspect = DataAspect())
    polys = [Polygon(triangle_points(coords, el2n, e)) for e in axes(el2n, 2)]
    pl = poly!(ax, polys; color = values, colormap = DIVERGING_CMAP,
        colorrange = (-vmax, vmax), strokewidth = 0)
    xlims!(ax, -15, 15)
    Colorbar(fig[1, 2], pl; label = "$label [mm]")
    path = joinpath(out_dir, "chamber_$(field_name).png")
    save(path, fig)
    return path
end

function budget_and_stations_figure(result_dir, name; out_dir = result_dir)
    result = deserialize(joinpath(result_dir, name * ".jls"))
    t_days = result.time[2:end] ./ day
    fig = Figure(size = (900, 700))
    ax1 = Axis(fig[1, 1]; xlabel = "time [days]", ylabel = "area rate [m²]",
        title = "$name: discrete continuity budget")
    lines!(ax1, t_days, result.recharge_area; label = "recharge")
    lines!(ax1, t_days, -result.withdrawal_area; label = "withdrawal")
    lines!(ax1, t_days, result.expansion; label = "chamber expansion")
    lines!(ax1, t_days, result.storage; label = "compression storage")
    lines!(ax1, t_days, result.budget; label = "residual", linestyle = :dash, color = :black)
    axislegend(ax1; position = :rt, framevisible = false)

    ax2 = Axis(fig[2, 1]; xlabel = "time [days]", ylabel = "pressure [MPa]",
        title = "$name: chamber pressure")
    lines!(ax2, result.time ./ day, result.pressure ./ 1.0e6)

    ax3 = Axis(fig[3, 1]; xlabel = "time [days]", ylabel = "uz [mm]",
        title = "$name: vertical surface displacement")
    for s in eachindex(result.station_x)
        lines!(ax3, result.time ./ day, result.uz[s, :] .* 1.0e3;
            label = "x = $(round(result.station_x[s] / 1.0e3, digits = 1)) km")
    end
    axislegend(ax3; position = :lt, framevisible = false, nbanks = 2)
    path = joinpath(out_dir, name * "_budget_stations.png")
    save(path, fig)
    return path
end

function elastic_comparison_figure(result_dir; out_dir = result_dir)
    elastic = deserialize(joinpath(result_dir, "elastic_host.jls"))
    reference = ChamberMemory2D.chamber_elastic_reference(elastic)
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1]; xlabel = "station x [km]", ylabel = "uz [mm]",
        title = "Elastic-host FEM vs same-area Mogi line source (final time)")
    x_km = elastic.station_x ./ 1.0e3
    scatterlines!(ax, x_km, elastic.uz[:, end] .* 1.0e3; label = "FEM", marker = :circle)
    scatterlines!(ax, x_km, reference.uz[:, end] .* 1.0e3; label = "Mogi line source",
        linestyle = :dash, marker = :xcross)
    axislegend(ax; position = :rt, framevisible = false)
    path = joinpath(out_dir, "elastic_vs_mogi.png")
    save(path, fig)
    return path
end

function timestep_sensitivity_figure(result_dir; out_dir = result_dir)
    baseline = deserialize(joinpath(result_dir, "viscoelastic.jls"))
    refined = deserialize(joinpath(result_dir, "half_timestep.jls"))
    common = [only(findall(==(t), refined.time)) for t in baseline.time]
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1]; xlabel = "time [days]", ylabel = "uz [mm]",
        title = "Time-step sensitivity (reference vs half-Δt)")
    station = cld(size(baseline.uz, 1), 2)
    lines!(ax, baseline.time ./ day, baseline.uz[station, :] .* 1.0e3; label = "Δt")
    lines!(ax, baseline.time ./ day, refined.uz[station, common] .* 1.0e3;
        label = "Δt/2 (resampled to Δt epochs)", linestyle = :dash)
    axislegend(ax; position = :lt, framevisible = false)
    path = joinpath(out_dir, "timestep_sensitivity.png")
    save(path, fig)
    return path
end

"""Parse chamber_memory_fd_plateau2D.jl's four-column `experiment,fd_step,sigma,tangent_error` CSV."""
function read_fd_plateau_csv(csv_path)
    lines = readlines(csv_path)[2:end]
    rows = [split(line, ',') for line in lines]
    return [(name = r[1], fd_step = parse(Float64, r[2]), σ = parse(Float64, r[3])) for r in rows]
end

function fd_plateau_figure(csv_path; out_dir = dirname(csv_path))
    rows = read_fd_plateau_csv(csv_path)
    fig = Figure(size = (850, 500))
    ax = Axis(fig[1, 1]; xlabel = "FD step size (log recharge / log viscosity)",
        ylabel = "σ(log Q) marginal", title = "FD step-size plateau (S0 item 3)",
        xscale = log10)
    for experiment in unique(r.name for r in rows)
        subset = sort(filter(r -> r.name == experiment, rows); by = r -> r.fd_step)
        scatterlines!(ax, [r.fd_step for r in subset], [r.σ for r in subset];
            label = experiment, marker = :circle)
    end
    Legend(fig[1, 2], ax; framevisible = false)
    path = joinpath(out_dir, "fd_plateau.png")
    save(path, fig)
    return path
end

"""Plot a convergence sweep's near-center final uz and budget-closure ratio
against the swept quantity (`xs`), both on a log x-axis."""
function convergence_figure(xs, cases, title; xlabel, out_path)
    fig = Figure(size = (700, 600))
    ax1 = Axis(fig[1, 1]; xlabel, ylabel = "uz(x≈0, final) [mm]", title, xscale = log10)
    scatterlines!(ax1, collect(xs), [1.0e3c.uz_final for c in cases]; marker = :circle)
    ax2 = Axis(fig[2, 1]; xlabel, ylabel = "budget / source", xscale = log10, yscale = log10)
    scatterlines!(ax2, collect(xs), [c.budget_error for c in cases]; marker = :circle)
    save(out_path, fig)
    return out_path
end

"""Bar chart comparing σ(logQ) when nuisance parameters (viscosity,
withdrawal amplitude) are naively fixed at nominal versus honestly
marginalized, from chamber_memory_identifiability2D.jl's metrics.csv."""
function identifiability_sigma_figure(csv_path; out_dir = dirname(csv_path))
    header, body... = readlines(csv_path)
    column = Dict(name => i for (i, name) in enumerate(split(header, ',')))
    rows = [split(line, ',') for line in body]
    names = [r[1] for r in rows]
    σ_fixed = [parse(Float64, r[column["sigma_logQ_fixed"]]) for r in rows]
    σ_marginal = [parse(Float64, r[column["sigma_logQ_marginal"]]) for r in rows]
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; ylabel = "σ(log recharge amplitude)",
        title = "Naive-fixed vs honestly-marginalized nuisance uncertainty",
        xticks = (1:length(names), names), xticklabelrotation = π / 8, yscale = log10)
    n = length(names)
    barplot!(ax, (1:n) .- 0.15, σ_fixed; width = 0.3, label = "nuisance fixed at nominal")
    barplot!(ax, (1:n) .+ 0.15, σ_marginal; width = 0.3, label = "nuisance marginalized")
    axislegend(ax; position = :lt, framevisible = false)
    path = joinpath(out_dir, "identifiability_sigma.png")
    save(path, fig)
    return path
end

"""σ(logQ) fixed vs marginal against cycle count, per design, from
chamber_memory_cycle_sweep2D.jl's dense-sampling rows."""
function cycle_sweep_figure(csv_path; out_dir = dirname(csv_path))
    header, body... = readlines(csv_path)
    column = Dict(name => i for (i, name) in enumerate(split(header, ',')))
    rows = filter(r -> r[column["experiment"]] == "all_components_all_epochs", split.(body, ','))
    fig = Figure(size = (750, 500))
    ax = Axis(fig[1, 1]; xlabel = "repeated cycles (shared η_shell)", ylabel = "σ(log recharge amplitude)",
        title = "More cycles shrink naive σ, not marginal σ", yscale = log10)
    for (design, color) in (("identical", PHASE_COLORS[1]), ("varied", PHASE_COLORS[2]))
        subset = sort(filter(r -> r[column["design"]] == design, rows); by = r -> parse(Int, r[column["cycles"]]))
        k = [parse(Int, r[column["cycles"]]) for r in subset]
        scatterlines!(ax, k, [parse(Float64, r[column["sigma_logQ_marginal"]]) for r in subset];
            color, marker = :circle, label = "$design, nuisance marginalized")
        scatterlines!(ax, k, [parse(Float64, r[column["sigma_logQ_fixed"]]) for r in subset];
            color, marker = :xcross, linestyle = :dash, label = "$design, nuisance fixed")
    end
    Legend(fig[1, 2], ax; framevisible = false)
    path = joinpath(out_dir, "cycle_sweep_sigma.png")
    save(path, fig)
    return path
end

"""Grouped bars: σ of the recent-supply ratio versus σ of the absolute recharge
amplitude, per marginalized nuisance set, from chamber_memory_supply_change2D.jl."""
function supply_change_figure(csv_path; out_dir = dirname(csv_path))
    header, body... = readlines(csv_path)
    column = Dict(name => i for (i, name) in enumerate(split(header, ',')))
    rows = filter(r -> r[column["experiment"]] == "all_components_all_epochs", split.(body, ','))
    labels = [replace(r[column["nuisance"]], "log_" => "") for r in rows]
    n = length(rows)
    fig = Figure(size = (900, 520))
    ax = Axis(fig[1, 1]; ylabel = "posterior σ (log)", yscale = log10,
        title = "Recent supply change vs absolute amplitude, by nuisance marginalized",
        xticks = (1:n, labels), xticklabelrotation = π / 8)
    barplot!(ax, (1:n) .- 0.18, [parse(Float64, r[column["sigma_ratio"]]) for r in rows];
        width = 0.34, color = PHASE_COLORS[1], label = "recent / earlier supply ratio")
    barplot!(ax, (1:n) .+ 0.18, [parse(Float64, r[column["sigma_amplitude"]]) for r in rows];
        width = 0.34, color = PHASE_COLORS[2], label = "absolute recharge amplitude")
    hlines!(ax, [0.5]; color = :gray, linestyle = :dash, label = "prior σ")
    axislegend(ax; position = :lt, framevisible = false)
    path = joinpath(out_dir, "supply_change_sigma.png")
    save(path, fig)
    return path
end

"""Gauss–Newton convergence: data misfit and target estimate per iteration,
from chamber_memory_gauss_newton2D.jl's `fit.jls`."""
function gauss_newton_figure(jls_path; out_dir = dirname(jls_path))
    fit = deserialize(jls_path)
    t = findfirst(==(fit.target), fit.parameters)
    k = [s.iteration for s in fit.steps]
    fig = Figure(size = (1350, 470))
    ax1 = Axis(fig[1, 1]; xlabel = "iteration", ylabel = "rms misfit / noise (before step)",
        yscale = log10, title = "Noise-free fit misfit", xticks = k)
    scatterlines!(ax1, k, [s.misfit for s in fit.steps]; color = PHASE_COLORS[1], marker = :circle)
    ax2 = Axis(fig[1, 2]; xlabel = "iteration", ylabel = string(fit.target),
        title = "Target estimate (after step)", xticks = k)
    scatterlines!(ax2, k, [s.θ[t] for s in fit.steps]; color = PHASE_COLORS[1], marker = :circle,
        label = "estimate")
    hlines!(ax2, [fit.θ_true[t]]; color = PHASE_COLORS[2], linestyle = :dash, label = "truth")
    axislegend(ax2; position = :rb, framevisible = false)
    n = length(fit.parameters)
    ax3 = Axis(fig[1, 3]; ylabel = "log offset from nominal", title = "Converged estimate vs truth",
        xticks = (1:n, [replace(string(p), "log_" => "") for p in fit.parameters]), xticklabelrotation = π / 4)
    if hasproperty(fit, :posterior_σ)
        errorbars!(ax3, 1:n, fit.θ, 1.96 .* fit.posterior_σ; color = PHASE_COLORS[1], whiskerwidth = 8)
    end
    scatter!(ax3, 1:n, fit.θ; color = PHASE_COLORS[1], markersize = 10, label = "estimate")
    scatter!(ax3, 1:n, fit.θ_true; color = PHASE_COLORS[2], marker = :xcross, markersize = 12, label = "truth")
    axislegend(ax3; position = :lb, framevisible = false)
    path = joinpath(out_dir, "gauss_newton.png")
    save(path, fig)
    return path
end

function main(;
        memory_dir = joinpath(@__DIR__, "output_chamber_memory"),
        fd_plateau_csv = joinpath(@__DIR__, "output_chamber_fd_plateau", "fd_plateau.csv"),
        convergence_dir = joinpath(@__DIR__, "output_chamber_convergence"),
        identifiability_csv = joinpath(@__DIR__, "output_chamber_identifiability", "metrics.csv"))
    paths = String[]
    viscoelastic = deserialize(joinpath(memory_dir, "viscoelastic.jls"))
    push!(paths, chamber_phase_figure(viscoelastic.geometry; out_dir = memory_dir))
    if hasproperty(viscoelastic, :uz_field)
        push!(paths, chamber_field_figure(viscoelastic, :uz_field, "vertical displacement"; out_dir = memory_dir))
        push!(paths, chamber_field_figure(viscoelastic, :ux_field, "horizontal displacement"; out_dir = memory_dir))
    end
    for name in ("viscoelastic", "elastic_host")
        push!(paths, budget_and_stations_figure(memory_dir, name))
    end
    push!(paths, elastic_comparison_figure(memory_dir))
    push!(paths, timestep_sensitivity_figure(memory_dir))
    isfile(fd_plateau_csv) && push!(paths, fd_plateau_figure(fd_plateau_csv))
    isfile(identifiability_csv) && push!(paths, identifiability_sigma_figure(identifiability_csv))
    cycle_csv = joinpath(@__DIR__, "output_chamber_cycle_sweep", "cycle_sweep.csv")
    isfile(cycle_csv) && push!(paths, cycle_sweep_figure(cycle_csv))
    supply_csv = joinpath(@__DIR__, "output_chamber_supply_change", "supply_change.csv")
    isfile(supply_csv) && push!(paths, supply_change_figure(supply_csv))
    fit_jls = joinpath(@__DIR__, "output_chamber_gauss_newton", "fit.jls")
    isfile(fit_jls) && push!(paths, gauss_newton_figure(fit_jls))
    mesh_sweep = joinpath(convergence_dir, "mesh_sweep.jls")
    if isfile(mesh_sweep)
        (; max_areas, mesh_cases) = deserialize(mesh_sweep)
        push!(paths, convergence_figure(max_areas, mesh_cases, "Mesh-resolution convergence";
            xlabel = "max_area [m²]", out_path = joinpath(convergence_dir, "mesh_convergence.png")))
    end
    domain_sweep = joinpath(convergence_dir, "domain_sweep.jls")
    if isfile(domain_sweep)
        (; widths, domain_cases) = deserialize(domain_sweep)
        push!(paths, convergence_figure(widths, domain_cases, "Domain-size convergence";
            xlabel = "domain width Lx [m]", out_path = joinpath(convergence_dir, "domain_convergence.png")))
    end
    foreach(p -> println("wrote ", p), paths)
    return paths
end

elapsed = @elapsed main()
println("figures done in ", round(elapsed, digits = 1), " s")
