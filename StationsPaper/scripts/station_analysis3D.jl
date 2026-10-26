# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/station_analysis3D.jl [impulse_dir] [out_dir] [elastic_dir]
# Stage F of StationsPaper/PLAN.md: do the 2-D information results transfer to
# the 17 Svartsengi stations? From the 3-D impulse response alone:
#   1. separability: singular values of H (result 1);
#   2. degrees of freedom by station subset under white and empirical noise,
#      7-day bins, known start and with unknown pre-window supply (results 4, 7);
#   3. reference stations: one station added to the near set, and the supply
#      bias of differencing against it (result 16);
#   4. elastic Mogi on noise-free constant supply that started `lead` days
#      before the window, with the exact compressibility factor
#      1 + 4G/(3K_magma) of a spherical cavity and with c = 1 (results 13, 17);
#      with an elastic-shell response in `elastic_dir`, the same for elastic truth.
# The window and pre-window lengths follow the impulse-response length.
# Writes subsets.csv, references.csv, mogi.csv, and station_analysis3D.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_information2D.jl"))

"""Vertical surface displacement of a Mogi point source per unit cavity volume [m per m³], at horizontal distance `r`."""
mogi_uz(r, depth, ν) = (1 - ν) * depth / (π * (r^2 + depth^2)^1.5)

function mogi_bias(H, xy, Δt, ν, c_true, Q_ref, noise_factor; window, width, lead, depths, centres)
    ns = length(xy)
    whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
    data = whiten(window_displacement(H, fill(Q_ref, lead + window), lead))
    best = (; misfit = Inf)
    for depth in depths, centre in centres
        g = [mogi_uz(hypot(x - centre[1], y - centre[2]), depth, ν) * Δt for (x, y) in xy]
        Kw = whiten(reshape(window_jacobian(repeat(g, 1, window), 1:ns; window, width).K, ns * window, :))
        rates = Kw \ data
        misfit = sum(abs2, data - Kw * rates)
        misfit < best.misfit && (best = (; misfit, depth, centre, rates))
    end
    ratio = mean(best.rates) / Q_ref
    return (; best.misfit, best.depth, best.centre, ratio_incompressible = ratio, ratio_exact = c_true * ratio,
        c_crossover = 1 / ratio)
end

function station_analysis3D_demo(; impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response3D"),
        out_dir = joinpath(@__DIR__, "..", "output", "station_analysis3D"), elastic_dir = nothing,
        width = 7, uplift = 0.24, noise_models = (("white", 7.0e-3, 0.0), ("empirical", 3.0e-3, 3.5e-3)),
        near_radius = 5.0e3, leads = (0, 30, 90, 180))
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    (; H, station_ids, station_xy, Δt, geometry) = response
    centre_xy = geometry.chamber_center[1:2]
    distance = [hypot(x - centre_xy[1], y - centre_xy[2]) for (x, y) in station_xy]
    order = sortperm(distance)
    nt = size(H, 2)
    window = min(365, nt ÷ 2)
    prewindow = nt - window
    closest = first(order)
    Q_ref = uplift * window / 365 / sum(H[closest, 1:window])
    @printf("%d stations, %d-step response: window %d days, pre-window %d days; Q_ref %.2f m³/s (%.2f m/yr at %s)\n",
        length(station_ids), nt, window, prewindow, Q_ref, uplift, station_ids[closest])

    singular = svdvals(H)
    @printf("separability: σ2/σ1 = %.2e, σ3/σ1 = %.2e\n", singular[2] / singular[1], singular[3] / singular[1])

    near = order[distance[order] .<= near_radius]
    far = order[distance[order] .> near_radius]
    named(ids...) = [findfirst(==(id), station_ids) for id in ids]
    subsets = ("closest station" => [closest], "near (≤ 5 km)" => near, "near + VOGC" => [near; named("VOGC")],
        "near + NYLA" => [near; named("NYLA")], "near + VOGC + NYLA" => [near; named("VOGC", "NYLA")],
        "all 17" => order)
    subset_rows = NamedTuple[]
    for (name, stations) in subsets, (noise, σ, σ_common) in noise_models, (memory, pre) in (("known start", 0), ("unknown earlier supply", prewindow))
        J = window_jacobian(H, stations; window, width, prewindow = pre)
        s = posterior_summary(J.K, J.nwindow; σ, σ_common, σ_prior = Q_ref)
        push!(subset_rows, (; subset = name, nstations = length(stations), noise, memory, s.dof, s.median_rdiag,
            s.centre_fwhm_bins))
    end
    println("degrees of freedom, $(width)-day bins:")
    for r in subset_rows
        @printf("  %-20s %2d  %-9s %-22s %6.1f  (median rdiag %.2f, fwhm %d bins)\n", r.subset, r.nstations, r.noise,
            r.memory, r.dof, r.median_rdiag, r.centre_fwhm_bins)
    end

    (_, σ, σ_common) = noise_models[end]
    J_near = window_jacobian(H, near; window, width)
    baseline = posterior_summary(J_near.K, J_near.nwindow; σ, σ_common, σ_prior = Q_ref).dof
    near_matrix = reshape(J_near.K, length(near) * window, :)
    reference_rows = map(far) do s
        J = window_jacobian(H, [near; s]; window, width)
        records = displacement_from_rates(H[[near; s], 1:window], fill(Q_ref, window))
        differenced = vec(records[1:(end - 1), :] .- records[end:end, :])
        (; station = station_ids[s], distance = distance[s],
            signal_fraction = sum(H[s, 1:window]) / sum(H[closest, 1:window]),
            dof_gain = posterior_summary(J.K, J.nwindow; σ, σ_common, σ_prior = Q_ref).dof - baseline,
            differencing_ratio = mean(near_matrix \ differenced) / Q_ref)
    end
    @printf("reference station added to the %d near stations (%.1f dof, empirical noise):\n", length(near), baseline)
    for r in reference_rows
        @printf("  %-5s %5.1f km  signal %.3f  +dof %.2f  differenced apparent/true %.3f\n", r.station,
            r.distance / 1.0e3, r.signal_fraction, r.dof_gain, r.differencing_ratio)
    end

    noise_factor = cholesky(Symmetric(σ^2 * I(length(order)) + σ_common^2 * ones(length(order), length(order)))).L
    depths = 2.5e3:100.0:6.0e3
    centres = [(centre_xy[1] + dx, centre_xy[2] + dy) for dx in -500.0:250.0:500.0, dy in -500.0:250.0:500.0]
    truths = ["Maxwell shell" => (H, response.material)]
    if !isnothing(elastic_dir) && isfile(joinpath(elastic_dir, "impulse_response.jls"))
        elastic = deserialize(joinpath(elastic_dir, "impulse_response.jls"))
        push!(truths, "elastic shell" => (elastic.H, elastic.material))
    end
    mogi_rows = NamedTuple[]
    for (truth, (Ht, material)) in truths, lead in leads
        lead + window <= size(Ht, 2) || continue
        (; G_host, K_host, K_magma) = material
        ν = (3K_host - 2G_host) / (2 * (3K_host + G_host))
        c_true = 1 + 4G_host / (3K_magma)
        fit = mogi_bias(Ht[order, :], station_xy[order], Δt, ν, c_true, Q_ref, noise_factor;
            window, width, lead, depths, centres)
        push!(mogi_rows, (; truth, lead, c_true, fit.depth, offset = hypot((fit.centre .- centre_xy)...),
            fit.misfit, fit.ratio_incompressible, fit.ratio_exact, fit.c_crossover))
    end
    println("elastic Mogi on noise-free constant supply (all 17 stations, empirical noise):")
    for r in mogi_rows
        @printf("  %-13s lead %3d d: depth %.1f km, offset %.2f km, Δχ² %8.1f;  c = 1 %.2f | exact (%.2f) %.2f;  c* %.2f\n",
            r.truth, r.lead, r.depth / 1.0e3, r.offset / 1.0e3, r.misfit, r.ratio_incompressible, r.c_true,
            r.ratio_exact, r.c_crossover)
    end

    write_rows(name, rows) = open(joinpath(out_dir, name), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        foreach(r -> println(io, join(values(r), ',')), rows)
    end
    write_rows("subsets.csv", subset_rows)
    write_rows("references.csv", reference_rows)
    isempty(mogi_rows) || write_rows("mogi.csv", mogi_rows)
    result = (; subset_rows, reference_rows, mogi_rows, singular, window, prewindow, width, Q_ref, station_ids,
        distance, impulse_dir)
    serialize(joinpath(out_dir, "station_analysis3D.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

station_analysis3D_demo(;
    impulse_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "impulse_response3D"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "output", "station_analysis3D"),
    elastic_dir = length(ARGS) > 2 ? ARGS[3] : nothing)
