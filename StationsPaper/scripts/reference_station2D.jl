# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/reference_station2D.jl [output_root] [out_dir]
# PLAN.md task 8: at what distance does a station count as a reference for the
# common mode? For each candidate reference x_ref on the 2-D profile (from the
# impulse response, gravity + advection unless a linear output is given):
#   - signal fraction: its one-year constant-supply uplift over that at x = 0;
#   - joint estimation (the treatment in station_information2D.jl): degrees of
#     freedom of 6 near stations plus x_ref under the empirical noise model,
#     7-day bins, known start, prior σ Q_ref;
#   - differencing (the common practice): noise-free 6-near records minus the
#     x_ref record, fitted with the undifferenced chamber Jacobian, so the
#     source signal at x_ref is read as common mode; window-mean apparent/true.
# Then the 3-D analogue for Svartsengi: the Mogi point-source fraction
# (1 + (r/d)²)^(-3/2) at each station of data/svartsengi_gnss_stations.csv, for
# the Parks et al. (2025) source position (OBSERVATION_MODEL.md). Writes
# profile.csv, svartsengi.csv, and reference_station.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_information2D.jl"))

"""Stations from the IMO inventory CSV: id and east/north offsets from SENG [m]."""
function read_stations(path)
    lines = filter(l -> !startswith(l, '#'), readlines(path))
    header = split(lines[1], ',')
    column(name) = findfirst(==(name), header)
    return map(lines[2:end]) do line
        f = split(line, ',')
        (; id = String(f[column("id")]), east = parse(Float64, f[column("east_m")]),
            north = parse(Float64, f[column("north_m")]))
    end
end

function reference_station_demo(; output_root = joinpath(@__DIR__, "..", "output"),
        impulse_dir = joinpath(output_root, "impulse_response2D_3yr"),
        out_dir = joinpath(output_root, "reference_station2D"),
        near = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3), window = 365, width = 7, uplift = 0.24,
        σ = 3.0e-3, σ_common = 3.5e-3,
        station_file = joinpath(@__DIR__, "..", "data", "svartsengi_gnss_stations.csv"),
        source_east = -1.0e3, source_north = -1.6e3, depths = (4.0e3, 4.35e3), thresholds = (0.05, 0.01))
    mkpath(out_dir)
    response = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    (; H, station_x) = response
    index(x) = findfirst(==(x), station_x)
    centre = index(0.0)
    year_uplift = vec(sum(H[:, 1:window]; dims = 2))
    Q_ref = uplift / year_uplift[centre]
    near_idx = index.(collect(near))
    jacobian(stations) = window_jacobian(H, stations; window, width)
    near_K = jacobian(near_idx)
    baseline = posterior_summary(near_K.K, near_K.nwindow; σ, σ_common, σ_prior = Q_ref).dof
    near_matrix = reshape(near_K.K, length(near_idx) * window, :)
    profile = NamedTuple[]
    for (s, x) in enumerate(station_x)
        x >= 4.0e3 && !(x in near) || continue
        joint = jacobian([near_idx; s])
        dof = posterior_summary(joint.K, joint.nwindow; σ, σ_common, σ_prior = Q_ref).dof
        records = displacement_from_rates(H[[near_idx; s], 1:window], fill(Q_ref, window))
        differenced = vec(records[1:(end - 1), :] .- records[end:end, :])
        ratio = mean(near_matrix \ differenced) / Q_ref
        push!(profile, (; x_ref = x, signal_fraction = year_uplift[s] / year_uplift[centre],
            dof_gain = dof - baseline, differencing_ratio = ratio))
    end

    # 3-D: vertical Mogi point-source displacement relative to the epicentre.
    fraction(r, d) = (1 + (r / d)^2)^(-1.5)
    radius(f, d) = d * sqrt(f^(-2 / 3) - 1)
    stations = read_stations(station_file)
    svartsengi = [(; s.id, r = hypot(s.east - source_east, s.north - source_north),
        fractions = Tuple(fraction(hypot(s.east - source_east, s.north - source_north), d) for d in depths))
        for s in stations]
    sort!(svartsengi; by = s -> s.r)

    @printf("2-D profile (%s): 6 near stations resolve %.1f degrees of freedom (empirical noise)\n",
        basename(normpath(impulse_dir)), baseline)
    @printf("  %8s %10s %10s %16s\n", "x_ref", "signal", "+dof joint", "differenced a/t")
    for p in profile
        @printf("  %5.1f km %10.3f %10.2f %16.3f\n", p.x_ref / 1.0e3, p.signal_fraction, p.dof_gain, p.differencing_ratio)
    end
    println("3-D Mogi: distance beyond which the vertical signal is below a fraction of the epicentral one")
    for d in depths, f in thresholds
        @printf("  depth %.2f km, %4.1f%%: r > %.1f km\n", d / 1.0e3, 100f, radius(f, d) / 1.0e3)
    end
    @printf("Svartsengi stations, distance from the Parks et al. (2025) Mogi position, vertical signal fraction (d = %s km)\n",
        join(string.(depths ./ 1.0e3), ", "))
    for s in svartsengi
        @printf("  %-5s %5.1f km  %s\n", s.id, s.r / 1.0e3, join([@sprintf("%.3f", f) for f in s.fractions], "  "))
    end

    open(joinpath(out_dir, "profile.csv"), "w") do io
        println(io, "x_ref_m,signal_fraction,dof_gain_joint,differencing_apparent_over_true")
        foreach(p -> println(io, join((p.x_ref, p.signal_fraction, p.dof_gain, p.differencing_ratio), ',')), profile)
    end
    open(joinpath(out_dir, "svartsengi.csv"), "w") do io
        println(io, "id,distance_from_source_m,", join(["mogi_fraction_depth_$(round(Int, d))m" for d in depths], ','))
        foreach(s -> println(io, join((s.id, s.r, s.fractions...), ',')), svartsengi)
    end
    result = (; profile, baseline_dof = baseline, svartsengi, depths = collect(depths), thresholds = collect(thresholds),
        threshold_radii = [radius(f, d) for d in depths, f in thresholds], impulse_dir, Q_ref, σ, σ_common)
    serialize(joinpath(out_dir, "reference_station.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

reference_station_demo(;
    output_root = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"), "reference_station2D"))
