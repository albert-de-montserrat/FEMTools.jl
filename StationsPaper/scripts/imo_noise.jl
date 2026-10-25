# julia --project=. StationsPaper/scripts/imo_noise.jl [out_dir]
# Empirical short-period noise of the IMO daily vertical GNSS solutions around
# Svartsengi, for the "empirical" noise model in station_information2D.jl.
# Downloads each station's post-processed series once into out_dir, removes a
# 31-day running median (so slower deformation and long-period noise are
# excluded), and splits the residual scatter into a common mode (daily median
# across stations) and a station-specific part. Stations whose residual scatter
# exceeds `outlier_mm` (series with jumps) are reported but left out of the
# common mode. Observed data are used only for this noise estimate.
using Dates
using Downloads
using Printf
using Statistics

const IMO_SERIES = "https://cdn.vedur.is/gps/timeseries/data/post-processing/timeseries"

"""Vertical displacement [mm] on each day of `days` from an IMO `-plate.NEU` file; missing days are NaN."""
function read_vertical(path, days)
    u = fill(NaN, length(days))
    for line in eachline(path)
        startswith(line, "#") && continue
        fields = split(line)
        length(fields) >= 8 || continue
        day = Date(fields[1], dateformat"yyyy/mm/dd")
        first(days) <= day <= last(days) && (u[Dates.value(day - first(days)) + 1] = parse(Float64, fields[7]))
    end
    return u
end

"""Residual after removing the running median of `2half + 1` days (NaN where fewer than 10 samples)."""
function high_pass(u; half = 15)
    return map(eachindex(u)) do k
        window = filter(!isnan, u[max(1, k - half):min(end, k + half)])
        isnan(u[k]) || length(window) < 10 ? NaN : u[k] - median(window)
    end
end

function imo_noise_demo(; out_dir = joinpath(@__DIR__, "..", "output", "imo_noise"),
        stations_csv = joinpath(@__DIR__, "..", "data", "svartsengi_gnss_stations.csv"),
        stop = Date(2026, 10, 23), ndays = 365, outlier_mm = 15.0)
    mkpath(out_dir)
    ids = [first(split(line, ',')) for line in eachline(stations_csv) if !startswith(line, '#') && !startswith(line, "id,")]
    days = (stop - Day(ndays - 1)):Day(1):stop
    residual = map(ids) do id
        path = joinpath(out_dir, id * "-plate.NEU")
        isfile(path) || Downloads.download("$IMO_SERIES/$id-plate.NEU", path)
        high_pass(read_vertical(path, days))
    end
    scatter = [std(filter(!isnan, r)) for r in residual]
    used = findall(<(outlier_mm), scatter)
    common = map(eachindex(days)) do k
        values = filter(!isnan, [residual[i][k] for i in used])
        length(values) >= 10 ? median(values) : NaN
    end
    @printf("%s to %s, %d stations (%d in the common mode)\n", first(days), last(days), length(ids), length(used))
    @printf("common mode: std %.2f mm, MAD-based %.2f mm\n", std(filter(!isnan, common)),
        1.4826 * median(abs.(filter(!isnan, common) .- median(filter(!isnan, common)))))
    open(joinpath(out_dir, "imo_noise.csv"), "w") do io
        println(io, "id,used_in_common_mode,scatter_mm,station_specific_mm,ndays")
        for (i, id) in enumerate(ids)
            both = .!isnan.(residual[i]) .& .!isnan.(common)
            specific = std(residual[i][both] .- common[both])
            println(io, join((id, i in used, scatter[i], specific, count(both)), ','))
            @printf("  %s  scatter %6.2f mm  station-specific %6.2f mm%s\n", id, scatter[i], specific,
                i in used ? "" : "  (excluded)")
        end
    end
    specific_used = [std(residual[i][.!isnan.(residual[i]) .& .!isnan.(common)] .-
        common[.!isnan.(residual[i]) .& .!isnan.(common)]) for i in used]
    @printf("median scatter %.2f mm; median station-specific %.2f mm\n", median(scatter[used]), median(specific_used))
    println("output: ", abspath(out_dir))
    return (; ids, scatter, used, common, days, residual)
end

if abspath(PROGRAM_FILE) == @__FILE__
    imo_noise_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "..", "output", "imo_noise") : first(ARGS))
end
