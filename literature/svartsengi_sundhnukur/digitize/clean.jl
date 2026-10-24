using Dates, Printf, Statistics, DelimitedFiles
raw = readdlm("fig6a_digitized_raw.csv", ',', Float64; skipstart = 1)
days_span = Dates.value(Date(2025, 2, 1) - Date(2023, 11, 1))
col2day(x) = (x - 134) / (1388 - 134) * days_span
event_day = col2day.([163.5, 267.5, 339.5, 408.5, 471.5, 511.5, 713.5, 946.5, 1191.5])
seg(d) = searchsortedfirst(event_day, d)           # 1 = before first event
d, v = raw[:, 1], raw[:, 2]
# Drop pixels more than 1.5e6 m3 from the running median of their segment.
keep = trues(length(d))
for i in eachindex(d)
    nb = [j for j in eachindex(d) if seg(d[j]) == seg(d[i]) && abs(d[j] - d[i]) <= 3]
    keep[i] = abs(v[i] - median(v[nb])) <= 1.5
end
println("dropped ", count(!, keep), " outlier pixels")
d, v = d[keep], v[keep]
# Daily bins (calendar day of the plot's x axis), median per bin and segment.
rows = []
for s in unique(seg.(d)), day in unique(floor.(Int, d[seg.(d) .== s]))
    sel = (seg.(d) .== s) .& (floor.(Int, d) .== day)
    push!(rows, (s, Date(2023, 11, 1) + Day(day), median(v[sel])))
end
sort!(rows; by = r -> (r[2], r[1]))
open("fig6a_volume_daily.csv", "w") do io
    println(io, "segment,date,volume_change_1e6m3")
    for (s, dt, val) in rows
        @printf(io, "%d,%s,%.2f\n", s, dt, val)
    end
end
println(length(rows), " daily values")
# Cross-check: slope of the digitized curve over each panel-b interval.
bars = readdlm("fig6b_digitized_rates.csv", ','; skipstart = 1)
println("interval                    panel b   slope of a  [m3/s]")
for k in axes(bars, 1)
    a, b = Date(bars[k, 1]), Date(bars[k, 2])
    sel = [(Date(2023, 11, 1) + Day(floor(Int, x))) for x in d]
    m = (sel .>= a) .& (sel .<= b)
    count(m) > 5 || continue
    x = d[m]; y = v[m]
    slope = cov(x, y) / var(x) * 1e6 / 86400
    @printf("%s – %s   %5.2f     %5.2f\n", a, b, bars[k, 3], slope)
end
