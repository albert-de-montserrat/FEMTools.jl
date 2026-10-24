using PNGFiles, Colors, Dates, Printf, Statistics
img = PNGFiles.load("p9-000.png")
R = Float64.(red.(img)) .* 255; G = Float64.(green.(img)) .* 255; B = Float64.(blue.(img)) .* 255
chroma = max.(R, G, B) .- min.(R, G, B)
# Pink eruption shading is (≈255, ≈200, ≈200); exclude it and anything near it.
pink = (R .> 240) .& (abs.(G .- B) .< 25) .& (G .> 150)
event_cols = [162:165; 266:269; 338:341; 407:410; 470:473; 510:513; 712:715; 945:948; 1190:1193]
mask = (chroma .> 40) .& .!pink
mask[:, [c for c in event_cols]] .= false
mask[40:110, 1085:1410] .= false          # legend
mask[690:end, :] .= false; mask[1:40, :] .= false; mask[:, 1:105] .= false

# Axis calibration from tick marks: y ticks 0, -20, -40, -60 at rows 165.5, 318, 470, 622.5;
# x ticks at month starts, 1 Nov 2023 (col 134) to 1 Feb 2025 (col 1388).
yrows = [165.5, 318.0, 470.0, 622.5]; yvals = [0.0, -20.0, -40.0, -60.0]
ya = sum((yrows .- mean(yrows)) .* (yvals .- mean(yvals))) / sum((yrows .- mean(yrows)) .^ 2)
yb = mean(yvals) - ya * mean(yrows)
t0 = DateTime(2023, 11, 1); days_span = Dates.value(Date(2025, 2, 1) - Date(2023, 11, 1))
col2day(x) = (x - 134) / (1388 - 134) * days_span
row2vol(y) = ya * y + yb

out = Tuple{Float64, Float64}[]
for x in axes(mask, 2)
    rows = findall(mask[:, x])
    isempty(rows) && continue
    # split into clusters separated by gaps > 12 px (a ring is ~9 px tall)
    start = 1
    for k in eachindex(rows)
        if k == lastindex(rows) || rows[k + 1] - rows[k] > 12
            cl = rows[start:k]
            length(cl) >= 2 && push!(out, (col2day(x), row2vol((first(cl) + last(cl)) / 2)))
            start = k + 1
        end
    end
end
open("fig6a_digitized_raw.csv", "w") do io
    println(io, "days_since_2023-11-01,volume_change_1e6m3")
    for (d, v) in out
        @printf(io, "%.3f,%.3f\n", d, v)
    end
end
println(length(out), " points; ya = ", ya, " (1e6 m3 / px)")
