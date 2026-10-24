using PNGFiles, Colors, Dates, Printf, Statistics
img = PNGFiles.load("p9-000.png")
R = Float64.(red.(img)) .* 255; G = Float64.(green.(img)) .* 255; B = Float64.(blue.(img)) .* 255
chroma = max.(R, G, B) .- min.(R, G, B)
pink = (R .> 240) .& (abs.(G .- B) .< 25) .& (G .> 150)
mask = falses(size(img))
mask[840:1480, 100:1400] .= ((chroma .> 60) .& .!pink)[840:1480, 100:1400]
# Event lines are vertical: drop columns that are mostly masked.
for x in axes(mask, 2)
    count(mask[840:1480, x]) > 100 && (mask[:, x] .= false)
end
# Connected components (4-neighbor flood fill).
label = zeros(Int, size(mask)); n = 0; comps = Vector{Vector{CartesianIndex{2}}}()
for I in CartesianIndices(mask)
    (mask[I] && label[I] == 0) || continue
    push!(comps, CartesianIndex{2}[]); n = length(comps); stack = [I]; label[I] = n; pix = CartesianIndex{2}[]
    while !isempty(stack)
        J = pop!(stack); push!(pix, J)
        for d in (CartesianIndex(1, 0), CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, -1))
            K = J + d
            checkbounds(Bool, mask, K) && mask[K] && label[K] == 0 && (label[K] = n; push!(stack, K))
        end
    end
    append!(comps[n], pix)
end
yrows = [900.5, 1018.5, 1136.0, 1254.0, 1371.5, 1489.5]; yvals = [10.0, 8, 6, 4, 2, 0]
ya = cov(yrows, yvals) / var(yrows); yb = mean(yvals) - ya * mean(yrows)
days_span = Dates.value(Date(2025, 1, 1) - Date(2023, 11, 1))
col2day(x) = (x - 128.5) / (1312 - 128.5) * days_span
bars = []
for pix in comps
    xs = [p[2] for p in pix]; ys = [p[1] for p in pix]
    maximum(xs) - minimum(xs) >= 15 || continue
    push!(bars, (col2day(minimum(xs)), col2day(maximum(xs)), ya * mean(ys) + yb))
end
sort!(bars)
open("fig6b_digitized_rates.csv", "w") do io
    println(io, "start_date,end_date,rate_m3_per_s")
    for (d0, d1, r) in bars
        @printf(io, "%s,%s,%.2f\n", Date(2023, 11, 1) + Day(round(Int, d0)), Date(2023, 11, 1) + Day(round(Int, d1)), r)
    end
end
print(read("fig6b_digitized_rates.csv", String))
