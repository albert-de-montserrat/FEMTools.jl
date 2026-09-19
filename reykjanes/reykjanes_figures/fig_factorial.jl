# Figure: the 2^4 factorial memory-switch design of the attribution experiments.
include("common.jl")

channels = ["Elastic stress", "Damage", "Temperature", "Frozen-dike geometry"]
ccol = [C_CLAMP, C_WEAK, C_REV, INK2]                    # channel colours as in the other figures

# runs grouped by the number of channels retained
combos = [Int[]]
for k in 1:4
    for c in Iterators.product(ntuple(_ -> (false, true), 4)...)
        sum(c) == k && push!(combos, [i for i in 1:4 if c[i]])
    end
end
# order inside each group: lexicographic on the retained set (single channels come out as S, D, T, G)
sort!(combos, by = c -> (length(c), c))

# x positions with a gap between groups
function place(combos)
    xs = Float64[]; x = 1.0; lastk = 0
    for c in combos
        if length(c) != lastk && !isempty(xs); x += 0.55; end
        push!(xs, x); x += 1.0; lastk = length(c)
    end
    xs
end
xs = place(combos)

fig = Figure(size = (720, 250))
ax = Axis(fig[1, 1]; yreversed = true, limits = (0.3, maximum(xs) + 0.7, -1.15, 5.55),
          yticks = (1:4, channels), yticklabelsize = 11.5, yticksvisible = false,
          leftspinevisible = false, bottomspinevisible = false, xticksvisible = false, xticklabelsvisible = false,
          yticklabelpad = 8)

# group bands and headers
groups = [(0, "none"), (1, "one channel"), (2, "two channels"), (3, "three"), (4, "all")]
for (k, lab) in groups
    idx = findall(c -> length(c) == k, combos)
    lo, hi = xs[first(idx)] - 0.5, xs[last(idx)] + 0.5
    iseven(k) && poly!(ax, Rect2f(lo, 0.45, hi - lo, 4.1); color = (NEUTRAL, 0.9))
    lines!(ax, [lo + 0.08, hi - 0.08], [0.1, 0.1]; color = INK2, linewidth = 1.2)
    text!(ax, (lo + hi) / 2, 0.0; text = lab, color = INK2, fontsize = 10.5, align = (:center, :bottom))
end

# switches: filled = channel retained after each event, open = field reset to its initial value
for (j, c) in enumerate(combos), i in 1:4
    on = i in c
    scatter!(ax, [xs[j]], [i]; markersize = 14,
             color = on ? ccol[i] : :white, strokecolor = on ? :white : BASE, strokewidth = on ? 1.5 : 1.3)
end

# experiment tags below the matrix
tag(j, s) = text!(ax, xs[j], 4.85; text = s, color = INK2, fontsize = 9.5, align = (:center, :top))
tag(1, "Exp. 1")
single = findall(c -> length(c) == 1, combos)          # S, D, T, G -> Exp. 2, 3, 4, 6
for (j, s) in zip(single, ("Exp. 2", "Exp. 3", "Exp. 4", "Exp. 6")); tag(j, s); end
multi = findall(c -> length(c) in (2, 3), combos)
lines!(ax, [xs[first(multi)] - 0.4, xs[last(multi)] + 0.4], [4.8, 4.8]; color = INK2, linewidth = 1.0)
text!(ax, (xs[first(multi)] + xs[last(multi)]) / 2, 4.85; text = "Exp. 7: all pairs and triples (10 runs)", color = INK2,
      fontsize = 9.5, align = (:center, :top))
tag(length(combos), "Exp. 5")

Legend(fig[2, 1], [MarkerElement(marker = :circle, color = INK, markersize = 12),
                   MarkerElement(marker = :circle, color = :white, strokecolor = BASE, strokewidth = 1.3, markersize = 12)],
       ["channel retained after each event", "channel reset to its initial value after each event"];
       orientation = :horizontal, tellwidth = false, colgap = 24)
rowgap!(fig.layout, 1, 4)
resize_to_layout!(fig)
save_fig("fig_factorial", fig)
