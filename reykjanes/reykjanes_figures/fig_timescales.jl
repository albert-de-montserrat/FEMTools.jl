# Figure: generic timescales of the memory channels against the recharge interval.
# Values are the generic estimates of the plan (kappa = 1e-6 m^2/s, G = 30 GPa, 2 cm/yr spreading).
include("common.jl")
using Statistics: mean

yr = 3.156e7
κ = 1e-6; G = 3e10
lg(x) = log10(x)

dike_freeze = (lg(1^2 / κ / yr), lg(5^2 / κ / yr))          # w = 1-5 m
maxwell     = (lg(1e17 / G / yr), lg(1e19 / G / yr))        # eta = 1e17-1e19 Pa s
heal        = (lg(0.01), lg(100))                            # unconstrained
diff_host   = (lg(10^2 / κ / yr), lg(100^2 / κ / yr))       # 10-100 m
diff_halo   = (lg(100^2 / κ / yr), lg(1000^2 / κ / yr))     # 0.1-1 km
reload_1m   = lg(1.0 / 0.02)                                 # 1 m of opening at 2 cm/yr
recharge    = (lg(0.05), lg(0.5))                            # weeks to months
sequence    = lg(5.0)

rows = [
    ("Dike freezing time",              "w²/κ, w = 1–5 m",                       dike_freeze, C_REV,   :bar),
    ("Maxwell time",                    "η/G, η = 10¹⁷–10¹⁹ Pa s",               maxwell,     C_CLAMP, :bar),
    ("Healing time",                    "unconstrained",                          heal,        C_WEAK,  :open),
    ("Heat diffusion into the host",    "ℓ²/κ, ℓ = 10–100 m",                    diff_host,   C_REV,   :bar),
    ("Heat diffusion around reservoir", "ℓ²/κ, ℓ = 0.1–1 km",                   diff_halo,   C_REV,   :bar),
    ("Tectonic reloading of 1 m opening", "",                                    (reload_1m, reload_1m), INK2, :point),
]

fig = Figure(size = (700, 330))
ax = Axis(fig[1, 1]; yreversed = true, limits = (-2.4, 4.9, -0.85, 6.6),
          xlabel = "time (years, logarithmic scale)",
          xgridvisible = true, xgridcolor = GRID, xgridwidth = 1,
          leftspinevisible = false, yticksvisible = false,
          xticks = ([-1.08, 0, 1, 2, 3, 4], ["1 month", "1 yr", "10 yr", "100 yr", "1 kyr", "10 kyr"]),
          yticks = (1:length(rows), [r[1] for r in rows]),
          yticklabelalign = (:right, :center), yticklabelpad = 8)

# recharge interval band and sequence length, drawn behind the bars
vspan!(ax, recharge...; color = (INK2, 0.13))
vlines!(ax, [recharge...]; color = (INK2, 0.55), linewidth = 1)
text!(ax, mean(recharge), -0.38; text = "recharge\ninterval", color = INK2, fontsize = 11, align = (:center, :center))
vlines!(ax, sequence; color = MUTED, linewidth = 1.2)
text!(ax, sequence + 0.08, -0.38; text = "sequence so far, ≈ 5 yr", color = INK2, fontsize = 11, align = (:left, :center))

h = 0.30
for (i, r) in enumerate(rows)
    lo, hi = r[3]; col = r[4]
    isempty(r[2]) || text!(ax, lo, i - 0.20; text = r[2], color = INK2, fontsize = 10, align = (:left, :bottom))
    if r[5] == :bar
        poly!(ax, Rect2f(lo, i - h / 2, hi - lo, h); color = col)
    elseif r[5] == :open
        poly!(ax, Rect2f(lo, i - h / 2, hi - lo, h); color = (col, 0.14), strokecolor = (col, 0.9), strokewidth = 1.2)
        text!(ax, (lo + hi) / 2, i; text = "?", color = INK2, fontsize = 13, align = (:center, :center))
    else
        scatter!(ax, [lo], [i]; marker = :diamond, markersize = 13, color = col, strokecolor = :white, strokewidth = 1.5)
        text!(ax, lo + 0.14, i; text = "≈ 50 yr at 2 cm/yr", color = INK2, fontsize = 11, align = (:left, :center))
    end
end

# legend: colour = memory channel
els = [PolyElement(color = C_REV), PolyElement(color = C_CLAMP), PolyElement(color = C_WEAK), MarkerElement(marker = :diamond, color = INK2, markersize = 11)]
Legend(fig[2, 1], els, ["thermal", "elastic stress relaxation", "damage healing", "tectonic loading"];
       orientation = :horizontal, tellwidth = false, colgap = 22, patchsize = (14, 10))
Label(fig[3, 1], "Generic estimates; to be replaced by measured values in Phase 0.";
      color = MUTED, fontsize = 10.5, halign = :left, tellwidth = false)
rowgap!(fig.layout, 1, 8); rowgap!(fig.layout, 2, 2)

save_fig("fig_timescales", fig)
