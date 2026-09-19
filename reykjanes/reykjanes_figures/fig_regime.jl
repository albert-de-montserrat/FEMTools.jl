# Figure: expected structure of the regime diagram in (De, Pi_h), with the memory-metric shape of each regime.
# Sketch only: boundaries are expected near unity and are to be found in Phase 6.
include("common.jl")
include("level0.jl")

fig = Figure(size = (720, 330))

# ------------------------------------------------------------------ a: regime map
axa = Axis(fig[1, 1]; width = 330, height = 275, title = "a   Expected regimes",
           limits = (-2, 4, -2, 3.6),
           xlabel = L"\mathrm{De} = t_M\,/\,t_\mathrm{recharge}", ylabel = L"\Pi_h = t_h\,/\,t_\mathrm{recharge}",
           xticks = ([-2, 0, 2, 4], [L"10^{-2}", L"1", L"10^{2}", L"10^{4}"]),
           yticks = ([-2, -1, 0, 1, 2, 3], [L"10^{-2}", L"10^{-1}", L"1", L"10", L"10^{2}", L"10^{3}"]))
poly!(axa, Rect2f(-2, -2, 2, 2);   color = (C_NONE, 0.10))
poly!(axa, Rect2f(-2, 0, 2, 3.6);  color = (C_WEAK, 0.16))
poly!(axa, Rect2f(0, -2, 4, 2);    color = (C_CLAMP, 0.16))
poly!(axa, Rect2f(0, 0, 4, 3.6);   color = (C_REV, 0.16))
lines!(axa, [0, 0], [-2, 3.6]; color = INK2, linewidth = 1.2)
lines!(axa, [-2, 4], [0, 0]; color = INK2, linewidth = 1.2)
text!(axa, -1.35, -1.0; text = "no\nmemory", color = INK, fontsize = 10.5, align = (:center, :center))
text!(axa, -1.35, 1.8;  text = "weak-\nening", color = INK, fontsize = 10.5, align = (:center, :center))
text!(axa, 3.2, -1.0;   text = "clamping", color = INK, fontsize = 10.5, align = (:center, :center))
text!(axa, 3.2, 1.8;    text = "both\nchannels", color = INK, fontsize = 10.5, align = (:center, :center))

# Reykjanes from the generic estimates of the plan: De = t_M / t_recharge spans about 0.2 to 200, Pi_h is unconstrained
de_lo, de_hi = log10(0.10 / 0.5), log10(10 / 0.05)
lines!(axa, [de_lo, de_hi, de_hi, de_lo, de_lo], [-2, -2, 3.6, 3.6, -2]; color = INK, linewidth = 1.4)
mid = (de_lo + de_hi) / 2
text!(axa, mid, 3.5; text = "Reykjanes\n(generic estimates)", color = INK, fontsize = 10, font = :bold, align = (:center, :top))
text!(axa, 1.25, -1.9; text = "De: about three decades", color = INK, fontsize = 9.5, align = (:center, :bottom))
lines!(axa, [mid, mid], [-1.25, 2.35]; color = INK2, linewidth = 1.2)
scatter!(axa, [mid, mid], [-1.4, 2.5]; marker = [:dtriangle, :utriangle], markersize = 9, color = INK2)
text!(axa, mid + 0.18, 0.55; text = rich("Π", subscript("h"), " unconstrained"), color = INK, fontsize = 10,
      align = (:center, :center), rotation = π / 2)

# ------------------------------------------------------------------ b: memory metric per regime
gb = GridLayout(fig[1, 2]; valign = :top)
Label(gb[1, 1], "b   Memory metric in each regime"; font = :bold, fontsize = 13, halign = :left, tellwidth = false)
panels = [
    (scenario_h0,    C_NONE,  rich("no memory: M", subscript("n"), " = 0")),
    (scenario_weak,  C_WEAK,  rich("weakening: M", subscript("n"), " < 0")),
    (scenario_clamp, C_CLAMP, rich("clamping: M", subscript("n"), " > 0")),
    (scenario_rev,   C_REV,   rich("both channels: sign set by the amplitudes; reversal possible")),
]
for (i, (s, col, ttl)) in enumerate(panels)
    last = i == length(panels)
    ax = Axis(gb[i + 1, 1]; width = 300, height = 50, limits = (0.7, 12.3, -0.5, 0.78),
              xlabel = rich("intrusion number ", rich("n", font = :italic)),
              xlabelvisible = last, xticklabelsvisible = last, xticksvisible = last, xticks = [1, 4, 8, 12],
              yticklabelsvisible = false, yticksvisible = false, leftspinevisible = false, bottomspinevisible = last)
    hlines!(ax, 0.0; color = BASE, linewidth = 1)
    lines!(ax, 1:12, s.M; color = col, linewidth = 2.2)
    scatter!(ax, 1:12, s.M; color = col, markersize = 5.5)
    text!(ax, 0.9, 0.76; text = ttl, color = INK2, fontsize = 10.5, align = (:left, :top))
end
rowgap!(gb, 1, 4)
for r in 2:4; rowgap!(gb, r, 4); end

Label(fig[2, 1:2], rich("Sketch: boundaries are expected near unity and are to be found in Phase 6. The thermal group Π", subscript("T"), " adds a third axis.");
      color = MUTED, fontsize = 10.5, halign = :left, tellwidth = false)
colgap!(fig.layout, 1, 26); rowgap!(fig.layout, 1, 6)
resize_to_layout!(fig)
save_fig("fig_regime", fig)
