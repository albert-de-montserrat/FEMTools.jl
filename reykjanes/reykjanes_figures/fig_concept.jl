# Figure: fixed threshold (H0) against state-dependent threshold, from the Level 0 placeholder law.
include("common.jl")
include("level0.jl")



weak, clamp, rev, h0 = scenario_weak, scenario_clamp, scenario_rev, scenario_h0
println("M_n weakening : ", round.(weak.M;  digits = 2))
println("M_n clamping  : ", round.(clamp.M; digits = 2))
println("M_n reversal  : ", round.(rev.M;   digits = 2))

# Sawtooth of overpressure against time: rises at a constant rate, returns to zero at each event.
function sawtooth(s; tmax)
    t = Float64[0.0]; p = Float64[0.0]
    for k in eachindex(s.tev)
        s.tev[k] > tmax && break
        push!(t, s.tev[k]); push!(p, s.dPc[k])
        push!(t, s.tev[k]); push!(p, 0.0)
    end
    t, p
end

fig = Figure(size = (700, 320))

# --- panel a: overpressure cycles ----------------------------------------------------------
axa = Axis(fig[1, 1]; title = "a   Overpressure cycles",
           xlabel = "time (units of the fixed-threshold recurrence interval)",
           ylabel = L"\Delta P_\mathrm{res}\,/\,\Delta P_\mathrm{crit}^{(1)}",
           limits = (0, 8.3, 0, 1.5), xticks = 0:2:8, yticks = 0:0.5:1)
tmax = 8.0
th0, ph0 = sawtooth(h0, tmax = tmax)
tw, pw = sawtooth(weak, tmax = tmax)
lines!(axa, th0, ph0; color = C_NONE, linewidth = 1.5)
lines!(axa, tw, pw; color = C_WEAK, linewidth = 2)
pk0 = [(t, p) for (t, p) in zip(h0.tev, h0.dPc) if t <= tmax]
pkw = [(t, p) for (t, p) in zip(weak.tev, weak.dPc) if t <= tmax]
scatter!(axa, first.(pk0), last.(pk0); color = C_NONE, markersize = 8, strokecolor = :white, strokewidth = 1.5)
scatter!(axa, first.(pkw), last.(pkw); color = C_WEAK, markersize = 8, strokecolor = :white, strokewidth = 1.5)
hlines!(axa, 1.0; color = BASE, linewidth = 1)
scatter!(axa, [0.22], [1.40]; color = C_NONE, markersize = 8)
text!(axa, 0.42, 1.40; text = "H0: equal peaks, equal spacing", color = INK2, fontsize = 11, align = (:left, :center))
scatter!(axa, [0.22], [1.25]; color = C_WEAK, markersize = 8)
text!(axa, 0.42, 1.25; text = "state-dependent: lower peaks, faster recurrence", color = INK2, fontsize = 11, align = (:left, :center))

# --- panel b: memory metric ------------------------------------------------------------------
axb = Axis(fig[1, 2]; title = "b   Memory metric",
           xlabel = rich("intrusion number ", rich("n", font = :italic)), ylabel = L"M_n",
           limits = (0.5, 17.5, -0.62, 0.62), xticks = [1, 4, 8, 12], yticks = -0.5:0.25:0.5)
ns = 1:12
hlines!(axb, 0.0; color = BASE, linewidth = 1)
# direct labels sit just right of the last point; the two upper ones are nudged apart
for (s, c, lab, dy) in ((h0, C_NONE, "H0", 0.0), (weak, C_WEAK, "weakening", 0.0),
                        (clamp, C_CLAMP, "clamping", 0.035), (rev, C_REV, "reversal", -0.035))
    lines!(axb, ns, s.M; color = c, linewidth = 2)
    scatter!(axb, ns, s.M; color = c, markersize = 7, strokecolor = :white, strokewidth = 1)
    text!(axb, 12.4, s.M[end] + dy; text = lab, color = INK2, fontsize = 11, align = (:left, :center))
end

colsize!(fig.layout, 1, Relative(0.55))

# --- shared legend and note -------------------------------------------------------------------
els = [LineElement(color = C_NONE, linewidth = 2), LineElement(color = C_WEAK, linewidth = 2),
       LineElement(color = C_CLAMP, linewidth = 2), LineElement(color = C_REV, linewidth = 2)]
Legend(fig[2, 1:2], els, ["H0: fixed threshold", "weakening", "clamping", "weakening, then reversal"];
       orientation = :horizontal, tellwidth = false, tellheight = true, colgap = 22, patchsize = (22, 8))
Label(fig[3, 1:2], "Illustrative parameters of the reduced-order placeholder law; not fitted to data.";
      color = MUTED, fontsize = 10.5, halign = :left, tellwidth = false)
rowgap!(fig.layout, 1, 8); rowgap!(fig.layout, 2, 2)

save_fig("fig_concept", fig)
