# Figure: the event protocol of the Level 1 model as a five-step storyboard.
include("common.jl")

const C_BRITTLE = "#f3f2ee"
const C_AUREOLE = "#eb6834"

ellipse(cx, cz, a, b; n = 100) = [Point2f(cx + a * cos(t), cz + b * sin(t)) for t in range(0, 2π, length = n)]
rect(x0, z0, x1, z1) = Point2f[(x0, z0), (x1, z0), (x1, z1), (x0, z1)]

PW = 122                                      # panel width in layout units
PH = PW * 7.2 / 6                             # panel height for x in [-3, 3], z in [-0.9, 6.3]
fig = Figure(size = (690, 340))

function panel!(col, k)
    ax = Axis(fig[1, col]; width = PW, height = PH, yreversed = true, limits = (-3, 3, -0.9, 6.3))
    hidedecorations!(ax); hidespines!(ax)
    poly!(ax, rect(-3, 0, 3, 6.3); color = C_BRITTLE)
    lines!(ax, [-3, 3], [0, 0]; color = INK, linewidth = 2.5)
    scatter!(ax, [-2.45], [-0.45]; color = INK, markersize = 17)
    text!(ax, -2.45, -0.45; text = string(k), color = :white, fontsize = 11, font = :bold, align = (:center, :center))
    ax
end

sill(ax; fill = (C_MAGMA, 0.35), b = 0.38) = poly!(ax, ellipse(0, 4.4, 2.1, b); color = fill, strokecolor = C_MAGMA, strokewidth = 1.2)
olddikes(ax) = (lines!(ax, [-0.5, -0.5], [4.02, 1.6]; color = INK2, linewidth = 2.5);
                lines!(ax, [0.5, 0.5], [4.02, 1.2]; color = INK2, linewidth = 2.5))

# 1 recharge and relaxation
a1 = panel!(1, 1)
olddikes(a1); sill(a1; fill = (C_MAGMA, 0.60))
arrow!(a1, (0, 6.1), (0, 4.95); color = C_MAGMA, lw = 2.2, head = 0.55)
text!(a1, 1.55, 3.3; text = L"\Delta P\ \uparrow", color = INK2, fontsize = 10.5, align = (:center, :center))

# 2 failure path
a2 = panel!(3, 2)
for x in -0.96:0.24:0.96; lines!(a2, [x, x], [0.5, 4.0]; color = (BASE, 0.45), linewidth = 0.5); end
for z in 0.5:0.24:4.0;    lines!(a2, [-0.96, 0.96], [z, z]; color = (BASE, 0.45), linewidth = 0.5); end
sill(a2; fill = (C_MAGMA, 0.35))
px = [0, 0, 0.24, 0.24, 0, 0, -0.24, -0.24, 0, 0, 0.24, 0.24, 0, 0]
pz = [3.78 - 0.24 * (i - 1) for i in 1:14]
for (x, z) in zip(px, pz)
    poly!(a2, rect(x - 0.12, z - 0.12, x + 0.12, z + 0.12); color = (C_YIELD, 0.85))
end

# 3 intrusion
a3 = panel!(5, 3)
poly!(a3, ellipse(0, 4.4, 2.1, 0.38); color = :transparent, strokecolor = BASE, strokewidth = 1.0)
sill(a3; fill = (C_MAGMA, 0.16), b = 0.24)
poly!(a3, rect(-0.11, 0.7, 0.11, 4.15); color = C_MAGMA)
arrow!(a3, (-0.3, 2.4), (-1.0, 2.4); color = INK2, lw = 1.6, head = 0.32)
arrow!(a3, (0.3, 2.4), (1.0, 2.4); color = INK2, lw = 1.6, head = 0.32)

# 4 heating and material assignment
a4 = panel!(7, 4)
sill(a4; fill = (C_MAGMA, 0.16), b = 0.24)
poly!(a4, rect(-0.45, 0.5, 0.45, 4.2); color = (C_AUREOLE, 0.35))
poly!(a4, rect(-0.11, 0.7, 0.11, 4.15); color = INK2)
text!(a4, 1.85, 2.3; text = L"T = T_\mathrm{m}", color = INK2, fontsize = 10.5, align = (:center, :center))

# 5 log and repeat
a5 = panel!(9, 5)
sill(a5; fill = (C_MAGMA, 0.20), b = 0.28)
for (x, top) in ((-0.55, 1.55), (0.0, 0.7), (0.55, 1.15))
    lines!(a5, [x, x], [4.05, top]; color = INK2, linewidth = 2.5)
end
poly!(a5, rect(-0.34, 0.55, 0.34, 4.2); color = (C_AUREOLE, 0.18))

# arrows between panels
for c in (2, 4, 6, 8)
    g = Axis(fig[1, c]; width = 14, height = PH, limits = (0, 1, 0, 1))
    hidedecorations!(g); hidespines!(g)
    scatter!(g, [0.5], [0.5]; marker = :rtriangle, markersize = 11, color = INK2)
end

# titles and descriptions (plain strings with manual line breaks)
titles = ["Recharge and\nrelaxation", "Failure\npath", "Intrusion\n ", "Heating and\nmaterials", "Log and\nrepeat"]
descr = [
    "Inject at the recharge\nrate while relaxation,\ncooling and healing run\nalongside. Stop at the\nfirst connected failure.\nRecords threshold\noverpressure, volume\nand interval.",
    "Graph search on the\nfailure indicator.\nRecords the path and\nits overlap with the\nprevious path.",
    "Open the path as an\neigenstrain band. Root\nfinding sets its\namplitude to reach the\narrest pressure. The\ntransferred volume is\nan output.",
    "Set the dike to magma\ntemperature, release\nlatent heat, assign\nfrozen-dike properties.",
    "Store the full state\nand the diagnostics,\nthen return to step 1.",
]
for (i, c) in enumerate((1, 3, 5, 7, 9))
    Label(fig[2, c], titles[i]; font = :bold, fontsize = 11, color = INK, halign = :left, valign = :top,
          justification = :left, tellwidth = false, width = PW)
    Label(fig[3, c], descr[i]; fontsize = 9.5, color = INK2, halign = :left, valign = :top, justification = :left,
          tellwidth = false, width = PW)
end

# return arrow: end of step 5 back to step 1
ret = Axis(fig[4, 1:9]; height = 34, limits = (0, 1, 0, 1))
hidedecorations!(ret); hidespines!(ret)
lines!(ret, [0.93, 0.93, 0.07, 0.07], [1.0, 0.25, 0.25, 0.92]; color = INK2, linewidth = 1.6)
scatter!(ret, [0.07], [0.98]; marker = :utriangle, markersize = 11, color = INK2)
text!(ret, 0.5, 0.42; text = "next recharge cycle", color = INK2, fontsize = 10.5, align = (:center, :bottom))

rowgap!(fig.layout, 1, 6); rowgap!(fig.layout, 2, 2); rowgap!(fig.layout, 3, 2)
colgap!(fig.layout, 3)
resize_to_layout!(fig)
save_fig("fig_protocol", fig)
