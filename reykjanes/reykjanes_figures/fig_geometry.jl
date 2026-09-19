# Figure: Level 1 cross-rift model domain (a) and zoom on the reservoir and dikes (b).
# Pictorial sketch: depths are illustrative and dike and aureole widths are exaggerated.
include("common.jl")

const C_BRITTLE = "#f3f2ee"
const C_VISCOUS = "#e4e2d9"
const C_AUREOLE = "#eb6834"

ellipse(cx, cz, a, b; n = 120) = [Point2f(cx + a * cos(t), cz + b * sin(t)) for t in range(0, 2π, length = n)]
rect(x0, z0, x1, z1) = Point2f[(x0, z0), (x1, z0), (x1, z1), (x0, z1)]

fig = Figure(size = (720, 330))
BDT = 8.0                       # illustrative depth of the brittle-ductile transition (km)
XW = 6.0                        # half-width of the zoom box (km)

# ---------------------------------------------------------------- a: overview
gl = GridLayout(fig[1, 1]; valign = :top)
axa = Axis(gl[1, 1]; width = 390, height = 390 * 27.2 / 66, yreversed = true, title = "a   Model domain, cross-rift section",
           limits = (-33, 33, -3.2, 24.0), ylabel = "depth (km)", yticks = [0, 10, 20])
hidexdecorations!(axa); hidespines!(axa, :t, :r, :b)
poly!(axa, rect(-25, 0, 25, BDT); color = C_BRITTLE)
poly!(axa, rect(-25, BDT, 25, 20); color = C_VISCOUS)
lines!(axa, [-25, 25], [BDT, BDT]; color = MUTED, linewidth = 1.2)
lines!(axa, [-25, 25], [0, 0]; color = INK, linewidth = 2.5)                 # free surface
lines!(axa, [-25, 25], [20, 20]; color = INK2, linewidth = 2.5)              # base
lines!(axa, [-25, -25], [0, 20]; color = BASE, linewidth = 1.2)
lines!(axa, [25, 25], [0, 20]; color = BASE, linewidth = 1.2)
text!(axa, 0, -0.7; text = "free surface", color = INK2, fontsize = 10.5, align = (:center, :bottom))
text!(axa, 24.5, 3.9; text = "brittle crust:\nplastic yielding", color = INK2, fontsize = 10.5, align = (:right, :center))
text!(axa, 24.5, 13.5; text = "viscous crust:\ncreep", color = INK2, fontsize = 10.5, align = (:right, :center))
text!(axa, -24.5, BDT - 0.55; text = "brittle–ductile\ntransition", color = INK2, fontsize = 10.5, align = (:left, :bottom))
text!(axa, 0, 21.0; text = "base: free slip or Winkler support (to be tested)", color = INK2, fontsize = 10.5, align = (:center, :top))

# regional extension
arrow!(axa, (-25.6, 10), (-30.4, 10); color = INK, lw = 2, head = 1.7)
arrow!(axa, (25.6, 10), (30.4, 10); color = INK, lw = 2, head = 1.7)
text!(axa, -28.0, 8.6; text = L"V_\mathrm{ext}/2", color = INK2, fontsize = 11, align = (:center, :bottom))
text!(axa, 28.0, 8.6; text = L"V_\mathrm{ext}/2", color = INK2, fontsize = 11, align = (:center, :bottom))

# reservoir, dike and zoom box
poly!(axa, ellipse(0, 4.5, 3.0, 0.45); color = (C_MAGMA, 0.35), strokecolor = C_MAGMA, strokewidth = 1.2)
lines!(axa, [0, 0], [4.05, 0.6]; color = C_MAGMA, linewidth = 2)
lines!(axa, [-XW, XW, XW, -XW, -XW], [0, 0, 10, 10, 0]; color = INK, linewidth = 1)
text!(axa, XW + 0.7, 9.7; text = "b", color = INK, fontsize = 12, font = :bold, align = (:left, :bottom))

# ---------------------------------------------------------------- b: zoom
axb = Axis(fig[1, 2]; width = 210, height = 210 * 12.2 / 12.2, yreversed = true, title = "b   Reservoir and dikes", valign = :top,
           limits = (-XW - 0.1, XW + 0.1, -2.0, 10.2), ylabel = "depth (km)", yticks = [0, 2, 4, 6, 8, 10])
hidexdecorations!(axb); hidespines!(axb, :t, :r, :b)
poly!(axb, rect(-XW, 0, XW, BDT); color = C_BRITTLE)
poly!(axb, rect(-XW, BDT, XW, 10); color = C_VISCOUS)
lines!(axb, [-XW, XW], [BDT, BDT]; color = MUTED, linewidth = 1.2)

# graded mesh hairlines, refined toward the dikes and the reservoir
xg = sort(unique(vcat(0.0, [s * 0.14 * 1.55^k for s in (-1, 1) for k in 0:9])))
xg = filter(x -> abs(x) <= XW, xg)
zg = sort(unique(vcat([4.5 + s * 0.13 * 1.5^k for s in (-1, 1) for k in 0:8], 0.0)))
zg = filter(z -> 0 <= z <= 10, zg)
for x in xg; lines!(axb, [x, x], [0, 10]; color = (BASE, 0.55), linewidth = 0.6); end
for z in zg; lines!(axb, [-XW, XW], [z, z]; color = (BASE, 0.55), linewidth = 0.6); end
lines!(axb, [-XW, XW], [0, 0]; color = INK, linewidth = 2.5)
text!(axb, -XW + 0.2, -0.35; text = "free surface", color = INK2, fontsize = 10.5, align = (:left, :bottom))

# thermal aureole around the active dike, then the dikes themselves
poly!(axb, rect(-0.34, 0.25, 0.34, 4.2); color = (C_AUREOLE, 0.32))
lines!(axb, [-0.5, -0.5], [4.05, 1.3]; color = INK2, linewidth = 3)
lines!(axb, [0.5, 0.5], [4.05, 0.9]; color = INK2, linewidth = 3)
lines!(axb, [0.0, 0.0], [4.05, 0.45]; color = C_MAGMA, linewidth = 3.5)

# reservoir and feed
poly!(axb, ellipse(0, 4.5, 3.0, 0.45); color = (C_MAGMA, 0.35), strokecolor = C_MAGMA, strokewidth = 1.2)
text!(axb, 0, 4.5; text = "reservoir (sill)", color = INK, fontsize = 10.5, align = (:center, :center))
arrow!(axb, (0, 9.7), (0, 5.15); color = C_MAGMA, lw = 2.5, head = 0.55)
text!(axb, 0.3, 7.6; text = L"\dot{V}_\mathrm{m}", color = INK2, fontsize = 12, align = (:left, :center))

# clamping arrows and labels (anchors chosen so labels do not touch arrows or leaders)
arrow!(axb, (-3.2, 2.7), (-1.5, 2.7); color = INK2, lw = 1.8, head = 0.5)
arrow!(axb, (3.2, 2.7), (1.5, 2.7); color = INK2, lw = 1.8, head = 0.5)
text!(axb, 2.35, 2.35; text = "clamping", color = INK2, fontsize = 10.5, align = (:center, :bottom))
text!(axb, -3.0, 0.75; text = "frozen\ndikes", color = INK2, fontsize = 10.5, align = (:center, :center))
lines!(axb, [-2.0, -0.62], [0.95, 1.4]; color = INK2, linewidth = 0.8)
text!(axb, 3.4, -0.55; text = "active dike\nand aureole", color = INK2, fontsize = 10.5, align = (:center, :bottom))
lines!(axb, [2.0, 0.1], [-0.35, 0.4]; color = INK2, linewidth = 0.8)

# ---------------------------------------------------------------- legend
els = [PolyElement(color = (C_MAGMA, 0.35), strokecolor = C_MAGMA, strokewidth = 1.2),
       LineElement(color = C_MAGMA, linewidth = 3), LineElement(color = INK2, linewidth = 3),
       PolyElement(color = (C_AUREOLE, 0.32)), PolyElement(color = (BASE, 0.6))]
Legend(gl[2, 1], els, ["reservoir", "active dike", "frozen dike", "thermal aureole", "graded mesh"];
       orientation = :horizontal, tellwidth = false, nbanks = 2, colgap = 16, rowgap = 1, patchsize = (18, 8))
Label(fig[2, 1:2], "Pictorial sketch: depths are illustrative, dike and aureole widths are exaggerated. Domain width 40–60 km, depth 15–25 km.";
      color = MUTED, fontsize = 10.5, halign = :left, tellwidth = false)
rowgap!(gl, 1, 4); rowgap!(fig.layout, 1, 8)
resize_to_layout!(fig)

save_fig("fig_geometry", fig)
