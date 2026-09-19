# Figure: dike-normal stress change around a dike opened by internal overpressure p.
# Westergaard solution for a pressurised crack of half-height a in an infinite elastic plane (plane strain).
# Compression positive, normalised by p. Checked on the mid-height line against the closed form
#   dsig_n(x, 0) = 1 - x^3 / (x^2 + a^2)^(3/2).
include("common.jl")

function dsig_n(x, s)                      # x across the dike, s along the dike, a = 1
    ζ = complex(s, x)
    sr = sqrt(ζ - 1) * sqrt(ζ + 1)
    Z = ζ / sr
    Zp = -1 / sr^3
    return 1 - real(Z) - x * imag(Zp)
end

# self-check against the closed form on the perpendicular bisector
for x in (0.5, 1.0, 2.0, 3.0)
    @assert isapprox(dsig_n(x, 0.0), 1 - x^3 / (x^2 + 1)^1.5; atol = 1e-12)
end

cmap = cgrad(Makie.to_color.([C_WEAK, NEUTRAL, C_MAGMA]))    # tension (blue) - neutral - compression (red)

fig = Figure(size = (720, 330))

# ------------------------------------------------------------ a: map of the stress change
xs = range(-2.2, 2.2, length = 441)
ss = range(-2.2, 2.2, length = 441)
F = [dsig_n(x, s) for x in xs, s in ss]
F = clamp.(F, -1, 1)
axa = Axis(fig[1, 1]; width = 270, height = 270, title = "a   Dike-normal stress change",
           xlabel = "across the dike, x / a", ylabel = "along the dike, z / a",
           xticks = -2:1:2, yticks = -2:1:2, limits = (-2.2, 2.2, -2.2, 2.2))
hm = heatmap!(axa, xs, ss, F; colormap = cmap, colorrange = (-1, 1))
contour!(axa, xs, ss, F; levels = [-0.25, 0.25, 0.5, 0.75], color = (:white, 0.7), linewidth = 0.8)
lines!(axa, [0, 0], [-1, 1]; color = INK, linewidth = 3.5)
scatter!(axa, [0, 0], [-1, 1]; color = :white, strokecolor = INK, strokewidth = 1.5, markersize = 9)
text!(axa, 0.14, 0.0; text = "dike", color = INK, fontsize = 11, align = (:left, :center), rotation = π / 2)
Colorbar(fig[1, 2], hm; height = 270, width = 12,
         ticks = ([-1, 0, 1], ["≤ −1  tension", "0", "1  compression"]), labelsize = 12, ticklabelsize = 10.5,
         tickcolor = BASE, spinewidth = 0, ticklabelcolor = INK2, labelcolor = INK2)

# ------------------------------------------------------------ b, c: profiles
gp = GridLayout(fig[1, 3]; valign = :top)
axb = Axis(gp[1, 1]; width = 230, height = 105, title = "b   Across the dike, mid-height",
           xlabel = "distance from the dike, x / a", ylabel = L"\Delta\sigma_n / p",
           limits = (0, 3, 0, 1.22), xticks = 0:1:3, yticks = 0:0.5:1)
xb = range(0, 3, length = 300)
yb = 1 .- xb .^ 3 ./ (xb .^ 2 .+ 1) .^ 1.5
band!(axb, xb, zeros(length(xb)), yb; color = (C_MAGMA, 0.20))
lines!(axb, xb, yb; color = C_MAGMA, linewidth = 2)
scatter!(axb, [2.0], [1 - 8 / 5^1.5]; color = C_MAGMA, markersize = 8, strokecolor = :white, strokewidth = 1.5)
text!(axb, 2.12, 0.42; text = "0.28 p at 2a", color = INK2, fontsize = 10.5, align = (:left, :center))
text!(axb, 0.1, 1.08; text = "p at the wall", color = INK2, fontsize = 10.5, align = (:left, :bottom))

axc = Axis(gp[2, 1]; width = 230, height = 105, title = "c   Beyond the tip",
           xlabel = "distance beyond the tip, (z − a) / a", ylabel = L"\Delta\sigma_n / p",
           limits = (0, 2, -3, 0.25), xticks = 0:0.5:2, yticks = -3:1:0)
xc = range(0.02, 2, length = 400)
yc = 1 .- (1 .+ xc) ./ sqrt.((1 .+ xc) .^ 2 .- 1)
band!(axc, xc, zeros(length(xc)), yc; color = (C_WEAK, 0.20))
lines!(axc, xc, yc; color = C_WEAK, linewidth = 2)
hlines!(axc, 0.0; color = BASE, linewidth = 1)
text!(axc, 1.95, -1.2; text = "singular at the tip", color = INK2, fontsize = 10.5, align = (:right, :center))
rowgap!(gp, 1, 30)

colgap!(fig.layout, 1, 6); colgap!(fig.layout, 2, 26)
resize_to_layout!(fig)
save_fig("fig_clamping", fig)
