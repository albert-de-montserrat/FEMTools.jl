# Shared style and helpers for the sketch figures of the Reykjanes project plan.
# Run any figure script with:  julia --project=@v1.13 fig_concept.jl
using GLMakie

GLMakie.activate!(visible = false)          # render off-screen, no window

const OUTDIR = @__DIR__

# Chart chrome and ink (documented reference palette)
const INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
const GRID, BASE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"

# Categorical slots 1-3 of the reference palette, one fixed entity per colour in every figure
const C_WEAK  = "#2a78d6"   # weakening (damage)
const C_CLAMP = "#1baf7a"   # clamping (elastic stress)
const C_REV   = "#eb6834"   # reversal / thermal
const C_NONE  = INK2        # no memory, fixed threshold (H0)

# Materials in the pictorial sketches
const C_MAGMA = "#e34948"
const C_YIELD = "#4a3aa7"

set_theme!(Theme(
    fontsize = 12,
    backgroundcolor = :white,
    figure_padding = (8, 10, 6, 6),
    Axis = (
        xgridvisible = false, ygridvisible = false,
        topspinevisible = false, rightspinevisible = false,
        leftspinecolor = BASE, bottomspinecolor = BASE,
        xtickcolor = BASE, ytickcolor = BASE,
        xticklabelcolor = INK2, yticklabelcolor = INK2,
        xlabelcolor = INK2, ylabelcolor = INK2,
        titlecolor = INK, titlealign = :left, titlesize = 13, titlefont = :bold,
        xticklabelsize = 11, yticklabelsize = 11, xlabelsize = 12, ylabelsize = 12,
    ),
    Legend = (framevisible = false, labelcolor = INK2, labelsize = 11.5, padding = (0, 0, 0, 0)),
    Lines = (linewidth = 2,),
))

save_fig(name, fig; ppu = 4) = (save(joinpath(OUTDIR, name * ".png"), fig; px_per_unit = ppu); println("saved ", name, ".png"))

# Straight arrow with a filled triangular head. Use on axes with DataAspect so the head is not distorted.
function arrow!(ax, p0, p1; color = INK2, lw = 1.5, head = 1.0, halfwidth = 0.38)
    a = Point2f(p0[1], p0[2]); b = Point2f(p1[1], p1[2])
    d = b - a; L = sqrt(d[1]^2 + d[2]^2); u = d / L; n = Point2f(-u[2], u[1])
    base = b - head * u
    lines!(ax, Point2f[a, base]; color = color, linewidth = lw)
    tip = Point2f[b, Point2f(base + halfwidth * head * n), Point2f(base - halfwidth * head * n)]
    poly!(ax, tip; color = color, strokewidth = 0)
end

# Empty axis for pictorial sketches
function sketch_axis(pos; kwargs...)
    ax = Axis(pos; aspect = DataAspect(), kwargs...)
    hidedecorations!(ax); hidespines!(ax)
    ax
end
