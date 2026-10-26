# Shared style of figures2D.jl and figures3D.jl.

# Reference palette (dataviz skill, references/palette.md): ordinal blue ramp for
# station distance, categorical slots in fixed order for station subsets.
const DISTANCE_RAMP = ("#104281", "#256abf", "#5598e7", "#86b6ef")
const SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
const INK, MUTED = "#0b0b0b", "#52514e"

paper_theme() = Theme(; fontsize = 11, figure_padding = 8,
    Axis = (; xgridvisible = false, ygridcolor = (:black, 0.08), spinewidth = 0.8,
        xtickwidth = 0.8, ytickwidth = 0.8, titlealign = :left, titlesize = 11, titlefont = :regular),
    Lines = (; linewidth = 2), Scatter = (; markersize = 9))

function save_both(dir, name, fig)
    foreach(ext -> save(joinpath(dir, name * ext), fig), (".png", ".pdf"))
    return joinpath(dir, name * ".png")
end
