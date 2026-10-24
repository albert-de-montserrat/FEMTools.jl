# Surface displacement of a 2D tensile dislocation (Okada, infinite strike) at
# several dips against the 2D Mogi line source: one of equal volume at the
# dislocation center and the best-fitting one in depth and strength.
#
#   julia --project=examples examples/stokes/volcano/elastic_sources2D_plot.jl

using GLMakie
using LinearAlgebra: dot, norm
using Printf

include(joinpath(@__DIR__, "elastic_dike2D.jl"))

"""
    best_line_source(x, uz; depths = 0.5e3:0.1e3:20e3, ν = 0.25) -> (; area_change, depth, misfit)

Line source under `x = 0` that best fits `uz` in least squares: depth by grid
search, strength by linear least squares at each depth.
"""
function best_line_source(x, uz; depths = 0.5e3:0.1e3:20.0e3, ν = 0.25)
    axes(x) == axes(uz) || throw(DimensionMismatch("x and uz must match: $(axes(x)) vs $(axes(uz))"))
    best = (; area_change = NaN, depth = NaN, misfit = Inf)
    for depth in depths
        g = [line_source_surface(xi; area_change = 1.0, depth, ν)[2] for xi in x]
        a = dot(uz, g) / dot(g, g)
        misfit = norm(uz .- a .* g) / norm(uz)
        misfit < best.misfit && (best = (; area_change = a, depth, misfit))
    end
    return best
end

function plot_elastic_sources(; opening = 1.0, center_depth = 5.0e3, length = 6.0e3,
        dips = (90.0, 60.0, 30.0, 0.0), half_width = 30.0e3, ν = 0.25)
    x = range(-half_width, half_width; length = 601)
    fig = Figure(size = (1500, 350 * Base.length(dips)))
    for (row, dip) in enumerate(dips)
        okada = [tensile_dislocation_surface(xi; opening, center_depth, length, dip) for xi in x]
        uz, ux = last.(okada), first.(okada)
        equal = [line_source_surface(xi; area_change = opening * length, depth = center_depth, ν) for xi in x]
        fit = best_line_source(x, uz; ν)
        fitted = [line_source_surface(xi; area_change = fit.area_change, depth = fit.depth, ν) for xi in x]
        for (col, (name, okada_u, k)) in enumerate((("uz", uz, 2), ("ux", ux, 1)))
            ax = Axis(fig[row, col]; xlabel = "x [km]", ylabel = "$name / opening",
                title = @sprintf("dip %g°: %s", dip, name))
            lines!(ax, x ./ 1.0e3, okada_u ./ opening; color = :black, label = "Okada")
            lines!(ax, x ./ 1.0e3, getindex.(equal, k) ./ opening; linestyle = :dash, label = "Mogi, equal volume at center")
            lines!(ax, x ./ 1.0e3, getindex.(fitted, k) ./ opening; label = @sprintf("Mogi best fit: %.1f km, ΔA/ΔV = %.2f, misfit %.2f",
                fit.depth / 1.0e3, fit.area_change / (opening * length), fit.misfit))
            col == 1 && axislegend(ax; position = :rt, labelsize = 11)
        end
    end
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    out_dir = joinpath(@__DIR__, "output_elastic_sources")
    mkpath(out_dir)
    save(joinpath(out_dir, "okada_vs_mogi.png"), plot_elastic_sources())
end
