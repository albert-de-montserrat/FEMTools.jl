import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GLMakie: Figure, Axis, Colorbar, Label, mesh!, lines!, save,
    Point2f, DataAspect, NoShading

include(joinpath(@__DIR__, "stokes_2D_compressible_inclusion.jl"))

# Sub-triangles of a T7 element, spanning its corner (1-3) and midside (4-6)
# nodes. The bubble node carries no vertex of the rendered triangulation.
const T7_SUBFACES = ((1, 4, 6), (4, 2, 5), (6, 5, 3), (4, 5, 6))

"""
    render_faces_v(el2n) -> Matrix{Int}

Triangulation of the velocity mesh for rendering, as a `nfaces × 3` matrix of
node indices. Splitting every element into [`T7_SUBFACES`](@ref) draws the
midside nodes as vertices, so a linearly interpolated picture follows the
quadratic field between corners instead of cutting across it.
"""
function render_faces_v(el2n)
    faces = Matrix{Int}(undef, length(T7_SUBFACES) * size(el2n, 2), 3)
    k = 0
    for iel in axes(el2n, 2), f in T7_SUBFACES
        k += 1
        faces[k, 1], faces[k, 2], faces[k, 3] = el2n[f[1], iel], el2n[f[2], iel], el2n[f[3], iel]
    end
    return faces
end

"""
    log10_error(e) -> Vector{Float64}

Base-10 logarithm of the elementwise errors `e`, with exact zeros raised to 16
decades below the largest entry. `-Inf` would otherwise collapse the colour
range of the error panels onto a single point.
"""
function log10_error(e)
    emax = maximum(e)
    floor_ = emax > 0 ? 1.0e-16 * emax : one(emax)
    return [log10(max(x, floor_)) for x in e]
end

"""
    field_panel!(fig, pos, pts, faces, vals, circle; title, colormap, colorrange)
        -> plot

Draw one nodal field on the unstructured triangulation `(pts, faces)` at grid
position `pos`, with the inclusion outline `circle` overlaid.
"""
function field_panel!(
        fig, pos, pts, faces, vals, circle;
        title, colormap, colorrange,
    )
    ax = Axis(fig[pos...]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
    h = mesh!(ax, pts, faces; color = vals, colormap, colorrange, shading = NoShading)
    lines!(ax, circle[1], circle[2]; color = :white, linewidth = 1.2, linestyle = :dash)
    return h
end

"""
    field_row!(fig, row, pts, faces, num, anal, circle; name) -> nothing

Fill one row of the comparison figure with the numerical field, the analytical
field, and the error between them.

The two solution panels share a colour range so that a difference between them
is visible as a difference in colour. That range is symmetric about zero
because `:vik` is a diverging map, whose neutral colour must sit on the sign
change of the field. The error panel gets its own logarithmic range, which
spans several decades between the interface and the far field.
"""
function field_row!(fig, row, pts, faces, num, anal, circle; name)
    amax = max(maximum(abs, num), maximum(abs, anal))
    clims = (-amax, amax)
    field_panel!(
        fig, (row, 1), pts, faces, num, circle;
        title = "$name  (numerical)", colormap = :vik, colorrange = clims,
    )
    field_panel!(
        fig, (row, 2), pts, faces, anal, circle;
        title = "$name  (analytical)", colormap = :vik, colorrange = clims,
    )
    Colorbar(fig[row, 3]; colormap = :vik, limits = clims, label = name, width = 12)

    logerr = log10_error(abs.(num .- anal))
    elims = extrema(logerr)
    field_panel!(
        fig, (row, 4), pts, faces, logerr, circle;
        title = "$name  error", colormap = :magma, colorrange = elims,
    )
    Colorbar(
        fig[row, 5]; colormap = :magma, limits = elims,
        label = "log₁₀|Δ$name|", width = 12,
    )
    return nothing
end

"""
    plot_compressible_inclusion(out; show_plot = true, savepath = nothing) -> Figure

Compare the computed velocity and pressure of the compressible host-inclusion
benchmark with the analytical solution, one field per row.

`out` is the result of [`main`](@ref). Velocity is continuous and is drawn on
the velocity nodes; pressure is P1-disc, so it is drawn on a triangulation of
its own per-element degrees of freedom, which keeps the jumps across element
edges visible rather than averaging them away.
"""
function plot_compressible_inclusion(out; show_plot = true, savepath = nothing)
    (; dr, mesh_stokes, centre, params) = out
    coords = Array(mesh_stokes.coords)
    el2n = Array(mesh_stokes.el2n)
    dofsP = Array(mesh_stokes.DoFsP)

    vx, vy, P = Array(dr.vx), Array(dr.vy), Array(dr.P)
    sol_v = [analytic_at(c, centre, params) for c in coords]
    vx_a = [s.V[1] for s in sol_v]
    vy_a = [s.V[2] for s in sol_v]

    pts_v = [Point2f(c) for c in coords]
    faces_v = render_faces_v(el2n)

    # Pressure DoFs duplicate the element corners, so each element's linear
    # pressure is rendered on its own triangle.
    pts_P = Vector{Point2f}(undef, mesh_stokes.nnodesP)
    P_a = Vector{Float64}(undef, mesh_stokes.nnodesP)
    for iel in axes(dofsP, 2), a in axes(dofsP, 1)
        c = coords[el2n[a, iel]]
        pts_P[dofsP[a, iel]] = Point2f(c)
        P_a[dofsP[a, iel]] = analytic_at(c, centre, params).p
    end
    faces_P = Matrix{Int}(permutedims(dofsP))

    θ = range(0, 2π, 300)
    circle = (centre[1] .+ params.ri .* cos.(θ), centre[2] .+ params.ri .* sin.(θ))

    fig = Figure(size = (1500, 1200))
    Label(
        fig[0, 1:5],
        @sprintf(
            "Compressible inclusion, T7/P1-disc: nels=%d, ‖Δv‖/‖v‖=%.3e, ‖ΔP‖/‖P‖=%.3e",
            out.nels, out.rel_v, out.rel_p
        );
        fontsize = 20,
    )
    field_row!(fig, 1, pts_v, faces_v, vx, vx_a, circle; name = "vx")
    field_row!(fig, 2, pts_v, faces_v, vy, vy_a, circle; name = "vy")
    field_row!(fig, 3, pts_P, faces_P, P, P_a, circle; name = "P")

    show_plot && display(fig)
    isnothing(savepath) || save(savepath, fig)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = main(; max_area = 1 / 32^2)
    plot_compressible_inclusion(
        out;
        show_plot = false,
        savepath = joinpath(@__DIR__, "stokes_2D_compressible_inclusion.png"),
    )
end
