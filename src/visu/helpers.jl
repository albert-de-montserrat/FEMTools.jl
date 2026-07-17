# Makie-based visualization helpers for mixed T7/P1-disc Stokes results.
#
# These methods extend the stub functions declared in `FEMTools`; they are
# included by `FEMToolsMakieExt`, so the Makie names (`Figure`, `poly!`, …) and
# `Point2f` resolve from that extension's imports. They are callable as
# `FEMTools.plot_summary(...)` once a Makie backend is loaded.

# Corner-node average of a nodal field over element `i`'s connectivity column.
_corner_mean(field, conn, i) = sum(field[conn[c, i]] for c in axes(conn, 1)) / size(conn, 1)

"""
    plotting_triangles(coords_v, el2nP) -> Vector{Vector{Point2f}}

Corner-triangle polygons for `poly!`, one per element, built from the P1 corner
connectivity `el2nP` (three rows) and vertex coordinates `coords_v`.
"""
function FEMTools.plotting_triangles(coords_v, el2nP)
    pts = [Point2f(c) for c in coords_v]
    return [
        [pts[el2nP[1, i]], pts[el2nP[2, i]], pts[el2nP[3, i]]]
        for i in axes(el2nP, 2)
    ]
end

# Dashed white outline of the material inclusion. A rectangular inclusion is
# `(xlo, xhi, ylo, yhi)`; a circle is `(; cx, cy, r)` and an ellipse is
# `(; cx, cy, rx, ry)`.
function _draw_inclusion!(ax, incl::Tuple)
    xlo, xhi, ylo, yhi = incl
    lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo];
        color = :white, linewidth = 1.5, linestyle = :dash)
    return nothing
end

function _draw_inclusion!(ax, incl::NamedTuple)
    rx = hasproperty(incl, :r) ? incl.r : incl.rx
    ry = hasproperty(incl, :r) ? incl.r : incl.ry
    if hasproperty(incl, :conduit_half_width)
        θr = acos(incl.conduit_half_width / rx)
        θ = range(π - θr, 2π + θr; length = 100)
        x = vcat(incl.cx .+ rx .* cos.(θ),
            incl.cx + incl.conduit_half_width,
            incl.cx - incl.conduit_half_width,
            incl.cx - incl.conduit_half_width)
        y = vcat(incl.cy .+ ry .* sin.(θ),
            incl.conduit_top, incl.conduit_top, incl.conduit_bottom)
    else
        θ = range(0, 2π; length = 100)
        x, y = incl.cx .+ rx .* cos.(θ), incl.cy .+ ry .* sin.(θ)
    end
    lines!(ax, x, y;
        color = :white, linewidth = 1.5, linestyle = :dash)
    return nothing
end

"""
    draw_geometry_boxes!(ax, incl, obs_bounds)

Overlay the material inclusion (white dashed outline) and the
objective/observation region (yellow solid box, `(xlo, xhi, ylo, yhi)`) on `ax`.
The inclusion is a rectangle `(xlo, xhi, ylo, yhi)`, circle `(; cx, cy, r)`,
or ellipse `(; cx, cy, rx, ry)`. Sharing this overlay keeps the figures
visually consistent.
"""
function FEMTools.draw_geometry_boxes!(ax, incl, obs_bounds)
    _draw_inclusion!(ax, incl)
    obs_xlo, obs_xhi, obs_ylo, obs_yhi = obs_bounds
    lines!(ax,
        [obs_xlo, obs_xhi, obs_xhi, obs_xlo, obs_xlo],
        [obs_ylo, obs_ylo, obs_yhi, obs_yhi, obs_ylo];
        color = :yellow, linewidth = 2)
    return nothing
end

"""
    plot_fields(vx_dofs, vy_dofs, P_dofs, titles, coords_v, el2nP, DoFsP,
                incl_bounds, obs_bounds) -> Figure

Three side-by-side element-averaged maps of `vx`, `vy`, and `P` on the current
mesh, each with the inclusion and observation overlays. `titles` gives the three
panel titles; velocities average over the P1 corners (`el2nP`) and pressure over
the discontinuous pressure DoFs (`DoFsP`).
"""
function FEMTools.plot_fields(vx_dofs, vy_dofs, P_dofs, titles,
        coords_v, el2nP, DoFsP, incl_bounds, obs_bounds)
    polys = FEMTools.plotting_triangles(coords_v, el2nP)
    el_vx = [_corner_mean(vx_dofs, el2nP, i) for i in axes(el2nP, 2)]
    el_vy = [_corner_mean(vy_dofs, el2nP, i) for i in axes(el2nP, 2)]
    el_P  = [_corner_mean(P_dofs, DoFsP, i) for i in axes(DoFsP, 2)]
    fig = Figure(size = (1400, 440))
    for (col, title, values, colormap) in (
            (1, titles[1], el_vx, :vik),
            (3, titles[2], el_vy, :vik),
            (5, titles[3], el_P,  :glasgow),
        )
        limits = extrema(values)
        ax = Axis(fig[1, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
        poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
        Colorbar(fig[1, col + 1]; colormap, limits, width = 15, tellheight = false)
        FEMTools.draw_geometry_boxes!(ax, incl_bounds, obs_bounds)
    end
    display(fig)
    return fig
end

"""
    plot_summary(density_sensitivity, viscosity_sensitivity, G_sensitivity,
                 K_sensitivity, fwd_hist, adj_hist, coords_v, el2nP,
                 element_area, incl_bounds, obs_bounds) -> Figure

Four per-area material-sensitivity maps (`ρ`, `η`, `G`, `K`) over the current
mesh plus the forward and adjoint residual traces. The sensitivities are raw
element integrals; dividing by `element_area` yields the mesh-independent
sensitivity density, free of the per-element speckle the area-weighted integrals
show on an irregular mesh.
"""
function FEMTools.plot_summary(density_sensitivity, viscosity_sensitivity,
        G_sensitivity, K_sensitivity, fwd_hist, adj_hist,
        coords_v, el2nP, element_area, incl_bounds, obs_bounds;
        log_scaled = false, Q_sensitivity = nothing)
    polys = FEMTools.plotting_triangles(coords_v, el2nP)
    fig = Figure(size = (1100, isnothing(Q_sensitivity) ? 900 : 1100))
    label = log_scaled ? "dlnJ/dln" : "dJ/d"
    fields = [
        (1, 1, "$(label)ρ (per area)", density_sensitivity ./ element_area, :vik),
        (1, 3, "$(label)η (per area)", viscosity_sensitivity ./ element_area, :vik),
        (2, 1, "$(label)G (per area)", G_sensitivity ./ element_area, :vik),
        (2, 3, "$(label)K (per area)", K_sensitivity ./ element_area, :vik),
    ]
    isnothing(Q_sensitivity) || push!(fields,
        (3, 1, "$(label)Q (per area)", Q_sensitivity ./ element_area, :vik))
    common_limit = max(maximum(maximum(abs, field[4]) for field in fields), eps())
    for (row, col, title, values, colormap) in fields
        limits = log_scaled ? (-common_limit, common_limit) : extrema(values)
        ax = Axis(fig[row, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
        poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
        Colorbar(fig[row, col + 1]; colormap, limits, width = 15, tellheight = false)
        FEMTools.draw_geometry_boxes!(ax, incl_bounds, obs_bounds)
    end

    residual_row = isnothing(Q_sensitivity) ? 3 : 4
    ax = Axis(fig[residual_row, 1:4]; xlabel = "iteration",
        ylabel = "log10 relative residual", title = "Relative residual evolution")
    logres(v) = log10.(max.(v, eps()))
    if !isempty(fwd_hist)
        it = Float64[h.iter for h in fwd_hist]
        lines!(ax, it, logres([h.err_v_rel for h in fwd_hist]); label = "forward V")
        lines!(ax, it, logres([h.err_P_rel for h in fwd_hist]); label = "forward P")
    end
    if !isempty(adj_hist)
        it = Float64[h.iter for h in adj_hist]
        # The PH/DYREL adjoint records a combined velocity residual `err_v`;
        # the experimental coupled solver records `err_vx`/`err_vy` separately.
        adj_v = [hasproperty(h, :err_v_rel) ? h.err_v_rel : h.err_rel for h in adj_hist]
        lines!(ax, it, logres(adj_v); label = "adjoint V", linestyle = :dash)
        adj_P = [hasproperty(h, :err_P_rel) ? h.err_P_rel : h.err_rel for h in adj_hist]
        lines!(ax, it, logres(adj_P); label = "adjoint P", linestyle = :dash)
    end
    (isempty(fwd_hist) && isempty(adj_hist)) || axislegend(ax; position = :rt)

    display(fig)
    return fig
end
