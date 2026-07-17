
using Printf: @sprintf

"""
    precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) -> Nothing

Fill per-element geometry data on the configured backend.

This wrapper derives the backend from `geo`, launches
`precompute_geometry_kernel!`, then synchronizes before returning.
"""
function precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
    backend = KernelAbstractions.get_backend(geo)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(
        geo, coords, el2n, ∂N∂ξq, ω, Val(N);
        ndrange = nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, half_width=nothing,
                                     radius=nothing, ellipse_half_width=nothing,
                                     ellipse_half_height=nothing, n_circle=48,
                                     max_area=nothing, max_area_at=nothing,
                                     split_background=false,
                                     topography=nothing, n_topo=80) -> Tuple

Build an unstructured T7 velocity mesh around an inclusion.

The inclusion is a rectangle of half-width `half_width`, a circle of radius
`radius`, or an ellipse with the supplied half-width and half-height, discretized
into `n_circle` segments; provide exactly one shape. Triangulate.jl
generates a second-order T6 PSLG mesh with the inclusion as a constrained
internal boundary. The local midpoint ordering is remapped to FEMTools' T6/T7
convention, then one centroid bubble node is appended per element.

For an ellipse, `conduit_half_width` optionally opens its upper arc and joins
the chamber to the topographic surface with two constrained vertical walls.

With `split_background=true`, a horizontal interface at `y = cy` (through the
inclusion centre) is added as a constrained boundary, splitting the background
into an upper and a lower half. The side walls and the inclusion boundary are
split at `y = cy` so the interface joins existing vertices and no element
crosses it — the same conformity the inclusion boundary already provides. A
circular inclusion then requires an even `n_circle`, so that vertices land at
`y = cy` on both sides.

`topography` is an optional `x -> y_top(x)` giving the top-surface height; the
flat top `y = 0` is used when it is `nothing`. It must satisfy
`topography(0) = topography(Lx) = 0` so the side walls stay vertical. The top is
sampled into `n_topo` segments from `x = Lx` to `x = 0`.

`max_area` is a uniform Triangle area bound. `max_area_at` is an optional
`(x, y) -> area` bound evaluated at each candidate triangle's centroid
(Triangle's `-u` switch), enabling spatially graded refinement; both may be
combined. When `max_area_at` is given, the topography polyline is sampled
with the local spacing `√(2 max_area_at(x, top(x)))` instead of `n_topo`
uniform samples, so the fixed top-surface vertices do not pin a fine
resolution where the area bound is coarse.
"""
function build_triangle_t7_inclusion_mesh(;
        Lx, Ly, cx, cy, half_width = nothing, radius = nothing,
        ellipse_half_width = nothing, ellipse_half_height = nothing,
        conduit_half_width = nothing, n_circle = 48,
        max_area = nothing, max_area_at = nothing, split_background = false,
        topography = nothing, n_topo = 80,
    )
    elliptical = !isnothing(ellipse_half_width) || !isnothing(ellipse_half_height)
    elliptical && (isnothing(ellipse_half_width) || isnothing(ellipse_half_height)) &&
        throw(ArgumentError("provide both `ellipse_half_width` and `ellipse_half_height`"))
    count((!isnothing(half_width), !isnothing(radius), elliptical)) == 1 ||
        throw(ArgumentError("provide exactly one rectangle, circle, or ellipse shape"))
    curved = !isnothing(radius) || elliptical
    !isnothing(conduit_half_width) && !elliptical &&
        throw(ArgumentError("`conduit_half_width` requires an elliptical inclusion"))
    !isnothing(conduit_half_width) && isnothing(topography) &&
        throw(ArgumentError("a conduit requires `topography` to locate the surface"))
    !isnothing(conduit_half_width) && conduit_half_width ≥ ellipse_half_width &&
        throw(ArgumentError("conduit half-width must be smaller than ellipse half-width"))
    curved && split_background && isodd(n_circle) &&
        throw(ArgumentError("split_background needs an even `n_circle` for a curved inclusion"))

    top = isnothing(topography) ? (x -> 0.0) : topography
    # A flat top only needs its two corners; a profile needs a refined polyline.
    # With a graded area bound the polyline spacing tracks the local target
    # edge length, so the top vertices stay sparse where the mesh is coarse.
    if isnothing(topography)
        xs_top = [Lx, 0.0]
    elseif isnothing(max_area_at)
        xs_top = collect(range(Lx, 0.0; length = n_topo))
    else
        xs_top = [Lx]
        while (x = xs_top[end]) > 0.0
            h = sqrt(2 * max_area_at(x, top(x)))
            h > 0 || throw(ArgumentError("max_area_at must be positive; got $(max_area_at(x, top(x))) at ($x, $(top(x)))"))
            # Snap the last step to x = 0 rather than leaving a sliver segment.
            push!(xs_top, x - h > h / 2 ? x - h : 0.0)
        end
    end
    if !isnothing(conduit_half_width)
        append!(xs_top, (cx - conduit_half_width, cx + conduit_half_width))
        sort!(unique!(xs_top); rev = true)
    end

    # Outer ring, counterclockwise from the bottom-left corner: up the right
    # wall (split at y=cy when requested), across the top surface from x=Lx to
    # x=0, then down the left wall.
    pts = NTuple{2, Cdouble}[(0.0, -Ly), (Lx, -Ly)]
    split_background && push!(pts, (Lx, cy))
    append!(pts, ((x, top(x)) for x in xs_top))
    split_background && push!(pts, (0.0, cy))
    n_outer = length(pts)
    i_rw = split_background ? 3 : 0          # right-wall interface vertex
    i_lw = split_background ? n_outer : 0    # left-wall interface vertex
    segs = Tuple{Cint, Cint}[(i, i == n_outer ? 1 : i + 1) for i in 1:n_outer]

    # Inclusion boundary. With split_background the boundary is split at y = cy
    # so the background interface joins existing inclusion vertices.
    if curved
        # Circle/ellipse vertices, θ = 0 first (rightmost point at y = cy). An even
        # n_circle then also lands a vertex at θ = π (leftmost point at y = cy).
        rx = elliptical ? ellipse_half_width : radius
        ry = elliptical ? ellipse_half_height : radius
        i0 = length(pts)
        if isnothing(conduit_half_width)
            θ = range(0.0, 2π; length = n_circle + 1)[1:(end - 1)]
            append!(pts, ((cx + rx * cos(t), cy + ry * sin(t)) for t in θ))
            append!(segs, [(i0 + k, i0 + (k == n_circle ? 1 : k + 1)) for k in 1:n_circle])
            i_right = i0 + 1
            i_left = i0 + n_circle ÷ 2 + 1
        else
            θr = acos(conduit_half_width / rx)
            θ = sort!(unique!(vcat(
                collect(range(π - θr, 2π + θr; length = n_circle + 1)),
                [π, 2π],
            )))
            append!(pts, ((cx + rx * cos(t), cy + ry * sin(t)) for t in θ))
            append!(segs, [(i0 + k, i0 + k + 1) for k in 1:(length(θ) - 1)])
            i_left = i0 + argmin(abs.(θ .- π))
            i_right = i0 + argmin(abs.(θ .- 2π))
            i_ellipse_left = i0 + 1
            i_ellipse_right = i0 + length(θ)
            i_top_left = findfirst(p -> p[1] == cx - conduit_half_width, pts)
            i_top_right = findfirst(p -> p[1] == cx + conduit_half_width, pts)
            append!(segs, [(i_ellipse_left, i_top_left), (i_ellipse_right, i_top_right)])
        end
        if split_background
            append!(segs, [(i_lw, i_left), (i_right, i_rw)])
        end
    else
        xlo, xhi = cx - half_width, cx + half_width
        ylo, yhi = cy - half_width, cy + half_width
        push!(pts, (xlo, ylo)); i_bl = length(pts)
        push!(pts, (xhi, ylo)); i_br = length(pts)
        if split_background
            push!(pts, (xhi, cy)); i_rm = length(pts)
        end
        push!(pts, (xhi, yhi)); i_tr = length(pts)
        push!(pts, (xlo, yhi)); i_tl = length(pts)
        if split_background
            push!(pts, (xlo, cy)); i_lm = length(pts)
            append!(segs, [(i_bl, i_br), (i_br, i_rm), (i_rm, i_tr),
                           (i_tr, i_tl), (i_tl, i_lm), (i_lm, i_bl),
                           (i_lw, i_lm), (i_rm, i_rw)])
        else
            append!(segs, [(i_bl, i_br), (i_br, i_tr), (i_tr, i_tl), (i_tl, i_bl)])
        end
    end

    tio = TriangulateIO()
    tio.pointlist = Cdouble[p[k] for k in 1:2, p in pts]
    tio.segmentlist = Cint[s[k] for k in 1:2, s in segs]

    # Triangle's switch parser does not accept scientific notation for the
    # numeric value following `a`. Julia prints sufficiently small Float64s in
    # exponent form (for example, 1 / 128^2 as `6.103515625e-5`), which Triangle
    # misreads as a much larger area. Force fixed-point notation here.
    area_switch = isnothing(max_area) ? "" : "a$(@sprintf("%.17f", max_area))"
    if !isnothing(max_area_at)
        triunsuitable!(
            (x1, y1, x2, y2, x3, y3, area) ->
                area > max_area_at((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)
        )
    end
    unsuitable_switch = isnothing(max_area_at) ? "" : "u"
    flags = "pq30o2$(area_switch)$(unsuitable_switch)Q"
    result, _ = triangulate(flags, tio)

    pts = result.pointlist
    tris_t6 = Matrix{Int32}(result.trianglelist)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]

    n_t6 = length(coords)
    nels = size(tris_t6, 2)
    el2n = Matrix{Int32}(undef, 7, nels)
    el2n[1:3, :] .= tris_t6[1:3, :]
    el2n[4, :] .= tris_t6[6, :] # FEMTools node 4 = mid(1, 2)
    el2n[5, :] .= tris_t6[4, :] # FEMTools node 5 = mid(2, 3)
    el2n[6, :] .= tris_t6[5, :] # FEMTools node 6 = mid(3, 1)
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        c1 = coords[tris_t6[1, iel]]
        c2 = coords[tris_t6[2, iel]]
        c3 = coords[tris_t6[3, iel]]
        push!(coords, (c1 + c2 + c3) / 3)
        el2n[7, iel] = Int32(n_t6 + iel)
    end

    tol = 100eps(Float64) * max(Lx, Ly)
    outer_nodes = Int32[
        i for i in 1:n_t6
        if abs(coords[i][1]) ≤ tol ||
           abs(coords[i][1] - Lx) ≤ tol ||
           abs(coords[i][2] - top(coords[i][1])) ≤ tol ||
           abs(coords[i][2] + Ly) ≤ tol
    ]
    interface_tol = 100eps(Float64) * max(Lx, Ly)
    if curved
        rx = elliptical ? ellipse_half_width : radius
        ry = elliptical ? ellipse_half_height : radius
        junction_y = isnothing(conduit_half_width) ? Inf :
            cy + ry * sqrt(1 - (conduit_half_width / rx)^2)
        interface_nodes = Int32[
            i for i in 1:n_t6
            if abs(hypot((coords[i][1] - cx) / rx, (coords[i][2] - cy) / ry) - 1) ≤
               interface_tol / max(rx, ry) ||
               (!isnothing(conduit_half_width) &&
                abs(abs(coords[i][1] - cx) - conduit_half_width) ≤ interface_tol &&
                coords[i][2] ≥ junction_y - interface_tol)
        ]
    else
        xlo, xhi = cx - half_width, cx + half_width
        ylo, yhi = cy - half_width, cy + half_width
        interface_nodes = Int32[
            i for i in 1:n_t6
            if ((abs(coords[i][1] - xlo) ≤ interface_tol || abs(coords[i][1] - xhi) ≤ interface_tol) &&
                ylo - interface_tol ≤ coords[i][2] ≤ yhi + interface_tol) ||
               ((abs(coords[i][2] - ylo) ≤ interface_tol || abs(coords[i][2] - yhi) ≤ interface_tol) &&
                xlo - interface_tol ≤ coords[i][1] ≤ xhi + interface_tol)
        ]
    end

    return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(interface_nodes))
end

"""
    volcano_profile(x; volcano_cx, volcano_half_span, volcano_peak_height,
                    volcano_rim_radius) -> Float64

Height of the volcano topography above the flat base level `y=0` at
horizontal position `x`.

The profile is flat at `y=0` outside `|x - volcano_cx| ≥ volcano_half_span`,
rises linearly to a flat summit plateau at the constant height
`y=volcano_peak_height`, held for the whole
`|x - volcano_cx| < volcano_rim_radius` interval (no central dip).
"""
function volcano_profile(
        x; volcano_cx, volcano_half_span, volcano_peak_height, volcano_rim_radius,
    )
    d = abs(x - volcano_cx)
    d < volcano_rim_radius && return volcano_peak_height
    d ≥ volcano_half_span && return 0.0
    return volcano_peak_height * (volcano_half_span - d) / (volcano_half_span - volcano_rim_radius)
end

"""
    build_triangle_t7_volcano_mesh(; Lx, Ly, cx, cy, r, max_area=nothing,
                                   volcano_cx=Lx/2, volcano_half_span=0.3Lx,
                                   volcano_peak_height=0.15Ly,
                                   volcano_rim_radius=0.15Lx,
                                   n_topo=80, n_circle=48) -> Tuple

Build an unstructured T7 velocity mesh with a circular inclusion and a
volcano-shaped top boundary.

The outer boundary is flat (`y=0`) on the bottom and sides; the top follows
`volcano_profile` — a flat base, a linear flank, and a flat summit plateau at
constant height, with no central dip — discretized into `n_topo` segments
from `x=Lx` down to `x=0`. The inclusion is a circle of radius `r` centered
at `(cx, cy)`, discretized into `n_circle` segments and passed to
Triangulate.jl as a constrained internal boundary, so no element crosses the
material interface. As in [`build_triangle_t7_inclusion_mesh`](@ref), the
resulting second-order T6 PSLG mesh has its local midpoint ordering remapped
to FEMTools' T6/T7 convention, then one centroid bubble node is appended per
element.
"""
function build_triangle_t7_volcano_mesh(;
        Lx, Ly, cx, cy, r, max_area = nothing,
        volcano_cx = Lx / 2, volcano_half_span = 0.3Lx,
        volcano_peak_height = 0.15Ly, volcano_rim_radius = 0.15Lx,
        n_topo = 80, n_circle = 48,
    )
    profile(x) = volcano_profile(
        x; volcano_cx, volcano_half_span, volcano_peak_height, volcano_rim_radius,
    )

    # Outer polygon, counterclockwise: bottom-left → bottom-right, then the
    # topography samples from x=Lx back down to x=0 (the first and last
    # samples double as the top-right/top-left corners since profile(0) =
    # profile(Lx) = 0 whenever volcano_half_span < min(volcano_cx, Lx - volcano_cx)).
    bottom_pts = Cdouble[0.0 Lx; -Ly -Ly]
    xs_top = collect(range(Lx, 0.0; length = n_topo))
    topo_pts = Cdouble.(permutedims(hcat(xs_top, profile.(xs_top))))
    outer_pts = hcat(bottom_pts, topo_pts)
    n_outer = size(outer_pts, 2)
    outer_segs = Matrix{Cint}(undef, 2, n_outer)
    for i in 1:n_outer
        outer_segs[1, i] = i
        outer_segs[2, i] = i == n_outer ? 1 : i + 1
    end

    θs = range(0, 2π; length = n_circle + 1)[1:(end - 1)]
    circle_pts = Cdouble.(vcat((cx .+ r .* cos.(θs))', (cy .+ r .* sin.(θs))'))
    circle_segs = Matrix{Cint}(undef, 2, n_circle)
    for i in 1:n_circle
        circle_segs[1, i] = n_outer + i
        circle_segs[2, i] = n_outer + (i == n_circle ? 1 : i + 1)
    end

    tio = TriangulateIO()
    tio.pointlist = hcat(outer_pts, circle_pts)
    tio.segmentlist = hcat(outer_segs, circle_segs)

    area_switch = isnothing(max_area) ? "" : "a$(@sprintf("%.17f", max_area))"
    flags = "pq30o2$(area_switch)Q"
    result, _ = triangulate(flags, tio)

    pts = result.pointlist
    tris_t6 = Matrix{Int32}(result.trianglelist)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]

    n_t6 = length(coords)
    nels = size(tris_t6, 2)
    el2n = Matrix{Int32}(undef, 7, nels)
    el2n[1:3, :] .= tris_t6[1:3, :]
    el2n[4, :] .= tris_t6[6, :] # FEMTools node 4 = mid(1, 2)
    el2n[5, :] .= tris_t6[4, :] # FEMTools node 5 = mid(2, 3)
    el2n[6, :] .= tris_t6[5, :] # FEMTools node 6 = mid(3, 1)
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        c1 = coords[tris_t6[1, iel]]
        c2 = coords[tris_t6[2, iel]]
        c3 = coords[tris_t6[3, iel]]
        push!(coords, (c1 + c2 + c3) / 3)
        el2n[7, iel] = Int32(n_t6 + iel)
    end

    # Diagnostic-only node lists (not used for boundary conditions — those
    # come from `Mesh`'s own topology-based boundary detection): loose
    # tolerances account for Triangle's piecewise-linear approximation of the
    # topography and circle when it inserts extra points during refinement.
    tol = 1.0e-3 * max(Lx, Ly)
    outer_nodes = Int32[
        i for i in 1:n_t6
        if abs(coords[i][1]) ≤ tol || abs(coords[i][1] - Lx) ≤ tol ||
           abs(coords[i][2] + Ly) ≤ tol || abs(coords[i][2] - profile(coords[i][1])) ≤ tol
    ]
    interface_nodes = Int32[
        i for i in 1:n_t6
        if abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) ≤ tol
    ]

    return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(interface_nodes))
end
