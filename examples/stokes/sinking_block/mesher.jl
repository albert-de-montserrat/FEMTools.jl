
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
    build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, half_width, max_area=nothing) -> Tuple

Build an unstructured T7 velocity mesh around a rectangular inclusion.

Triangulate.jl generates a second-order T6 PSLG mesh with the rectangle as a
constrained internal boundary. The local midpoint ordering is remapped to
FEMTools' T6/T7 convention, then one centroid bubble node is appended per
element.
"""
function build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, half_width, max_area = nothing)
    rect_pts  = Cdouble[0.0 Lx  Lx 0.0;
                        -Ly -Ly 0.0 0.0 ]
    rect_segs = Cint[1 2; 2 3; 3 4; 4 1]'

    xlo, xhi = cx - half_width, cx + half_width
    ylo, yhi = cy - half_width, cy + half_width
    incl_pts = Cdouble[xlo xhi xhi xlo;
                       ylo ylo yhi yhi]
    incl_segs = Cint[5 6; 6 7; 7 8; 8 5]'

    tio = TriangulateIO()
    tio.pointlist = hcat(rect_pts, incl_pts)
    tio.segmentlist = hcat(rect_segs, incl_segs)

    # Triangle's switch parser does not accept scientific notation for the
    # numeric value following `a`. Julia prints sufficiently small Float64s in
    # exponent form (for example, 1 / 128^2 as `6.103515625e-5`), which Triangle
    # misreads as a much larger area. Force fixed-point notation here.
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

    tol = 100eps(Float64) * max(Lx, Ly)
    outer_nodes = Int32[
        i for i in 1:n_t6
        if abs(coords[i][1]) ≤ tol ||
           abs(coords[i][1] - Lx) ≤ tol ||
           abs(coords[i][2]) ≤ tol ||
           abs(coords[i][2] + Ly) ≤ tol
    ]
    interface_tol = 100eps(Float64) * max(Lx, Ly)
    interface_nodes = Int32[
        i for i in 1:n_t6
        if ((abs(coords[i][1] - xlo) ≤ interface_tol || abs(coords[i][1] - xhi) ≤ interface_tol) &&
            ylo - interface_tol ≤ coords[i][2] ≤ yhi + interface_tol) ||
           ((abs(coords[i][2] - ylo) ≤ interface_tol || abs(coords[i][2] - yhi) ≤ interface_tol) &&
            xlo - interface_tol ≤ coords[i][1] ≤ xhi + interface_tol)
    ]

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
