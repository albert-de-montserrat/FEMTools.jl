using FEMTools
using StaticArrays
using Triangulate

"""
    triangle_output_to_t7(out) -> (coords, el2n, phase)

Convert the quadratic (`o2`) output of Triangle into T7 nodes and connectivity in FEMTools order,
and return the region attribute of each element as `phase`.
"""
function triangle_output_to_t7(out)
    points = out.pointlist
    coords = [SVector{2, Float64}(points[1, i], points[2, i]) for i in axes(points, 2)]
    # Triangle numbers each mid-edge node after the corner opposite to it; FEMTools
    # numbers them along edges 1-2, 2-3 and 3-1.
    el2n_t6 = Int32.(out.trianglelist[[1, 2, 3, 6, 4, 5], :])
    coords, el2n = FEMTools.add_t7_bubbles!(coords, el2n_t6)
    return coords, el2n, Int.(vec(out.triangleattributelist))
end

"""
    build_triangulate_t7_sill_mesh(; kwargs...) -> (coords, el2n, groups)

Build a T7 (Crouzeix-Raviart velocity) Triangle mesh of a flat-surfaced crustal
cross-section `[-Lx/2, Lx/2] × [-depth, 0]` that holds an elliptical magma sill
as a conforming material interface. Lengths are in any consistent unit and the
ground surface is at `y = 0`; the defaults are a Reykjanes-like cross-rift
section in meters.

`max_area` bounds the corner-triangle area in the crust. The sill is meshed with
`max_area / refinement^2`, and its boundary vertices are spaced to match, so the
crust coarsens away from the sill under Triangle's 30° quality constraint. The
interface is a polygon inscribed in the ellipse: its vertices lie on the ellipse
and its edge nodes on the chords.

`groups` is a `NamedTuple` of node indices: `Γnodes` (every boundary node),
`surface`, `bottom`, `left`, `right`, and `sill` (the material interface), plus
`phase`, the material of each element (`1` crust, `2` sill).
"""
function build_triangulate_t7_sill_mesh(;
        Lx = 40.0e3,
        depth = 20.0e3,
        sill_center = (0.0, -4.5e3),
        sill_radii = (2.5e3, 0.5e3),
        max_area = (max(Lx, depth) / 32)^2 / 2,
        refinement = 4,
    )
    cx, cy = sill_center
    rx, ry = sill_radii
    rx > 0 && ry > 0 || throw(ArgumentError("sill_radii must be positive"))
    abs(cx) + rx < Lx / 2 ||
        throw(ArgumentError("the sill must lie strictly inside the domain width Lx"))
    cy + ry < 0 && -depth < cy - ry ||
        throw(ArgumentError("the sill must lie strictly below the surface and above the base"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))
    # Triangle reads a non-positive area constraint as "unconstrained".
    max_area > 0 || throw(ArgumentError("max_area must be positive"))

    x1 = Lx / 2
    sill_max_area = max_area / refinement^2

    # Uniform in the ellipse angle, the vertex spacing is widest at the top and
    # bottom, where it is `2π max(rx, ry) / n`; at least 16 vertices keep a coarse
    # mesh from collapsing the ellipse into a triangle.
    n_sill = max(16, ceil(Int, 2π * max(rx, ry) / sqrt(2 * sill_max_area)))
    sill_angles = range(0, 2π; length = n_sill + 1)[1:(end - 1)]
    vertices = vcat(
        [(-x1, -depth), (x1, -depth), (x1, 0.0), (-x1, 0.0)],
        [(cx + rx * cos(θ), cy + ry * sin(θ)) for θ in sill_angles],
    )
    loop(first, last) = [(i, i == last ? first : i + 1) for i in first:last]
    segments = vcat(loop(1, 4), loop(5, 4 + n_sill))

    tio = TriangulateIO()
    tio.pointlist = Cdouble[v[i] for i in 1:2, v in vertices]
    tio.segmentlist = Cint[s[i] for i in 1:2, s in segments]
    # One seed point per region: `[x; y; phase; max_area]`. The crust seed sits
    # between the sill and the base.
    crust_seed_y = (cy - ry - depth) / 2
    tio.regionlist = Cdouble[cx cx; crust_seed_y cy; 1 2; max_area sill_max_area]
    # `p` PSLG, `q30` minimum angle, `A` region attributes, `a` per-region area
    # limits (the values are in the region list because Triangle would read
    # `a1.0e6` as `a1.0` followed by `e`), `o2` quadratic, `Q` quiet.
    out, _ = triangulate("pq30Aao2Q", tio)
    coords, el2n, phase = triangle_output_to_t7(out)

    tol = sqrt(eps(Float64)) * max(Lx, depth)
    select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
    surface = select(c -> abs(c[2]) <= tol)
    bottom = select(c -> abs(c[2] + depth) <= tol)
    left = select(c -> abs(c[1] + x1) <= tol)
    right = select(c -> abs(c[1] - x1) <= tol)
    Γnodes = sort!(union(surface, bottom, left, right))
    # Elements never straddle the interface, so its nodes are the vertex and edge
    # nodes that both phases share.
    sill = sort!(intersect(vec(el2n[1:6, phase .== 1]), vec(el2n[1:6, phase .== 2])))
    return coords, el2n, (; Γnodes, surface, bottom, left, right, sill, phase)
end

"""
    build_triangulate_t7_cavity_mesh(; kwargs...) -> (coords, el2n, groups)

Build a T7 (Crouzeix-Raviart velocity) Triangle mesh of a disc of radius `radius` centred on
the origin that holds a concentric elliptical inclusion, with semi-axes `cavity_radii` along
`x` and `y`, as a conforming material interface. It is the geometry of the pressurised-cavity
benchmark: the inclusion is the magma body and the disc is a truncated infinite host.

`max_area` bounds the corner-triangle area in the host. The inclusion is meshed with
`max_area / refinement^2`, and the interface and outer-circle vertices are spaced to match, so the
host coarsens away from the inclusion under Triangle's 30° quality constraint. Both boundaries
are polygons inscribed in their curves, with the edge nodes on the chords.

`groups` is a `NamedTuple` of node indices: `Γnodes` (every node on the outer circle) and
`cavity` (the material interface), plus `phase`, the material of each element (`1` host,
`2` inclusion).
"""
function build_triangulate_t7_cavity_mesh(;
        radius = 15.0e3,
        cavity_radii = (2.5e3, 0.5e3),
        max_area = (radius / 12)^2 / 2,
        refinement = 4,
    )
    rx, ry = cavity_radii
    rx > 0 && ry > 0 || throw(ArgumentError("cavity_radii must be positive"))
    max(rx, ry) < radius || throw(ArgumentError("the cavity must lie strictly inside the disc"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))
    max_area > 0 || throw(ArgumentError("max_area must be positive"))

    cavity_max_area = max_area / refinement^2
    n_outer = max(32, ceil(Int, 2π * radius / sqrt(2 * max_area)))
    n_cavity = max(16, ceil(Int, 2π * max(rx, ry) / sqrt(2 * cavity_max_area)))
    outer = [(radius * cos(θ), radius * sin(θ)) for θ in range(0, 2π; length = n_outer + 1)[1:(end - 1)]]
    cavity = [(rx * cos(θ), ry * sin(θ)) for θ in range(0, 2π; length = n_cavity + 1)[1:(end - 1)]]
    loop(first, last) = [(i, i == last ? first : i + 1) for i in first:last]
    segments = vcat(loop(1, n_outer), loop(n_outer + 1, n_outer + n_cavity))

    tio = TriangulateIO()
    tio.pointlist = Cdouble[v[i] for i in 1:2, v in vcat(outer, cavity)]
    tio.segmentlist = Cint[s[i] for i in 1:2, s in segments]
    tio.segmentmarkerlist = Cint[fill(1, n_outer); fill(2, n_cavity)]
    # One seed point per region: `[x; y; phase; max_area]`; see the sill mesher for the switches.
    tio.regionlist = Cdouble[0 0; (ry + radius) / 2 0; 1 2; max_area cavity_max_area]
    out, _ = triangulate("pq30Aao2Q", tio)
    coords, el2n, phase = triangle_output_to_t7(out)

    # The bubble nodes `add_t7_bubbles!` appends have no marker and are never on the boundary.
    Γnodes = Int32[i for i in axes(out.pointlist, 2) if out.pointmarkerlist[i] == 1]
    # Elements never straddle the interface, so its nodes are the vertex and edge
    # nodes that both phases share.
    cavity_nodes = sort!(intersect(vec(el2n[1:6, phase .== 1]), vec(el2n[1:6, phase .== 2])))
    return coords, el2n, (; Γnodes, cavity = cavity_nodes, phase)
end
