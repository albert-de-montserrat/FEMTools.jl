module FEMToolsTriangulateExt

using Triangulate
using StaticArrays
using FEMTools

# Triangle is a double-precision C library, so the returned coordinates are
# always `Float64` regardless of the precision used to describe the boundary.
function FEMTools.triangulate_t7_mesh(
        points::AbstractVector{<:StaticVector{2}};
        max_area, min_angle = 30.0, segments = nothing, regions = (),
    )
    npoints = length(points)
    npoints >= 3 ||
        throw(ArgumentError("a planar straight-line graph needs at least 3 points, got $npoints"))
    max_area > 0 || throw(ArgumentError("max_area must be positive"))
    # Triangle only guarantees termination well below 34 degrees; beyond that the
    # refinement loop can run forever instead of failing, so reject it up front.
    0 < min_angle < 34 ||
        throw(ArgumentError("min_angle must lie in (0, 34) degrees, got $min_angle"))

    tio = TriangulateIO()
    tio.pointlist = Cdouble[p[c] for c in 1:2, p in points]
    tio.segmentlist = _segment_list(segments, npoints)
    isempty(regions) || (tio.regionlist = _region_list(regions, max_area))
    # p: read the PSLG, q: minimum-angle quality bound, o2: second-order (T6)
    # output supplying the midside nodes, a: area constraint, Q: quiet. A adds
    # the regional attribute that tells the caller which region an element is in.
    # With regions Triangle takes the area constraint from the region list, so
    # `a` carries no number there; `_region_list` defaults each region to
    # `max_area`. A numeric `a` would override the per-region constraints.
    flags = isempty(regions) ? "pq$(min_angle)o2a$(max_area)Q" :
        "pq$(min_angle)o2aQA"
    result, _ = triangulate(flags, tio)

    tris = result.trianglelist
    size(tris, 1) == 6 ||
        throw(ArgumentError("Triangle returned $(size(tris, 1)) nodes per element, expected 6"))
    nels = size(tris, 2)
    nels > 0 ||
        throw(ArgumentError("Triangle produced no elements; check that the segments bound a region"))

    pts = result.pointlist
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]
    # Triangle lists the midside node opposite each corner; FEMTools expects the
    # node on the edge following that corner.
    el2n_t6 = Matrix{Int32}(undef, 6, nels)
    el2n_t6[1:3, :] .= tris[1:3, :]
    el2n_t6[4, :] .= tris[6, :]
    el2n_t6[5, :] .= tris[4, :]
    el2n_t6[6, :] .= tris[5, :]

    coords, el2n = FEMTools.add_t7_bubbles!(coords, el2n_t6)
    FEMTools.orient_triangle_elements!(coords, el2n)
    return coords, el2n, _attribute_vector(result.triangleattributelist, nels)
end

function _segment_list(::Nothing, npoints)
    # No explicit graph: close the point list into a single polygon.
    list = Matrix{Cint}(undef, 2, npoints)
    for i in 1:npoints
        list[1, i] = i
        list[2, i] = i % npoints + 1
    end
    return list
end

function _segment_list(segments, npoints)
    isempty(segments) && throw(ArgumentError("segments must not be empty"))
    list = Matrix{Cint}(undef, 2, length(segments))
    for (i, segment) in enumerate(segments)
        length(segment) == 2 ||
            throw(ArgumentError("segment $i must hold two point indices"))
        for (c, index) in enumerate(segment)
            1 <= index <= npoints ||
                throw(ArgumentError("segment $i refers to point $index, outside 1:$npoints"))
            list[c, i] = index
        end
    end
    return list
end

function _region_list(regions, max_area)
    list = Matrix{Cdouble}(undef, 4, length(regions))
    for (i, region) in enumerate(regions)
        length(region) == 3 || length(region) == 4 ||
            throw(ArgumentError("region $i must be (x, y, attribute[, max_area])"))
        area = length(region) == 4 ? region[4] : max_area
        area > 0 || throw(ArgumentError("region $i has a non-positive max_area"))
        list[1, i] = region[1]
        list[2, i] = region[2]
        list[3, i] = region[3]
        list[4, i] = area
    end
    return list
end

# Without regions Triangle writes no attributes, and every element belongs to
# the single implied region.
_attribute_vector(attributes, nels) =
    isempty(attributes) ? ones(Int32, nels) : Int32[round(Int32, a) for a in vec(attributes)]

end # module FEMToolsTriangulateExt
