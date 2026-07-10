
"""
    precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) -> Nothing

Fill per-element geometry data on the configured backend.

This wrapper launches `precompute_geometry_kernel!` with the example-wide
`backend` and `workgroup` constants, then synchronizes before returning.
"""
function precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
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

    flags = isnothing(max_area) ? "pqo2Q" : "pq30o2a$(max_area)Q"
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
