
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

"""
    build_gmsh_hex27_mesh(; Lx=1, Ly=1, Lz=1, mesh_size=0.15, nz=6)

Generate a recombined, vertically extruded Gmsh mesh and return it in
FEMTools' Hex27 ordering. `mesh_size` controls the in-plane target edge length
and `nz` the number of extrusion layers. The domain is
`[0,Lx] × [-Ly,0] × [0,Lz]`.

Returns `(coords, el2n, boundary_nodes)`: three-dimensional coordinates,
`27 × nels` element connectivity, and the sorted velocity-node indices on
the six exterior faces. Mixed Gmsh volume-element output is rejected.
"""
function build_gmsh_hex27_mesh(; Lx = 1.0, Ly = 1.0, Lz = 1.0, mesh_size = 0.15, nz = 6)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.model.add("sinking_block_3d")

        points = (
            gmsh.model.geo.addPoint(0, -Ly, 0, mesh_size),
            gmsh.model.geo.addPoint(Lx, -Ly, 0, mesh_size),
            gmsh.model.geo.addPoint(Lx, 0, 0, mesh_size),
            gmsh.model.geo.addPoint(0, 0, 0, mesh_size),
        )
        lines = ntuple(i -> gmsh.model.geo.addLine(points[i], points[mod1(i + 1, 4)]), 4)
        surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(collect(lines))])
        gmsh.model.geo.mesh.setRecombine(2, surface)
        gmsh.model.geo.extrude([(2, surface)], 0, 0, Lz, [nz], [1.0], true)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        tag_to_node = Dict(tag => Int32(i) for (i, tag) in enumerate(node_tags))
        coords = [SVector{3, Float64}(xyz[3i - 2], xyz[3i - 1], xyz[3i]) for i in eachindex(node_tags)]

        element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
        element_types == [12] || error("expected only Gmsh Hex27 (type 12), got $element_types")
        nodes = only(element_nodes)
        gmsh_el2n = reshape(Int32[tag_to_node[tag] for tag in nodes], 27, :)
        gmsh_to_fem = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 10, 17, 19,
                       20, 18, 11, 13, 15, 16, 21, 22, 24, 25, 23, 26, 27)
        el2n = gmsh_el2n[collect(gmsh_to_fem), :]
        tol = 32eps(Float64) * max(Lx, Ly, Lz)
        boundary_nodes = Int32[i for i in eachindex(coords) if
            abs(coords[i][1]) ≤ tol || abs(coords[i][1] - Lx) ≤ tol ||
            abs(coords[i][2]) ≤ tol || abs(coords[i][2] + Ly) ≤ tol ||
            abs(coords[i][3]) ≤ tol || abs(coords[i][3] - Lz) ≤ tol]
        return coords, el2n, boundary_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end
