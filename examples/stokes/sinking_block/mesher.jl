
include(joinpath(@__DIR__, "..", "..", "gmsh_meshing.jl"))

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
    build_gmsh_hex27_mesh(; Lx=1, Ly=1, Lz=1, mesh_size=0.15, nz=6)

Generate a recombined, vertically extruded Gmsh mesh and return it in
FEMTools' Hex27 ordering. `mesh_size` controls the horizontal target edge length
and `nz` the number of extrusion layers. The domain is
`[0,Lx] × [0,Ly] × [0,Lz]`, with `z` the vertical axis.

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

        # Counterclockwise base rectangle in the horizontal x-y plane at z = 0;
        # the extrusion below raises it to z = Lz.
        points = (
            gmsh.model.geo.addPoint(0, 0, 0, mesh_size),
            gmsh.model.geo.addPoint(Lx, 0, 0, mesh_size),
            gmsh.model.geo.addPoint(Lx, Ly, 0, mesh_size),
            gmsh.model.geo.addPoint(0, Ly, 0, mesh_size),
        )
        lines = ntuple(i -> gmsh.model.geo.addLine(points[i], points[mod1(i + 1, 4)]), 4)
        surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(collect(lines))])
        gmsh.model.geo.mesh.setRecombine(2, surface)
        gmsh.model.geo.extrude([(2, surface)], 0, 0, Lz, [nz], [1.0], true)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        coords = [SVector{3, Float64}(xyz[3i - 2], xyz[3i - 1], xyz[3i]) for i in eachindex(node_tags)]

        element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
        element_types == [12] || error("expected only Gmsh Hex27 (type 12), got $element_types")
        gmsh_el2n = FEMTools.renumber_connectivity(node_tags, only(element_nodes), 27)
        gmsh_to_fem = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 10, 17, 19,
                       20, 18, 11, 13, 15, 16, 21, 22, 24, 25, 23, 26, 27)
        el2n = gmsh_el2n[collect(gmsh_to_fem), :]
        tol = 32eps(Float64) * max(Lx, Ly, Lz)
        boundary_nodes = Int32[i for i in eachindex(coords) if
            abs(coords[i][1]) ≤ tol || abs(coords[i][1] - Lx) ≤ tol ||
            abs(coords[i][2]) ≤ tol || abs(coords[i][2] - Ly) ≤ tol ||
            abs(coords[i][3]) ≤ tol || abs(coords[i][3] - Lz) ≤ tol]
        return coords, el2n, boundary_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end
