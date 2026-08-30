using KernelAbstractions: CPU

@testset "3-D volcano mesh conforms to chamber" begin
    include(joinpath(pkgdir(FEMTools), "examples", "stokes", "volcano", "volcano_mesh_3D.jl"))

    center = (1.0, -1.0, -3.0)
    radii = (1.0, 0.8, 0.5)
    coords, el2n, groups = build_tet11_volcano_mesh(;
        Lx = 10.0, Ly = 8.0, depth = 6.0,
        cone_base = 2.0, cone_top = 0.2, cone_height = 0.5,
        chamber_center = center, chamber_radii = radii,
        mesh_size = 1.5, refinement = 3.0,
    )

    @test size(el2n, 1) == 11
    @test !isempty(groups.chamber)
    @test all(i -> isapprox(_ellipsoid_level(coords[i], center, radii), 1; atol = 1e-10),
        groups.chamber)
    @test Set(groups.phase) == Set((1, 2))
    @test all(iel -> begin
        levels = [_ellipsoid_level(coords[el2n[a, iel]], center, radii) for a in 1:4]
        groups.phase[iel] == 2 ? maximum(levels) <= 1 + 1e-10 : minimum(levels) >= 1 - 1e-10
    end, axes(el2n, 2))

    velocity_element = ReferenceElement(QuadraticElement{3, 11, Float64})
    pressure_element = ReferenceElement(LinearElement{3, 4, Float64})
    mesh = MixedMesh(Mesh(coords, el2n; order = 2), pressure_element)
    @test mesh.el2nP == el2n[1:4, :]

    # A T10 edge-node permutation that disagrees with the element ordering
    # inverts the isoparametric map on a few elements rather than failing.
    cache = MixedMeshCache(CPU(), 1, mesh, velocity_element, pressure_element)
    @test all(qp -> last(qp) > 0, Iterators.flatten(cache.geo_v))

    path, io = mktemp()
    close(io)
    try
        write_vtk(path, mesh)
        text = read(path, String)
        @test occursin("CELLS $(mesh.nels) $(5mesh.nels)", text)
        @test occursin("CELL_TYPES $(mesh.nels)\n10", text)
    finally
        rm(path; force = true)
    end
end
