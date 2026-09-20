using KernelAbstractions: CPU

@testset "2-D sill mesh conforms to the reservoir" begin
    include(joinpath(pkgdir(FEMTools), "examples", "gmsh_meshing.jl"))

    Lx, depth = 10.0, 5.0
    center, radii = (0.5, -1.5), (2.0, 0.4)
    coords, el2n, groups = build_gmsh_t7_sill_mesh(;
        Lx, depth, sill_center = center, sill_radii = radii, max_area = 0.25, refinement = 4,
    )
    level(c) = hypot((c[1] - center[1]) / radii[1], (c[2] - center[2]) / radii[2])

    @test size(el2n, 1) == 7
    @test Set(groups.phase) == Set((1, 2))
    @test !isempty(groups.sill)
    @test all(i -> isapprox(level(coords[i]), 1; atol = 1.0e-8), groups.sill)
    # No element straddles the interface: its vertex and edge nodes lie on one side.
    @test all(axes(el2n, 2)) do iel
        levels = [level(coords[el2n[a, iel]]) for a in 1:6]
        groups.phase[iel] == 2 ? maximum(levels) <= 1 + 1.0e-8 : minimum(levels) >= 1 - 1.0e-8
    end

    # Counter-clockwise corner triangles tile the rectangle, and those of the sill
    # inscribe its ellipse.
    area(iel) = begin
        a, b, c = coords[el2n[1, iel]], coords[el2n[2, iel]], coords[el2n[3, iel]]
        ((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
    end
    areas = area.(axes(el2n, 2))
    @test all(>(0), areas)
    @test sum(areas) ≈ Lx * depth
    @test isapprox(sum(areas[groups.phase .== 2]), π * radii[1] * radii[2]; rtol = 0.02)

    @test all(i -> isapprox(coords[i][2], 0; atol = 1.0e-9), groups.surface)
    @test all(i -> isapprox(coords[i][2], -depth; atol = 1.0e-9), groups.bottom)
    @test all(i -> isapprox(coords[i][1], -Lx / 2; atol = 1.0e-9), groups.left)
    @test all(i -> isapprox(coords[i][1], Lx / 2; atol = 1.0e-9), groups.right)

    @test_throws ArgumentError build_gmsh_t7_sill_mesh(; sill_radii = (0.5e3, 2.5e3))
    @test_throws ArgumentError build_gmsh_t7_sill_mesh(; sill_center = (0.0, -0.2e3))
end

@testset "2-D Triangle sill mesh conforms to the reservoir" begin
    include(joinpath(pkgdir(FEMTools), "examples", "triangulate_meshing.jl"))

    Lx, depth = 10.0, 5.0
    center, radii = (0.5, -1.5), (2.0, 0.4)
    max_area, refinement = 0.25, 4
    coords, el2n, groups = build_triangulate_t7_sill_mesh(;
        Lx, depth, sill_center = center, sill_radii = radii, max_area, refinement,
    )
    level(c) = hypot((c[1] - center[1]) / radii[1], (c[2] - center[2]) / radii[2])

    @test size(el2n, 1) == 7
    @test Set(groups.phase) == Set((1, 2))
    # The interface is a polygon inscribed in the ellipse: its vertices lie on the
    # ellipse and its edge nodes on the chords, just inside.
    @test !isempty(groups.sill)
    @test all(i -> 0.99 <= level(coords[i]) <= 1 + 1.0e-12, groups.sill)
    @test all(findall(==(2), groups.phase)) do iel
        all(a -> level(coords[el2n[a, iel]]) <= 1 + 1.0e-12, 1:6)
    end

    area(iel) = begin
        a, b, c = coords[el2n[1, iel]], coords[el2n[2, iel]], coords[el2n[3, iel]]
        ((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
    end
    areas = area.(axes(el2n, 2))
    @test all(>(0), areas)
    @test sum(areas) ≈ Lx * depth
    @test isapprox(sum(areas[groups.phase .== 2]), π * radii[1] * radii[2]; rtol = 0.01)
    # A dropped `a` switch would leave every region unconstrained and the mesh coarse.
    @test all(<=(max_area * (1 + 1.0e-9)), areas[groups.phase .== 1])
    @test all(<=(max_area / refinement^2 * (1 + 1.0e-9)), areas[groups.phase .== 2])

    @test all(i -> isapprox(coords[i][2], 0; atol = 1.0e-9), groups.surface)
    @test all(i -> isapprox(coords[i][2], -depth; atol = 1.0e-9), groups.bottom)
    @test all(i -> isapprox(coords[i][1], -Lx / 2; atol = 1.0e-9), groups.left)
    @test all(i -> isapprox(coords[i][1], Lx / 2; atol = 1.0e-9), groups.right)

    @test_throws ArgumentError build_triangulate_t7_sill_mesh(; sill_center = (0.0, -0.2e3))
    @test_throws ArgumentError build_triangulate_t7_sill_mesh(; refinement = 0.5)
    @test_throws ArgumentError build_triangulate_t7_sill_mesh(; max_area = 0.0)
end

@testset "2-D Triangle cavity mesh conforms to the inclusion" begin
    include(joinpath(pkgdir(FEMTools), "examples", "triangulate_meshing.jl"))

    radius, radii = 5.0, (1.0, 0.3)
    max_area, refinement = 0.25, 4
    coords, el2n, groups = build_triangulate_t7_cavity_mesh(;
        radius, cavity_radii = radii, max_area, refinement,
    )
    level(c) = hypot(c[1] / radii[1], c[2] / radii[2])

    @test size(el2n, 1) == 7
    @test Set(groups.phase) == Set((1, 2))
    # Both boundaries are inscribed polygons: vertices on the curve, chord nodes just inside.
    @test !isempty(groups.cavity)
    @test all(i -> 0.99 <= level(coords[i]) <= 1 + 1.0e-12, groups.cavity)
    @test all(i -> 0.995 * radius <= norm(coords[i]) <= radius * (1 + 1.0e-12), groups.Γnodes)
    @test all(i -> norm(coords[i]) < 0.995 * radius, setdiff(eachindex(coords), groups.Γnodes))

    area(iel) = begin
        a, b, c = coords[el2n[1, iel]], coords[el2n[2, iel]], coords[el2n[3, iel]]
        ((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
    end
    areas = area.(axes(el2n, 2))
    @test all(>(0), areas)
    @test isapprox(sum(areas), π * radius^2; rtol = 0.005)
    @test isapprox(sum(areas[groups.phase .== 2]), π * radii[1] * radii[2]; rtol = 0.01)
    @test all(<=(max_area * (1 + 1.0e-9)), areas[groups.phase .== 1])
    @test all(<=(max_area / refinement^2 * (1 + 1.0e-9)), areas[groups.phase .== 2])

    @test_throws ArgumentError build_triangulate_t7_cavity_mesh(; radius = 5.0, cavity_radii = (6.0, 1.0))
    @test_throws ArgumentError build_triangulate_t7_cavity_mesh(; cavity_radii = (1.0, 0.0))
    @test_throws ArgumentError build_triangulate_t7_cavity_mesh(; refinement = 0.5)
    @test_throws ArgumentError build_triangulate_t7_cavity_mesh(; max_area = 0.0)
end

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
