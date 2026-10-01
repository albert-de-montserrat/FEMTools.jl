using Test

using FEMTools
using StaticArrays
using Triangulate

@testset "Triangulate T7 meshing extension" begin
    square = [
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0),
    ]
    max_area = 0.05
    coords, el2n, attributes = FEMTools.triangulate_t7_mesh(square; max_area)

    @test eltype(coords) === SVector{2, Float64}
    @test eltype(el2n) === Int32
    @test size(el2n, 1) == 7
    @test size(el2n, 2) > 0
    # One bubble node per element sits on top of the T6 node set.
    @test length(coords) > size(el2n, 2)

    signed_area(p1, p2, p3) =
        ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2])) / 2
    areas = [
        signed_area(coords[el2n[1, i]], coords[el2n[2, i]], coords[el2n[3, i]])
            for i in axes(el2n, 2)
    ]
    # Positive areas mean counter-clockwise corners; the total recovers the
    # unit square exactly because the boundary is a straight-edged polygon.
    @test all(>(0), areas)
    @test sum(areas) ≈ 1.0
    @test maximum(areas) <= max_area

    for i in axes(el2n, 2)
        c1, c2, c3 = coords[el2n[1, i]], coords[el2n[2, i]], coords[el2n[3, i]]
        @test coords[el2n[4, i]] ≈ (c1 + c2) / 2
        @test coords[el2n[5, i]] ≈ (c2 + c3) / 2
        @test coords[el2n[6, i]] ≈ (c3 + c1) / 2
        @test coords[el2n[7, i]] ≈ (c1 + c2 + c3) / 3
    end

    # A finer area constraint must produce a finer mesh.
    _, el2n_fine, _ = FEMTools.triangulate_t7_mesh(square; max_area = max_area / 10)
    @test size(el2n_fine, 2) > size(el2n, 2)

    # The mesh feeds the T7 velocity element without further conversion.
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    mesh = Mesh(
        element_v, nothing, nothing, coords, Int32.(1:length(coords)), el2n,
        FEMTools.rectangle_boundary_nodes(coords, 0.0, 1.0, 0.0, 1.0),
    )
    @test mesh.nnodes == length(coords)
    @test mesh.nels == size(el2n, 2)

    @test attributes == ones(Int32, size(el2n, 2))

    @test_throws "at least 3 points" FEMTools.triangulate_t7_mesh(square[1:2]; max_area)
    @test_throws "max_area must be positive" FEMTools.triangulate_t7_mesh(square; max_area = 0.0)
    @test_throws "min_angle must lie in" FEMTools.triangulate_t7_mesh(
        square; max_area, min_angle = 45.0
    )
end

@testset "Triangulate conforming region attributes" begin
    # A unit square split in two by an interior segment across its middle. The
    # mesh must conform to that segment and report which side each element is on.
    points = [
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 0.5),
        SVector(1.0, 1.0), SVector(0.0, 1.0), SVector(0.0, 0.5),
    ]
    segments = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (3, 6)]
    regions = ((0.5, 0.25, 1), (0.5, 0.75, 2))
    coords, el2n, attributes =
        FEMTools.triangulate_t7_mesh(points; max_area = 0.02, segments, regions)

    @test eltype(attributes) === Int32
    @test length(attributes) == size(el2n, 2)
    @test sort(unique(attributes)) == Int32[1, 2]

    # Conformity: no element straddles the interface, so every corner of a
    # region-1 element sits on or below y = 0.5, and the reverse for region 2.
    for iel in axes(el2n, 2)
        ys = (coords[el2n[1, iel]][2], coords[el2n[2, iel]][2], coords[el2n[3, iel]][2])
        if attributes[iel] == 1
            @test maximum(ys) <= 0.5 + 1.0e-12
        else
            @test minimum(ys) >= 0.5 - 1.0e-12
        end
    end

    signed_area(p1, p2, p3) =
        ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2])) / 2
    areas = [
        signed_area(coords[el2n[1, i]], coords[el2n[2, i]], coords[el2n[3, i]])
            for i in axes(el2n, 2)
    ]
    # Each region is half the square, which also confirms the attributes are not
    # swapped between the two sides.
    @test sum(areas[attributes .== 1]) ≈ 0.5
    @test sum(areas[attributes .== 2]) ≈ 0.5

    # A per-region area constraint binds the region it names. Quality refinement
    # still carries some of it across the interface, so the neighbouring region
    # is not left untouched; what must hold is the constraint itself.
    coords_split, el2n_split, attr_split = FEMTools.triangulate_t7_mesh(
        points; max_area = 0.02, segments,
        regions = ((0.5, 0.25, 1), (0.5, 0.75, 2, 0.002)),
    )
    @test size(el2n_split, 1) == 7
    areas_split = [
        signed_area(coords_split[el2n_split[1, i]], coords_split[el2n_split[2, i]],
            coords_split[el2n_split[3, i]])
            for i in axes(el2n_split, 2)
    ]
    @test maximum(areas_split[attr_split .== 2]) <= 0.002
    @test maximum(areas_split[attr_split .== 1]) <= 0.02
    @test sum(areas_split[attr_split .== 2]) ≈ 0.5

    @test_throws "two point indices" FEMTools.triangulate_t7_mesh(
        points; max_area = 0.02, segments = [(1, 2, 3)]
    )
    @test_throws "outside 1:6" FEMTools.triangulate_t7_mesh(
        points; max_area = 0.02, segments = [(1, 2), (2, 99)]
    )
    @test_throws "segments must not be empty" FEMTools.triangulate_t7_mesh(
        points; max_area = 0.02, segments = ()
    )
    @test_throws "(x, y, attribute" FEMTools.triangulate_t7_mesh(
        points; max_area = 0.02, segments, regions = ((0.5, 0.25),)
    )
    @test_throws "non-positive max_area" FEMTools.triangulate_t7_mesh(
        points; max_area = 0.02, segments, regions = ((0.5, 0.25, 1, 0.0),)
    )
end
