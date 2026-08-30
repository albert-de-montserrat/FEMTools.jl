using Test

using FEMTools
using StaticArrays

@testset "external mesh utilities" begin
    tags = [10, 20, 30, 40, 50, 60]
    el2n = FEMTools.renumber_connectivity(tags, [10, 30, 20, 40, 50, 60], 6)
    @test el2n == reshape(Int32[1, 3, 2, 4, 5, 6], 6, 1)
    @test_throws ArgumentError FEMTools.renumber_connectivity([10, 10], [10, 10, 10], 3)
    @test_throws ArgumentError FEMTools.renumber_connectivity(tags, [10, 20, 99], 3)
    @test_throws DimensionMismatch FEMTools.renumber_connectivity(tags, [10, 20], 3)

    coords = SVector{2, Float32}.([(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)])
    FEMTools.orient_triangle_elements!(coords, el2n)
    @test el2n[:, 1] == Int32[1, 2, 3, 6, 5, 4]

    coords_t7, el2n_t7 = FEMTools.add_t7_bubbles!(copy(coords), el2n)
    @test el2n_t7[:, 1] == Int32[1, 2, 3, 6, 5, 4, 7]
    @test coords_t7[7] == SVector{2, Float32}(1 / 3, 1 / 3)
    fill!(@view(coords_t7[4:7]), zero(SVector{2, Float32}))
    FEMTools.straighten_t7_geometry!(coords_t7, el2n_t7)
    @test coords_t7[el2n_t7[4:7, 1]] ==
        SVector{2, Float32}.([(0.5, 0), (0.5, 0.5), (0, 0.5), (1 / 3, 1 / 3)])

    boundary_coords = SVector{2, Float64}.([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (0.5, 0), (0, 2)])
    @test FEMTools.rectangle_boundary_nodes(boundary_coords, 0, 1, 0, 1) == Int32[1, 2, 3, 4, 6]
    circle_coords = SVector{2, Float64}.([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])
    @test FEMTools.circle_boundary_nodes(circle_coords, 0, 0, 1) == Int32[1, 2, 3, 4]

    @test_throws "nodes_per_element must be positive" FEMTools.renumber_connectivity(tags, [10], 0)
    @test_throws "3, 6, or 7 local nodes" FEMTools.orient_triangle_elements!(coords, ones(Int32, 4, 1))
    collinear = SVector{2, Float32}.([(0, 0), (1, 0), (2, 0)])
    @test_throws "degenerate triangle at element 1" FEMTools.orient_triangle_elements!(
        collinear, reshape(Int32[1, 2, 3], 3, 1))
    @test_throws "T6 connectivity must have 6 rows" FEMTools.add_t7_bubbles!(coords, reshape(Int32[1, 2, 3], 3, 1))
    @test_throws "T7 connectivity must have 7 rows" FEMTools.straighten_t7_geometry!(coords, el2n)
    @test_throws "bounds must be ordered" FEMTools.rectangle_boundary_nodes(boundary_coords, 1, 0, 0, 1)
    @test_throws "radius must be positive" FEMTools.circle_boundary_nodes(circle_coords, 0, 0, 0)
    @test_throws "atol must be non-negative" FEMTools.circle_boundary_nodes(circle_coords, 0, 0, 1; atol = -1)
end
