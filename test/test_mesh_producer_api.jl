using Test

using FEMTools
using FEMTools: generate_element_groups, generate_discontinuous_linear_mesh
using KernelAbstractions: CPU
using StaticArrays

@testset "generate_element_groups replaces build_element_groups" begin
    backend = CPU()
    colors = [1, 2, 1, 3, 2]

    groups_new = generate_element_groups(backend, colors)
    groups_old = @test_deprecated build_element_groups(backend, colors)

    @test groups_new == [[1, 3], [2, 5], [4]]
    @test groups_old == groups_new
end

@testset "generate_discontinuous_linear_mesh replaces build_discontinuous_linear_mesh" begin
    coords = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
    el2n = Int32[1 2; 2 4; 3 3]

    new_result = generate_discontinuous_linear_mesh(coords, el2n)
    old_result = @test_deprecated build_discontinuous_linear_mesh(coords, el2n)

    @test old_result == new_result
end
