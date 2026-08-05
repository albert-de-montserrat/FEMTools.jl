using Test

using FEMTools
using FEMTools: generate_element_groups, generate_discontinuous_linear_mesh
using KernelAbstractions: CPU
using StaticArrays

@testset "generate_element_groups groups elements by color" begin
    backend = CPU()
    colors = [1, 2, 1, 3, 2]

    @test generate_element_groups(backend, colors) == [[1, 3], [2, 5], [4]]
end

@testset "generate_discontinuous_linear_mesh duplicates shared nodes" begin
    coords = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
    el2n = Int32[1 2; 2 4; 3 3]
    nels = size(el2n, 2)

    p_el2n, p_el2dof, p_dof_coords = generate_discontinuous_linear_mesh(coords, el2n)

    # Corner connectivity is preserved, while every element receives its own
    # three degrees of freedom carrying that corner's coordinates.
    @test p_el2n == el2n[1:3, :]
    @test p_el2dof == reshape(Int32.(1:(3 * nels)), 3, nels)
    @test length(p_dof_coords) == 3 * nels
    @test all(p_dof_coords[p_el2dof[a, iel]] == coords[el2n[a, iel]]
              for iel in 1:nels, a in 1:3)
end
