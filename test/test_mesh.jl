using Test

using DomainSets
using FEMTools
using SparseArrays

@testset "mesh coordinates" begin
    Ω = 0.0..10.0

    linear = ReferenceElement(LinearElement{1, 2})
    @test generate_coordinates(linear, Ω, 4) == [0.0, 2.5, 5.0, 7.5, 10.0]

    quadratic = ReferenceElement(QuadraticElement{1, 3})
    @test generate_coordinates(quadratic, Ω, 2) == [0.0, 2.5, 5.0, 7.5, 10.0]
end

@testset "mesh connectivity" begin
    linear = ReferenceElement(LinearElement{1, 2})
    el2n = generate_element2node(linear, 3)
    @test el2n == Int32[1 2 3; 2 3 4]

    n2el = generate_node2element(el2n)
    @test n2el == Vector{Int32}[[1], [1, 2], [2, 3], [3]]

    quadratic = ReferenceElement(QuadraticElement{1, 3})
    el2n = generate_element2node(quadratic, 3)
    @test el2n == Int32[1 3 5; 2 4 6; 3 5 7]

    n2el = generate_node2element(el2n)
    @test n2el == Vector{Int32}[[1], [1], [1, 2], [2], [2, 3], [3], [3]]
    @test generate_boundary_elements(Int32[1, 7], n2el) == Int32[1, 3]
end

@testset "mesh constructor" begin
    Ω = 0.0..10.0
    mesh = Mesh(Ω, 3)

    @test mesh isa FEMTools.AbstractMesh
    @test mesh isa Mesh{1, 2}
    @test mesh.Ω == Ω
    @test mesh.Γ == boundary(Ω)
    @test mesh.coords ≈ [0.0, 10 / 6, 10 / 3, 5.0, 20 / 3, 50 / 6, 10.0]
    @test mesh.DoFs == Int32[1, 2, 3, 4, 5, 6, 7]
    @test mesh.el2n == Int32[1 3 5; 2 4 6; 3 5 7]
    @test mesh.n2el == Vector{Int32}[[1], [1], [1, 2], [2], [2, 3], [3], [3]]
    @test mesh.Γnodes == Int32[1, 7]
    @test mesh.Γels == Int32[1, 3]
    @test mesh.nnodes == 7
    @test mesh.nels == 3
end

@testset "mesh compatibility helpers" begin
    Ω = 0.0..1.0
    element = ReferenceElement(QuadraticElement{1, 3})

    @test FEMTools.generete_coordinates(element, Ω, 1) == generate_coordinates(element, Ω, 1)
    @test FEMTools.generete_DoFs(element, 3) == generate_dofs(element, 3)
end

@testset "mesh sparsity pattern" begin
    mesh = Mesh(0.0..10.0, 3)
    pattern = generate_sparsity_pattern(mesh)

    @test pattern isa SparseMatrixCSC{Bool, Int}
    @test size(pattern) == (mesh.nnodes, mesh.nnodes)
    @test pattern == pattern'
    @test nnz(pattern) == 25

    for iel in 1:mesh.nels
        local_nodes = @view mesh.el2n[:, iel]
        for j in local_nodes, i in local_nodes
            @test pattern[i, j]
        end
    end
end

@testset "mesh coloring" begin
    mesh = Mesh(0.0..10.0, 6)
    greedy_colors = color_mesh_greedy(mesh)
    colors = color_mesh(mesh)

    function is_valid_coloring(mesh, colors)
        for iel in 1:mesh.nels
            for node in @view mesh.el2n[:, iel]
                for jel in mesh.n2el[node]
                    if jel != iel && colors[jel] == colors[iel]
                        return false
                    end
                end
            end
        end
        return true
    end

    @test colors == [1, 2, 1, 2, 1, 2]
    @test maximum(colors) == 2
    @test is_valid_coloring(mesh, colors)
    @test is_valid_coloring(mesh, greedy_colors)
end
