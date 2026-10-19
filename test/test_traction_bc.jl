using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU
using StaticArrays

facet_pairs(f) = Set(zip(Array(f.elements), Array(f.local_edges)))

@testset "boundary facets" begin
    element = ReferenceElement(QuadraticElement{2, 6, Float64})
    mesh = Mesh(CPU(), (0.0 .. 2.0) × (0.0 .. 1.0), element, (4, 2); workgroup = 1)
    coords = Array(mesh.coords)

    top = boundary_facets(mesh, x -> x[2] ≈ 1.0)
    @test top isa BoundaryFacets
    @test length(top) == 4

    top_nodes = findall(x -> x[2] ≈ 1.0, coords)
    @test facet_pairs(boundary_facets(mesh, top_nodes)) == facet_pairs(top)

    # A right-wall edge touches the top set only at its corner node.
    right_wall_corner = findfirst(x -> x[1] ≈ 2.0 && x[2] ≈ 0.75, coords)
    @test length(boundary_facets(mesh, vcat(top_nodes, right_wall_corner))) == 4

    @test_throws "no boundary edges have all their nodes" boundary_facets(mesh, x -> false)
    @test_throws "no boundary edges have all their nodes" boundary_facets(
        mesh, x -> 0.2 < x[2] < 0.8,
    )

    mixed = MixedMesh(mesh, ReferenceElement(LinearElement{2, 3, Float64}); workgroup = 1)
    @test facet_pairs(boundary_facets(mixed, x -> x[2] ≈ 1.0)) == facet_pairs(top)
end
