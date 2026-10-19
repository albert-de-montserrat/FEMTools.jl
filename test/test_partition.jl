using Test, FEMTools, DomainSets, StaticArrays, Metis
using DomainSets: ×

# Invariants every partition must satisfy, checked against the serial mesh.
function check_partition(mesh, assignment, nparts)
    parts = FEMTools.partition_mesh(mesh, assignment, nparts)
    el2n, Γnodes = Array(mesh.el2n), Set(Array(mesh.Γnodes))
    incident = [Set{Int}() for _ in 1:mesh.nnodes]
    for e in 1:mesh.nels, n in el2n[:, e]
        push!(incident[n], assignment[e])
    end
    @test length(parts) == nparts
    @test sort!(reduce(vcat, [p.elements for p in parts])) == 1:mesh.nels
    owned_ids = [p.nodes[1:p.nowned] for p in parts]
    @test FEMTools.reconstruct_owned(mesh.nnodes, owned_ids, owned_ids) == 1:mesh.nnodes
    for p in parts
        @test all(e -> assignment[e] == p.part, p.elements)
        @test p.nodes[p.el2n] == el2n[:, p.elements]
        @test eltype(p.el2n) == eltype(el2n)
        @test p.coords == mesh.coords[p.nodes]
        @test all(i -> minimum(incident[p.nodes[i]]) == p.part, 1:p.nowned)
        @test all(i -> minimum(incident[p.nodes[i]]) != p.part, (p.nowned + 1):length(p.nodes))
        # Interface nodes are not physical boundary nodes.
        @test Set(p.nodes[p.Γnodes]) == intersect(Γnodes, p.nodes)
        @test p.nodes[p.halo] == filter(n -> length(incident[n]) > 1, p.nodes)
        @test p.halo_owners == [minimum(incident[n]) for n in p.nodes[p.halo]]
    end
    # Every part sharing an interface node sees the same ID, and its owner lists it as owned.
    shared_nodes = Dict{Int, Int}()
    for p in parts, (i, id) in zip(p.halo, p.halo_ids)
        @test get!(shared_nodes, id, p.nodes[i]) == p.nodes[i]
        owner = parts[p.halo_owners[findfirst(==(i), p.halo)]]
        @test id in owner.halo_ids[owner.halo .<= owner.nowned]
    end
    @test sort!(collect(keys(shared_nodes))) == 1:only(unique(p.nshared for p in parts))
    return parts
end

@testset "mesh partition metadata" begin
    for FP in (Float32, Float64)
        mesh = Mesh((FP(0) .. FP(1)) × (FP(0) .. FP(1)), ReferenceElement(LinearElement{2, 3, FP}), (2, 2))
        # Part 3 owns one element whose nodes all belong to lower parts; part 4 owns nothing.
        assignment = [1, 1, 2, 1, 2, 3, 2, 2]
        parts = check_partition(mesh, assignment, 4)
        @test eltype(parts[1].coords) == SVector{2, FP}
        @test length(parts[3].elements) == 1 && parts[3].nowned == 0
        @test isempty(parts[4].nodes) && size(parts[4].el2n) == (3, 0)
        @test eltype(parts[4].coords) == SVector{2, FP}
    end

    # High-order nodes, vertex-only sharing, and four sharers of one vertex.
    q9 = Mesh((0.0 .. 1.0) × (0.0 .. 1.0), ReferenceElement(QuadraticElement{2, 9, Float64}), (2, 2))
    check_partition(q9, [1, 2, 3, 4], 4)
    check_partition(q9, [1, 2, 2, 1], 2)

    # Ownership follows part incidence, not node numbering.
    perm = [7, 19, 3, 25, 11, 1, 16, 22, 5, 13, 9, 24, 2, 18, 14, 6, 21, 10, 23, 4, 15, 12, 20, 8, 17]
    position = invperm(perm)
    shuffled = Mesh(q9.coords[perm], position[Array(q9.el2n)]; order = 2)
    for (a, b) in zip(FEMTools.partition_mesh(q9, [1, 2, 3, 4], 4), check_partition(shuffled, [1, 2, 3, 4], 4))
        @test sort(a.nodes[1:a.nowned]) == sort(perm[b.nodes[1:b.nowned]])
    end

    mesh = Mesh((0.0 .. 1.0) × (0.0 .. 1.0), ReferenceElement(LinearElement{2, 3, Float64}), (2, 2))
    @test_throws "assignment has 3 entries for 8 elements" FEMTools.partition_mesh(mesh, [1, 1, 1], 1)
    @test_throws "element 2 is assigned to part 3, outside 1:2" FEMTools.partition_mesh(mesh, [1, 3, 1, 1, 1, 1, 1, 1], 2)
    orphaned = Mesh(push!(Array(mesh.coords), SVector(2.0, 2.0)), Array(mesh.el2n))
    @test_throws "node 10 is referenced by no element" FEMTools.partition_mesh(orphaned, ones(Int, 8), 1)
end

@testset "partition local meshes" begin
    for (FP, element) in ((Float32, LinearElement{2, 3, Float32}), (Float64, QuadraticElement{2, 9, Float64}))
        mesh = Mesh((FP(0) .. FP(1)) × (FP(0) .. FP(1)), ReferenceElement(element), (2, 2))
        for part in FEMTools.partition_mesh(mesh, [mod1(e, 3) for e in 1:mesh.nels], 4)
            local_mesh = FEMTools.local_mesh(part, mesh.element)
            @test local_mesh.nnodes == length(part.nodes) && local_mesh.nels == length(part.elements)
            @test local_mesh.el2n == part.el2n && local_mesh.coords == part.coords
            # Physical boundary only: interface nodes stay interior.
            @test local_mesh.Γnodes == part.Γnodes
            @test local_mesh.geometry == mesh.geometry[part.elements]
        end
    end
    # Without MPI loaded, the execution layer names the missing packages.
    @test_throws "run `using MPI, PartitionedArrays`" FEMTools.distribute_mesh(nothing, nothing, nothing)
end

@testset "partitioned Dirichlet constraints and reconstruction" begin
    mesh = Mesh((0.0 .. 1.0) × (0.0 .. 1.0), ReferenceElement(LinearElement{2, 3, Float64}), (2, 2))
    parts = FEMTools.partition_mesh(mesh, [1, 1, 2, 1, 2, 3, 2, 2], 3)
    Γnodes = Array(mesh.Γnodes)
    bc = DirichletBoundaryCondition(vcat(Γnodes, Γnodes[1]), vcat(Float64.(Γnodes), Γnodes[1]))
    local_bcs = map(p -> FEMTools.localize_dirichlet(p, bc), parts)
    @test all(((p, b),) -> all(<=(p.nowned), b.DoFs), zip(parts, local_bcs))
    ids = [p.nodes[b.DoFs] for (p, b) in zip(parts, local_bcs)]
    @test sort!(reduce(vcat, ids)) == sort(Γnodes)
    @test all(((i, b),) -> b.vals == i, zip(ids, local_bcs))
    conflicting = DirichletBoundaryCondition([1, 1], [0.0, 1.0])
    @test_throws "conflicting boundary values at node 1" FEMTools.localize_dirichlet(parts[1], conflicting)
    @test_throws "boundary node 10 is outside 1:9" FEMTools.localize_dirichlet(parts[1], DirichletBoundaryCondition([10], [0.0]))

    @test_throws "node 2 is supplied by more than one part" FEMTools.reconstruct_owned(2, [[1, 2], [2]], [[0.0, 0.0], [0.0]])
    @test_throws "node 2 is supplied by no part" FEMTools.reconstruct_owned(2, [[1]], [[0.0]])
end

@testset "Metis element partitioning extension" begin
    mesh = Mesh((0.0 .. 1.0) × (0.0 .. 1.0), ReferenceElement(LinearElement{2, 3, Float64}), (8, 8))
    assignment = FEMTools.partition_elements(mesh, 4)
    @test length(assignment) == mesh.nels
    sizes = [count(==(p), assignment) for p in 1:4]
    @test sum(sizes) == mesh.nels
    @test maximum(sizes) <= 1.1 * mesh.nels / 4
    check_partition(mesh, assignment, 4)
    small = Mesh((0.0 .. 1.0) × (0.0 .. 1.0), ReferenceElement(LinearElement{2, 3, Float64}), (1, 1))
    # Metis may leave parts empty on a graph this small; labels stay within the requested parts.
    @test all(in(1:2), FEMTools.partition_elements(small, 4))
    @test FEMTools.partition_elements(small, 1) == [1, 1]
end
