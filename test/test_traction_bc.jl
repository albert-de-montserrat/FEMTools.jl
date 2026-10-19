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

@testset "boundary load" begin
    element = ReferenceElement(QuadraticElement{2, 6, Float64})
    mesh = Mesh(CPU(), (0.0 .. 2.0) × (0.0 .. 1.0), element, (4, 2); workgroup = 1)
    coords = Array(mesh.coords)
    el2n = Array(mesh.el2n)
    top = boundary_facets(mesh, x -> x[2] ≈ 1.0)
    bottom = boundary_facets(mesh, x -> x[2] ≈ 0.0)
    p = 3.0
    load(bc, m = mesh) = map(Array, FEMTools.assemble_boundary_load(bc, m; workgroup = 1))

    F = load(PressureBoundaryCondition(top, p))
    @test sum(F[2]) ≈ -p * 2
    @test abs(sum(F[1])) < 1.0e-12
    @test sum(load(PressureBoundaryCondition(top, x -> x[1]))[2]) ≈ -2.0
    # The outward normal of the base points in -y.
    @test sum(load(PressureBoundaryCondition(bottom, p))[2]) ≈ p * 2
    @test sum(load((PressureBoundaryCondition(top, p), PressureBoundaryCondition(bottom, p)))[2]) ≈ 0 atol = 1.0e-12

    Ft = load(TractionBoundaryCondition(top, SVector(1.0, 0.0)))
    @test sum(Ft[1]) ≈ 2.0
    @test abs(sum(Ft[2])) < 1.0e-12
    @test load(TractionBoundaryCondition(top, (1.0, 0.0))) == Ft
    @test_throws "traction must be" TractionBoundaryCondition(top, 1.0)
    @test_throws "pressure must be" PressureBoundaryCondition(top, SVector(1.0, 0.0))

    # Consistent load of a straight quadratic edge: h/6, 2h/3, h/6.
    edge_nodes = findall(x -> x[2] ≈ 1.0 && x[1] <= 0.5 + 1.0e-12, coords)
    F1 = load(PressureBoundaryCondition(boundary_facets(mesh, edge_nodes), p))
    h = 0.5
    expected = Dict(0.0 => h / 6, 0.25 => 2h / 3, 0.5 => h / 6)
    @test all(F1[2][n] ≈ -p * expected[coords[n][1]] for n in edge_nodes)
    @test count(!iszero, F1[2]) == 3

    top_nodes = findall(x -> x[2] ≈ 1.0, coords)
    remesh(c) = Mesh(CPU(), c, el2n, element; workgroup = 1)

    # A symmetric bump on one edge leaves the total load unchanged but pushes
    # the edge's corners apart.
    bumped = copy(coords)
    mid = findfirst(x -> x ≈ SVector(0.25, 1.0), coords)
    bumped[mid] = SVector(0.25, 1.1)
    bumped_mesh = remesh(bumped)
    Fb = load(PressureBoundaryCondition(boundary_facets(bumped_mesh, top_nodes), p), bumped_mesh)
    @test sum(Fb[2]) ≈ -p * 2
    @test abs(sum(Fb[1])) < 1.0e-12
    left, right = findfirst(x -> x ≈ SVector(0.0, 1.0), coords), findfirst(x -> x ≈ SVector(0.5, 1.0), coords)
    @test Fb[1][left] * Fb[1][right] < 0

    # Raising the top-right corner by 0.2 tilts the last edge: ΣFx = p Δy.
    tilted = copy(coords)
    corner = findfirst(x -> x ≈ SVector(2.0, 1.0), coords)
    edge_mid = findfirst(x -> x ≈ SVector(1.75, 1.0), coords)
    tilted[corner] = SVector(2.0, 1.2)
    tilted[edge_mid] = SVector(1.75, 1.1)
    tilted_mesh = remesh(tilted)
    Ftilt = load(PressureBoundaryCondition(boundary_facets(tilted_mesh, top_nodes), p), tilted_mesh)
    @test sum(Ftilt[1]) ≈ p * 0.2

    element32 = ReferenceElement(QuadraticElement{2, 6, Float32})
    mesh32 = Mesh(CPU(), (0.0f0 .. 2.0f0) × (0.0f0 .. 1.0f0), element32, (4, 2); workgroup = 1)
    F32 = load(PressureBoundaryCondition(boundary_facets(mesh32, x -> x[2] ≈ 1), 3.0f0), mesh32)
    @test eltype(F32[1]) == Float32
    @test sum(F32[2]) ≈ -6.0f0
end

@testset "stokes under surface pressure" begin
    backend = CPU()
    workgroup = 1
    Δt = 1.0
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (4, 4); workgroup)
    mesh = MixedMesh(mesh_v, element_P; workgroup)
    coords = Array(mesh.coords)
    p = 2.0

    # Free slip on the walls and the base; the top carries the load.
    wall_nodes = findall(x -> x[1] ≈ 0 || x[1] ≈ 1, coords)
    base_nodes = findall(x -> x[2] ≈ 0, coords)
    bc_vx = DirichletBoundaryCondition(nothing, wall_nodes, zeros(length(wall_nodes)))
    bc_vy = DirichletBoundaryCondition(nothing, base_nodes, zeros(length(base_nodes)))
    top = boundary_facets(mesh, x -> x[2] ≈ 1)

    function solve(traction; ρ0 = (0.0,), g = (0.0, 0.0))
        stokes = StokesDR(
            backend, mesh.nnodes, mesh.nnodesP,
            StokesMaterial(; η = (1.0,), ηb = (Inf,), G = (Inf,), α = (0.0,), ρ0, g),
        )
        γP = zeros(mesh.nnodesP)
        assemble_viscosity_weighted_pressure_scaling!(γP, stokes, mesh, 1.0, Δt; workgroup)
        stats = solve_stokes_dyrel!(
            stokes, mesh, bc_vx, bc_vy, Δt, γP;
            traction, workgroup, ncheck = 10, ϵ_tol = 1.0e-9,
            iterMax = 20_000, total_iterMax = 20_000, verbose = false,
        )
        return stats, stokes
    end

    stats, stokes = solve(PressureBoundaryCondition(top, p))
    @test stats.converged
    @test all(P -> isapprox(P, p; rtol = 1.0e-5), stokes.P)
    @test maximum(abs, stokes.v.y) < 1.0e-6
    # The top corners are wall nodes, so their vx stays prescribed.
    @test all(iszero, Array(stokes.v.x)[wall_nodes])

    left_top = boundary_facets(mesh, x -> x[2] ≈ 1 && x[1] <= 0.5 + 1.0e-12)
    right_top = boundary_facets(mesh, x -> x[2] ≈ 1 && x[1] >= 0.5 - 1.0e-12)
    _, split = solve((PressureBoundaryCondition(left_top, p), PressureBoundaryCondition(right_top, p)))
    @test split.P ≈ stokes.P

    # Under gravity the static pressure is the load plus the overburden.
    stats_g, gravity = solve(PressureBoundaryCondition(top, p); ρ0 = (1.0,), g = (0.0, -1.0))
    @test stats_g.converged
    P_exact = [p + 1 - coords[n][2] for n in Array(mesh.el2nP)]
    @test Array(gravity.P)[Array(mesh.DoFsP)] ≈ P_exact rtol = 1.0e-4

    element3 = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh3 = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0) × (0.0 .. 1.0), element3, (1, 1, 1))
    @test_throws "implemented only in 2-D" solve_stokes_dyrel!(
        ntuple(_ -> zeros(mesh3.nnodes), 3), zeros(4, mesh3.nels), mesh3,
        ones(Int, mesh3.nels), (1.0,), (1.0,), (0.0, 0.0, 0.0),
        ntuple(_ -> mesh3.Γnodes, 3);
        traction = PressureBoundaryCondition(top, p), verbose = false,
    )
end
