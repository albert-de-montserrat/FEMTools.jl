using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU

@testset "3D prescribed velocities enter the first residual" begin
    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    domain = (0.0 .. 1.0) × (0.0 .. 1.0) × (0.0 .. 1.0)
    mesh = Mesh(CPU(), domain, element, (1, 1, 1))
    bc_values = ntuple(3) do component
        [component == 1 ? c[1] : 0.0 for c in mesh.coords[mesh.Γnodes]]
    end
    bc_v = map(vals -> DirichletBoundaryCondition(mesh.Γnodes, vals), bc_values)
    material = StokesMaterial(; η = 1.0, ρ0 = 1.0, g = (0.0, 0.0, 0.0))

    # Expanding x-boundaries produce a nonzero continuity residual even with
    # zero initial fields and no gravity. Checking after one iteration must
    # not report convergence from the initial all-zero state.
    dr = CellPressureStokesDR(mesh, material)
    @test_throws "3-D Stokes solve did not converge" solve!(
        dr, mesh, bc_v; check_interval = 1, max_iterations = 1, verbose = false,
    )
    dr = CellPressureStokesDR(mesh, material)
    stats = solve!(
        dr, mesh, bc_v; check_interval = 1, max_iterations = 1, verbose = false,
        collect_history = true, throw_on_failure = false,
    )
    @test !stats.converged
    @test stats.reached_total_iter
    @test stats.err_P > 1e-5
    @test only(stats.history).err_P == stats.err_P
    for component in 1:3
        @test Tuple(dr.v)[component][mesh.Γnodes] == bc_values[component]
    end

    # Starting from an already constrained guess must produce the same first
    # iteration as letting the solver apply those constraints itself.
    constrained = CellPressureStokesDR(mesh, material)
    for component in 1:3
        Tuple(constrained.v)[component][mesh.Γnodes] .= bc_values[component]
    end
    constrained_stats = solve!(
        constrained, mesh, bc_v; check_interval = 1, max_iterations = 1,
        verbose = false, throw_on_failure = false,
    )
    @test stats.err_P ≈ constrained_stats.err_P
    @test dr.P ≈ constrained.P
    for component in 1:3
        @test Tuple(dr.v)[component] ≈ Tuple(constrained.v)[component]
    end

    @test_throws "phases has 2 entries" CellPressureStokesDR(mesh, material; phases = [1, 1])
    @test_throws "phases must lie in 1:1" CellPressureStokesDR(mesh, material; phases = [2])
    @test_throws "purely viscous" CellPressureStokesDR(
        mesh, StokesMaterial(; η = 1.0, G = 1.0, g = (0.0, 0.0, 0.0)),
    )
    @test_throws "requires a Hex27 mesh" CellPressureStokesDR(
        Mesh(CPU(), domain, ReferenceElement(LinearElement{3, 8, Float64}), (1, 1, 1)), material,
    )
end
