using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU

@testset "3D prescribed velocities enter the first residual" begin
    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    domain = (0.0 .. 1.0) × (0.0 .. 1.0) × (0.0 .. 1.0)
    mesh = Mesh(CPU(), domain, element, (1, 1, 1))
    fixed_nodes = ntuple(_ -> mesh.Γnodes, 3)
    bc_values = ntuple(3) do component
        [component == 1 ? c[1] : 0.0 for c in mesh.coords[mesh.Γnodes]]
    end
    velocity = ntuple(_ -> zeros(mesh.nnodes), 3)
    pressure = zeros(4, mesh.nels)

    # Expanding x-boundaries produce a nonzero continuity residual even with
    # zero initial fields and no gravity. Checking after one iteration must
    # not report convergence from the initial all-zero state.
    stats = solve_stokes_dyrel!(
        velocity, pressure, mesh, ones(Int, mesh.nels), (1.0,), (1.0,),
        (0.0, 0.0, 0.0), fixed_nodes;
        bc_values, ncheck = 1, total_iterMax = 1, verbose = false,
    )
    @test !stats.converged
    @test stats.reached_total_iter
    @test stats.err_P > 1e-5
    for component in 1:3
        @test velocity[component][fixed_nodes[component]] == bc_values[component]
    end

    # Starting from an already constrained guess must produce the same first
    # iteration as letting the solver apply those constraints itself.
    constrained_velocity = ntuple(_ -> zeros(mesh.nnodes), 3)
    for component in 1:3
        constrained_velocity[component][fixed_nodes[component]] .= bc_values[component]
    end
    constrained_pressure = zeros(4, mesh.nels)
    constrained_stats = solve_stokes_dyrel!(
        constrained_velocity, constrained_pressure, mesh, ones(Int, mesh.nels),
        (1.0,), (1.0,), (0.0, 0.0, 0.0), fixed_nodes;
        bc_values, ncheck = 1, total_iterMax = 1, verbose = false,
    )
    @test stats.err_P ≈ constrained_stats.err_P
    @test pressure ≈ constrained_pressure
    for component in 1:3
        @test velocity[component] ≈ constrained_velocity[component]
    end

    bad_values = (bc_values[1][1:(end - 1)], bc_values[2], bc_values[3])
    @test_throws DimensionMismatch solve_stokes_dyrel!(
        ntuple(_ -> zeros(mesh.nnodes), 3), zeros(4, mesh.nels), mesh,
        ones(Int, mesh.nels), (1.0,), (1.0,), (0.0, 0.0, 0.0), fixed_nodes;
        bc_values = bad_values, total_iterMax = 1, verbose = false,
    )
end
