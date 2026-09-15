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
end
