include(joinpath(pkgdir(FEMTools), "examples", "stokes", "sinking_block", "sinking_block_3D_adj.jl"))

@testset "3D sinking-block DYREL and sparse reference" begin
    forward = run_sinking_block_3d(;
        mesh_size = 0.25, nz = 2, half_width = 0.25, write_output = false,
        verbose = false, build_reference = true,
    )
    residual = forward.A * forward.solution - forward.rhs
    @test forward.solve_stats.converged
    @test norm(residual[forward.free]) < 2e-5
    @test Set(forward.cell_phase) == Set((1, 2))
    @test all(iszero, forward.velocity[1][forward.fixed_nodes[1]])
    @test all(iszero, forward.velocity[2][forward.fixed_nodes[2]])
    @test all(iszero, forward.velocity[3][forward.fixed_nodes[3]])

    momentum = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    continuity = zeros(4, forward.mesh.nels)
    velocity = forward.velocity
    FEMTools.assemble_stokes_momentum_residual_3d!(
        momentum, velocity, forward.pressure, forward.mesh,
        forward.cell_phase, forward.η, forward.ρ, forward.g,
    )
    FEMTools.assemble_stokes_pressure_residual_3d!(continuity, velocity, forward.mesh)
    assembled = vcat(vec(stack(momentum; dims = 1)), vec(continuity))
    @test assembled ≈ residual rtol = 1e-11 atol = 1e-11

    coords = forward.mesh.coords
    fixed_nodes = (
        findall(c -> iszero(c[1]) || c[1] == 1, coords),
        findall(c -> iszero(c[2]) || c[2] == -1, coords),
        findall(c -> iszero(c[3]) || c[3] == 1, coords),
    )
    iterative_velocity = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    iterative_pressure = zeros(4, forward.mesh.nels)
    stats = solve_stokes_dyrel!(
        iterative_velocity, iterative_pressure, forward.mesh, forward.cell_phase,
        forward.η, forward.ρ, forward.g, fixed_nodes,
        verbose = false,
    )
    @test stats.converged
    @test norm(vec(stack(iterative_velocity; dims = 1)) -
               vec(stack(forward.velocity; dims = 1))) < 2e-4

    adjoint = solve_sinking_block_adjoint_3d(forward)
    block_nodes = unique(vec(Array(forward.mesh.el2n)[:, forward.cell_phase .== 2]))
    objective_load = ntuple(i -> begin
        load = zeros(forward.mesh.nnodes)
        i == 2 && (load[block_nodes] .= 1 / length(block_nodes))
        load
    end, 3)
    iterative_adjoint = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    iterative_adjoint_pressure = zeros(4, forward.mesh.nels)
    adjoint_stats = solve_stokes_adjoint_dyrel!(
        iterative_adjoint, iterative_adjoint_pressure, objective_load,
        forward.mesh, forward.cell_phase, forward.η, fixed_nodes,
        iterMax = 5000, verbose = false,
    )
    objective_vector = vcat(vec(stack(objective_load; dims = 1)), zeros(4forward.mesh.nels))
    exact_adjoint = zeros(length(forward.rhs))
    exact_adjoint[forward.free] =
        transpose(forward.A[forward.free, forward.free]) \ objective_vector[forward.free]
    exact_adjoint_velocity = reshape(@view(exact_adjoint[1:(3forward.mesh.nnodes)]), 3, :)
    @test adjoint.adjoint_stats.converged
    @test adjoint_stats.converged
    @test norm(vec(stack(iterative_adjoint; dims = 1)) - vec(exact_adjoint_velocity)) < 2e-4
    gradients = stokes_material_gradient_3d(
        iterative_velocity, iterative_adjoint, forward.mesh, forward.cell_phase,
        forward.η, forward.ρ, forward.g,
    )
    @test gradients.density_gradient ≈ adjoint.density_gradient rtol = 2e-3
    @test gradients.viscosity_gradient ≈ adjoint.viscosity_gradient rtol = 2e-3
    @test adjoint.density_relative_error < 2e-3
    @test adjoint.viscosity_relative_error < 2e-3
end
