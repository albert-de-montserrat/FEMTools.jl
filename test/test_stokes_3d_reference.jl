include(joinpath(pkgdir(FEMTools), "examples", "stokes", "sinking_block", "sinking_block_3D_adj.jl"))

@testset "3D sinking-block sparse reference" begin
    forward = run_sinking_block_3d(;
        mesh_size = 0.25, nz = 2, half_width = 0.25, write_output = false,
    )
    residual = forward.A * forward.solution - forward.rhs
    @test norm(residual[forward.free]) < 1e-10
    @test Set(forward.cell_phase) == Set((1, 2))

    momentum = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    continuity = zeros(4, forward.mesh.nels)
    velocity = Tuple(eachrow(forward.velocity))
    FEMTools.assemble_stokes_momentum_residual_3d!(
        momentum, velocity, forward.pressure, forward.mesh,
        forward.cell_phase, forward.η, forward.ρ, forward.g,
    )
    FEMTools.assemble_stokes_pressure_residual_3d!(continuity, velocity, forward.mesh)
    assembled = vcat(vec(stack(momentum; dims = 1)), vec(continuity))
    @test assembled ≈ residual rtol = 1e-11 atol = 1e-11

    adjoint = solve_sinking_block_adjoint_3d(forward)
    @test adjoint.density_relative_error < 1e-6
    @test adjoint.viscosity_relative_error < 1e-6
end
