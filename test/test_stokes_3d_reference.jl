include(joinpath(pkgdir(FEMTools), "examples", "stokes", "sinking_block", "sinking_block_3D_adj.jl"))

@testset "3D sinking-block sparse reference" begin
    forward = run_sinking_block_3d(;
        mesh_size = 0.25, nz = 2, half_width = 0.25, write_output = false,
    )
    residual = forward.A * forward.solution - forward.rhs
    @test norm(residual[forward.free]) < 1e-10
    @test Set(forward.cell_phase) == Set((1, 2))

    adjoint = solve_sinking_block_adjoint_3d(forward)
    @test adjoint.density_relative_error < 1e-6
    @test adjoint.viscosity_relative_error < 1e-6
end
