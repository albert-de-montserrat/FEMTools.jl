@testset "dimension-generic Stokes residual scaling" begin
    R = ([2.0, 4.0], [0.0, 0.0])
    PC = ([1.0, 2.0], [1.0, 1.0])
    err = FEMTools._scaled_velocity_residual_nd(R, PC)

    @test err ≈ 2.0
    @test FEMTools._scaled_velocity_residual_nd(1e-9 .* R, 1e-9 .* PC) ≈ err
end
