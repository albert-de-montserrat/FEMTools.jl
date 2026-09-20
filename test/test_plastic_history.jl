@testset "integration-point plastic history" begin
    λ = reshape(Float32[0.0, 2.0, 1.5, 0.5], 2, 2)
    εpl = zeros(Float32, 2, 2)
    D = fill(Float32(0.25), 2, 2)
    history = FEMTools.IntegrationPointPlasticHistory(λ, εpl, D)

    @test FEMTools.update_plastic_history!(history, Float32(0.2)) === history
    @test εpl == Float32[0.0 0.3; 0.4 0.1]
    @test D == fill(Float32(0.25), 2, 2)

    @test_throws DimensionMismatch FEMTools.update_plastic_history!(
        FEMTools.IntegrationPointPlasticHistory(λ, zeros(Float32, 2, 1), D), 0.1,
    )
    @test_throws ArgumentError FEMTools.update_plastic_history!(history, -1.0)

    λout = zeros(Float32, 2, 2)
    output = FEMTools.IntegrationPointPlasticHistoryOutput(
        λout, εpl, D, 2,
    )
    FEMTools.store_plastic_multiplier_at_ip!(output, 1, 3.0f0)
    @test λout[:, 2] == Float32[3.0, 0.0]
    @test FEMTools.store_plastic_multiplier_at_ip!(nothing, 1, 2.0, 0.5) === nothing

    dNdx = @SMatrix [1.0 0.0]
    v = (SA[2.0], SA[0.0])
    plastic = FEMTools.DruckerPrager((0.0,), (0.0,), (0.0,), (1.0,), (1.0,))
    λassembled = zeros(Float64, 1, 1)
    FEMTools.integrate_momentum_residual(
        v, SA[0.0], ((dNdx, 1.0),), SA[1],
        (1.0,), (Inf,), 1.0, (SA[1.0],), (SA[1.0],),
        nothing, plastic, nothing,
        FEMTools.IntegrationPointPlasticMultiplierOutput(λassembled, 1),
    )
    @test λassembled[1, 1] > 0

    history_assembled = FEMTools.IntegrationPointPlasticHistory(
        zeros(Float64, 1, 1), zeros(Float64, 1, 1), zeros(Float64, 1, 1),
    )
    FEMTools.integrate_momentum_residual(
        v, SA[0.0], ((dNdx, 1.0),), SA[1],
        (1.0,), (Inf,), 1.0, (SA[1.0],), (SA[1.0],),
        nothing, plastic, nothing,
        FEMTools.IntegrationPointPlasticHistoryOutput(history_assembled, 1, 0.5),
    )
    @test history_assembled.λ[1, 1] > 0
    @test history_assembled.εpl[1, 1] > 0

    shear_rate = FEMTools.plastic_strain_rate_invariant(
        1.0, (0.0, 0.0, 1.0),
    )
    @test shear_rate ≈ sqrt(4 / 3)
end
