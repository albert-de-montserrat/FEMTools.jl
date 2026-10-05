@testset "integration-point plastic history" begin
    λ = zeros(Float32, 2, 2)
    ε̇pl = reshape(Float32[0.0, 2.0, 1.5, 0.5], 2, 2)
    εpl = zeros(Float32, 2, 2)
    history = FEMTools.IntegrationPointPlasticHistory(λ, ε̇pl, εpl)

    @test FEMTools.update_plastic_history!(history, Float32(0.2)) === history
    @test εpl == Float32[0.0 0.3; 0.4 0.1]

    @test_throws DimensionMismatch FEMTools.update_plastic_history!(
        FEMTools.IntegrationPointPlasticHistory(λ, ε̇pl, zeros(Float32, 2, 1)), 0.1,
    )
    @test_throws ArgumentError FEMTools.update_plastic_history!(history, -1.0)

    output = FEMTools.IntegrationPointPlasticHistoryOutput(history, 2)
    FEMTools.store_plastic_multiplier_at_ip!(output, 1, 3.0f0, 4.0f0)
    @test λ[:, 2] == Float32[3.0, 0.0]
    @test ε̇pl[1, 2] == 4.0f0
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

    # Assembly only refreshes λ and ε̇pl, so repeating it leaves εpl untouched.
    history_assembled = FEMTools.IntegrationPointPlasticHistory(
        zeros(Float64, 1, 1), zeros(Float64, 1, 1), zeros(Float64, 1, 1),
    )
    for _ in 1:2
        FEMTools.integrate_momentum_residual(
            v, SA[0.0], ((dNdx, 1.0),), SA[1],
            (1.0,), (Inf,), 1.0, (SA[1.0],), (SA[1.0],),
            nothing, plastic, nothing,
            FEMTools.IntegrationPointPlasticHistoryOutput(history_assembled, 1),
        )
    end
    @test history_assembled.λ[1, 1] > 0
    @test history_assembled.ε̇pl[1, 1] > 0
    @test iszero(history_assembled.εpl[1, 1])
    FEMTools.update_plastic_history!(history_assembled, 0.5)
    @test history_assembled.εpl[1, 1] ≈ 0.5 * history_assembled.ε̇pl[1, 1]

    @test FEMTools.plastic_strain_rate_invariant(1.0, (0.0, 0.0, 1.0)) ≈ sqrt(4 / 3)
    @test FEMTools.plastic_strain_rate_invariant(1.0, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)) ≈ sqrt(4 / 3)

    # The 3-D momentum residual writes the multiplier store too.
    dNdx3 = @SMatrix [1.0 0.0 0.0]
    v3 = (SA[2.0], SA[0.0], SA[0.0])
    history3 = FEMTools.IntegrationPointPlasticHistory(
        zeros(Float64, 1, 1), zeros(Float64, 1, 1), zeros(Float64, 1, 1),
    )
    FEMTools.integrate_momentum_residual(
        v3, SA[0.0], nothing, SA[0.0], ((dNdx3, 1.0),), SA[1],
        (1.0,), (Inf,), (0.0,), (1.0,), (Inf,), (0.0, 0.0, 0.0), 0.0, 1.0,
        (SA[1.0],), (SA[1.0],),
        nothing, plastic, nothing,
        FEMTools.IntegrationPointPlasticHistoryOutput(history3, 1),
    )
    @test history3.λ[1, 1] > 0
    @test history3.ε̇pl[1, 1] > 0
end
