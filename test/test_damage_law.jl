@testset "lagged damage law" begin
    law = FEMTools.DamageLaw((1.0f0,), (0.5f0,), (0.25f0,), (2.0f0,))
    plastic = FEMTools.DruckerPrager(
        (Float32(π / 6),), (0.0f0,), (10.0f0,), (1.0f0,), (2.0f0,);
        damage = law,
    )

    cϕ0, sϕ0, C0 = FEMTools.weakened_drucker_prager_parameters(plastic, 0.0f0, 1)
    cϕ, sϕ, C = FEMTools.weakened_drucker_prager_parameters(plastic, 0.4f0, 1)
    @test cϕ0 ≈ plastic.cosϕ[1]
    @test sϕ0 ≈ plastic.sinϕ[1]
    @test C0 == plastic.C[1]
    @test C < C0
    @test sϕ < sϕ0
    @test cϕ^2 + sϕ^2 ≈ 1.0f0
    phases_ip = Int[1 2; 2 1]
    law_two = FEMTools.DamageLaw(
        (1.0f0, 2.0f0), (0.5f0, 0.5f0), (0.25f0, 0.25f0), (2.0f0, Inf32),
    )
    εc_ip, th_ip = FEMTools.damage_update_parameters(phases_ip, law_two)
    @test εc_ip == Float32[1.0 2.0; 2.0 1.0]
    @test th_ip == Float32[2.0 Inf32; Inf32 2.0]
    @test_throws ArgumentError FEMTools.damage_update_parameters(Int[3;;], law_two)

    εc_q, th_q = FEMTools.damage_update_parameters(
        SA[0.25, 0.75], SA[1, 2], law_two,
    )
    @test εc_q ≈ 1.75f0
    @test th_q ≈ Inf32

    dNdx = @SMatrix [0.0 1.0]
    v = (SA[2.0], SA[0.0])
    plastic_yield = FEMTools.DruckerPrager(
        (Float32(π / 6),), (0.0f0,), (0.0f0,), (1.0f0,), (2.0f0,);
        damage = law,
    )
    τ0 = FEMTools.deviatoric_stress(
        v, dNdx, SA[1.0], (1.0,), (Inf,), SA[1], 1.0,
        (0.0, 0.0, 0.0), -10.0, plastic_yield,
    )
    τD = FEMTools.deviatoric_stress(
        v, dNdx, SA[1.0], (1.0,), (Inf,), SA[1], 1.0,
        (0.0, 0.0, 0.0), -10.0, plastic_yield, 0.8,
    )
    @test τD != τ0
    dNdx3 = @SMatrix [0.0 1.0 0.0]
    v3 = (SA[2.0], SA[0.0], SA[0.0])
    τ3 = FEMTools.deviatoric_stress(
        v3, dNdx3, SA[1.0], (1.0,), (Inf,), SA[1], 1.0,
        ntuple(_ -> 0.0, 6), -10.0, plastic_yield, 0.8,
    )
    @test all(isfinite, τ3)

    D = zeros(Float32, 2, 1)
    Δεpl = fill(0.2f0, 2, 1)
    εc = fill(1.0f0, 2, 1)
    th = fill(2.0f0, 2, 1)
    FEMTools.update_damage!(D, Δεpl, εc, th, 1.0f0)
    @test all(D .≈ (0.2f0 / 1.5f0))

    history = FEMTools.IntegrationPointPlasticHistory(
        fill(1.0f0, 2, 1), zeros(Float32, 2, 1), zeros(Float32, 2, 1),
    )
    old_εpl = copy(history.εpl)
    history.εpl .= 0.2f0
    FEMTools.update_damage_from_history!(
        history, old_εpl, εc, th, 1.0f0,
    )
    @test all(history.D .≈ (0.2f0 / 1.5f0))

    Dheal = fill(0.8f0, 1, 1)
    FEMTools.update_damage!(Dheal, zeros(Float32, 1, 1), εc[1:1, :], th[1:1, :], 1.0f0)
    @test Dheal[1] ≈ 0.8f0 / 1.5f0

    Dsat = fill(0.9f0, 1, 1)
    FEMTools.update_damage!(
        Dsat, fill(2.0f0, 1, 1), εc[1:1, :], fill(Float32(Inf), 1, 1), 1.0f0,
    )
    @test Dsat[1] == 1.0f0

    # The unclipped implicit update is first-order accurate for constant
    # plastic production with finite healing time.
    damage_after(dt) = begin
        n = round(Int, 1 / dt)
        Dref = zeros(Float64, 1, 1)
        for _ in 1:n
            FEMTools.update_damage!(Dref, fill(0.4 * dt, 1, 1),
                fill(1.0, 1, 1), fill(2.0, 1, 1), dt)
        end
        Dref[1]
    end
    exact = 0.4 * 2.0 * (1 - exp(-1 / 2.0))
    err_coarse = abs(damage_after(0.1) - exact)
    err_fine = abs(damage_after(0.05) - exact)
    @test err_fine < err_coarse
    @test err_fine / err_coarse < 0.6

    @test_throws DimensionMismatch FEMTools.update_damage!(D, Δεpl, εc[1:1, :], th, 1.0f0)
    @test_throws ArgumentError FEMTools.update_damage!(D, Δεpl, εc, th, -1.0f0)
    @test_throws ArgumentError FEMTools.update_damage!(D, Δεpl, εc, zeros(Float32, 2, 1), 1.0f0)

    @test_throws ArgumentError FEMTools.DamageLaw((0.0f0,), (0.5f0,), (0.5f0,), (1.0f0,))
    @test_throws ArgumentError FEMTools.DamageLaw((1.0f0,), (1.1f0,), (0.5f0,), (1.0f0,))
end
