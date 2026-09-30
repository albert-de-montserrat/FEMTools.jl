# Small material-point gate for the 2-D Popov tensile-cap benchmark.
# The spatial example should not be added until this local behavior is stable.

using Test
using FEMTools
using StaticArrays

@testset "Popov 2-D tensile cap material point" begin
    ϕ = deg2rad(30.0)
    ψ = deg2rad(0.0)
    k = sin(ϕ)
    kq = sin(ψ)
    c = 20.0 * cos(ϕ)
    pT = -1.0
    geom = FEMTools.cap_geometry(k, kq, c, pT)

    @test FEMTools.cap_yield_function(0.0, pT, k, c, geom) ≈ 0.0 atol = 1e-12

    elastic = FEMTools.cap_return_map(
        0.0, -0.5, 1.0, 64.0, k, kq, c, pT, 0.1,
    )
    @test elastic.converged
    @test elastic.λ == 0.0
    @test elastic.P == -0.5

    yielded = FEMTools.cap_return_map(
        0.0, -1.2, 1.0, 64.0, k, kq, c, pT, 0.1,
    )
    @test yielded.converged
    @test yielded.λ > 0
    # The regularized update moves the tensile pressure toward pT without
    # crossing the cap in one local step.
    @test -1.2 < yielded.P < pT

    @testset "constitutive pressure reaches momentum path" begin
        dNdx = @SMatrix [1.0 0.0; 0.0 1.0; 0.0 0.0]
        Nv = SA[1.0, 0.0, 0.0]
        phase = SA[1, 1, 1]
        plastic = FEMTools.DruckerPragerCap(
            (ϕ,), (ψ,), (20.0,), (pT,), (0.1,), (64.0,),
        )
        τ, P_local = FEMTools.deviatoric_stress_and_pressure(
            (SA[0.0, 0.0, 0.0], SA[0.0, 0.0, 0.0]),
            dNdx, Nv, (1.0,), (1.0,), phase, 1.0,
            (0.0, 0.0, 0.0), -1.2, plastic,
        )
        @test τ == (0.0, 0.0, 0.0)
        @test P_local ≈ yielded.P
    end

    @testset "corrected pressure reaches the integration-point store" begin
        τxx, τyy, τxy = (zeros(2, 3) for _ in 1:3)
        P_store = fill(NaN, 2, 3)

        @test FEMTools.store_stress_at_ip!(nothing, 1, 1.0, 2.0, 3.0, -1.5) === nothing

        # Three matrices keep the stress-only behaviour: no pressure recorded.
        out3 = FEMTools._stress_output((τxx, τyy, τxy), 2)
        @test out3.P === nothing
        FEMTools.store_stress_at_ip!(out3, 1, 1.0, 2.0, 3.0, -1.5)
        @test τxx[1, 2] == 1.0
        @test all(isnan, P_store)

        # A fourth matrix records the cap's corrected pressure, not the trial.
        out4 = FEMTools._stress_output((τxx, τyy, τxy, P_store), 2)
        FEMTools.store_stress_at_ip!(out4, 1, 1.0, 2.0, 3.0, yielded.P)
        @test P_store[1, 2] == yielded.P
        @test P_store[1, 2] != -1.2
        @test isnan(P_store[2, 3])
    end

    @testset "cohesion softening law" begin
        @test FEMTools.softened_cohesion(20.0, 5.0, -3.0, 0.0) == 20.0
        @test FEMTools.softened_cohesion(20.0, 5.0, -3.0, 2.0) == 14.0
        @test FEMTools.softened_cohesion(20.0, 5.0, -3.0, 10.0) == 5.0
        soft = FEMTools.DruckerPragerCap(
            (ϕ,), (ψ,), (20.0,), (pT,), (0.1,), (64.0,);
            C_min = (5.0,), H_C = (-3.0,),
        )
        @test soft.C_min == (5.0,)
        @test soft.H_C == (-3.0,)

        dNdx = @SMatrix [1.0 0.0; 0.0 1.0; 0.0 0.0]
        Nv = SA[1.0, 0.0, 0.0]
        phase = SA[1, 1, 1]
        _, P_soft = FEMTools.deviatoric_stress_and_pressure(
            (SA[0.0, 0.0, 0.0], SA[0.0, 0.0, 0.0]),
            dNdx, Nv, (1.0,), (1.0,), phase, 1.0,
            (0.0, 0.0, 0.0), -1.2, soft, 2.0,
        )
        softened = FEMTools.cap_return_map(
            0.0, -1.2, 1.0, 64.0, k, kq, 14.0 * cos(ϕ), pT, 0.1,
        )
        @test P_soft ≈ softened.P
    end
end
