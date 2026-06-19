@testset "interp2ip" begin
    N = SA[0.25, 0.75]
    v = SA[2.0, 6.0]

    @test FEMTools.interp2ip(N, v) ≈ 5.0

    ρ0 = SA[2.0, 4.0]
    T = SA[300.0, 310.0]
    α = SA[0.1, 0.2]
    P = SA[0.0, 2.0]
    K = SA[100.0, 200.0]
    Tref = 300.0

    ρ = FEMTools.interp2ip(
        N,
        (ρ0i, Ti, αi, Pi, Ki) -> ρ0i * (1 - αi * (Ti - Tref) + Pi / Ki),
        (ρ0, T, α, P, K),
    )

    @test ρ ≈ 0.25 * 2.0 + 0.75 * (4.0 * (1 - 0.2 * 10.0 + 2.0 / 200.0))
end

@testset "interp2ip_phase" begin
    N = SA[0.25, 0.75]
    phase = SA[2, 1]
    props = (2.0, 10.0)

    @test FEMTools.interp2ip_phase(N, props, phase) ≈ 0.25 * 10.0 + 0.75 * 2.0

    N32 = SA[0.25f0, 0.75f0]
    props32 = (2.0f0, 10.0f0)
    q32 = FEMTools.interp2ip_phase(N32, props32, phase)

    @test q32 isa Float32
    @test q32 ≈ 0.25f0 * 10.0f0 + 0.75f0 * 2.0f0
end
