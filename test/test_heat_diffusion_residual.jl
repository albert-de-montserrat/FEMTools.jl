@testset "heat diffusion residual Tref parameter" begin
    dNdx = @SMatrix [0.0 0.0; 0.0 0.0]
    geo_el = ((dNdx, 1.0),)
    Nq = (SA[0.5, 0.5],)

    Tloc = SA[300.0, 300.0]
    T0loc = SA[300.0, 300.0]
    sloc = SA[1.0, 1.0]
    phase_loc = SA[1, 1]
    k = (1.0,)
    Cp = (1.0,)
    ρ0 = (2.0,)
    α = (0.01,)
    K = (100.0,)
    Ploc = SA[0.0, 0.0]
    Δt = 1.0

    r1 = FEMTools.integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, 300.0, Nq, Val(2))
    r2 = FEMTools.integrate_residual(Tloc, T0loc, geo_el, sloc, phase_loc, k, Cp, ρ0, α, K, Ploc, Δt, 250.0, Nq, Val(2))

    @test r1 ≈ SA[0.25, 0.25]
    @test r2 ≈ 2 .* r1
end
