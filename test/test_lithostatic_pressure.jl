@testset "lithostatic residual parameters" begin
    dNdx = @SMatrix [0.0 1.0; 0.0 -1.0]
    geo_el = ((dNdx, 1.0),)
    Nq = (SA[0.5, 0.5],)

    Ploc = SA[0.0, 0.0]
    Tloc = SA[300.0, 300.0]
    phase_loc = SA[1, 1]
    ρ0 = (2.0,)
    α = (0.1,)
    K = (100.0,)

    r1 = FEMTools.lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, 300.0, SA[0.0, -1.0], Nq, Val(2))
    r2 = FEMTools.lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, 300.0, SA[0.0, -2.0], Nq, Val(2))
    r3 = FEMTools.lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, 290.0, SA[0.0, -1.0], Nq, Val(2))

    @test r1 ≈ SA[-2.0, 2.0]
    @test r2 ≈ 2 .* r1
    @test r3 ≈ zero(r3)
end
