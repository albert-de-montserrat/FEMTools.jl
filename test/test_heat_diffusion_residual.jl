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

@testset "incompressible limit stays finite where shape functions are negative" begin
    # A quadratic shape-function value below zero makes an interpolated K=Inf
    # produce Inf - Inf. Interpolating the compressibility 1/K instead keeps
    # the incompressible limit at zero.
    dNdx = @SMatrix [0.0 1.0; 0.0 -1.0]
    geo_el = ((dNdx, 1.0),)
    Nq = (SA[-0.25, 1.25],)          # partition of unity, one negative weight

    Tloc = SA[300.0, 300.0]
    T0loc = SA[300.0, 300.0]
    sloc = SA[1.0, 1.0]
    Ploc = SA[1.0e6, 1.0e6]

    incompressible = FEMTools.integrate_residual(
        Tloc, T0loc, geo_el, sloc, SA[1, 1], (1.0,), (1.0,), (2.0,), (0.01,), (Inf,),
        Ploc, 1.0, 300.0, Nq, Val(2),
    )
    @test all(isfinite, incompressible)

    # Two phases at K=Inf must reach the same incompressible limit.
    two_phase = FEMTools.integrate_residual(
        Tloc, T0loc, geo_el, sloc, SA[1, 2], (1.0, 1.0), (1.0, 1.0), (2.0, 2.0),
        (0.01, 0.01), (Inf, Inf), Ploc, 1.0, 300.0, Nq, Val(2),
    )
    @test all(isfinite, two_phase)
    @test two_phase ≈ incompressible

    # A very stiff but finite K approaches the incompressible result.
    stiff = FEMTools.integrate_residual(
        Tloc, T0loc, geo_el, sloc, SA[1, 1], (1.0,), (1.0,), (2.0,), (0.01,), (1.0e30,),
        Ploc, 1.0, 300.0, Nq, Val(2),
    )
    @test stiff ≈ incompressible
end

@testset "quadratic-tetrahedron thermal step converges when the mass term dominates" begin
    # The row-sum lumped mass ∫Nᵢ dΩ of the Tet11 element is negative at its
    # corners, so a mass-dominated (small Δt) step requires the consistent mass.
    include(joinpath(@__DIR__, "tet11_box_mesh.jl"))
    element = ReferenceElement(QuadraticElement{3, 11, Float64})
    coords, el2n, groups = build_tet11_inclusion_mesh()
    mesh = Mesh(CPU(), coords, el2n, element; workgroup = 1)
    thermal = ThermalDiffusionDR(CPU(), mesh.nnodes, (1.0,), (1.0,), (1.0,), (0.0,), (Inf,))
    hot = Set(el2n[:, findall(==(2), groups.phase)])
    copyto!(thermal.T, [n in hot ? 1.0 : 0.0 for n in eachindex(coords)])
    copyto!(thermal.T0, thermal.T)
    fixed = Int32.(vcat(groups.surface, groups.bottom))
    bc = DirichletBoundaryCondition(nothing, fixed, zeros(length(fixed)))

    solver!(thermal, 1.0e-4, mesh, bc; workgroup = 1, verbose = false, iterMax = 20_000)

    @test all(isfinite, thermal.T)
    @test maximum(abs, thermal.T) < 1.1
end
