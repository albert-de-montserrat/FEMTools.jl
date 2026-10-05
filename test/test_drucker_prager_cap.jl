# Drucker-Prager tensile cap geometry and composite yield surface.
# Reference: Popov, Berlie & Kaus (2025), Geosci. Model Dev. 18, 7035-7058,
# doi:10.5194/gmd-18-7035-2025, Eqs. 13-22.

using ForwardDiff
using FEMTools: cap_geometry, cap_yield_function, cap_return_map

# Unit-scale parameters keep the finite-difference gradient checks well
# conditioned; a Pa-scale set is exercised separately below.
const CAP_ϕ  = deg2rad(30.0)
const CAP_Ψ  = deg2rad(10.0)
const CAP_k  = sin(CAP_ϕ)
const CAP_kq = sin(CAP_Ψ)
const CAP_c  = 10.0 * cos(CAP_ϕ)   # c = C cos(ϕ)
const CAP_pT = -5.0                # tensile strength, compression-positive

# Central difference of the yield function in the meridional plane.
function cap_∇F(τII, P, k, c, geom, h)
    ∂τ = (cap_yield_function(τII + h, P, k, c, geom) -
          cap_yield_function(τII - h, P, k, c, geom)) / 2h
    ∂P = (cap_yield_function(τII, P + h, k, c, geom) -
          cap_yield_function(τII, P - h, k, c, geom)) / 2h
    return ∂τ, ∂P
end

@testset "Cap scalar coefficients and softened geometry" begin
    for T in (Float32, Float64)
        ϕ, ψ = T(π / 6), T(π / 18)
        k, kq, cosϕ = sin(ϕ), sin(ψ), cos(ϕ)
        C, pT = T(10), T(-5)
        c = C * cosϕ
        geom = cap_geometry(k, kq, c, pT)
        # Opposite yield/potential branches on either side of their delimiter.
        probes = ((T(1), T(-6)), (T(30), T(20)),
            (geom.τ_d / 2, geom.p_y - k * geom.τ_d / 2 - T(0.1)),
            (2geom.τ_d, geom.p_y - 2k * geom.τ_d + T(0.1)))
        for (s, p) in probes[3:4]
            @test (p + k*s ≥ geom.p_y) != (p + kq*s ≥ geom.p_q)
        end
        for (s, p) in probes
            F, Aτ, Ap = @inferred FEMTools.cap_invariants(s, p, k, kq, c, pT)
            @test (F, Aτ, Ap) isa NTuple{3, T}
            expected_F = p + k*s ≥ geom.p_y ? s - k*p - c :
                geom.a * (hypot(s, p - geom.p_y) - geom.R_y)
            @test F ≈ expected_F
            if p + kq*s ≥ geom.p_q
                @test Aτ == T(0.5)
                @test Ap == kq
            else
                Rq = hypot(s, p - geom.p_q)
                @test 2Aτ ≈ geom.b * s / Rq
                @test Ap ≈ -geom.b * (p - geom.p_q) / Rq
            end
        end
        _, Aτ, Ap = FEMTools.cap_invariants(zero(T), T(-6), k, kq, c, pT)
        @test iszero(Aτ)
        @test Ap ≈ geom.b
        # The reference chooses the shear branch at the potential centre.
        _, Aτ, Ap = FEMTools.cap_invariants(zero(T), geom.p_q, k, kq, c, pT)
        @test (Aτ, Ap) == (T(0.5), kq)

        material = DruckerPragerCap((ϕ,), (ψ,), (C,), (pT,), (T(0.1),), (T(4),);
            C_min = (T(4),), H_C = (T(-2),))
        for (γ, softened_C) in ((zero(T), C), (T(2), T(6)), (T(100), T(4)))
            result = @inferred FEMTools.cap_invariants(material, 1, T(1), T(-6), γ)
            @test result isa NTuple{3, T}
            @test result == FEMTools.cap_invariants(T(1), T(-6), k, kq, softened_C*cosϕ, pT)
            @test cap_geometry(k, kq, softened_C*cosϕ, pT).R_y > 0
        end
        @test FEMTools.cap_invariants(material, 1, T(1), T(-6), nothing) ==
            FEMTools.cap_invariants(material, 1, T(1), T(-6), zero(T))

        # Initial cohesion is valid, but the softened cap would have negative radius.
        @test_throws ArgumentError DruckerPragerCap((ϕ,), (ψ,), (C,), (pT,), (T(0.1),), (T(4),);
            C_min = (T(1),), H_C = (T(-2),))
        for Kb in (zero(T), T(-1), T(Inf), T(NaN))
            @test_throws ArgumentError DruckerPragerCap((ϕ,), (ψ,), (C,), (pT,), (T(0.1),), (Kb,))
        end
        for bad in (T(Inf), T(NaN))
            @test_throws ArgumentError DruckerPragerCap((ϕ,), (ψ,), (bad,), (pT,), (T(0.1),), (T(4),))
            @test_throws ArgumentError DruckerPragerCap((ϕ,), (ψ,), (C,), (pT,), (bad,), (T(4),))
        end
    end
    # Differentiate an independent cap potential: scalar Aτ is half dQ/ds.
    geom = cap_geometry(CAP_k, CAP_kq, CAP_c, CAP_pT)
    x = SVector(1.0, -6.0)
    Q(y) = geom.b * hypot(y[1], y[2] - geom.p_q)
    ∇Q = ForwardDiff.gradient(Q, x)
    _, Aτ, Ap = FEMTools.cap_invariants(x..., CAP_k, CAP_kq, CAP_c, CAP_pT)
    @test 2Aτ ≈ ∇Q[1]
    @test Ap ≈ -∇Q[2]
    coefficients(y) = SVector(FEMTools.cap_invariants(y..., CAP_k, CAP_kq, CAP_c, CAP_pT))
    J = ForwardDiff.jacobian(coefficients, x)
    h = 1.0e-5
    for j in 1:2
        step = SVector(ntuple(i -> i == j ? h : 0.0, 2))
        @test J[:, j] ≈ (coefficients(x + step) - coefficients(x - step)) / (2h) rtol = 1.0e-7
    end
end

@testset "Drucker-Prager tensile cap" begin
    geom = cap_geometry(CAP_k, CAP_kq, CAP_c, CAP_pT)

    @testset "cap geometry identities" begin
        @test geom.a ≈ √(1 + CAP_k^2)
        @test geom.b ≈ √(1 + CAP_kq^2)

        # The two conditions that fix the cap: it meets the pressure axis at pT,
        # and it is tangent to the shear line τII = k P + c.  Both must give the
        # same radius, which is what pins p_y (Eq. 15).
        @test geom.R_y ≈ geom.p_y - CAP_pT
        @test geom.R_y ≈ (CAP_k * geom.p_y + CAP_c) / geom.a

        # Eq. 16 defines τ_d as k p_d + c; tangency makes that R_y/a.
        @test geom.τ_d ≈ CAP_k * geom.p_d + CAP_c
        @test geom.τ_d ≈ geom.R_y / geom.a

        # The delimiter sits on the cap circle and on the shear line at once.
        @test √(geom.τ_d^2 + (geom.p_d - geom.p_y)^2) ≈ geom.R_y

        # Flow-potential centre placed so its own transition passes through the
        # same delimiter (Eq. 17).
        @test geom.p_q ≈ geom.p_d + CAP_kq * geom.τ_d

        # Ordering: apex is the most tensile point, delimiter sits between apex
        # and centre, and a positive radius needs c + k pT > 0.
        @test CAP_c + CAP_k * CAP_pT > 0
        @test geom.R_y > 0
        @test CAP_pT < geom.p_d < geom.p_y
        @test geom.τ_d > 0
    end

    @testset "yield surface is the zero level set" begin
        # Apex: pure tensile failure at P = pT.
        @test cap_yield_function(0.0, CAP_pT, CAP_k, CAP_c, geom) ≈ 0 atol = 1e-12

        # On the cap circle, between the apex (θ = π) and the delimiter.
        θ_d = atan(1 / geom.a, -CAP_k / geom.a)
        for θ in range(θ_d, π; length = 9)
            P   = geom.p_y + geom.R_y * cos(θ)
            τII = geom.R_y * sin(θ)
            @test cap_yield_function(τII, P, CAP_k, CAP_c, geom) ≈ 0 atol = 1e-10
            # Every such point must actually select the cap branch.
            @test P + CAP_k * τII ≤ geom.p_y + 1e-10
        end

        # On the Drucker-Prager line, for pressures above the delimiter.
        for P in range(geom.p_d, geom.p_d + 20; length = 9)
            τII = CAP_k * P + CAP_c
            @test cap_yield_function(τII, P, CAP_k, CAP_c, geom) ≈ 0 atol = 1e-10
        end

        # Inside the surface F < 0, outside F > 0, on both branches.
        @test cap_yield_function(0.0, geom.p_y, CAP_k, CAP_c, geom) < 0
        @test cap_yield_function(0.0, CAP_pT - 1.0, CAP_k, CAP_c, geom) > 0
        @test cap_yield_function(2geom.τ_d, geom.p_d + 20, CAP_k, CAP_c, geom) < 0
    end

    @testset "branches agree on the switching ray" begin
        # The domain test switches on P + k τII = p_y, and the delimiter lies on
        # that line.  Both branches reduce to a(a τII − R_y) along the whole ray,
        # so they agree exactly, not merely at the delimiter.
        for τII in range(0.0, 3geom.τ_d; length = 11)
            P = geom.p_y - CAP_k * τII
            shear = τII - CAP_k * P - CAP_c
            cap   = geom.a * (√(τII^2 + (P - geom.p_y)^2) - geom.R_y)
            @test shear ≈ cap atol = 1e-10
            @test cap_yield_function(τII, P, CAP_k, CAP_c, geom) ≈ shear atol = 1e-10
        end

        # The delimiter is on the ray.
        @test geom.p_d + CAP_k * geom.τ_d ≈ geom.p_y
    end

    @testset "gradient magnitude is a on both branches" begin
        h = 1.0e-5
        # Shear side, tensile-cap side, and straddling the delimiter.
        probes = (
            (2geom.τ_d, geom.p_d + 10.0),          # shear domain
            (0.2geom.τ_d, CAP_pT - 0.5),           # cap domain
            (geom.τ_d, geom.p_d + 0.05),           # just inside shear
            (geom.τ_d, geom.p_d - 0.05),           # just inside cap
        )
        for (τII, P) in probes
            ∂τ, ∂P = cap_∇F(τII, P, CAP_k, CAP_c, geom, h)
            @test √(∂τ^2 + ∂P^2) ≈ geom.a rtol = 1e-6
        end

        # Analytic gradients, checked against the same differences.
        τII, P = 2geom.τ_d, geom.p_d + 10.0
        ∂τ, ∂P = cap_∇F(τII, P, CAP_k, CAP_c, geom, h)
        @test ∂τ ≈ 1 rtol = 1e-6
        @test ∂P ≈ -CAP_k rtol = 1e-6

        τII, P = 0.2geom.τ_d, CAP_pT - 0.5
        R̂y = √(τII^2 + (P - geom.p_y)^2)
        ∂τ, ∂P = cap_∇F(τII, P, CAP_k, CAP_c, geom, h)
        @test ∂τ ≈ geom.a * τII / R̂y rtol = 1e-6
        @test ∂P ≈ geom.a * (P - geom.p_y) / R̂y rtol = 1e-6
    end

    @testset "gradient is continuous across the delimiter" begin
        h, δ = 1.0e-5, 1.0e-4
        # Approach the delimiter from the shear side and the cap side along the
        # surface normal direction; the gradients must converge to each other.
        ∂τ⁺, ∂P⁺ = cap_∇F(geom.τ_d, geom.p_d + δ, CAP_k, CAP_c, geom, h)
        ∂τ⁻, ∂P⁻ = cap_∇F(geom.τ_d, geom.p_d - δ, CAP_k, CAP_c, geom, h)
        @test ∂τ⁺ ≈ ∂τ⁻ rtol = 1e-4
        @test ∂P⁺ ≈ ∂P⁻ rtol = 1e-4
        @test ∂τ⁺ ≈ 1 rtol = 1e-4
        @test ∂P⁺ ≈ -CAP_k rtol = 1e-4
    end

    @testset "reduces to Drucker-Prager in the shear domain" begin
        # The shear branch is bit-for-bit the surface the existing return map
        # uses, F = τII − C cos(ϕ) − P sin(ϕ), for any cap that is out of reach.
        C = 10.0
        for pT in (-5.0, -50.0, -5.0e3)
            g = cap_geometry(CAP_k, CAP_kq, C * cos(CAP_ϕ), pT)
            for (τII, P) in ((5.0, 20.0), (30.0, 100.0), (1.0, 60.0))
                if P + CAP_k * τII ≥ g.p_y
                    @test cap_yield_function(τII, P, CAP_k, C * cos(CAP_ϕ), g) ==
                        τII - C * cos(CAP_ϕ) - P * sin(CAP_ϕ)
                end
            end
        end

        # A far tensile strength pushes the cap out of the working range entirely.
        g = cap_geometry(CAP_k, CAP_kq, CAP_c, -1.0e6)
        @test 0.0 + CAP_k * 0.0 ≥ g.p_y
    end

    @testset "Pa-scale parameters and Float32" begin
        # Realistic magnitudes: C = 10 MPa, tensile strength 5 MPa.
        for FP in (FP64, FP32)
            k  = FP(sin(deg2rad(30)))
            kq = FP(sin(deg2rad(10)))
            c  = FP(10.0e6 * cos(deg2rad(30)))
            pT = FP(-5.0e6)
            g  = cap_geometry(k, kq, c, pT)

            @test g isa NamedTuple
            @test all(x -> x isa FP, values(g))
            @test g.R_y ≈ g.p_y - pT
            @test g.R_y ≈ (k * g.p_y + c) / g.a rtol = sqrt(eps(FP))
            @test g.τ_d ≈ k * g.p_d + c rtol = sqrt(eps(FP))

            F = cap_yield_function(zero(FP), pT, k, c, g)
            @test F isa FP
            @test abs(F) ≤ sqrt(eps(FP)) * g.R_y

        end
    end

    @testset "inference" begin
        @test (@inferred cap_geometry(CAP_k, CAP_kq, CAP_c, CAP_pT)) isa NamedTuple
        @test (@inferred cap_yield_function(1.0, 1.0, CAP_k, CAP_c, geom)) isa Float64
    end

    @testset "return-map derivative matches central differences" begin
        ηve, KΔt, η_reg = 2.0, 10.0, 1.0
        cases = ((30.0, 0.0), (0.5, -6.0), (geom.τ_d, geom.p_d - 1.0e-3))
        for (τ_trial, P_trial) in cases
            fτ(τ) = cap_return_map(
                τ, P_trial, ηve, KΔt, CAP_k, CAP_kq, CAP_c, CAP_pT, η_reg,
            ).τII
            fP(P) = cap_return_map(
                τ_trial, P, ηve, KΔt, CAP_k, CAP_kq, CAP_c, CAP_pT, η_reg,
            ).τII
            h = 1.0e-5
            dτ_fd = (fτ(τ_trial + h) - fτ(τ_trial - h)) / (2h)
            dP_fd = (fP(P_trial + h) - fP(P_trial - h)) / (2h)
            @test ForwardDiff.derivative(fτ, τ_trial) ≈ dτ_fd rtol = 1.0e-5 atol = 1.0e-7
            @test ForwardDiff.derivative(fP, P_trial) ≈ dP_fd rtol = 1.0e-5 atol = 1.0e-7
        end
    end
end

@testset "Coupled cap return map" begin
    # Reference material point: C = 1, ϕ = 30°, pT = -0.5, ηve = G = 1, K = 4, Δt = 1.
    ϕ = deg2rad(30.0)
    k, c, pT, ηve, KΔt = sin(ϕ), cos(ϕ), -0.5, 1.0, 4.0
    a = sqrt(1 + k^2)
    residual(ret, s, p, kq, η_reg) = begin
        F, Aτ, Ap = FEMTools.cap_invariants(ret.τII, ret.P, k, kq, c, pT)
        (ret.τII - s + 2ηve * ret.λ * Aτ, ret.P - p - KΔt * ret.λ * Ap, F - η_reg * ret.λ)
    end

    @testset "ψ = $(rad2deg(ψ))°, η_reg = $η_reg" for ψ in (0.0, deg2rad(5.0)), η_reg in (0.1, 0.0)
        kq = sin(ψ)
        # Hydrostatic tension, mixed tension, compression/shear.
        for (s, p) in ((0.0, -1.0), (0.0, -0.8), (1.0, -1.0), (1.0, -0.8), (3.0, 2.0))
            ret = cap_return_map(s, p, ηve, KΔt, k, kq, c, pT, η_reg)
            @test ret.converged
            @test ret.λ > 0
            @test all(r -> abs(r) ≤ 1.0e-12, residual(ret, s, p, kq, η_reg))
            p < 0 && @test ret.Ap > 0   # tension opens
        end
        # Pure tension with ψ = 0 has a closed form and no deviatoric stress.
        if kq == 0
            ret = cap_return_map(0.0, -1.0, ηve, KΔt, k, kq, c, pT, η_reg)
            @test ret.λ ≈ a * (pT + 1.0) / (η_reg + a * KΔt)
            @test ret.τII == 0
        end
        # Radial Drucker-Prager limit on the shear branch: s = s_trial - ηve λ.
        s, p = 3.0, 2.0
        λ = (s - k * p - c) / (ηve + η_reg + KΔt * k * kq)
        ret = cap_return_map(s, p, ηve, KΔt, k, kq, c, pT, η_reg)
        @test ret.λ ≈ λ
        @test ret.τII ≈ s - ηve * λ
        @test ret.P ≈ p + KΔt * kq * λ
    end

    @testset "elastic unloading" begin
        ret = cap_return_map(0.2, 0.0, ηve, KΔt, k, 0.0, c, pT, 0.1)
        @test ret.converged
        @test (ret.τII, ret.P, ret.λ) == (0.2, 0.0, 0.0)
    end

    @testset "yield and potential branches switch independently" begin
        kq = sin(deg2rad(20.0))
        geom = cap_geometry(k, kq, c, pT)
        # Trial state on the shear side of the yield ray but the cap side of the
        # potential ray, and the converse.
        for (s, p) in ((geom.τ_d / 2 + 1, geom.p_y - k * geom.τ_d / 2 + 0.01),
                       (2geom.τ_d, geom.p_y - 2k * geom.τ_d - 0.01))
            ret = cap_return_map(s, p, ηve, KΔt, k, kq, c, pT, 0.1)
            @test ret.converged
            @test all(r -> abs(r) ≤ 1.0e-12, residual(ret, s, p, kq, 0.1))
        end
    end

    @testset "tensor update is radial and rotation invariant" begin
        τ = (1.2, -0.3, 0.4)
        out = FEMTools.cap_local_update(τ, -0.6, ηve, KΔt, k, 0.1, c, pT, 0.1)
        ret = cap_return_map(FEMTools.second_invariant(τ), -0.6, ηve, KΔt, k, 0.1, c, pT, 0.1)
        @test FEMTools.second_invariant(out.τ) ≈ ret.τII
        @test out.P ≈ ret.P
        @test out.γdot ≈ ret.λ * ret.Aτ
        @test out.θdot ≈ ret.λ * ret.Ap
        rotate(t, θ) = begin
            R = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
            M = R * SMatrix{2, 2}(t[1], t[3], t[3], t[2]) * R'
            (M[1, 1], M[2, 2], M[1, 2])
        end
        rot = FEMTools.cap_local_update(rotate(τ, 0.7), -0.6, ηve, KΔt, k, 0.1, c, pT, 0.1)
        @test all(rot.τ .≈ rotate(out.τ, 0.7))
        @test rot.P ≈ out.P
        # Hydrostatic tension through the AD-safe invariant floor.
        zero_state = FEMTools.cap_local_update((0.0, 0.0, 0.0), -1.0, ηve, KΔt, k, 0.0, c, pT, 0.1)
        @test zero_state.τ == (0.0, 0.0, 0.0)
        @test zero_state.P ≈ cap_return_map(0.0, -1.0, ηve, KΔt, k, 0.0, c, pT, 0.1).P
    end

    @testset "Float32" begin
        ret = @inferred cap_return_map(1.0f0, -0.8f0, 1.0f0, 4.0f0, Float32(k), 0.0f0, Float32(c), -0.5f0, 0.1f0)
        @test ret.converged
        @test ret.λ isa Float32
    end

    @testset "failures are explicit" begin
        for (η_, K_) in ((0.0, KΔt), (ηve, 0.0), (ηve, Inf))
            @test !cap_return_map(0.0, -1.0, η_, K_, k, 0.0, c, pT, 0.1).converged
        end
        @test !cap_return_map(0.0, -1.0, ηve, KΔt, k, 0.0, c, pT, 0.1; maxiter = 0).converged
        failed = FEMTools.cap_local_update((0.0, 0.0, 0.0), -1.0, 0.0, KΔt, k, 0.0, c, pT, 0.1)
        @test all(isnan, failed.τ) && isnan(failed.P) && isnan(failed.γdot)
    end
end

@testset "Pinned JustRelax cap fixtures" begin
    # Generated from JustRelax 14ffc40ca59716bdac68cf0dc3342cc9e0a462c8,
    # test/test_cap_return_mapping.jl units: C=η=G=Δt=1, K=4, ϕ=30°,
    # pT=-0.5. Its Maxwell viscosity is ηve=0.5. Columns after the inputs are
    # corrected (τII, P, λ, θdot).
    fixtures = (
        (0.0, 0.0, 0.0, -1.0, 0.0, -0.5, 0.125, 0.125),
        (0.0, 0.0, 0.0, -0.8, 0.0, -0.5, 0.07500000000000002, 0.07500000000000002),
        (0.0, 0.0, 1.0, -1.0, 0.7138809973230174, -0.1988680769707268, 0.6062751909426288, 0.2002829807573183),
        (0.0, 0.0, 1.0, -0.8, 0.7393352034857215, -0.1717513436572687, 0.5444750388719386, 0.15706216408568285),
        (0.0, 0.1, 0.0, -1.0, 0.0, -0.5109358077913947, 0.12226604805215133, 0.12226604805215133),
        (0.0, 0.1, 0.0, -0.8, 0.0, -0.5065614846748367, 0.07335962883129082, 0.07335962883129082),
        (0.0, 0.1, 1.0, -1.0, 0.7478011906745448, -0.2332127198290175, 0.5395967276728152, 0.19169682004274563),
        (0.0, 0.1, 1.0, -0.8, 0.770966851846044, -0.201037213621191, 0.4819201261930389, 0.14974069659470224),
        (5.0, 0.0, 0.0, -1.0, 0.0, -0.5, 0.1245279300120768, 0.12499999999999997),
        (5.0, 0.0, 0.0, -0.8, 0.0, -0.5, 0.07471675800724613, 0.07500000000000002),
        (5.0, 0.0, 1.0, -1.0, 0.7415662149212339, -0.1692756291438945, 0.5549273268904352, 0.20768109271402638),
        (5.0, 0.0, 1.0, -0.8, 0.7660214876335598, -0.1409973422964033, 0.49423784294486456, 0.16475066442589917),
        (5.0, 0.1, 0.0, -1.0, 0.0, -0.510895408013067, 0.12181436479906836, 0.12227614799673324),
        (5.0, 0.1, 0.0, -0.8, 0.0, -0.5065372448078401, 0.07308861887944106, 0.07336568879803998),
        (5.0, 0.1, 1.0, -1.0, 0.7700437546899073, -0.2043524795101405, 0.499191884741223, 0.19891188012246488),
        (5.0, 0.1, 1.0, -0.8, 0.792336244059665, -0.171203715822286, 0.4424045245893892, 0.1571990710444285),
    )
    k, c = sinpi(1 / 6), cospi(1 / 6)
    for (ψ, η_reg, s_trial, p_trial, s, p, λ, θdot) in fixtures
        ret = cap_return_map(
            s_trial, p_trial, 0.5, 4.0, k, sind(ψ), c, -0.5, η_reg,
        )
        @test ret.converged
        @test SVector(ret.τII, ret.P, ret.λ, ret.λ * ret.Ap) ≈
            SVector(s, p, λ, θdot) rtol = 1.0e-12 atol = 1.0e-14
    end
end

@testset "3-D momentum residual uses the cap-corrected pressure" begin
    # Two nodes whose gradients span x and y give a homogeneous plane extension
    # with ε̇zz = 0, so the 3-D element must reproduce the plane-strain one.
    plastic = DruckerPragerCap(
        (deg2rad(30.0),), (deg2rad(5.0),), (1.0,), (-0.5,), (0.1,), (1.0,),
    )
    vx, vy = SA[0.5, 0.0], SA[0.0, 0.2]
    P_loc, Pf_loc = SA[-1.0], SA[0.3]
    Nq, NqP = (SA[0.5, 0.5],), (SA[1.0],)
    common = (SA[1, 1], (1.0e3,), (1.0,), (0.0,), (1.0,), (Inf,))
    out(n) = ntuple(_ -> zeros(1, 1), n)

    τP_2D = out(4)
    FEMTools.integrate_momentum_residual(
        (vx, vy), P_loc, nothing, SA[0.0], ((@SMatrix([1.0 0.0; 0.0 1.0]), 1.0),),
        common..., (0.0, 0.0), 0.0, 1.0, Nq, NqP,
        nothing, plastic, FEMTools._stress_output(τP_2D, 1),
        nothing, nothing, Pf_loc,
    )
    τP_3D = out(7)
    FEMTools.integrate_momentum_residual(
        (vx, vy, SA[0.0, 0.0]), P_loc, nothing, SA[0.0],
        ((@SMatrix([1.0 0.0 0.0; 0.0 1.0 0.0]), 1.0),),
        common..., (0.0, 0.0, 0.0), 0.0, 1.0, Nq, NqP,
        nothing, plastic, FEMTools._stress_output(τP_3D, 1),
        nothing, nothing, Pf_loc,
    )

    τxx, τyy, τxy, P = map(only, τP_2D)
    τxx3, τyy3, τzz3, τxy3, τxz3, τyz3, P3 = map(only, τP_3D)
    @test P != only(P_loc)
    @test all(isapprox.((τxx3, τyy3, τxy3, P3), (τxx, τyy, τxy, P); rtol = 1.0e-12))
    @test τzz3 ≈ -(τxx + τyy)
    @test τxz3 == τyz3 == 0
end
