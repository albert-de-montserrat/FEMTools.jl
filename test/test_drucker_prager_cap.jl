# Drucker-Prager tensile cap geometry and composite yield surface.
# Reference: Popov, Berlie & Kaus (2025), Geosci. Model Dev. 18, 7035-7058,
# doi:10.5194/gmd-18-7035-2025, Eqs. 13-22.

using ForwardDiff
using FEMTools: cap_geometry, cap_yield_function, cap_flow_direction, cap_return_map

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

    @testset "flow direction" begin
        # Shear domain: the classical Drucker-Prager flow prefactors.
        τII, P = 2geom.τ_d, geom.p_q + 10.0
        Bτ, Bp = cap_flow_direction(τII, P, CAP_kq, geom)
        @test Bτ ≈ 1 / (2τII) rtol = 1e-12
        @test Bp ≈ CAP_kq / 3 rtol = 1e-12

        # Cap domain: Eq. 22, second row.
        τII, P = 0.2geom.τ_d, CAP_pT - 0.5
        R̂q = √(τII^2 + (P - geom.p_q)^2)
        Bτ, Bp = cap_flow_direction(τII, P, CAP_kq, geom)
        @test Bτ ≈ geom.b / (2R̂q) rtol = 1e-12
        @test Bp ≈ -geom.b * (P - geom.p_q) / (3R̂q) rtol = 1e-12

        # The potential's own switching ray is P + kq τII = p_q, and both rows of
        # Eq. 22 collapse to the shear values along it, exactly.
        for τII in range(0.1geom.τ_d, 3geom.τ_d; length = 11)
            P = geom.p_q - CAP_kq * τII
            Bτ, Bp = cap_flow_direction(τII, P, CAP_kq, geom)
            @test Bτ ≈ 1 / (2τII) rtol = 1e-10
            @test Bp ≈ CAP_kq / 3 rtol = 1e-10
        end

        # That ray passes through the delimiter, so the return direction is
        # single-valued where the yield branches meet.
        @test geom.p_d + CAP_kq * geom.τ_d ≈ geom.p_q

        # Volumetric part: ε̇_vol = 3 λ Bp reduces to λ sin(Ψ) in the shear domain,
        # matching ∂Q∂P = −sinΨ of the existing Drucker-Prager return map.
        Bτ, Bp = cap_flow_direction(2geom.τ_d, geom.p_q + 10.0, CAP_kq, geom)
        @test 3Bp ≈ CAP_kq

        # Zero dilation is explicitly supported.
        geom0 = cap_geometry(CAP_k, 0.0, CAP_c, CAP_pT)
        @test geom0.b ≈ 1
        @test geom0.p_q ≈ geom0.p_d
        Bτ0, Bp0 = cap_flow_direction(2geom0.τ_d, geom0.p_q + 10.0, 0.0, geom0)
        @test Bp0 ≈ 0 atol = 1e-15
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

            Bτ, Bp = cap_flow_direction(2g.τ_d, g.p_q + g.τ_d, kq, g)
            @test Bτ isa FP
            @test Bp ≈ kq / 3 rtol = sqrt(eps(FP))
        end
    end

    @testset "inference" begin
        @test (@inferred cap_geometry(CAP_k, CAP_kq, CAP_c, CAP_pT)) isa NamedTuple
        @test (@inferred cap_yield_function(1.0, 1.0, CAP_k, CAP_c, geom)) isa Float64
        @test (@inferred cap_flow_direction(1.0, 1.0, CAP_kq, geom)) isa Tuple{Float64, Float64}
    end

    @testset "return-map derivative matches central differences" begin
        ηve, KΔt, η_reg = 2.0, 10.0, 1.0
        cases = ((30.0, 0.0), (0.5, -6.0), (geom.τ_d, geom.p_d - 1.0e-3))
        for (τ_trial, P_trial) in cases
            fτ(τ) = cap_return_map(
                τ, P_trial, ηve, KΔt, CAP_k, CAP_kq, CAP_c, CAP_pT, η_reg, Val(50),
            ).τII
            fP(P) = cap_return_map(
                τ_trial, P, ηve, KΔt, CAP_k, CAP_kq, CAP_c, CAP_pT, η_reg, Val(50),
            ).τII
            h = 1.0e-5
            dτ_fd = (fτ(τ_trial + h) - fτ(τ_trial - h)) / (2h)
            dP_fd = (fP(P_trial + h) - fP(P_trial - h)) / (2h)
            @test ForwardDiff.derivative(fτ, τ_trial) ≈ dτ_fd rtol = 1.0e-5 atol = 1.0e-7
            @test ForwardDiff.derivative(fP, P_trial) ≈ dP_fd rtol = 1.0e-5 atol = 1.0e-7
        end
    end
end
