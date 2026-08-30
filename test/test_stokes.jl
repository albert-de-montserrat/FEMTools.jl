using Test

using DomainSets
using DomainSets: ×
using FEMTools
using FEMTools: assemble_momentum_residual_matrices_atomix!,
    assemble_viscosity_weighted_pressure_scaling!
using ForwardDiff
using KernelAbstractions: CPU, synchronize
using StaticArrays

if !isdefined(@__MODULE__, :FP64)
    const FP64 = Float64
end
if !isdefined(@__MODULE__, :FP32)
    const FP32 = Float32
end

# ---------------------------------------------------------------------------
# Helper: precompute physical-space geometry for a mesh element set
# ---------------------------------------------------------------------------

function _stokes_geo(coords, el2n, nels, element::ReferenceElement{E}) where {E <: AbstractElement{2, NV}} where NV
    ip    = element.integration_points
    NQ    = length(ip.ω)
    FP    = eltype(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = Vector{NTuple{NQ, Tuple{SMatrix{NV, 2, FP, 2NV}, FP}}}(undef, nels)
    FEMTools.precompute_geometry_kernel!(CPU(), 1)(
        geo, coords, el2n, ∂N∂ξq, ip.ω, Val(NV); ndrange = nels,
    )
    synchronize(CPU())
    return geo
end

# ---------------------------------------------------------------------------
# StokesDR constructor
# ---------------------------------------------------------------------------

@testset "StokesMaterial" begin
    material = StokesMaterial(;
        η = (1.0, 2.0), ηb = (3.0, 4.0), G = (5.0, 6.0),
        α = (0.0, 0.0), ρ0 = (7.0, 8.0), K = (Inf, Inf),
        g = (0.0, -9.81), Tref = 273.0,
    )
    dr = StokesDR(CPU(), 3, 2, material)
    @test dr.G === material.G
    @test dr.η === material.η
    @test dr.g === material.g
    @test_throws DimensionMismatch StokesMaterial(; η = (1.0, 2.0), ηb = (1.0,))
end

@testset "StokesDR constructor — defaults" begin
    for FP in (FP32, FP64)
        η  = NTuple{2, FP}((1.0, 10.0))
        ηb = NTuple{2, FP}((1.0, 10.0))
        α  = NTuple{2, FP}((0.0,  0.0))
        dr = StokesDR(CPU(), 10, 12, η, ηb, α)

        @test dr.ρ0   == NTuple{2, FP}((1.0, 1.0))
        @test dr.K    == NTuple{2, FP}((Inf, Inf))
        @test dr.G    == NTuple{2, FP}((Inf, Inf))
        @test dr.g    == (FP(0), FP(0))
        @test dr.Tref == FP(0)
        @test dr.η    == η
        @test dr.ηb   == ηb
        @test dr.α    == α
        @test eltype(dr.vx) == FP
        @test eltype(dr.P)  == FP
        @test length(dr.vx) == 10
        @test length(dr.P)  == 12
        @test length(dr.M_P) == 12
        @test length(dr.Pnum) == 12
        @test all(==(1), Array(dr.phases_v))
        @test all(==(1), Array(dr.phases_P))
        @test length(dr.τxx) == 10
        @test length(dr.τyy) == 10
        @test length(dr.τxy) == 10
        @test length(dr.τxx_old) == 10
        @test length(dr.τyy_old) == 10
        @test length(dr.τxy_old) == 10
        @test all(iszero, Array(dr.τxx))
        @test all(iszero, Array(dr.τyy))
        @test all(iszero, Array(dr.τxy))
        @test all(iszero, Array(dr.τxx_old))
        @test all(iszero, Array(dr.τyy_old))
        @test all(iszero, Array(dr.τxy_old))
        @test all(iszero, Array(dr.M_P))
        @test all(iszero, Array(dr.Pnum))

        vx, vy = FEMTools.velocity(dr)
        τxx, τyy, τxy = FEMTools.stress(dr)
        @test vx === dr.vx
        @test vy === dr.vy
        @test τxx === dr.τxx
        @test τyy === dr.τyy
        @test τxy === dr.τxy
        @test FEMTools.pressure(dr) === dr.P
        @test FEMTools.temperature(dr) === dr.T
    end
end

@testset "StokesDR constructor — integration-point stress storage" begin
    for FP in (FP32, FP64)
        η  = NTuple{2, FP}((1.0, 10.0))
        ηb = NTuple{2, FP}((1.0, 10.0))
        α  = NTuple{2, FP}((0.0,  0.0))
        dr = StokesDR(CPU(), 10, 12, η, ηb, α; stress_size = (3, 4))

        @test size(dr.τxx) == (3, 4)
        @test size(dr.τyy) == (3, 4)
        @test size(dr.τxy) == (3, 4)
        @test size(dr.τxx_old) == (3, 4)
        @test size(dr.τyy_old) == (3, 4)
        @test size(dr.τxy_old) == (3, 4)
        @test all(iszero, Array(dr.τxx))
        @test all(iszero, Array(dr.τxx_old))
    end
end

@testset "StokesDR constructor — explicit ρ0 / K / g / Tref" begin
    for FP in (FP32, FP64)
        η    = NTuple{2, FP}((1.0,    10.0))
        ηb   = NTuple{2, FP}((1.0,    10.0))
        α    = NTuple{2, FP}((1e-5,   2e-5))
        ρ0   = NTuple{2, FP}((2700.0, 3000.0))
        K    = NTuple{2, FP}((1e10,   2e10))
        g    = (FP(0), FP(-9.81))
        Tref = FP(1600)
        dr   = StokesDR(CPU(), 10, 12, η, ηb, α; ρ0, K, g, Tref)

        @test dr.ρ0   == ρ0
        @test dr.K    == K
        @test dr.g    == g
        @test dr.Tref == Tref
    end
end

@testset "3D viscous momentum element" begin
    dNdx = @SMatrix [0.2 0.3 0.4]
    Nv = SA[1.0]
    v = (SA[1.0], SA[2.0], SA[3.0])
    P = SA[5.0]
    Pnum = SA[0.5]
    T = SA[0.0]
    g = (0.0, -2.0, 0.0)
    R = FEMTools.integrate_momentum_residual(
        v, P, Pnum, T, ((dNdx, 2.0),), SA[1],
        (4.0,), (Inf,), (0.0,), (3.0,), (Inf,), g, 0.0, 1.0,
        (Nv,), (Nv,),
    )

    grad_v = @SMatrix [0.2 0.3 0.4; 0.4 0.6 0.8; 0.6 0.9 1.2]
    strain = (grad_v + grad_v') / 2
    stress = 8 .* (strain - tr(strain) / 3 .* I)
    expected = ntuple(i -> SA[dot(dNdx[1, :], stress[:, i]) - dNdx[1, i] * 5.5 - 3g[i]] .* 2, 3)
    @test all(isapprox.(R, expected))
end

@testset "3D plastic momentum supports component-wise AD" begin
    dNdx = @SMatrix [0.2 0.3 0.4]
    Nv = SA[1.0]
    v = (SA[1.0], SA[2.0], SA[3.0])
    plastic = DruckerPrager((0.0,), (0.0,), (0.0,), (1.0,), (1.0,))
    momentum(vx) = first(FEMTools.integrate_momentum_residual(
        (vx, v[2], v[3]), SA[0.0], nothing, SA[0.0], ((dNdx, 1.0),), SA[1],
        (1.0,), (Inf,), (0.0,), (1.0,), (Inf,), (0.0, 0.0, 0.0), 0.0, 1.0,
        (Nv,), (Nv,), ntuple(_ -> 0.0, 6), plastic,
    ))

    @test all(isfinite, ForwardDiff.jacobian(momentum, v[1]))
end

@testset "3D pressure element divergence" begin
    dNdx = @SMatrix [0.2 0.3 0.4]
    v = (SA[1.0], SA[2.0], SA[3.0])
    residual = FEMTools.integrate_PH_pressure_residual(
        v, SA[0.0], SA[0.0], SA[0.0], SA[0.0],
        ((dNdx, 2.0),), ((dNdx, 2.0),), SA[1],
        (0.0,), (Inf,), 1.0, (SA[1.0],),
    )
    @test residual ≈ SA[-4.0]
end

@testset "DruckerPrager constructor precomputes phase parameters" begin
    for FP in (FP32, FP64)
        ϕ     = NTuple{2, FP}((π / 6, π / 4))
        Ψ     = NTuple{2, FP}((π / 12, π / 8))
        C     = NTuple{2, FP}((10.0, 20.0))
        η_reg = NTuple{2, FP}((1.0e18, 2.0e18))
        Kb    = NTuple{2, FP}((1.0e10, 2.0e10))

        plastic = DruckerPrager(ϕ, Ψ, C, η_reg, Kb)

        @test all(isapprox.(plastic.cosϕ, map(cos, ϕ)))
        @test all(isapprox.(plastic.sinϕ, map(sin, ϕ)))
        @test all(isapprox.(plastic.sinΨ, map(sin, Ψ)))
        @test plastic.C == C
        @test plastic.η_reg == η_reg
        @test plastic.Kb == Kb
    end
end

@testset "symmetric tensor display" begin
    @test sprint(show, FEMTools.SymmetricTensor(1.0, 2.0, 3.0)) ==
        "SymmetricTensor2D(xx=1.0, yy=2.0, xy=3.0, II=0.0)"
    @test sprint(show, FEMTools.SymmetricTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)) ==
        "SymmetricTensor3D(xx=1.0, yy=2.0, zz=3.0, yz=4.0, xz=5.0, xy=6.0, II=0.0)"
    @test sprint(show, FEMTools.SymmetricTensor(zeros(2, 3), zeros(2, 3), zeros(2, 3))) ==
        "SymmetricTensor2D(xx=2×3 Matrix{Float64}, yy=2×3 Matrix{Float64}, xy=2×3 Matrix{Float64}, II=2×3 Matrix{Float64})"
end

@testset "DruckerPrager return uses plane-strain invariant gradient" begin
    dNdx = @SMatrix [1.0 0.0; 0.0 1.0; 0.0 0.0]
    Nv = SA[1.0, 0.0, 0.0]
    vx = SA[2.0, 0.0, 0.0]
    vy = SA[0.0, 0.0, 0.0]
    phase_loc = SA[1, 1, 1]
    τ_old = (0.0, 0.0, 0.0)
    plastic = DruckerPrager((0.0,), (0.0,), (0.0,), (1.0,), (1.0,))

    τxx, τyy, τxy = FEMTools.deviatoric_stress(
        (vx, vy), dNdx, Nv, (1.0,), (Inf,), phase_loc, 1.0, τ_old, 0.0, plastic,
    )

    τxx_trial = 8 / 3
    τyy_trial = -4 / 3
    τxy_trial = 0.0
    τII = sqrt((τxx_trial^2 + τyy_trial^2 + (-τxx_trial - τyy_trial)^2) / 2 + τxy_trial^2)
    λ = τII / 2
    @test τxx ≈ τxx_trial - 2 * λ * (2 * τxx_trial + τyy_trial) / (2 * τII)
    @test τyy ≈ τyy_trial - 2 * λ * (τxx_trial + 2 * τyy_trial) / (2 * τII)
    @test τxy ≈ τxy_trial
end

@testset "DruckerPrager pressure derivative sign in plastic denominator" begin
    dNdx = @SMatrix [1.0 0.0; 0.0 1.0; 0.0 0.0]
    Nv = SA[1.0, 0.0, 0.0]
    vx = SA[2.0, 0.0, 0.0]
    vy = SA[0.0, 0.0, 0.0]
    phase_loc = SA[1, 1, 1]
    τ_old = (0.0, 0.0, 0.0)
    plastic = DruckerPrager((π / 6,), (π / 6,), (0.0,), (3.0,), (2.0,))

    τxx, τyy, τxy = FEMTools.deviatoric_stress(
        (vx, vy), dNdx, Nv, (1.0,), (Inf,), phase_loc, 1.0, τ_old, 0.0, plastic,
    )

    τxx_trial = 8 / 3
    τyy_trial = -4 / 3
    τxy_trial = 0.0
    τII = sqrt((τxx_trial^2 + τyy_trial^2 + (-τxx_trial - τyy_trial)^2) / 2 + τxy_trial^2)
    ∂Q∂τxx = (2 * τxx_trial + τyy_trial) / (2 * τII)
    ∂Q∂τyy = (τxx_trial + 2 * τyy_trial) / (2 * τII)
    ∂Q∂τxy = τxy_trial / τII
    λ = τII / (1.0 + 3.0 + 2.0 * sin(π / 6) * sin(π / 6))

    @test τxx ≈ τxx_trial - 2 * λ * ∂Q∂τxx
    @test τyy ≈ τyy_trial - 2 * λ * ∂Q∂τyy
    @test τxy ≈ τxy_trial - 2 * λ * ∂Q∂τxy
end

@testset "Dilation stiffens the plastic return for a stiff bulk modulus" begin
    dNdx = @SMatrix [1.0 0.0; 0.0 1.0; 0.0 0.0]
    Nv = SA[1.0, 0.0, 0.0]
    vx = SA[2.0, 0.0, 0.0]
    vy = SA[0.0, 0.0, 0.0]
    phase_loc = SA[1, 1, 1]
    τ_old = (0.0, 0.0, 0.0)
    Δt = 0.1

    # Kb Δt sinΨ sinϕ exceeds ηve here, so the dilation term dominates the
    # denominator of the plastic multiplier. The point must still yield, and
    # dilation must only stiffen the return, leaving more stress rather than less.
    τII(Ψ) = FEMTools.second_invariant(
        FEMTools.deviatoric_stress(
            (vx, vy), dNdx, Nv, (1.0,), (1.0,), phase_loc, Δt, τ_old, 0.0,
            DruckerPrager((deg2rad(30),), (Ψ,), (0.1,), (1.0e-3,), (100.0,)),
        )
    )
    τII_trial = FEMTools.second_invariant(
        FEMTools.deviatoric_stress((vx, vy), dNdx, Nv, (1.0,), (1.0,), phase_loc, Δt, τ_old)
    )

    @test τII(deg2rad(3)) < τII_trial
    @test τII(deg2rad(3)) > τII(0.0)
end

# ---------------------------------------------------------------------------
# Unit tests for integrate_momentum_residual / _x_ / _y_
#
# Fixture: 3-node linear triangle, 1 quad point at centroid.
# Zero velocity gradients (∂N∂x = 0) to isolate the body-force branch.
# ---------------------------------------------------------------------------

let
    Nv_c    = SA[1/3, 1/3, 1/3]
    NqP_c   = SA[1/3, 1/3, 1/3]
    dNdx_0  = @SMatrix zeros(3, 2)
    geo_el  = ((dNdx_0, 0.5),)          # dΩ = 0.5 (reference triangle area)
    Nq      = (Nv_c,)
    NqP_v   = (NqP_c,)

    vx0       = SA[0.0, 0.0, 0.0]
    vy0       = SA[0.0, 0.0, 0.0]
    P0_loc    = SA[0.0, 0.0, 0.0]
    phase_loc = SA[1, 1, 1]             # single homogeneous phase

    @testset "pressure rates interpolate nodal increments" begin
        for FP in (Float32, Float64)
            Nv = SVector{3, FP}(0.2, 0.3, 0.5)
            P, P0 = SVector{3, FP}(3, 5, 8), SVector{3, FP}(1, 2, 3)
            T, T0 = SVector{3, FP}(7, 4, 2), SVector{3, FP}(2, 1, 1)
            Δt, ηb, α, dΩ = FP(2), (FP(4),), (FP(0.25),), FP(0.5)
            residual = FEMTools.integrate_PH_pressure_residual(
                (zero(P), zero(P)), P, P0, T, T0,
                ((@SMatrix(zeros(FP, 3, 2)), dΩ),),
                ((@SMatrix(zeros(FP, 3, 2)), dΩ),),
                SA[1, 1, 1], α, ηb, Δt, (Nv,),
            )
            rate = -sum(Nv .* (P - P0)) / (ηb[1] * Δt) +
                   α[1] * sum(Nv .* (T - T0)) / Δt
            @test residual ≈ Nv * rate * dΩ
        end
    end

    @testset "integrate_momentum_residual — zero gravity vanishes" begin
        T_loc = SA[0.0, 0.0, 0.0]
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.0,), (3.0,), (Inf,), (0.0, 0.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ zero(Rv_x)
        @test Rv_y ≈ zero(Rv_y)
    end

    @testset "integrate_momentum_residual — body force with T = Tref" begin
        # ρq = ρ0[1] = 3; Rv_y = −Nv·(ρ0·g_y·dΩ) = −[⅓,⅓,⅓]·(3·(−10)·0.5) = [5,5,5]
        T_loc = SA[0.0, 0.0, 0.0]
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.0,), (3.0,), (Inf,), (0.0, -10.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ SA[0.0, 0.0, 0.0]
        @test Rv_y ≈ SA[5.0, 5.0, 5.0]
    end

    @testset "integrate_momentum_residual — horizontal body force" begin
        # g_x = 5, g_y = 0; Rv_x = −[⅓,⅓,⅓]·(2·5·0.5) = [−1.67,−1.67,−1.67]
        T_loc = SA[0.0, 0.0, 0.0]
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.0,), (2.0,), (Inf,), (5.0, 0.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ SA[-5/3, -5/3, -5/3]   # −Nv·(2·5·0.5) = −5/3
        @test Rv_y ≈ SA[0.0,  0.0,  0.0]
    end

    @testset "integrate_momentum_residual — thermal EOS reduces density" begin
        # α=0.1, T=1, Tref=0 → αq·(Tq−Tref)=0.1 → ρq = 4·0.9 = 3.6
        # Rv_y = −[⅓,⅓,⅓]·(3.6·(−10)·0.5) = [6,6,6]
        T_loc = SA[1.0, 1.0, 1.0]
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.1,), (4.0,), (Inf,), (0.0, -10.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ SA[0.0, 0.0, 0.0]
        @test Rv_y ≈ SA[6.0, 6.0, 6.0]
    end

    @testset "integrate_momentum_residual — pressure EOS increases density" begin
        # K=4, P=2 → Pq/Kq=0.5 → ρq = 2·1.5 = 3; Rv_y = −[⅓,⅓,⅓]·(3·(−10)·0.5) = [5,5,5]
        P_loc = SA[2.0, 2.0, 2.0]
        T_loc = SA[0.0, 0.0, 0.0]
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.0,), (2.0,), (4.0,), (0.0, -10.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ SA[0.0, 0.0, 0.0]
        @test Rv_y ≈ SA[5.0, 5.0, 5.0]
    end

    @testset "integrate_momentum_residual — two-phase body force" begin
        # phase_loc=[1,1,2], ρ0=(1,3) → ρ0q = Nv·[1,1,3] = (1+1+3)/3 = 5/3
        # g=(0,−6): Rv_y = −[⅓,⅓,⅓]·(5/3·(−6)·0.5) = [⅓,⅓,⅓]·5 = [5/3,5/3,5/3]
        ph2 = SA[1, 1, 2]
        T_loc = SA[0.0, 0.0, 0.0]
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, ph2, (1.0, 1.0), (Inf, Inf), (0.0, 0.0), (1.0, 3.0), (Inf, Inf), (0.0, -6.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ SA[0.0, 0.0, 0.0]
        @test Rv_y ≈ SA[5/3, 5/3, 5/3]
    end

    @testset "integrate_momentum_residual — Pnum correction shifts residual" begin
        # With nonzero ∂N∂x the pressure term is nonzero; Pnum adds to it.
        dNdx_nz = @SMatrix [-1.0 -1.0; 1.0 0.0; 0.0 1.0]   # reference-triangle gradients
        geo_nz  = ((dNdx_nz, 1.0),)
        T_loc   = SA[0.0, 0.0, 0.0]
        P_loc   = SA[1.0, 1.0, 1.0]
        Pn_loc  = SA[0.5, 0.5, 0.5]
        args    = (
            (vx0, vy0), P_loc, nothing, T_loc,
            geo_nz, phase_loc, (1.0,), (Inf,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 1.0, Nq, NqP_v,
        )
        args_pn = (
            (vx0, vy0), P_loc, Pn_loc, T_loc,
            geo_nz, phase_loc, (1.0,), (Inf,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 1.0, Nq, NqP_v,
        )
        R0_x, _ = FEMTools.integrate_momentum_residual(args...)
        Rn_x, _ = FEMTools.integrate_momentum_residual(args_pn...)
        @test !isapprox(R0_x, Rn_x)          # Pnum shifts the residual
        # zero Pnum ≡ nothing
        args_z = (
            (vx0, vy0), P_loc, SA[0.0, 0.0, 0.0], T_loc,
            geo_nz, phase_loc, (1.0,), (Inf,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 1.0, Nq, NqP_v,
        )
        Rz_x, _ = FEMTools.integrate_momentum_residual(args_z...)
        @test Rz_x ≈ R0_x
    end

    @testset "integrate_momentum_x/y_residual — inline pressure correction matches explicit Pnum" begin
        dNdx_nz = @SMatrix [-1.0 -1.0; 1.0 0.0; 0.0 1.0]
        geo_nz  = ((dNdx_nz, 1.0),)
        vx_loc  = SA[0.2, -0.1, 0.4]
        vy_loc  = SA[-0.3, 0.5, 0.1]
        P_loc   = SA[0.7, 0.2, -0.1]
        P0loc   = SA[0.1, -0.2, 0.3]
        T_loc   = SA[2.0, 3.0, 1.0]
        T0loc   = SA[1.5, 2.0, 0.5]
        MP_loc  = SA[1.0, 2.0, 4.0]
        γ_eff   = SA[0.5, 0.75, 1.0]
        η       = (2.0,)
        G       = (Inf,)
        α       = (0.05,)
        ρ0      = (3.0,)
        K       = (Inf,)
        ηb      = (4.0,)
        g       = (1.0, -2.0)
        Tref    = 0.0
        Δt      = 0.25

        Pnum_loc = FEMTools.pressure_scale(
            γ_eff,
            FEMTools.integrate_PH_pressure_residual(
                (vx_loc, vy_loc), P_loc, P0loc, T_loc, T0loc,
                geo_nz, geo_nz, phase_loc, α, ηb, Δt, NqP_v,
            ),
            MP_loc,
        )
        explicit_x = FEMTools.integrate_momentum_x_residual(
            (vx_loc, vy_loc), P_loc, Pnum_loc, T_loc,
            geo_nz, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP_v,
        )
        explicit_y = FEMTools.integrate_momentum_y_residual(
            (vx_loc, vy_loc), P_loc, Pnum_loc, T_loc,
            geo_nz, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP_v,
        )

        augmented_args = (
            (vx_loc, vy_loc), P_loc, P0loc, T_loc, T0loc,
            geo_nz, geo_nz, phase_loc, phase_loc,
            η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, Nq, NqP_v,
        )
        @test FEMTools.integrate_momentum_x_residual(augmented_args...) ≈ explicit_x
        @test FEMTools.integrate_momentum_y_residual(augmented_args...) ≈ explicit_y
    end

    @testset "x/y component functions are consistent with combined residual" begin
        # Arbitrary parameters; combined output must equal per-component outputs.
        T_loc = SA[2.0, 3.0, 1.0]
        args  = (
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.05,), (3.0,), (Inf,), (3.0, -7.0), 5.0, 1.0, Nq, NqP_v,
        )
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(args...)
        Rx_only    = FEMTools.integrate_momentum_x_residual(args...)
        Ry_only    = FEMTools.integrate_momentum_y_residual(args...)
        @test Rv_x ≈ Rx_only
        @test Rv_y ≈ Ry_only
    end

    @testset "x/y component functions match combined finite-G stress" begin
        dNdx_nz = @SMatrix [-1.0 -1.0; 1.0 0.0; 0.0 1.0]
        geo_nz  = ((dNdx_nz, 1.0),)
        vx_loc  = SA[0.2, -0.1, 0.4]
        vy_loc  = SA[-0.3, 0.5, 0.1]
        P_loc   = SA[0.7, 0.2, -0.1]
        T_loc   = SA[0.0, 0.0, 0.0]
        args = (
            (vx_loc, vy_loc), P_loc, nothing, T_loc,
            geo_nz, phase_loc, (2.0,), (4.0,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 0.25, Nq, NqP_v,
        )
        Rv_x, Rv_y = FEMTools.integrate_momentum_residual(args...)
        @test FEMTools.integrate_momentum_x_residual(args...) ≈ Rv_x
        @test FEMTools.integrate_momentum_y_residual(args...) ≈ Rv_y
    end

    @testset "integrate_momentum_residual — finite-G old stress contributes" begin
        dNdx_nz = @SMatrix [-1.0 -1.0; 1.0 0.0; 0.0 1.0]
        geo_nz  = ((dNdx_nz, 1.0),)
        T_loc   = SA[0.0, 0.0, 0.0]
        τ_old   = (SA[0.3, 0.3, 0.3], SA[-0.1, -0.1, -0.1], SA[0.2, 0.2, 0.2])
        R0_x, R0_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_nz, phase_loc, (2.0,), (4.0,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 0.25, Nq, NqP_v,
        )
        Rτ_x, Rτ_y = FEMTools.integrate_momentum_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_nz, phase_loc, (2.0,), (4.0,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 0.25, Nq, NqP_v, τ_old,
        )
        @test !isapprox(Rτ_x, R0_x)
        @test !isapprox(Rτ_y, R0_y)
    end

    @testset "integrate_momentum_x/y_residual — body force" begin
        T_loc = SA[0.0, 0.0, 0.0]
        # x-component: g=(4,0) → Rv_x = −[⅓,⅓,⅓]·(2·4·0.5) = [−4/3,−4/3,−4/3]
        Rv_x = FEMTools.integrate_momentum_x_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.0,), (2.0,), (Inf,), (4.0, 0.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_x ≈ SA[-4/3, -4/3, -4/3]
        # y-component: g=(0,−6) → Rv_y = [⅓,⅓,⅓]·(2·6·0.5) = [2,2,2]
        Rv_y = FEMTools.integrate_momentum_y_residual(
            (vx0, vy0), P0_loc, nothing, T_loc,
            geo_el, phase_loc, (1.0,), (Inf,), (0.0,), (2.0,), (Inf,), (0.0, -6.0), 0.0, 1.0, Nq, NqP_v,
        )
        @test Rv_y ≈ SA[2.0, 2.0, 2.0]
    end
end

# ---------------------------------------------------------------------------
# Assembler integration tests (small 2×2 T6 mesh)
# ---------------------------------------------------------------------------

@testset "assemble_viscosity_weighted_pressure_scaling! — homogeneous viscosity" begin
    FP        = Float64
    element_v = ReferenceElement(QuadraticElement{2, 6, FP})
    element_P = ReferenceElement(LinearElement{2, 3, FP})
    mesh_v    = Mesh(CPU(), (0.0..1.0) × (0.0..1.0), element_v, (2, 2))
    mesh      = MixedMesh(mesh_v, element_P)
    geo_P     = _stokes_geo(mesh.coords, mesh.el2n, mesh.nels, element_v)

    M_P      = zeros(FP, mesh.nnodesP)
    γP       = zeros(FP, mesh.nnodesP)
    phases_v = ones(Int, mesh.nnodes)

    assemble_viscosity_weighted_pressure_scaling!(
        M_P, γP,
        mesh.el2n, mesh.DoFsP, geo_P, mesh.nels,
        element_v, element_P,
        phases_v, (6.0, 99.0), 2.0,
        CPU(), 1,
    )

    @test sum(M_P) ≈ one(FP) atol = 1e-12
    @test all(>(0), M_P)
    @test γP ≈ fill(6.0, mesh.nnodesP) atol = 1e-12
end

@testset "assemble_viscosity_weighted_pressure_scaling! — finite bulk modulus" begin
    FP        = Float64
    element_v = ReferenceElement(QuadraticElement{2, 6, FP})
    element_P = ReferenceElement(LinearElement{2, 3, FP})
    mesh_v    = Mesh(CPU(), (0.0..1.0) × (0.0..1.0), element_v, (2, 2))
    mesh      = MixedMesh(mesh_v, element_P)
    geo_P     = _stokes_geo(mesh.coords, mesh.el2n, mesh.nels, element_v)

    γP       = zeros(FP, mesh.nnodesP)
    dr       = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP, (6.0, 99.0), (1.0, 1.0), (0.0, 0.0))

    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, geo_P, element_v, element_P,
        2.0, 0.5, CPU(), 1; K = (4.0, 99.0),
    )

    @test sum(dr.M_P) ≈ one(FP) atol = 1e-12
    @test all(>(0), dr.M_P)
    @test γP ≈ fill(12 / 7, mesh.nnodesP) atol = 1e-12
end

@testset "assemble_momentum_residual_matrices_atomix! — zero velocity + zero gravity" begin
    FP        = Float64
    element_v = ReferenceElement(QuadraticElement{2, 6, FP})
    element_P = ReferenceElement(LinearElement{2, 3, FP})
    mesh_v    = Mesh(CPU(), (0.0..1.0) × (0.0..1.0), element_v, (2, 2))
    mesh      = MixedMesh(mesh_v, element_P)
    geo_v     = _stokes_geo(mesh.coords, mesh.el2n, mesh.nels, element_v)

    vx     = zeros(FP, mesh.nnodes);  vy  = zeros(FP, mesh.nnodes)
    P      = zeros(FP, mesh.nnodesP); T   = zeros(FP, mesh.nnodesP)
    phases = ones(Int, mesh.nnodes)
    Rv_x   = zeros(FP, mesh.nnodes);  Rv_y = zeros(FP, mesh.nnodes)

    # Pass DoFsP (sequential 1..3*nels) — the assembler uses it to index into P and T
    assemble_momentum_residual_matrices_atomix!(
        Rv_x, Rv_y, vx, vy, P, T, nothing,
        mesh.el2n, mesh.DoFsP, geo_v, mesh.nels,
        element_v, element_P,
        phases, nothing, nothing, nothing, (1.0, 1.0), (Inf, Inf), (0.0, 0.0), (1.0, 1.0), (Inf, Inf),
        (0.0, 0.0), FP(0), FP(1),
        CPU(), 1,
    )

    @test Rv_x ≈ zeros(FP, mesh.nnodes) atol = 1e-14
    @test Rv_y ≈ zeros(FP, mesh.nnodes) atol = 1e-14
end

@testset "assemble_momentum_residual_matrices_atomix! — quadrature old stress" begin
    FP        = Float64
    element_v = ReferenceElement(QuadraticElement{2, 6, FP})
    element_P = ReferenceElement(LinearElement{2, 3, FP})
    mesh_v    = Mesh(CPU(), (0.0..1.0) × (0.0..1.0), element_v, (2, 2))
    mesh      = MixedMesh(mesh_v, element_P)
    geo_v     = _stokes_geo(mesh.coords, mesh.el2n, mesh.nels, element_v)
    nq        = length(element_v.integration_points.ω)

    vx     = zeros(FP, mesh.nnodes);  vy  = zeros(FP, mesh.nnodes)
    P      = zeros(FP, mesh.nnodesP); T   = zeros(FP, mesh.nnodesP)
    phases = ones(Int, mesh.nnodes)
    τ_old  = (fill(FP(0.2), nq, mesh.nels), fill(FP(-0.1), nq, mesh.nels), fill(FP(0.15), nq, mesh.nels))
    τ_new  = (zeros(FP, nq, mesh.nels), zeros(FP, nq, mesh.nels), zeros(FP, nq, mesh.nels))
    Rv_x   = zeros(FP, mesh.nnodes);  Rv_y = zeros(FP, mesh.nnodes)

    assemble_momentum_residual_matrices_atomix!(
        Rv_x, Rv_y, vx, vy, P, T, nothing,
        mesh.el2n, mesh.DoFsP, geo_v, mesh.nels,
        element_v, element_P,
        phases, τ_old, nothing, τ_new, (1.0, 1.0), (4.0, 4.0), (0.0, 0.0), (1.0, 1.0), (Inf, Inf),
        (0.0, 0.0), FP(0), FP(0.25),
        CPU(), 1,
    )

    @test sum(abs2, Rv_x) > 0
    @test sum(abs2, Rv_y) > 0
    @test sum(abs2, τ_new[1]) > 0
    @test sum(abs2, τ_new[2]) > 0
    @test sum(abs2, τ_new[3]) > 0
end

@testset "assemble_momentum_residual_matrices_atomix! — gravity body force" begin
    FP        = Float64
    element_v = ReferenceElement(QuadraticElement{2, 6, FP})
    element_P = ReferenceElement(LinearElement{2, 3, FP})
    mesh_v    = Mesh(CPU(), (0.0..1.0) × (0.0..1.0), element_v, (2, 2))
    mesh      = MixedMesh(mesh_v, element_P)
    geo_v     = _stokes_geo(mesh.coords, mesh.el2n, mesh.nels, element_v)

    ρ0 = 1.0; gy = -9.81
    vx     = zeros(FP, mesh.nnodes);  vy  = zeros(FP, mesh.nnodes)
    P      = zeros(FP, mesh.nnodesP); T   = zeros(FP, mesh.nnodesP)
    phases = ones(Int, mesh.nnodes)
    Rv_x   = zeros(FP, mesh.nnodes);  Rv_y = zeros(FP, mesh.nnodes)

    assemble_momentum_residual_matrices_atomix!(
        Rv_x, Rv_y, vx, vy, P, T, nothing,
        mesh.el2n, mesh.DoFsP, geo_v, mesh.nels,
        element_v, element_P,
        phases, nothing, nothing, nothing, (1.0, 1.0), (Inf, Inf), (0.0, 0.0), (ρ0, ρ0), (Inf, Inf),
        (0.0, gy), FP(0), FP(1),
        CPU(), 1,
    )

    # Horizontal component must be zero (no horizontal gravity)
    @test Rv_x ≈ zeros(FP, mesh.nnodes) atol = 1e-14
    # All vertical residuals carry upward body force (−ρ·g_y > 0 since g_y < 0)
    @test all(Rv_y .>= 0)
    # ∑ᵢ Rv_y[i] = −ρ·g_y·Area (partition of unity: ∑ᵢ Nᵢ = 1 over the domain)
    @test sum(Rv_y) ≈ -ρ0 * gy * 1.0   atol = 1e-10
end
