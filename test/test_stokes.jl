using KernelAbstractions: CPU, synchronize
using DomainSets: ×

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

@testset "StokesDR constructor — defaults" begin
    for FP in (FP32, FP64)
        η  = NTuple{2, FP}((1.0, 10.0))
        ηb = NTuple{2, FP}((1.0, 10.0))
        α  = NTuple{2, FP}((0.0,  0.0))
        dr = StokesDR(CPU(), 10, 12, η, ηb, α)

        @test dr.ρ0   == NTuple{2, FP}((1.0, 1.0))
        @test dr.K    == NTuple{2, FP}((Inf, Inf))
        @test dr.g    == (FP(0), FP(0))
        @test dr.Tref == FP(0)
        @test dr.η    == η
        @test dr.ηb   == ηb
        @test dr.α    == α
        @test eltype(dr.vx) == FP
        @test eltype(dr.P)  == FP
        @test length(dr.vx) == 10
        @test length(dr.P)  == 12
        @test length(dr.τxx_old) == 10
        @test length(dr.τyy_old) == 10
        @test length(dr.τxy_old) == 10
        @test all(iszero, Array(dr.τxx_old))
        @test all(iszero, Array(dr.τyy_old))
        @test all(iszero, Array(dr.τxy_old))
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
            geo_nz, phase_loc, (2.0,), (4.0,), (0.0,), (1.0,), (Inf,), (0.0, 0.0), 0.0, 0.25, τ_old, Nq, NqP_v,
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
        phases, (1.0, 1.0), (Inf, Inf), (0.0, 0.0), (1.0, 1.0), (Inf, Inf),
        (0.0, 0.0), FP(0), FP(1),
        CPU(), 1,
    )

    @test Rv_x ≈ zeros(FP, mesh.nnodes) atol = 1e-14
    @test Rv_y ≈ zeros(FP, mesh.nnodes) atol = 1e-14
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
        phases, (1.0, 1.0), (Inf, Inf), (0.0, 0.0), (ρ0, ρ0), (Inf, Inf),
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
