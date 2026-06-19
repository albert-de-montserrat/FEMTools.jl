using KernelAbstractions: CPU

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

@testset "lithostatic residual FP32" begin
    dNdx = @SMatrix Float32[0.0 1.0; 0.0 -1.0]
    geo_el = ((dNdx, 1.0f0),)
    Nq = (SA[0.5f0, 0.5f0],)
    Ploc = SA[0.0f0, 0.0f0]
    Tloc = SA[300.0f0, 300.0f0]
    phase_loc = SA[1, 1]
    ρ0 = (2.0f0,)
    α = (0.1f0,)
    K = (100.0f0,)

    r = FEMTools.lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, 300.0f0, SA[0.0f0, -1.0f0], Nq, Val(2))
    @test r isa SVector{2, Float32}
    @test r ≈ SA[-2.0f0, 2.0f0]
end

@testset "lithostatic residual stiffness term" begin
    # Non-zero P exercises the ∇P·∇v stiffness contribution.
    dNdx = @SMatrix [0.0 1.0; 0.0 -1.0]
    geo_el = ((dNdx, 1.0),)
    Nq = (SA[0.5, 0.5],)
    # Ploc non-zero: Pq = 1.0, so ρ = ρ0*(1 + Pq/K) = 2*(1.01) = 2.02
    Ploc = SA[2.0, 0.0]
    Tloc = SA[300.0, 300.0]
    phase_loc = SA[1, 1]
    ρ0 = (2.0,)
    α = (0.1,)
    K = (100.0,)

    r = FEMTools.lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, 300.0, SA[0.0, -1.0], Nq, Val(2))
    # ∇P_N = dNdx*(dNdx'*Ploc) = dNdx*[0,2] = [2,-2]
    # ρg_N = 2.02*dNdx*[0,-1] = 2.02*[-1,1] = [-2.02,2.02]
    # Re = ρg_N - ∇P_N = [-4.02, 4.02]
    @test r ≈ SA[-4.02, 4.02]
end

@testset "lithostatic element jacobian includes EOS pressure term" begin
    dNdx = @SMatrix [0.0 1.0; 0.0 -1.0]
    geo = (((dNdx, 1.0),),)
    Nq = (SA[0.5, 0.5],)
    el2n = reshape(Int32[1, 2], 2, 1)
    T = [300.0, 300.0]
    P = [2.0, 0.0]
    phases = [1, 1]
    ρ0 = (2.0,)
    α = (0.1,)
    K = (100.0,)

    local_nodes, rowsums, diags = FEMTools.lp_element_jacobian(
        T, P, el2n, geo, phases, ρ0, α, K, 300.0, SA[0.0, -1.0], Nq, 1, Val(2),
    )

    @test local_nodes == SA[1, 2]
    @test rowsums ≈ SA[2.0, 2.0]
    @test diags ≈ SA[1.01, 0.99]
end

@testset "lithostatic residual multi-phase" begin
    dNdx = @SMatrix [0.0 1.0; 0.0 -1.0]
    geo_el = ((dNdx, 1.0),)
    Nq = (SA[0.5, 0.5],)
    Ploc = SA[0.0, 0.0]
    Tloc = SA[300.0, 300.0]
    # node 1 = phase 1 (ρ0=2), node 2 = phase 2 (ρ0=4)
    phase_loc = SA[1, 2]
    ρ0 = (2.0, 4.0)
    α = (0.1, 0.1)
    K = (100.0, 100.0)

    r = FEMTools.lp_integrate_residual(Ploc, Tloc, geo_el, phase_loc, ρ0, α, K, 300.0, SA[0.0, -1.0], Nq, Val(2))
    # ρ0q = 0.5*2 + 0.5*4 = 3.0, ρq = 3.0 at Tref with P=0
    @test r ≈ SA[-3.0, 3.0]
end

for FP in (FP32, FP64)
    @testset "LithostaticPressureDR – $FP" begin
        nnodes  = 10
        nphases = 2
        ρ0 = ntuple(_ -> FP(3300.0), nphases)
        α  = ntuple(_ -> FP(3e-5),   nphases)
        K  = ntuple(_ -> FP(1.3e11), nphases)

        @testset "default constructor (CPU, $FP)" begin
            dr = LithostaticPressureDR(CPU(), nnodes, ρ0, α, K)
            @test dr isa LithostaticPressureDR{nphases}
        end

        @testset "convenience constructor (no backend, $FP)" begin
            dr = LithostaticPressureDR(nnodes, ρ0, α, K)
            @test dr isa LithostaticPressureDR{nphases}
        end

        dr = LithostaticPressureDR(CPU(), nnodes, ρ0, α, K;
                                    CFL=FP(0.98), c_fact=FP(0.9), ϵ=FP(1e-6))

        @testset "default solver parameters" begin
            @test dr.CFL    === FP(0.98)
            @test dr.c_fact === FP(0.9)
            @test dr.ϵ      === FP(1e-6)
        end

        @testset "custom solver parameters" begin
            dr2 = LithostaticPressureDR(nnodes, ρ0, α, K;
                                         CFL=FP(0.5), c_fact=FP(0.75), ϵ=FP(1e-8))
            @test dr2.CFL    === FP(0.5)
            @test dr2.c_fact === FP(0.75)
            @test dr2.ϵ      === FP(1e-8)
        end

        @testset "per-phase material properties" begin
            @test dr.ρ0 === ρ0
            @test dr.α  === α
            @test dr.K  === K
            @test length(dr.ρ0) == nphases
        end

        nodal_fields = (:R, :R0, :∂R∂P, :PC, :P, :∂P∂τ, :T)

        @testset "nodal array element type" begin
            for f in nodal_fields
                @test eltype(getfield(dr, f)) === FP
            end
        end

        @testset "nodal array length" begin
            for f in nodal_fields
                @test length(getfield(dr, f)) == nnodes
            end
        end

        @testset "nodal arrays zero-initialized" begin
            for f in nodal_fields
                @test iszero(getfield(dr, f))
            end
        end

        @testset "phases field" begin
            @test eltype(dr.phases) === Int
            @test length(dr.phases) == nnodes
            @test all(==(1), dr.phases)
        end
    end
end
