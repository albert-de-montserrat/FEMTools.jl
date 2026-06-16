using KernelAbstractions: CPU

for FP in (FP32, FP64)
    @testset "ThermalDiffusionDR – $FP" begin
        nnodes  = 10
        nphases = 2
        k  = ntuple(_ -> FP(2.5),    nphases)
        Cp = ntuple(_ -> FP(1000.0), nphases)
        ρ0 = ntuple(_ -> FP(3300.0), nphases)
        α  = ntuple(_ -> FP(3e-5),   nphases)
        K  = ntuple(_ -> FP(1.3e11), nphases)

        # --- construction ---

        @testset "default constructor (CPU, $FP)" begin
            dr = ThermalDiffusionDR(CPU(), nnodes, k, Cp, ρ0, α, K)
            @test dr isa ThermalDiffusionDR{nphases}
        end

        @testset "convenience constructor (no backend, $FP)" begin
            dr = ThermalDiffusionDR(nnodes, k, Cp, ρ0, α, K)
            @test dr isa ThermalDiffusionDR{nphases}
        end

        dr = ThermalDiffusionDR(CPU(), nnodes, k, Cp, ρ0, α, K;
                                 CFL=FP(0.98), c_fact=FP(0.9), ϵ=FP(1e-6))

        # --- solver scalar parameters ---

        @testset "default solver parameters" begin
            @test dr.CFL    === FP(0.98)
            @test dr.c_fact === FP(0.9)
            @test dr.ϵ      === FP(1e-6)
        end

        @testset "custom solver parameters" begin
            dr2 = ThermalDiffusionDR(nnodes, k, Cp, ρ0, α, K;
                                      CFL=FP(0.5), c_fact=FP(0.75), ϵ=FP(1e-8))
            @test dr2.CFL    === FP(0.5)
            @test dr2.c_fact === FP(0.75)
            @test dr2.ϵ      === FP(1e-8)
        end

        # --- per-phase scalar tuples ---

        @testset "per-phase material properties" begin
            @test dr.k  === k
            @test dr.Cp === Cp
            @test dr.ρ0 === ρ0
            @test dr.α  === α
            @test dr.K  === K
            @test length(dr.k)  == nphases
            @test length(dr.Cp) == nphases
        end

        # --- nodal array fields ---

        nodal_fields = (:R, :R0, :∂R∂T, :PC, :T, :T0, :∂T∂τ, :P, :source)

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

        # --- phases integer array ---

        @testset "phases field" begin
            @test eltype(dr.phases) === Int
            @test length(dr.phases) == nnodes
            @test all(==(1), dr.phases)   # defaults to phase 1 everywhere
        end
    end
end
