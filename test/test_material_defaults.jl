using FEMTools, Test

@testset "material scalars and typed defaults" begin
    @test ThermalMaterial().k === (1.0,)
    @test StokesMaterial().ηb === (1.0,)
    @test (@inferred ThermalMaterial(; k = 2)).k === (2.0,)
    for FP in (Float32, Float64)
        thermal = @inferred ThermalMaterial(; k = FP(3))
        @test thermal isa ThermalMaterial{1, FP}
        @test thermal.k === (FP(3),)
        @test thermal.Cp === thermal.ρ0 === thermal.α === thermal.K === (FP(1),)
        phases = @inferred ThermalMaterial(; Cp = (FP(2), FP(3)), k = FP(4))
        @test phases isa ThermalMaterial{2, FP}
        @test phases.k === (FP(4), FP(4))
        @test phases.ρ0 === phases.α === phases.K === (FP(1), FP(1))
        stokes = @inferred StokesMaterial(; η = (FP(2), FP(3)), ηb = FP(Inf))
        @test stokes isa StokesMaterial{2, 2, FP}
        @test stokes.ηb === stokes.G === stokes.K === (FP(Inf), FP(Inf))
        @test stokes.α === (FP(0), FP(0))
        @test stokes.ρ0 === (FP(1), FP(1))
        @test stokes.g === (FP(0), FP(0))
        @test stokes.Tref === FP(0)
        gravity = @inferred StokesMaterial(; g = (FP(0), FP(0), FP(-1)))
        @test gravity isa StokesMaterial{1, 3, FP}
        @test gravity.η === gravity.ηb === gravity.ρ0 === (FP(1),)
        @test (@inferred StokesMaterial(; Tref = FP(2))).Tref === FP(2)
        @test_throws DimensionMismatch ThermalMaterial(; k = (FP(1), FP(2)), Cp = (FP(1),))
        @test_throws DimensionMismatch StokesMaterial(; η = (FP(1), FP(2)), G = (FP(1),))
    end
    @test_throws MethodError ThermalMaterial(; k = 1f0, Cp = 1.0)
    @test_throws MethodError StokesMaterial(; η = 1f0, ηb = 1.0)
    @test_throws MethodError StokesMaterial(; η = 1f0, g = (0.0, 0.0))
    @test_throws MethodError StokesMaterial(; η = 1f0, Tref = 0.0)
    @test_throws ArgumentError ThermalMaterial(; k = ())
    @test_throws ArgumentError StokesMaterial(; η = ())
    @test_throws ArgumentError StokesMaterial(; g = ())
    @test_throws ArgumentError StokesMaterial(; g = (0.0,))
end
