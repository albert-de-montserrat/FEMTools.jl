using Test

using FEMTools

@testset "ExplicitImports" begin
    if Base.find_package("ExplicitImports") === nothing
        @test_skip false
    else
        @eval using ExplicitImports
        test_explicit_imports(
            FEMTools;
            ignore = (:jacobian, :ones, :zeros, Symbol("@atomic")),
        )
    end
end
