using Test

using FEMTools

@testset "Aqua.jl" begin
    if Base.find_package("Aqua") === nothing
        @test_skip false
    else
        @eval using Aqua
        Aqua.test_all(
            FEMTools;
            unbound_args = false,
            deps_compat = (check_weakdeps = false,),
            stale_deps = (ignore = [:Triangulate],),
        )
    end
end
