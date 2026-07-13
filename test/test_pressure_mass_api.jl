using Test

using FEMTools
using FEMTools: pressure_mass, velocity_mass
using KernelAbstractions: CPU

@testset "mass accessors dispatch on StokesDR" begin
    η  = (1.0,)
    ηb = (1.0,)
    α  = (0.0,)
    dr = StokesDR(CPU(), 10, 12, η, ηb, α)

    @test pressure_mass(dr) === dr.M_P
    @test velocity_mass(dr) === dr.M_V

    # Non-solver inputs fail fast rather than probing for fields.
    @test_throws MethodError pressure_mass((M_P = [1.0],))
    @test_throws MethodError velocity_mass((M_V = [1.0],))
end
