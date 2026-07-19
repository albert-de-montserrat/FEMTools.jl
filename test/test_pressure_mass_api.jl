using Test

using FEMTools
using FEMTools: pressure_mass
using KernelAbstractions: CPU

@testset "pressure_mass dispatches on StokesDR" begin
    η  = (1.0,)
    ηb = (1.0,)
    α  = (0.0,)
    dr = StokesDR(CPU(), 10, 12, η, ηb, ξ, α)

    @test pressure_mass(dr) === dr.M_P

    # Non-solver inputs fail fast rather than probing for fields.
    @test_throws MethodError pressure_mass((M_P = [1.0],))
end
