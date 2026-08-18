using Test

using FEMTools
using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(FEMTools)
end
