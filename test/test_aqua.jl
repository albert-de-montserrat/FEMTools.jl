using Test

using FEMTools
using Aqua

@testset "Aqua.jl" begin
    # Aqua's persistent-task check walks every dependency's Project.toml;
    # InternedStrings (via GeoParams) ships only a REQUIRE file, so the walk errors.
    Aqua.test_all(FEMTools; persistent_tasks = false)
end
