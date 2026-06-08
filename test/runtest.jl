using Test

@testset "FEMTools tests" begin
    test_dir = @__DIR__

    for test_file in sort(filter(startswith("test_"), readdir(test_dir)))
        include(joinpath(test_dir, test_file))
    end
end
