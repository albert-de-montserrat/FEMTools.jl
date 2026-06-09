using Test

using DomainSets
using DomainSets: ×
using FEMTools
using SparseArrays
using StaticArrays

const FP64 = Float64
const FP32 = Float32

@testset "FEMTools tests" begin
    test_dir = @__DIR__

    for test_file in sort(filter(startswith("test_"), readdir(test_dir)))
        include(joinpath(test_dir, test_file))
    end
end
