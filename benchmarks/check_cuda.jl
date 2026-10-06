using Test, CUDA
module SolKzBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "stokes", "solkz2D", "SolKz2D_triangle.jl"))
end
end
module SolCxBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "stokes", "solcx2D", "SolCx2D_triangle.jl"))
end
end
module ThermalDiffusionBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "thermal", "thermal_diffusion2D", "ThermalDiffusion2D.jl"))
end
end

module SolKzQuadBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "stokes", "solkz2D", "SolKz2D_quad.jl"))
end
end
module SolCxQuadBenchmark
withenv("FEMTOOLS_BENCHMARK_PLOTS" => "false", "FEMTOOLS_BENCHMARK_HISTORY" => "false") do
    include(joinpath(@__DIR__, "stokes", "solcx2D", "SolCx2D_quad.jl"))
end
end

@testset "exact-field CPU/CUDA agreement" begin
    if CUDA.functional()
        CUDA.allowscalar(false)
        for (case, benchmark) in ((:SolCx, SolCxBenchmark), (:SolKz, SolKzBenchmark),
                                  (:SolCx, SolCxQuadBenchmark), (:SolKz, SolKzQuadBenchmark))
            contrast = case == :SolCx ? 2.0 : 10.0
            cpu = benchmark.main(; save_history = false, resolution = 4, contrast)
            gpu = benchmark.main(; save_history = false, resolution = 4, contrast, backend = CUDA.CUDABackend())
            @test cpu.stats.converged && gpu.stats.converged
            @test gpu.velocity[1] ≈ cpu.velocity[1] rtol = 1e-4 atol = 1e-8
            @test gpu.velocity[2] ≈ cpu.velocity[2] rtol = 1e-4 atol = 1e-8
            @test gpu.numerical_p ≈ cpu.numerical_p rtol = 1e-4 atol = 1e-8
        end
        cpu = ThermalDiffusionBenchmark.main(; save_history = false, resolution = 4, nsteps = 4)
        gpu = ThermalDiffusionBenchmark.main(; save_history = false, resolution = 4, nsteps = 4, backend = CUDA.CUDABackend())
        @test gpu.temperature ≈ cpu.temperature rtol = 1e-5 atol = 1e-8
    else
        @test_skip CUDA.functional()
    end
end
