using Test, FEMTools, JLD2
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

@testset "exact-field benchmark refinement" begin
    for (case, benchmark) in ((:SolCx, SolCxBenchmark), (:SolKz, SolKzBenchmark),
                              (:SolCx, SolCxQuadBenchmark), (:SolKz, SolKzQuadBenchmark))
        contrast = case == :SolCx ? 2.0 : 10.0
        coarse = benchmark.main(; save_history = false, resolution = 4, contrast)
        fine = benchmark.main(; save_history = false, resolution = 8, contrast)
        @test coarse.stats.converged && fine.stats.converged
        @test all(h -> length(h.err_v_components) == 2 && all(isfinite, h.err_v_components) &&
                       isfinite(h.err_P) && maximum(h.err_v_components) ≈ h.err_v, fine.stats.history)
        @test fine.errors.velocity.relative < coarse.errors.velocity.relative / 2
        @test fine.errors.pressure.relative < coarse.errors.pressure.relative / 2
        # Changing the pressure gauge must not affect the reported error.
        numerical = fine.numerical_p .+ 7
        analytical = fine.analytical_p .- 3
        weights = fine.samples.weights
        numerical .-= sum(weights .* numerical) / sum(weights)
        analytical .-= sum(weights .* analytical) / sum(weights)
        shifted_error = sqrt(sum(weights .* abs2.(numerical .- analytical)))
        @test shifted_error ≈ fine.errors.pressure.absolute
        if case == :SolCx
            nodes = Array(fine.mesh.el2n)
            coords = Array(fine.mesh.coords)
            @test all(e -> maximum(coords[n][1] for n in nodes[1:(size(nodes, 1) == 9 ? 4 : 3), e]) <= 0.5 ||
                           minimum(coords[n][1] for n in nodes[1:(size(nodes, 1) == 9 ? 4 : 3), e]) >= 0.5,
                      1:fine.mesh.nels)
            @test Set(fine.phases) == Set([1, 2])
        end
        high_contrast = benchmark.main(; save_history = false, resolution = 8, contrast = 1e6)
        @test high_contrast.stats.converged
        @test high_contrast.errors.velocity.relative < 0.15
        @test high_contrast.errors.pressure.relative < 0.25
    end
    coarse = ThermalDiffusionBenchmark.main(; save_history = false, resolution = 8, nsteps = 4)
    fine = ThermalDiffusionBenchmark.main(; save_history = false, resolution = 8, nsteps = 8)
    @test coarse.converged && fine.converged
    @test issorted([h.iter for h in fine.convergence_history]) &&
          Set(h.step for h in fine.convergence_history) == Set(1:8)
    @test last(fine.convergence_history).residual ≈ fine.errors.residual
    @test fine.errors.relative < 0.6 * coarse.errors.relative
    @test fine.metadata.params.K == 1.0
    @test_throws ArgumentError ThermalDiffusionBenchmark.main(; save_history = false, nsteps = 0)
    @test_throws ArgumentError SolCxBenchmark.main(; save_history = false, resolution = 3)
    @test_throws ErrorException SolCxBenchmark.main(; save_history = false, resolution = 4, total_iterMax = 1)
    mktempdir() do dir
        # Exercise VTK output without creating plots in the headless check.
        result = ThermalDiffusionBenchmark.main(; save_history = false, resolution = 2, nsteps = 1)
        FEMTools.write_vtk(joinpath(dir, "temperature.vtk"), result.mesh;
                          point_data = (; temperature = result.temperature))
        @test isfile(joinpath(dir, "temperature.vtk"))
    end
end

@testset "benchmark history JLD2 round-trip" begin
    for benchmark in (SolKzBenchmark, SolCxBenchmark, SolKzQuadBenchmark, SolCxQuadBenchmark,
                      ThermalDiffusionBenchmark)
        mktempdir() do dir
            r = benchmark === ThermalDiffusionBenchmark ?
                benchmark.main(; resolution = 4, nsteps = 4, output_dir = dir) :
                benchmark.main(; resolution = 4, contrast = 10.0, output_dir = dir)
            data = JLD2.load(r.history_path)
            history = hasproperty(r, :stats) ? r.stats.history : r.convergence_history
            @test data["convergence_history"] == history
            @test data["metadata"] == r.metadata
            if hasproperty(r, :stats)
                @test r.metadata.dofs == (; vx = r.mesh.nnodes, vy = r.mesh.nnodes,
                                          p = r.mesh.nnodesP, total = 2 * r.mesh.nnodes + r.mesh.nnodesP)
            else
                @test r.metadata.dofs == (; T = r.mesh.nnodes, total = r.mesh.nnodes)
            end
            # Saving histories does not require loading/creating figures or VTK.
            @test readdir(dir) == [basename(r.history_path)]
        end
    end
end
