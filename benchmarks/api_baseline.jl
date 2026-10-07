using FEMTools
using KernelAbstractions
using DomainSets
using DomainSets: ×
using Test

# CUDA is optional. Run with --cuda in an environment containing CUDA.
if "--cuda" in ARGS
    using CUDA
    CUDA.functional() || error("CUDA baseline requires a functional device")
    CUDA.allowscalar(false)
end

function thermal_case(backend, FP)
    element = ReferenceElement(QuadraticElement{2, 9, FP})
    domain = (FP(0) .. FP(1)) × (FP(0) .. FP(1))
    mesh = Mesh(backend, domain, element, (2, 2))
    material = ThermalMaterial(; k = FP(1), α = FP(0), K = FP(Inf))
    tolerance = FP === Float32 ? FP(1e-5) : FP(1e-9)
    state = ThermalDiffusionDR(mesh, material; ϵ = tolerance)
    fill!(state.source, FP(1))
    bc = DirichletBoundaryCondition(
        mesh.Γnodes, KernelAbstractions.zeros(backend, FP, length(mesh.Γnodes)),
    )
    stats = solve!(state, mesh, bc; dt = FP(0.01),
        workgroup = 32, check_interval = 10, max_iterations = 10_000, Tref = FP(0), verbose = false)
    return (; field = Array(state.T), residual = stats.residual)
end

function stokes_case(backend, FP)
    element = ReferenceElement(QuadraticElement{2, 7, FP})
    pressure_element = ReferenceElement(LinearElement{2, 3, FP})
    domain = (FP(0) .. FP(1)) × (FP(0) .. FP(1))
    velocity_mesh = Mesh(backend, domain, element, (3, 3))
    mesh = MixedMesh(velocity_mesh, pressure_element; workgroup = 32)
    material = StokesMaterial(;
        η = (FP(1), FP(10)), ηb = FP(Inf), ρ0 = (FP(1), FP(2)),
        g = (FP(0), FP(-1)),
    )
    state = StokesDR(mesh, material)
    # Prepare cell phases on the host, then transfer once. Live solver arrays
    # remain on the selected backend, including boundary indices and scaling.
    coords, nodes = Array(mesh.coords), Array(mesh.el2n)
    phases = [sum(coords[nodes[a, e]][1] for a in 1:3) / 3 < FP(0.5) ? Int32(2) : Int32(1)
              for e in 1:mesh.nels]
    phases_device = FEMTools.TA(backend)(reshape(phases, 1, :))
    bc = ntuple(2) do _
        DirichletBoundaryCondition(velocity_mesh.Γnodes,
            KernelAbstractions.zeros(backend, FP, length(velocity_mesh.Γnodes)))
    end
    scale = KernelAbstractions.zeros(backend, FP, mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(scale, state, mesh, FP(50), FP(1);
        phases_v = phases_device, workgroup = 32)
    tolerance = FP === Float32 ? FP(1e-4) : FP(1e-8)
    stats = solve_stokes_dyrel!(state, mesh, bc, FP(1), scale;
        phases_v = phases_device, phases_P = phases_device, workgroup = 32,
        ncheck = 25, ϵ_tol = tolerance, total_iterMax = 50_000,
        verbose = false, verbose_inner = false)
    @test stats.converged
    return (; velocity = map(Array, FEMTools.velocity(state)), pressure = Array(state.P), stats)
end

function main(; gpu_backend = nothing)
    @testset "API baseline CPU and optional CUDA" begin
        for FP in (Float32, Float64)
            cpu = thermal_case(CPU(), FP)
            cpu_stokes = stokes_case(CPU(), FP)
            @test eltype(cpu.field) === FP
            println(FP, " CPU thermal/Stokes residuals: ", cpu.residual, " / ", cpu_stokes.stats.err_abs)
            if gpu_backend !== nothing
                gpu = thermal_case(gpu_backend, FP)
                gpu_stokes = stokes_case(gpu_backend, FP)
                @test gpu.field ≈ cpu.field rtol = FP === Float32 ? 2e-4 : 1e-7 atol = 1e-8
                @test eltype(gpu.field) === FP
                @test gpu_stokes.pressure ≈ cpu_stokes.pressure rtol = FP === Float32 ? 2e-3 : 1e-6 atol = FP === Float32 ? 2e-4 : 1e-8
                for component in 1:2
                    @test gpu_stokes.velocity[component] ≈ cpu_stokes.velocity[component] rtol = FP === Float32 ? 5e-3 : 1e-5 atol = FP === Float32 ? 2e-5 : 1e-8
                end
                println(FP, " CUDA thermal/Stokes residuals: ", gpu.residual, " / ", gpu_stokes.stats.err_abs)
            end
        end
    end
end

main(; gpu_backend = "--cuda" in ARGS ? CUDA.CUDABackend() : nothing)
