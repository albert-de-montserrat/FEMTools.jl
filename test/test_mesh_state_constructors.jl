using Test, FEMTools, KernelAbstractions, DomainSets
using DomainSets: ×
using FEMTools: temperature, velocity, stress

function check_mesh_state_constructors(backend)
    for FP in (Float32, Float64)
        domain = (FP(0) .. FP(1)) × (FP(0) .. FP(1))
        element = ReferenceElement(QuadraticElement{2, 9, FP})
        mesh = Mesh(backend, domain, element, (2, 2))
        thermal = ThermalMaterial(; k = FP(1), α = FP(0), K = FP(Inf))
        heat = @inferred ThermalDiffusionDR(mesh, thermal; CFL = FP(0.8))
        pressure = @inferred LithostaticPressureDR(mesh, thermal)
        @test length(temperature(heat)) == mesh.nnodes
        @test length(FEMTools.pressure(pressure)) == mesh.nnodes
        @test heat.CFL === FP(0.8)
        for field in (heat.T, heat.phases, pressure.P, pressure.phases)
            @test KernelAbstractions.get_backend(field) == backend
        end
        @test eltype(heat.T) === FP
        @test eltype(pressure.P) === FP
        mixed = MixedMesh(mesh, ReferenceElement(LinearElement{2, 3, FP}))
        material = StokesMaterial(; η = FP(1), ηb = FP(Inf), g = (FP(0), FP(-1)))
        state = @inferred StokesDR(mixed, material)
        @test length(state.P) == mixed.nnodesP
        @test all(v -> length(v) == mixed.nnodes, velocity(state))
        @test all(t -> size(t) == (length(element.integration_points.ω), mixed.nels), stress(state))
        @test all(a -> KernelAbstractions.get_backend(a) == backend, (velocity(state)..., state.P, stress(state)...))
        @test eltype(state.P) === FP
        @test StokesDR(mixed, material; stress_size = :none).τ === nothing
        element3 = ReferenceElement(QuadraticElement{3, 27, FP})
        mesh3 = Mesh(backend, domain × (FP(0) .. FP(1)), element3, (1, 1, 1))
        mixed3 = MixedMesh(mesh3, ReferenceElement(LinearElement{3, 4, FP}))
        material3 = StokesMaterial(; η = material.η, ηb = material.ηb,
            G = material.G, α = material.α, ρ0 = material.ρ0, K = material.K,
            g = (FP(0), FP(0), FP(-1)), Tref = FP(0))
        state3 = @inferred StokesDR(mixed3, material3)
        @test length(velocity(state3)) == 3
        @test all(t -> size(t) == (length(element3.integration_points.ω), mixed3.nels), stress(state3))
        @test all(a -> KernelAbstractions.get_backend(a) == backend, (velocity(state3)..., state3.P, stress(state3)...))
        @test_throws DimensionMismatch StokesDR(mixed3, material)
        nodes = mesh.Γnodes
        values = KernelAbstractions.ones(backend, FP, length(nodes))
        bc = @inferred DirichletBoundaryCondition(nodes, values)
        @test bc.DoFs === nodes
        @test bc.vals === values
        @test bc.Γ === nothing
        @test all(iszero, Array(bc.zero_vals))
        @test_throws DimensionMismatch DirichletBoundaryCondition(nodes, similar(values, 0))
        @test_throws DimensionMismatch DirichletBoundaryCondition(:wall, nodes, similar(values, 0))
        if FP === Float32
            @test_throws ArgumentError ThermalDiffusionDR(mesh, ThermalMaterial())
            @test_throws ArgumentError LithostaticPressureDR(mesh, ThermalMaterial())
            @test_throws ArgumentError StokesDR(mixed, StokesMaterial())
        end
    end
end

@testset "mesh-aware state constructors" begin
    check_mesh_state_constructors(CPU())
end
