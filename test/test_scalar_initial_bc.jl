using Test, FEMTools, DomainSets, KernelAbstractions
using DomainSets: ×

@testset "scalar DR constrains the initial state before assembly" begin
    for FP in (Float32, Float64)
        element = ReferenceElement(LinearElement{2, 3, FP})
        mesh = Mesh(CPU(), (FP(0) .. FP(1)) × (FP(0) .. FP(1)), element, (2, 2))
        material = ThermalMaterial(; k = FP(1), Cp = FP(1), ρ0 = FP(1), α = FP(0), K = FP(Inf))
        bc = DirichletBoundaryCondition(mesh.Γnodes, fill(FP(2), length(mesh.Γnodes)))
        unconstrained = ThermalDiffusionDR(mesh, material)
        constrained = ThermalDiffusionDR(mesh, material)
        FEMTools.apply_dirichlet!(constrained.T, bc.DoFs, bc.vals, CPU(), 1)
        initial_history = copy(unconstrained.T0)
        for dr in (unconstrained, constrained)
            solve!(dr, mesh, bc; dt = FP(0.1), workgroup = 1, check_interval = 2,
                max_iterations = 3, throw_on_failure = false, verbose = false)
        end
        @test unconstrained.T == constrained.T
        @test unconstrained.T0 == initial_history
    end
end
