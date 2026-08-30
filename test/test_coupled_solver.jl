using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU

@testset "coupled thermal--Stokes DYREL" begin
    backend = CPU()
    workgroup = 1
    Δt = 1.0
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    thermal_mesh = Mesh(
        backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (1, 1);
        workgroup,
    )
    stokes_mesh = MixedMesh(thermal_mesh, element_P)
    cache = MixedMeshCache(backend, workgroup, stokes_mesh, element_v, element_P)

    thermal_material = ThermalMaterial(;
        k = (1.0,), Cp = (1.0,), ρ0 = (1.0,), α = (0.0,), K = (Inf,),
    )
    thermal = ThermalDiffusionDR(
        backend, thermal_mesh.nnodes, thermal_material; ϵ = 1.0e-8,
    )
    thermal_ref = ThermalDiffusionDR(
        backend, thermal_mesh.nnodes, thermal_material; ϵ = 1.0e-8,
    )
    fill!(thermal.source, 1.0)
    fill!(thermal_ref.source, 1.0)

    Γnodes = thermal_mesh.Γnodes
    zero_bc = zeros(length(Γnodes))
    bc_T = DirichletBoundaryCondition(nothing, Γnodes, zero_bc)
    solver!(
        thermal_ref, Δt, thermal_mesh, bc_T;
        workgroup, ncheck = 1, iterMax = 2000, verbose = false,
    )

    stokes = StokesDR(
        backend, stokes_mesh.nnodes, stokes_mesh.nnodesP,
        StokesMaterial(; η = (1.0,), ηb = (1.0,), G = (Inf,), α = (0.0,));
        ϵ = 1.0e-8,
    )
    bc_v = DirichletBoundaryCondition(nothing, Γnodes, zero_bc)
    γP = zeros(stokes_mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, stokes, stokes_mesh, cache, 1.0, Δt; workgroup,
    )

    stats = solve_coupled_dyrel!(
        thermal, stokes, thermal_mesh, stokes_mesh, cache,
        bc_T, bc_v, bc_v, Δt, γP;
        workgroup, ncheck = 1, iterMax = 2000, total_iterMax = 2000,
        max_ph_iterations = 5, ϵ_tol = 1.0e-8, verbose = false,
    )

    @test stats.converged
    @test stats.err_T < thermal.ϵ
    @test stats.thermal_iterations == stats.iter
    @test thermal.T ≈ thermal_ref.T
    @test stokes.T[stokes_mesh.DoFsP] ≈ thermal.T[stokes_mesh.el2nP]

    coupled(thermal_state, mesh_T) = solve_coupled_dyrel!(
        thermal_state, stokes, mesh_T, stokes_mesh, cache,
        bc_T, bc_v, bc_v, Δt, γP; workgroup, verbose = false,
    )
    topology_only = Mesh(backend, Array(thermal_mesh.coords), Array(thermal_mesh.el2n); order = 2)
    @test_throws "thermal mesh has no geometry" coupled(thermal, topology_only)
    coarser = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2); workgroup)
    @test_throws "must match the Stokes velocity-node layout" coupled(thermal, coarser)
    oversized = ThermalDiffusionDR(backend, thermal_mesh.nnodes + 1, thermal_material)
    @test_throws "thermal state size must match" coupled(oversized, thermal_mesh)
end
