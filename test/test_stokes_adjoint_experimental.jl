using Test
using FEMTools
using DomainSets
using DomainSets: ×
using KernelAbstractions: CPU

@testset "experimental coupled Stokes adjoint DR" begin
    backend = CPU()
    workgroup = 1
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2))
    mesh = MixedMesh(mesh_v, element_P)
    cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)
    phases = ones(Int, 1, mesh.nels)

    η = (1.0,)
    ηb = (Inf,)
    α = (0.0,)
    Δt = 1.0
    nq = length(element_v.integration_points.ω)
    dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, η, ηb, α;
        ρ0 = (1.0,), K = (Inf,), g = (0.0, 0.0), Tref = 0.0,
        CFL_v = 0.35, CFL_P = 0.35, c_fact = 0.7,
        stress_size = (nq, mesh.nels))
    γP = zeros(mesh.nnodesP)
    FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache.geo_P, element_v, element_P, 10.0, Δt,
        backend, workgroup; phases_v = phases, η)
    τ_old = ntuple(_ -> zeros(nq, mesh.nels), 3)

    tol = 1.0e-10
    on_x_wall(c) = abs(c[1]) ≤ tol || abs(c[1] - 1.0) ≤ tol
    on_y_wall(c) = abs(c[2]) ≤ tol || abs(c[2] - 1.0) ≤ tol
    vx_nodes = Int32[n for n in mesh_v.Γnodes if on_x_wall(mesh.coords[n])]
    vy_nodes = Int32[n for n in mesh_v.Γnodes if on_y_wall(mesh.coords[n])]

    objective_vx = zeros(mesh.nnodes)
    objective_vy = zeros(mesh.nnodes)
    free_vy = setdiff(collect(1:mesh.nnodes), Int.(vy_nodes))
    objective_vy[first(free_vy)] = 1.0
    λvx = zeros(mesh.nnodes)
    λvy = zeros(mesh.nnodes)
    λP = zeros(mesh.nnodesP)

    stats = solve_stokes_adjoint_coupled_experimental!(
        dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P,
        phases, phases, τ_old, nothing, (Inf,), Δt, γP,
        objective_vx, objective_vy, λvx, λvy, λP,
        backend, workgroup;
        vx_nodes, vy_nodes, ncheck = 25, adjoint_tol = 1.0e-7,
        iterMax = 20_000, verbose = false, collect_history = true,
    )

    @test all(isfinite, values(stats.λmax))
    @test all(>(0), values(stats.λmax))
    @test all(isfinite, values(stats.λmin))
    @test maximum(stats.spectral_rowsums.P) > maximum(stats.spectral_diagonals.P)
    @test !isempty(stats.history)
    @test stats.history[end].err < stats.history[1].err
    @test stats.converged
    @test any(!iszero, λP)
    @test maximum(abs, λvx[vx_nodes]) == 0
    @test maximum(abs, λvy[vy_nodes]) == 0
    @test abs(sum(λP .* dr.M_P)) ≤ 1.0e-10
end
