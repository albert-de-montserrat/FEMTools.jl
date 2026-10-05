using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU
using StaticArrays

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
    stokes_mesh = MixedMesh(thermal_mesh, element_P; workgroup)

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
        γP, stokes, stokes_mesh, 1.0, Δt; workgroup,
    )

    stats = solve_coupled_dyrel!(
        thermal, stokes, thermal_mesh, stokes_mesh,
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
        thermal_state, stokes, mesh_T, stokes_mesh,
        bc_T, bc_v, bc_v, Δt, γP; workgroup, verbose = false,
    )
    topology_only = Mesh(backend, Array(thermal_mesh.coords), Array(thermal_mesh.el2n); order = 2)
    @test_throws "thermal mesh has no geometry" coupled(thermal, topology_only)
    coarser = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2); workgroup)
    @test_throws "must match the Stokes velocity-node layout" coupled(thermal, coarser)
    oversized = ThermalDiffusionDR(backend, thermal_mesh.nnodes + 1, thermal_material)
    @test_throws "thermal state size must match" coupled(oversized, thermal_mesh)
end

@testset "coupled shear heating" begin
    backend = CPU()
    workgroup = 1
    Δt = 0.5
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    thermal_mesh = Mesh(
        backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2);
        workgroup,
    )
    stokes_mesh = MixedMesh(thermal_mesh, element_P; workgroup)
    coords = thermal_mesh.coords
    Γnodes = thermal_mesh.Γnodes

    # Simple shear vx = γ̇ y carries the uniform Maxwell stress τxy = ηve γ̇, whose
    # dissipation τxy²/η excludes the elastically stored power.
    η, G, γ̇ = 2.0, 1.0, 1.0
    ηve = inv(inv(η) + inv(G * Δt))
    Φ = (ηve * γ̇)^2 / η
    T_expected = Δt * Φ

    function solve(shear_heating)
        thermal = ThermalDiffusionDR(
            backend, thermal_mesh.nnodes,
            ThermalMaterial(; k = (1.0,), Cp = (1.0,), ρ0 = (1.0,), α = (0.0,), K = (Inf,));
            ϵ = 1.0e-10,
        )
        stokes = StokesDR(
            backend, stokes_mesh.nnodes, stokes_mesh.nnodesP,
            StokesMaterial(; η = (η,), ηb = (1.0,), G = (G,), α = (0.0,));
            ϵ = 1.0e-10,
        )
        # Near, but not at, the solution: the relaxation needs a nonzero residual.
        stokes.v.x .= [γ̇ * c[2] + 0.1 * sinpi(c[1]) * sinpi(c[2]) for c in coords]
        bc_vx = DirichletBoundaryCondition(nothing, Γnodes, [γ̇ * coords[n][2] for n in Γnodes])
        bc_vy = DirichletBoundaryCondition(nothing, Γnodes, zeros(length(Γnodes)))
        T_Γ = shear_heating ? T_expected : 0.0
        bc_T = DirichletBoundaryCondition(nothing, Γnodes, fill(T_Γ, length(Γnodes)))
        γP = zeros(stokes_mesh.nnodesP)
        assemble_viscosity_weighted_pressure_scaling!(γP, stokes, stokes_mesh, 1.0, Δt; workgroup)
        stats = solve_coupled_dyrel!(
            thermal, stokes, thermal_mesh, stokes_mesh,
            bc_T, bc_vx, bc_vy, Δt, γP;
            workgroup, shear_heating, ncheck = 10, iterMax = 20_000,
            total_iterMax = 20_000, ϵ_tol = 1.0e-10, verbose = false,
        )
        return stats, thermal
    end

    stats, thermal = solve(true)
    @test stats.converged
    @test all(T -> isapprox(T, T_expected; atol = 1.0e-8), thermal.T)
    @test all(iszero, thermal.source)

    stats_off, thermal_off = solve(false)
    @test stats_off.converged
    @test all(T -> isapprox(T, 0.0; atol = 1.0e-12), thermal_off.T)
end

@testset "3-D shear dissipation" begin
    # Viscous simple shear vx = γ̇ y: τxy = η γ̇ and Φ = τ:ε̇ = η γ̇².
    η, γ̇ = 2.0, 0.3
    ∇v = (SA[0.0, γ̇, 0.0], SA[0.0, 0.0, 0.0], SA[0.0, 0.0, 0.0])
    τ = (0.0, 0.0, 0.0, η * γ̇, 0.0, 0.0)
    @test FEMTools.shear_dissipation(τ, τ, ∇v, 0.0) ≈ η * γ̇^2

    # A plane flow embedded in 3-D dissipates as in plane strain.
    τ2, τ2_o = (0.7, -0.4, 0.3), (0.2, 0.1, -0.5)
    ∇v2 = (SA[0.4, -0.2], SA[0.6, -0.1])
    embed((xx, yy, xy)) = (xx, yy, -(xx + yy), xy, 0.0, 0.0)
    ∇v3 = (SA[∇v2[1]..., 0.0], SA[∇v2[2]..., 0.0], SA[0.0, 0.0, 0.0])
    @test FEMTools.shear_dissipation(embed(τ2), embed(τ2_o), ∇v3, 1.7) ≈
        FEMTools.shear_dissipation(τ2, τ2_o, ∇v2, 1.7)
end
