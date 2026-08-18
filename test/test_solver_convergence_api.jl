using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU, synchronize
using StaticArrays

function _convergence_geometry(coords, el2n, nels, element::ReferenceElement{E}) where {E <: AbstractElement{2, NV}} where {NV}
    ip = element.integration_points
    NQ = length(ip.ω)
    FP = eltype(ip.ω)
    ξq = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo = Vector{NTuple{NQ, Tuple{SMatrix{NV, 2, FP, 2NV}, FP}}}(undef, nels)
    FEMTools.precompute_geometry_kernel!(CPU(), 1)(
        geo, coords, el2n, ∂N∂ξq, ip.ω, Val(NV);
        ndrange = nels,
    )
    synchronize(CPU())
    return geo
end

@testset "Stokes PH uses per-cycle targets and fresh final residuals" begin
    backend = CPU()
    workgroup = 1
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2))
    mesh = MixedMesh(mesh_v, element_P)
    cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)
    nq = length(element_v.integration_points.ω)
    dr = StokesDR(
        backend, mesh.nnodes, mesh.nnodesP, (1.0,), (Inf,), (0.0,);
        ρ0 = (1.0,), K = (Inf,), g = (0.0, -1.0),
        stress_size = (nq, mesh.nels),
    )
    γP = zeros(Float64, mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache.geo_P, element_v, element_P,
        20.0, 1.0, backend, workgroup,
    )
    τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)

    coords = Array(mesh.coords)
    Γnodes = Array(mesh_v.Γnodes)
    vx_nodes = Int32[n for n in Γnodes if coords[n][1] ≈ 0.0 || coords[n][1] ≈ 1.0]
    vy_nodes = Int32[n for n in Γnodes if coords[n][2] ≈ 0.0 || coords[n][2] ≈ 1.0]
    bc_vx = zeros(Float64, length(vx_nodes))
    bc_vy = zeros(Float64, length(vy_nodes))

    stats = solve_stokes_dyrel!(
        dr, mesh, cache, element_v, element_P,
        dr.phases_v, dr.phases_P, τ_old, nothing, (Inf,), 1.0, γP,
        Int32.(Γnodes), bc_vx, bc_vy, backend, workgroup;
        ncheck = 1,
        ϵ_tol = 0.0,
        iterMax = 3,
        total_iterMax = 5,
        max_ph_iterations = 5,
        rel_drop0 = 0.5,
        verbose = false,
        verbose_inner = false,
        vx_nodes,
        vy_nodes,
        collect_history = true,
    )

    @test stats.iter == 5
    @test stats.reached_total_iter
    @test !stats.converged
    @test all(h.itPT ≤ 3 for h in stats.history)
    for itPH in unique(h.itPH for h in stats.history)
        cycle = filter(h -> h.itPH == itPH, stats.history)
        @test all(h.target_v == first(cycle).target_v for h in cycle)
    end

    total_mass_V = sum(dr.M_V)
    free_mass_vx = total_mass_V - sum(dr.M_V[vx_nodes])
    free_mass_vy = total_mass_V - sum(dr.M_V[vy_nodes])
    expected_err_v = max(
        FEMTools._mass_weighted_rms(dr.Rv_x, dr.M_V, free_mass_vx),
        FEMTools._mass_weighted_rms(dr.Rv_y, dr.M_V, free_mass_vy),
    ) / 2
    expected_err_P = FEMTools._mass_weighted_rms(dr.RP, dr.M_P, sum(dr.M_P))
    @test stats.err_v ≈ expected_err_v
    @test stats.err_P ≈ expected_err_P
end

function _tiny_triangle_mesh()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = Mesh(CPU(), (0.0 .. 1.0) × (0.0 .. 1.0), element, (1, 1))
    geo = _convergence_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    return (; mesh, geo, element)
end

function _orphan_triangle_mesh()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(2.0, 2.0),
    ]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    mesh = Mesh(CPU(), coords, el2n)
    geo = _convergence_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    return (; mesh, geo, element)
end

function _orphan_stokes_case()
    backend = CPU()
    workgroup = 1
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(0.5, 0.0),
        SVector(0.5, 0.5),
        SVector(0.0, 0.5),
        SVector(2.0, 2.0),
    ]
    el2n = reshape(Int32[1, 2, 3, 4, 5, 6], 6, 1)
    mesh_v = Mesh(backend, coords, el2n; order = 2)
    mesh = MixedMesh(mesh_v, element_P)
    cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)
    nq = length(element_v.integration_points.ω)
    material = StokesMaterial(; η = (1.0,), ηb = (1.0,), G = (Inf,), α = (0.0,))
    dr = StokesDR(
        backend, mesh.nnodes, mesh.nnodesP, material;
        stress_size = (nq, mesh.nels),
    )
    γP = zeros(Float64, mesh.nnodesP)
    FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache, 1.0, 1.0; workgroup,
    )
    τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)
    return (; dr, mesh, cache, element_v, element_P, τ_old, γP, backend, workgroup)
end

function _caught_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "DR solvers report non-convergence" begin
    (; mesh, geo, element) = _tiny_triangle_mesh()
    empty_i = Int32[]
    empty_v = Float64[]

    thermal = ThermalDiffusionDR(CPU(), mesh.nnodes, (1.0,), (1.0,), (1.0,), (0.0,), (Inf,); ϵ = 0.0)
    fill!(thermal.source, 1.0)
    copyto!(thermal.T0, thermal.T)
    thermal_err = _caught_error() do
        solver!(
            thermal, 1.0, mesh, geo, element, empty_i, empty_v, empty_v, CPU(), 1;
            ncheck = 1,
            iterMax = 1,
            verbose = false,
        )
    end
    @test thermal_err isa ErrorException
    @test occursin("Thermal diffusion DR solver did not converge", sprint(showerror, thermal_err))

    litho = LithostaticPressureDR(CPU(), mesh.nnodes, (1.0,), (0.0,), (Inf,); ϵ = 0.0)
    litho_err = _caught_error() do
        solver!(
            litho, mesh, geo, element, empty_i, empty_v, empty_v, CPU(), 1;
            ncheck = 1,
            iterMax = 1,
            verbose = false,
            g = SVector(0.0, -1.0),
        )
    end
    @test litho_err isa ErrorException
    @test occursin("Lithostatic pressure DR solver did not converge", sprint(showerror, litho_err))
end

@testset "DR solvers accept mesh-owned geometry and BC objects" begin
    element = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(2.0, 2.0),
    ]
    mesh = Mesh(CPU(), coords, reshape(Int32[1, 2, 3], 3, 1), element; workgroup = 1)
    bc = DirichletBoundaryCondition(nothing, Int32[], Float64[])

    thermal = ThermalDiffusionDR(CPU(), mesh.nnodes, (1.0,), (1.0,), (1.0,), (0.0,), (Inf,))
    thermal_err = _caught_error() do
        solver!(thermal, 1.0, mesh, bc; workgroup = 1, verbose = false)
    end
    @test occursin("thermal diffusion preconditioner produced invalid λmax", sprint(showerror, thermal_err))

    litho = LithostaticPressureDR(CPU(), mesh.nnodes, (1.0,), (0.0,), (Inf,))
    litho_err = _caught_error() do
        solver!(litho, mesh, bc; workgroup = 1, verbose = false, g = SVector(0.0, -1.0))
    end
    @test occursin("lithostatic pressure preconditioner produced invalid λmax", sprint(showerror, litho_err))
end

@testset "DR solvers reject zero preconditioners" begin
    (; mesh, geo, element) = _orphan_triangle_mesh()
    empty_i = Int32[]
    empty_v = Float64[]

    thermal = ThermalDiffusionDR(CPU(), mesh.nnodes, (1.0,), (1.0,), (1.0,), (0.0,), (Inf,))
    thermal_err = _caught_error() do
        solver!(
            thermal, 1.0, mesh, geo, element, empty_i, empty_v, empty_v, CPU(), 1;
            verbose = false,
        )
    end
    @test thermal_err isa ErrorException
    @test occursin("thermal diffusion preconditioner produced invalid λmax", sprint(showerror, thermal_err))

    litho = LithostaticPressureDR(CPU(), mesh.nnodes, (1.0,), (0.0,), (Inf,))
    litho_err = _caught_error() do
        solver!(
            litho, mesh, geo, element, empty_i, empty_v, empty_v, CPU(), 1;
            verbose = false,
            g = SVector(0.0, -1.0),
        )
    end
    @test litho_err isa ErrorException
    @test occursin("lithostatic pressure preconditioner produced invalid λmax", sprint(showerror, litho_err))

    (; dr, mesh, cache, element_v, element_P, τ_old, γP, backend, workgroup) = _orphan_stokes_case()
    stokes_err = _caught_error() do
        solve_stokes_dyrel!(
            dr, mesh, cache, element_v, element_P,
            dr.phases_v, dr.phases_P, τ_old, nothing, (Inf,), 1.0, γP,
            empty_i, empty_v, empty_v, backend, workgroup;
            verbose = false,
            verbose_inner = false,
        )
    end
    @test stokes_err isa ErrorException
    @test occursin("stokes vx preconditioner produced invalid λmax", sprint(showerror, stokes_err))
end
