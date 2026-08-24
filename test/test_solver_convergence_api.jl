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
    geo = Matrix{Tuple{SMatrix{NV, 2, FP, 2NV}, FP}}(undef, NQ, nels)
    FEMTools.precompute_geometry_kernel!(CPU(), 1)(
        geo, coords, el2n, ∂N∂ξq, ip.ω, Val(NV);
        ndrange = nels,
    )
    synchronize(CPU())
    return geo
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

# Unit square with a denser lower half under free-slip walls. Small enough to
# converge quickly, buoyant enough that the velocity field is nontrivial.
function _buoyancy_stokes_case()
    backend = CPU()
    workgroup = 1
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (3, 3))
    mesh = MixedMesh(mesh_v, element_P)
    cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)

    coords = mesh.coords
    centroid(i) = sum(coords[mesh.el2nP[a, i]] for a in 1:3) / 3
    phases = reshape([centroid(i)[2] < 0.5 ? 2 : 1 for i in 1:mesh.nels], 1, :)

    Γ = mesh_v.Γnodes
    tol = 1.0e-9
    vx_nodes = Int32[n for n in Γ if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - 1.0) ≤ tol]
    vy_nodes = Int32[n for n in Γ if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - 1.0) ≤ tol]
    return (; backend, workgroup, element_v, element_P, mesh, cache, phases,
        Γ, vx_nodes, vy_nodes)
end

@testset "solve_stokes_dyrel! measures λmax by power iteration" begin
    (; backend, workgroup, element_v, element_P, mesh, cache, phases,
        Γ, vx_nodes, vy_nodes) = _buoyancy_stokes_case()

    η = (1.0, 1.0)
    Δt = 1.0
    nq = length(element_v.integration_points.ω)
    bc = zeros(length(Γ))

    function solve(; measure_λmax)
        dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, η, (Inf, Inf), (0.0, 0.0);
            ρ0 = (1.0, 2.0), K = (Inf, Inf), g = SVector(0.0, -1.0), Tref = 0.0,
            CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.7, stress_size = (nq, mesh.nels))
        γP = zeros(Float64, mesh.nnodesP)
        FEMTools.assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh, cache.geo_P, element_v, element_P, 20.0, Δt, backend, workgroup;
            phases_v = phases, η)
        τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)
        stats = solve_stokes_dyrel!(
            dr, mesh, cache, element_v, element_P,
            phases, phases, τ_old, nothing, (Inf, Inf), Δt, γP,
            Γ, bc, bc, backend, workgroup;
            ncheck = 100, ϵ_tol = 1.0e-9, rel_drop0 = 0.1,
            verbose = false, verbose_inner = false, vx_nodes, vy_nodes, measure_λmax)
        return dr, stats
    end

    dr_bound, bound = solve(measure_λmax = false)
    @test bound.converged
    @test bound.λmax_iterations == 0
    @test bound.λmax == bound.λmax_gershgorin

    dr_power, power = solve(measure_λmax = true)
    @test power.converged
    @test power.λmax_iterations > 0
    # The measured value is capped by the Gershgorin bound, never exceeds it,
    # and a loose bound is what the power iteration exists to tighten.
    @test power.λmax ≤ power.λmax_gershgorin
    @test power.λmax_gershgorin ≈ bound.λmax_gershgorin

    # Both spectral estimates only set the pseudo-time step, so the converged
    # solution is the same either way.
    @test Array(dr_power.v.y) ≈ Array(dr_bound.v.y) rtol = 1.0e-5
    @test Array(dr_power.v.x) ≈ Array(dr_bound.v.x) rtol = 1.0e-5
    @test any(!iszero, Array(dr_power.v.y))
end
