using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU, synchronize
using StaticArrays

function _capture_stdout(f)
    return mktemp() do _, io
        redirect_stdout(io) do
            f()
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
end

function _thermal_geometry(coords, el2n, nels, element::ReferenceElement{E}) where {E <: AbstractElement{2, NV}} where {NV}
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

function _thermal_case()
    backend = CPU()
    workgroup = 1
    element = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element, (1, 1))
    geo = _thermal_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    dr = ThermalDiffusionDR(
        backend, mesh.nnodes, (1.0,), (1.0,), (1.0,), (0.0,), (Inf,);
        ϵ = 2.0,
    )
    fill!(dr.source, 1.0)
    copyto!(dr.T0, dr.T)
    return (; dr, mesh, geo, element, backend, workgroup)
end

function _thermal_output(; verbose)
    (; dr, mesh, geo, element, backend, workgroup) = _thermal_case()
    empty_i = Int32[]
    empty_v = Float64[]
    return _capture_stdout() do
        solver!(
            dr, 1.0, mesh, geo, element, empty_i, empty_v, empty_v,
            backend, workgroup;
            ncheck = 1,
            verbose,
        )
    end
end

function _stokes_case()
    backend = CPU()
    workgroup = 1
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (1, 1))
    Γnodes = mesh_v.Γnodes
    mesh = MixedMesh(mesh_v, element_P)
    cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)
    dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, (1.0,), (1.0,), (0.0,); ϵ = 2.0)
    γP = zeros(Float64, mesh.nnodesP)
    FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache.geo_P, element_v, element_P,
        1.0, 1.0, backend, workgroup,
    )
    nq = length(element_v.integration_points.ω)
    τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)
    bc = zeros(Float64, length(Γnodes))
    return (; dr, mesh, cache, element_v, element_P, τ_old, γP, Γnodes, bc, backend, workgroup)
end

function _stokes_output(; kwargs...)
    (; dr, mesh, cache, element_v, element_P, τ_old, γP, Γnodes, bc, backend, workgroup) = _stokes_case()
    return _capture_stdout() do
        solve_stokes_dyrel!(
            dr, mesh, cache, element_v, element_P,
            dr.phases_v, dr.phases_P, τ_old, nothing, (Inf,), 1.0, γP,
            Γnodes, bc, bc, backend, workgroup;
            ncheck = 1,
            ϵ_tol = 2.0,
            iterMax = 0,
            total_iterMax = 0,
            max_ph_iterations = 1,
            kwargs...,
        )
    end
end

function _stokes_split_bc_output()
    (; dr, mesh, cache, element_v, element_P, τ_old, γP, Γnodes, backend, workgroup) = _stokes_case()
    coords = Array(mesh.coords)
    vx_nodes = Int32[n for n in Γnodes if coords[n][1] ≈ 0.0 || coords[n][1] ≈ 1.0]
    vy_nodes = Int32[n for n in Γnodes if coords[n][2] ≈ 0.0 || coords[n][2] ≈ 1.0]
    return _capture_stdout() do
        solve_stokes_dyrel!(
            dr, mesh, cache, element_v, element_P,
            dr.phases_v, dr.phases_P, τ_old, nothing, (Inf,), 1.0, γP,
            Γnodes, zeros(Float64, length(vx_nodes)), zeros(Float64, length(vy_nodes)), backend, workgroup;
            ncheck = 1,
            ϵ_tol = 2.0,
            iterMax = 0,
            total_iterMax = 0,
            max_ph_iterations = 1,
            verbose = false,
            verbose_inner = false,
            vx_nodes = vx_nodes,
            vy_nodes = vy_nodes,
        )
    end
end

@testset "solver verbosity API" begin
    @test _thermal_output(verbose = false) == ""
    @test occursin("PT", _thermal_output(verbose = true))

    @test _stokes_output(verbose = false, verbose_inner = false) == ""
    @test _stokes_split_bc_output() == ""
    @test_logs (:info, "Initial momentum preconditioner") begin
        @test occursin("itPH", _stokes_output(verbose = true, verbose_inner = false))
    end

    @test (@test_deprecated _stokes_output(verbose_PH = false, verbose_inner = false)) == ""
    @test (@test_deprecated _stokes_output(verbose = false, verbose_DR = false)) == ""
end
