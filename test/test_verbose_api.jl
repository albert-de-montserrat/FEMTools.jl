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

_thermal_geometry(coords, el2n, nels, element) =
    precompute_geometry(coords, el2n, element; backend = CPU(), workgroup = 1)

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
    mesh = MixedMesh(mesh_v, element_P; workgroup)
    material = StokesMaterial(; η = (1.0,), ηb = (1.0,), G = (Inf,))
    dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, material; ϵ = 2.0)
    γP = zeros(Float64, mesh.nnodesP)
    FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, 1.0, 1.0; workgroup,
    )
    nq = length(element_v.integration_points.ω)
    τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)
    bc = DirichletBoundaryCondition(nothing, Γnodes, zeros(Float64, length(Γnodes)))
    return (; dr, mesh, τ_old, γP, Γnodes, bc, workgroup)
end

function _stokes_output(; kwargs...)
    (; dr, mesh, τ_old, γP, bc, workgroup) = _stokes_case()
    return _capture_stdout() do
        solve_stokes_dyrel!(
            dr, mesh, bc, bc, 1.0, γP; τ_old, workgroup,
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
    (; dr, mesh, τ_old, γP, Γnodes, workgroup) = _stokes_case()
    coords = Array(mesh.coords)
    vx_nodes = Int32[n for n in Γnodes if coords[n][1] ≈ 0.0 || coords[n][1] ≈ 1.0]
    vy_nodes = Int32[n for n in Γnodes if coords[n][2] ≈ 0.0 || coords[n][2] ≈ 1.0]
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, zeros(Float64, length(vx_nodes)))
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, zeros(Float64, length(vy_nodes)))
    return _capture_stdout() do
        solve_stokes_dyrel!(
            dr, mesh, bc_vx, bc_vy, 1.0, γP; τ_old, workgroup,
            ncheck = 1,
            ϵ_tol = 2.0,
            iterMax = 0,
            total_iterMax = 0,
            max_ph_iterations = 1,
            verbose = false,
            verbose_inner = false,
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
end
