using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU, synchronize
using StaticArrays

_postprocess_geometry(coords, el2n, nels, element) =
    precompute_geometry(coords, el2n, element; backend = CPU(), workgroup = 1)

function _postprocess_fixture()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = Mesh(CPU(), (0.0 .. 1.0) × (0.0 .. 1.0), element, (1, 1))
    geo = _postprocess_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    vx = zeros(Float64, mesh.nnodes)
    vy = zeros(Float64, mesh.nnodes)
    return (; mesh, geo, element, vx, vy)
end

@testset "strain-rate stress postprocess exposes one tauII field" begin
    (; mesh, geo, element, vx, vy) = _postprocess_fixture()
    nq = length(element.integration_points.ω)
    τ_ip = (
        fill(2.0, nq, mesh.nels),
        fill(-1.0, nq, mesh.nels),
        fill(0.5, nq, mesh.nels),
    )

    post = compute_strain_rate_stress_postprocess(vx, vy, mesh.el2n, geo, τ_ip, element)
    @test hasproperty(post, :tauII)
    @test !hasproperty(post, :τII)
    @test post.tauII ≈ fill(sqrt(3.25), mesh.nels)

    τ_old = ntuple(_ -> zeros(Float64, mesh.nnodes), 3)
    post_ve = compute_strain_rate_stress_postprocess(
        vx, vy, mesh.el2n, geo, ones(Int, mesh.nnodes), τ_old, (1.0,), (Inf,), 1.0, element,
    )
    @test hasproperty(post_ve, :tauII)
    @test !hasproperty(post_ve, :τII)
    @test all(iszero, post_ve.tauII)

    post_ve_matrix_phase = compute_strain_rate_stress_postprocess(
        vx, vy, mesh.el2n, geo, ones(Int, size(mesh.el2n)), τ_old, (1.0,), (Inf,), 1.0, element,
    )
    @test post_ve_matrix_phase.tauII == post_ve.tauII
end

@testset "stokes diagnostics preserve Float32" begin
    element = ReferenceElement(LinearElement{2, 3, Float32})
    mesh = Mesh(CPU(), (0.0f0 .. 1.0f0) × (0.0f0 .. 1.0f0), element, (1, 1))
    geo = _postprocess_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    vx = zeros(Float32, mesh.nnodes)
    vy = zeros(Float32, mesh.nnodes)
    nq = length(element.integration_points.ω)
    τ_ip = ntuple(_ -> zeros(Float32, nq, mesh.nels), 3)

    post = compute_strain_rate_stress_postprocess(vx, vy, mesh.el2n, geo, τ_ip, element)
    @test eltype(post.τxx) === Float32
    @test eltype(post.tauII) === Float32

    τ_old = ntuple(_ -> zeros(Float32, mesh.nnodes), 3)
    post_ve = compute_strain_rate_stress_postprocess(
        vx, vy, mesh.el2n, geo, ones(Int, mesh.nnodes), τ_old, (1.0f0,), (Inf32,), 1.0f0, element,
    )
    @test eltype(post_ve.τxx) === Float32
    @test eltype(post_ve.tauII) === Float32

    update_old_stress_from_cells!(τ_old, post_ve, mesh.el2n, mesh.nnodes)
    @test eltype(τ_old[1]) === Float32

    λmin = FEMTools._stokes_λmin(1.0f0, zeros(Float32, 2), zeros(Float32, 2), zeros(Float32, 2), ones(Float32, 2))
    α, β = FEMTools._stokes_cheb(1.0f0, λmin, 0.9f0)
    @test typeof(λmin) === Float32
    @test typeof(α) === Float32
    @test typeof(β) === Float32
end
