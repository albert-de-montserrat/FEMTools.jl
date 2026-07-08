using Test
using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU, synchronize
using StaticArrays

function _postprocess_geometry(coords, el2n, nels, element::ReferenceElement{E}) where {E <: AbstractElement{2, NV}} where {NV}
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
end
