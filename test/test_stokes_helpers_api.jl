using Test

using FEMTools
using FEMTools: precompute_stokes_geometry!, stokes_update_rate!, stokes_update_variable!
using KernelAbstractions
using KernelAbstractions: CPU
using StaticArrays

@testset "stokes_update_rate! trailing backend/workgroup" begin
    backend, workgroup = CPU(), 4
    R = [2.0, -4.0, 6.0]
    PC = [1.0, 2.0, 3.0]
    β = 0.5

    ∂u∂τ = [1.0, 1.0, 1.0]
    stokes_update_rate!(∂u∂τ, R, PC, β, length(R), backend, workgroup)

    @test ∂u∂τ ≈ R ./ PC .+ β        # β·1 initial rate
end

@testset "stokes_update_variable! trailing backend/workgroup" begin
    backend, workgroup = CPU(), 4
    ∂u∂τ = [0.5, -1.0, 2.0]
    α_dr = 0.25

    u = [10.0, 20.0, 30.0]
    stokes_update_variable!(u, ∂u∂τ, α_dr, length(u), backend, workgroup)

    @test u ≈ [10.0, 20.0, 30.0] .+ α_dr .* ∂u∂τ
end

@testset "precompute_stokes_geometry! trailing backend/workgroup" begin
    backend, workgroup = CPU(), 4
    element = ReferenceElement(LinearElement{2, 3, Float64})
    ip = element.integration_points
    NQ = length(ip.ω)
    ξq = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> FEMTools.eval_shape_function_jacobian(element, ξq[q]), NQ)

    coords = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    nels = 1
    geotype = NTuple{NQ, Tuple{SMatrix{3, 2, Float64, 6}, Float64}}

    geo = KernelAbstractions.allocate(backend, geotype, nels)
    precompute_stokes_geometry!(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(3), nels, backend, workgroup)

    # Unit reference triangle: quadrature weights sum to its area, and the
    # linear shape-function gradients are the same at every point.
    @test sum(last, geo[1]) ≈ 0.5
    @test all(∂N∂x ≈ first(geo[1])[1] for (∂N∂x, _) in geo[1])
end
