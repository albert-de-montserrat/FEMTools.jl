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

    ∂u∂τ_new = [1.0, 1.0, 1.0]
    stokes_update_rate!(∂u∂τ_new, R, PC, β, length(R), backend, workgroup)

    ∂u∂τ_old = [1.0, 1.0, 1.0]
    @test_deprecated stokes_update_rate!(backend, workgroup, ∂u∂τ_old, R, PC, β, length(R))

    @test ∂u∂τ_new ≈ R ./ PC .+ β        # β·1 initial rate
    @test ∂u∂τ_old ≈ ∂u∂τ_new
end

@testset "stokes_update_variable! trailing backend/workgroup" begin
    backend, workgroup = CPU(), 4
    ∂u∂τ = [0.5, -1.0, 2.0]
    α_dr = 0.25

    u_new = [10.0, 20.0, 30.0]
    stokes_update_variable!(u_new, ∂u∂τ, α_dr, length(u_new), backend, workgroup)

    u_old = [10.0, 20.0, 30.0]
    @test_deprecated stokes_update_variable!(backend, workgroup, u_old, ∂u∂τ, α_dr, length(u_old))

    @test u_new ≈ [10.0, 20.0, 30.0] .+ α_dr .* ∂u∂τ
    @test u_old ≈ u_new
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

    geo_new = KernelAbstractions.allocate(backend, geotype, nels)
    precompute_stokes_geometry!(geo_new, coords, el2n, ∂N∂ξq, ip.ω, Val(3), nels, backend, workgroup)

    geo_old = KernelAbstractions.allocate(backend, geotype, nels)
    @test_deprecated precompute_stokes_geometry!(backend, workgroup, geo_old, coords, el2n, ∂N∂ξq, ip.ω, Val(3), nels)

    @test geo_old == geo_new
end
