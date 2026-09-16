using Test

using FEMTools
using FEMTools: MixedMeshCache
using KernelAbstractions: CPU, synchronize
using StaticArrays

# Single reference triangle. Linear velocity fields are reproduced exactly on
# it, so the vorticity seen by every quadrature point is the analytic one.
function _rotation_fixture(vfun; Δt = 0.25, τ0 = (0.7, -0.4, 0.3))
    element = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[SVector(0, 0), SVector(1, 0), SVector(0, 1)]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    mesh = Mesh(CPU(), coords, el2n, element; workgroup = 1)
    nq = length(element.integration_points.ω)

    dr = StokesDR(CPU(), 3, 3, (1.0,), (1.0,), (0.0,); stress_size = (nq, 1))
    for (component, field) in enumerate(Tuple(dr.v))
        copyto!(field, [vfun(c)[component] for c in coords])
    end
    for (slot, value) in zip(Tuple(dr.τ), τ0)
        fill!(slot, value)
    end
    return (; dr, mesh, element, Δt, nq, τ0)
end

# Rotation of a symmetric 2-D tensor by θ, written independently of the
# implementation under test.
function _rotate_reference(τxx, τyy, τxy, θ)
    R = @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)]
    τ = @SMatrix [τxx τxy; τxy τyy]
    rotated = R * τ * R'
    return rotated[1, 1], rotated[2, 2], rotated[1, 2]
end

@testset "rotate_stress! leaves stress untouched without vorticity" begin
    # Pure shear: ∂vx/∂y and ∂vy/∂x both vanish, so ω = 0.
    ε̇ = 1.5
    f = _rotation_fixture(c -> (ε̇ * c[1], -ε̇ * c[2]))

    rotate_stress!(f.dr, f.mesh, f.mesh.geometry, f.element, f.Δt)

    for (slot, value) in zip(Tuple(f.dr.τ_old), f.τ0)
        @test all(≈(value), slot)
    end
end

@testset "rotate_stress! applies the solid-body rotation" begin
    # vx = -Ω y, vy = Ω x gives ω = (∂vx/∂y - ∂vy/∂x)/2 = -Ω.
    Ω = 0.8
    f = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1]))
    θ = -Ω * f.Δt

    rotate_stress!(f.dr, f.mesh, f.mesh.geometry, f.element, f.Δt)

    expected = _rotate_reference(f.τ0..., θ)
    for (slot, value) in zip(Tuple(f.dr.τ_old), expected)
        @test all(≈(value), slot)
    end

    # A rotation is an orthogonal similarity, so trace and determinant survive.
    τxx, τyy, τxy = (first(slot) for slot in Tuple(f.dr.τ_old))
    @test τxx + τyy ≈ f.τ0[1] + f.τ0[2]
    @test τxx * τyy - τxy^2 ≈ f.τ0[1] * f.τ0[2] - f.τ0[3]^2
end

@testset "rotate_stress! is the identity for a vanishing step" begin
    Ω = 0.8
    f = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1]))

    rotate_stress!(f.dr, f.mesh, f.mesh.geometry, f.element, 0.0)

    for (slot, value) in zip(Tuple(f.dr.τ_old), f.τ0)
        @test all(≈(value), slot)
    end
end

@testset "rotate_stress! accepts a MixedMeshCache" begin
    Ω = 0.8
    from_cache = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1]))
    from_geo = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1]))

    cache = MixedMeshCache(from_cache.mesh.geometry, from_cache.mesh.geometry)
    rotate_stress!(from_cache.dr, from_cache.mesh, cache, from_cache.element, from_cache.Δt)
    rotate_stress!(from_geo.dr, from_geo.mesh, from_geo.mesh.geometry, from_geo.element, from_geo.Δt)

    for (cached, raw) in zip(Tuple(from_cache.dr.τ_old), Tuple(from_geo.dr.τ_old))
        @test cached ≈ raw
    end
end
