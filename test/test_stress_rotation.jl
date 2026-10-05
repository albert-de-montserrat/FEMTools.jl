using Test

using FEMTools
using FEMTools: MixedMeshCache
using KernelAbstractions: CPU, synchronize
using StaticArrays
using LinearAlgebra: cross, det, tr

# Single reference triangle (2-D) or tetrahedron (3-D). Linear velocity fields
# are reproduced exactly on it, so the vorticity seen by every quadrature point
# is the analytic one.
function _rotation_fixture(vfun; Δt = 0.25, τ0 = (0.7, -0.4, 0.3), dim = 2)
    if dim == 2
        element = ReferenceElement(LinearElement{2, 3, Float64})
        coords = SVector{2, Float64}[SVector(0, 0), SVector(1, 0), SVector(0, 1)]
        g = (0.0, 0.0)
    else
        element = ReferenceElement(LinearElement{3, 4, Float64})
        coords = SVector{3, Float64}[
            SVector(0, 0, 0), SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1),
        ]
        g = (0.0, 0.0, 0.0)
    end
    nv = length(coords)
    el2n = reshape(Int32.(1:nv), nv, 1)
    mesh = Mesh(CPU(), coords, el2n, element; workgroup = 1)
    nq = length(element.integration_points.ω)

    dr = StokesDR(CPU(), nv, nv, (1.0,), (1.0,), (0.0,); g, stress_size = (nq, 1))
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

@testset "rotate_stress! reads a MixedMesh's geometry" begin
    Ω = 0.8
    from_mixed = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1]))
    from_geo = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1]))

    mixed = MixedMesh(from_mixed.mesh, from_mixed.element; workgroup = 1)
    rotate_stress!(from_mixed.dr, mixed, from_mixed.Δt)
    rotate_stress!(from_geo.dr, from_geo.mesh, from_geo.mesh.geometry, from_geo.element, from_geo.Δt)

    for (mixed_τ, raw) in zip(Tuple(from_mixed.dr.τ_old), Tuple(from_geo.dr.τ_old))
        @test mixed_τ ≈ raw
    end
end

# Deviatoric 3-D stress in SymmetricTensor3D order (xx, yy, zz, yz, xz, xy).
const _τ0_3D = (0.7, -0.4, -0.3, 0.2, -0.1, 0.3)

_tensor3D((xx, yy, zz, yz, xz, xy)) = @SMatrix [xx xy xz; xy yy yz; xz yz zz]

@testset "3-D rotate_stress! leaves stress untouched without vorticity" begin
    ε̇ = 1.5
    f = _rotation_fixture(c -> (ε̇ * c[1], -2ε̇ * c[2], ε̇ * c[3]); dim = 3, τ0 = _τ0_3D)

    rotate_stress!(f.dr, f.mesh, f.mesh.geometry, f.element, f.Δt)

    for (slot, value) in zip(Tuple(f.dr.τ_old), f.τ0)
        @test all(≈(value), slot)
    end
end

@testset "3-D rotate_stress! co-rotates a plane flow about z" begin
    # vx = -Ω y, vy = Ω x has curl (0, 0, 2Ω); the stress turns by +ΩΔt, the
    # opposite sense to the 2-D convention for the same flow.
    Ω = 0.8
    f = _rotation_fixture(c -> (-Ω * c[2], Ω * c[1], 0.0); dim = 3, τ0 = _τ0_3D)

    rotate_stress!(f.dr, f.mesh, f.mesh.geometry, f.element, f.Δt)

    R = @SMatrix [cos(Ω * f.Δt) -sin(Ω * f.Δt) 0; sin(Ω * f.Δt) cos(Ω * f.Δt) 0; 0 0 1]
    expected = R * _tensor3D(f.τ0) * R'
    xx, yy, zz, yz, xz, xy = Tuple(f.dr.τ_old)
    @test all(≈(expected[1, 1]), xx)
    @test all(≈(expected[2, 2]), yy)
    @test all(≈(expected[3, 3]), zz)
    @test all(≈(expected[2, 3]), yz)
    @test all(≈(expected[1, 3]), xz)
    @test all(≈(expected[1, 2]), xy)
end

@testset "3-D rotate_stress! preserves invariants of a rigid rotation" begin
    # v = a × x is a rigid rotation about an oblique axis a.
    a = SVector(0.3, -0.5, 0.7)
    f = _rotation_fixture(c -> Tuple(cross(a, c)); dim = 3, τ0 = _τ0_3D)

    rotate_stress!(f.dr, f.mesh, f.mesh.geometry, f.element, f.Δt)

    before = _tensor3D(f.τ0)
    after = _tensor3D(map(first, Tuple(f.dr.τ_old)))
    @test !(after ≈ before)
    @test tr(after) ≈ tr(before) atol = 1.0e-14
    @test sum(abs2, after) ≈ sum(abs2, before)
    @test det(after) ≈ det(before)
end
