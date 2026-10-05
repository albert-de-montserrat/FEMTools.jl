"""
    PrincipalStresses(values, directions)

Principal stress fields in two or three dimensions. `values[k]` is the array
of kth principal values in descending algebraic order; `directions[k][c]` is
Cartesian component `c` of the corresponding unit direction. Each field has
the shape, floating-point type, and backend of the sampled stress.

The constructor wraps caller-owned tuples of arrays without copying them.
The container is immutable; its arrays are updated by
[`compute_principal_stresses!`](@ref), which validates the buffer layout before
computing. [`compute_principal_stresses`](@ref) allocates and fills a container.
"""
struct PrincipalStresses{V, D}
    values::V
    directions::D
end

_principal_components(τ::Tuple) = τ
_principal_components(τ::SymmetricTensor2D) = (τ.xx, τ.yy, τ.xy)
_principal_components(τ::SymmetricTensor3D) = (τ.xx, τ.yy, τ.zz, τ.xy, τ.xz, τ.yz)

function _check_principal_array(a, τ₀)
    a isa AbstractArray || throw(ArgumentError("stress, pressure, and output components must be arrays"))
    Base.require_one_based_indexing(a)
    axes(a) == axes(τ₀) || throw(DimensionMismatch("principal-stress arrays must have matching axes"))
    eltype(a) === eltype(τ₀) || throw(ArgumentError("principal-stress arrays must have matching floating-point types"))
    KA.get_backend(a) == KA.get_backend(τ₀) || throw(ArgumentError("principal-stress arrays must share a backend"))
    return nothing
end

function _check_principal_inputs(τ::Tuple{Vararg{Any, N}}, P, workgroup) where {N}
    N in (3, 6) || throw(ArgumentError("expected 3 (2D) or 6 (3D) stress components"))
    τ₀ = first(τ)
    τ₀ isa AbstractArray || throw(ArgumentError("stress components must be arrays"))
    eltype(τ₀) in (Float32, Float64) || throw(ArgumentError("principal stresses require Float32 or Float64 arrays"))
    workgroup isa Integer && workgroup > 0 || throw(ArgumentError("workgroup must be a positive integer"))
    foreach(a -> _check_principal_array(a, τ₀), τ)
    P === nothing || _check_principal_array(P, τ₀)
    return Val(N == 3 ? 2 : 3)
end

"""
    compute_principal_stresses(τ; pressure=nothing, workgroup=256)

Compute pointwise principal stress values and unit directions in two or three
dimensions. `τ` is a `FEMTools.SymmetricTensor2D`/`SymmetricTensor3D`, or a tuple
of component arrays in [`FEMTools.stress`](@ref) order: `(xx, yy, xy)` in 2D,
`(xx, yy, zz, xy, xz, yz)` in 3D. The tensor container's `II` field is ignored.

Return a [`PrincipalStresses`](@ref) container. `values[k]` is the kth principal
value array, ordered from largest to smallest algebraically;
`directions[k][c]` is Cartesian component `c` of its unit direction. All arrays
preserve the input shape, Float32/Float64 type, and backend. Computation uses
KernelAbstractions and synchronizes before returning, without host copies.

With a matching physical pressure array, return eigenvalues of `σ = τ - P I`:
pressure shifts the values and leaves directions unchanged. Positive Cauchy
stress denotes tension, and pressure is positive in compression. Pressure must
already be sampled at the same locations; this operation does not interpolate
or average. Cap plasticity requires corrected physical pressure, not trial
pressure or the solver relaxation term `Pnum`.

2D returns two **in-plane** eigenpairs. To include the out-of-plane principal
stress in plane strain, supply a 3D tuple with `τzz = -(τxx + τyy)` for
deviatoric stress and zero `τxz`, `τyz`.

Directions are axes (`v` and `-v` are equivalent); the largest-magnitude
component is made nonnegative, with ties resolved in Cartesian order. Repeated
values permit any orthonormal basis of their eigenspace, with no promise of
unique or spatially continuous directions. Eigenpairs of an averaged tensor
are different from averages of pointwise eigenpairs.

Inputs must have matching one-based axes, floating-point types, and backends.
Invalid shapes raise `DimensionMismatch`; unsupported types, backends, or
workgroup sizes raise `ArgumentError`. Nonfinite input, unrepresentable values,
or failed local eigenpair checks raise `DomainError` after the kernel completes.
"""
function compute_principal_stresses(τ; pressure = nothing, workgroup = 256)
    τᵢ = _principal_components(τ)
    D = _check_principal_inputs(τᵢ, pressure, workgroup)
    return _allocate_principal_stresses(τᵢ, pressure, workgroup, D)
end

function _allocate_principal_stresses(τ, P, workgroup, D::Val{d}) where {d}
    σ = PrincipalStresses(
        ntuple(_ -> similar(first(τ)), D),
        ntuple(_ -> ntuple(_ -> similar(first(τ)), D), D),
    )
    return _compute_principal_stresses!(σ, τ, P, workgroup, D)
end

"""
    compute_principal_stresses!(out::PrincipalStresses, τ; pressure=nothing, workgroup=256)

Compute principal stresses in-place in a [`PrincipalStresses`](@ref) container
and return the same `out`, reusing all its value and direction arrays.
The layout and numerical conventions match [`compute_principal_stresses`](@ref).
`out.values` must contain two or three arrays; `out.directions` must contain
the same number of direction tuples, each with two or three component arrays.
All buffers must match input axes, type, and backend, and must not alias inputs,
pressure, or one another. Aliasing raises `ArgumentError` before mutation.

The backend is synchronized before returning. A numerical failure raises
`DomainError`; output is then partially updated, with failed samples set to
NaN, and must not be used as a successful result. Structural validation occurs
before the launch. Empty arrays return without launching a kernel.
"""
function compute_principal_stresses!(σ::PrincipalStresses, τ; pressure = nothing, workgroup = 256)
    τᵢ = _principal_components(τ)
    D = _check_principal_inputs(τᵢ, pressure, workgroup)
    return _compute_principal_stresses!(σ, τᵢ, pressure, workgroup, D)
end

function _compute_principal_stresses!(σ, τ, P, workgroup, ::Val{D}) where {D}
    σ.values isa Tuple && length(σ.values) == D || throw(DimensionMismatch("expected $D principal-value arrays"))
    σ.directions isa Tuple && length(σ.directions) == D || throw(DimensionMismatch("expected $D principal directions"))
    foreach(σ.directions) do n
        n isa Tuple && length(n) == D || throw(DimensionMismatch("each principal direction needs $D components"))
    end
    buffers = (σ.values..., ntuple(i -> σ.directions[(i - 1) ÷ D + 1][(i - 1) % D + 1], Val(D^2))...)
    inputs = P === nothing ? τ : (τ..., P)
    _check_principal_buffers(buffers, inputs)
    isempty(first(τ)) && return σ
    backend = KA.get_backend(first(τ))
    # Runtime workgroup size keeps the kernel type concrete, unlike encoding
    # an arbitrary workgroup integer in the kernel's static type parameters.
    _principal_stresses_kernel!(backend)(σ.values, σ.directions, τ, P, Val(D); ndrange = length(first(τ)), workgroupsize = workgroup)
    KA.synchronize(backend)
    # Failed samples carry NaN in every value/direction, so one reduction
    # reports device-side failure without copying the fields to CPU.
    all(isfinite, first(σ.values)) || throw(DomainError(nothing, "principal-stress computation failed: nonfinite input, overflow, or inaccurate eigenpairs"))
    return σ
end

_check_principal_buffers(::Tuple{}, ::Tuple) = nothing
function _check_principal_buffers(buffers::Tuple, inputs::Tuple)
    a, rest = first(buffers), Base.tail(buffers)
    _check_principal_array(a, first(inputs))
    any(b -> Base.mightalias(a, b), inputs) && throw(ArgumentError("principal-stress outputs must not alias inputs"))
    any(b -> Base.mightalias(a, b), rest) && throw(ArgumentError("principal-stress output buffers must not alias one another"))
    return _check_principal_buffers(rest, inputs)
end

@inline function _principal_direction_sign(n::SVector)
    j = argmax(abs.(n))
    return n[j] < zero(eltype(n)) ? -n : n
end

@inline function _principal_eigenpairs(τ::NTuple{3, T}) where {T}
    α = maximum(abs, τ)
    α == zero(T) && return zeros(SVector{2, T}), SMatrix{2, 2, T}(I), true
    τ₁₁, τ₂₂, τ₁₂ = map(x -> x / α, τ)
    μ = (τ₁₁ + τ₂₂) / 2
    δ = (τ₁₁ - τ₂₂) / 2
    ρ = hypot(δ, τ₁₂)
    θ = atan(τ₁₂, δ) / 2
    s, c = sincos(θ)
    n₁ = _principal_direction_sign(SVector(c, s))
    n₂ = _principal_direction_sign(SVector(-s, c))
    return SVector((μ + ρ) * α, (μ - ρ) * α), hcat(n₁, n₂), true
end

@inline function _principal_eigenpairs(τ::NTuple{6, T}) where {T}
    α = maximum(abs, τ)
    α == zero(T) && return zeros(SVector{3, T}), SMatrix{3, 3, T}(I), true
    τ₁₁, τ₂₂, τ₃₃, τ₁₂, τ₁₃, τ₂₃ = map(x -> x / α, τ)
    A = SMatrix{3, 3, T}(τ₁₁, τ₁₂, τ₁₃, τ₁₂, τ₂₂, τ₂₃, τ₁₃, τ₂₃, τ₃₃)
    # StaticArrays specializes symmetric 3×3 eigenpairs without host LAPACK.
    E = eigen(Symmetric(A))
    λ = reverse(E.values)
    Q = hcat(ntuple(k -> _principal_direction_sign(E.vectors[:, 4 - k]), Val(3))...)
    # The library has a bounded iteration loop but no exposed convergence flag.
    # Verify its result in scaled coordinates before accepting the eigenpairs.
    R = A * Q - Q * Diagonal(λ)
    Δ = Q' * Q - SMatrix{3, 3, T}(I)
    ϵ = T(64) * eps(T)
    return λ * α, Q, maximum(abs, R) <= ϵ && maximum(abs, Δ) <= ϵ
end

@inline _principal_pressure(::Nothing, _, ::Type{T}) where {T} = zero(T)
@inline _principal_pressure(P, i, _) = P[i]

@kernel function _principal_stresses_kernel!(σ, n, τ, pressure, ::Val{D}) where {D}
    i = @index(Global, Linear)
    T = eltype(first(τ))
    τᵢ = map(a -> a[i], τ)
    P = _principal_pressure(pressure, i, T)
    λ = zeros(SVector{D, T})
    Q = SMatrix{D, D, T}(I)
    valid = all(isfinite, τᵢ) && isfinite(P)
    if valid
        λ, Q, valid = _principal_eigenpairs(τᵢ)
        λ = λ .- P
        valid = valid && all(isfinite, λ) && all(isfinite, Q)
    end
    for k in 1:D
        σ[k][i] = valid ? λ[k] : T(NaN)
        for c in 1:D
            n[k][c][i] = valid ? Q[c, k] : T(NaN)
        end
    end
end
