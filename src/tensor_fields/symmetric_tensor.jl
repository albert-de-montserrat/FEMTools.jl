"""
    AbstractTensor{T}

Supertype for the tensor containers of FEMTools. `T` is the storage type of a
single component: a scalar for a tensor sampled at one point, or an array for a
whole field of tensors.
"""
abstract type AbstractTensor{T} end

"""
    AbstractSymmetricTensor{T} <: AbstractTensor{T}

Supertype for symmetric second-order tensors stored component-wise (struct of
arrays). Concrete subtypes hold the independent components together with an
invariant slot `II`.
"""
abstract type AbstractSymmetricTensor{T} <: AbstractTensor{T} end

"""
    AbstractVoigtTensor{T} <: AbstractTensor{T}

Supertype for a single symmetric second-order tensor whose independent
components are scalars of type `T`, held in Voigt order.
"""
abstract type AbstractVoigtTensor{T} <: AbstractTensor{T} end

"""
    SymmetricTensor2D{T} <: AbstractSymmetricTensor{T}
    SymmetricTensor2D(xx, yy, xy, II)
    SymmetricTensor2D(backend, [T = Float64], nnods)

Two-dimensional symmetric second-order tensor storing the independent
components `xx`, `yy`, `xy` and an invariant slot `II`, in Voigt order.

Each component may be a scalar or an array of sample values; all four share the
storage type `T`. The `backend` form allocates four zeroed `nnods`-element
arrays on the given `KernelAbstractions` backend. Use [`SymmetricTensor`](@ref)
to build one from its independent components alone, leaving `II` zeroed.

`A[i]` returns the independent components at sample `i` as an `SVector`;
`A[q, iel]` does the same for components stored as `nq × nels` matrices of
integration-point values.
"""
struct SymmetricTensor2D{T} <: AbstractSymmetricTensor{T}
    xx::T
    yy::T
    xy::T
    II::T

    SymmetricTensor2D{T}(xx, yy, xy, II) where {T} = new{T}(xx, yy, xy, II)
end

"""
    SymmetricTensor3D{T} <: AbstractSymmetricTensor{T}
    SymmetricTensor3D(xx, yy, zz, yz, xz, xy, II)
    SymmetricTensor3D(backend, [T = Float64], nnods)

Three-dimensional symmetric second-order tensor storing the independent
components `xx`, `yy`, `zz`, `yz`, `xz`, `xy` and an invariant slot `II`, in
Voigt order.

Component storage and indexing follow [`SymmetricTensor2D`](@ref).
"""
struct SymmetricTensor3D{T} <: AbstractSymmetricTensor{T}
    xx::T
    yy::T
    zz::T
    yz::T
    xz::T
    xy::T
    II::T

    SymmetricTensor3D{T}(xx, yy, zz, yz, xz, xy, II) where {T} =
        new{T}(xx, yy, zz, yz, xz, xy, II)
end

SymmetricTensor2D(xx::T, yy::T, xy::T, II::T) where {T} = SymmetricTensor2D{T}(xx, yy, xy, II)
SymmetricTensor3D(xx::T, yy::T, zz::T, yz::T, xz::T, xy::T, II::T) where {T} =
    SymmetricTensor3D{T}(xx, yy, zz, yz, xz, xy, II)

function SymmetricTensor2D(backend::Backend, ::Type{T}, nnods::Integer) where {T}
    return SymmetricTensor2D(ntuple(i -> KernelAbstractions.zeros(backend, T, nnods), Val(4))...)
end
function SymmetricTensor3D(backend::Backend, ::Type{T}, nnods::Integer) where {T}
    return SymmetricTensor3D(ntuple(i -> KernelAbstractions.zeros(backend, T, nnods), Val(7))...)
end

SymmetricTensor2D(backend::Backend, nnods::Integer) = SymmetricTensor2D(backend, Float64, nnods)
SymmetricTensor3D(backend::Backend, nnods::Integer) = SymmetricTensor3D(backend, Float64, nnods)

"""
    SymmetricTensor(xx, yy, xy) -> SymmetricTensor2D
    SymmetricTensor(xx, yy, zz, yz, xz, xy) -> SymmetricTensor3D

Construct a symmetric tensor from its independent components in Voigt order,
initializing the invariant slot `II` to zero.
"""
@inline SymmetricTensor(xx, yy, xy) = SymmetricTensor2D(xx, yy, xy, zero(xx))
@inline SymmetricTensor(xx, yy, zz, yz, xz, xy) =
    SymmetricTensor3D(xx, yy, zz, yz, xz, xy, zero(xx))

Base.eltype(::Type{<:AbstractSymmetricTensor{T}}) where {T} = eltype(T)
Base.size(A::AbstractSymmetricTensor) = size(A.xx)
Base.length(A::AbstractSymmetricTensor) = length(A.xx)
Base.axes(A::AbstractSymmetricTensor) = axes(A.xx)
Base.eachindex(A::AbstractSymmetricTensor) = eachindex(A.xx)
Base.firstindex(A::AbstractSymmetricTensor) = firstindex(A.xx)
Base.lastindex(A::AbstractSymmetricTensor) = lastindex(A.xx)

@inline Base.getindex(A::SymmetricTensor2D, i::Vararg{Integer}) =
    SA[A.xx[i...], A.yy[i...], A.xy[i...]]
@inline Base.getindex(A::SymmetricTensor3D, i::Vararg{Integer}) =
    SA[A.xx[i...], A.yy[i...], A.zz[i...], A.yz[i...], A.xz[i...], A.xy[i...]]

# `II` is derived from the components and is left untouched by a component write.
@inline function Base.setindex!(A::SymmetricTensor2D, x, i::Vararg{Integer})
    v = SVector(x)
    A.xx[i...], A.yy[i...], A.xy[i...] = v[1], v[2], v[3]
    return x
end
@inline function Base.setindex!(A::SymmetricTensor3D, x, i::Vararg{Integer})
    v = SVector(x)
    A.xx[i...], A.yy[i...], A.zz[i...], A.yz[i...], A.xz[i...], A.xy[i...] =
        v[1], v[2], v[3], v[4], v[5], v[6]
    return x
end

# Unpack the independent components (dropping the derived `II`) so the struct
# plugs straight into assemblers that dispatch on `NTuple{3}` / `NTuple{6}`.
@inline Base.Tuple(A::SymmetricTensor2D) = (A.xx, A.yy, A.xy)
@inline Base.Tuple(A::SymmetricTensor3D) = (A.xx, A.yy, A.zz, A.yz, A.xz, A.xy)

# In-place component-wise copy, e.g. `copyto!(τ_old, τ)` to advance stress history.
function Base.copyto!(dst::SymmetricTensor2D, src::SymmetricTensor2D)
    copyto!(dst.xx, src.xx)
    copyto!(dst.yy, src.yy)
    copyto!(dst.xy, src.xy)
    copyto!(dst.II, src.II)
    return dst
end
function Base.copyto!(dst::SymmetricTensor3D, src::SymmetricTensor3D)
    copyto!(dst.xx, src.xx)
    copyto!(dst.yy, src.yy)
    copyto!(dst.zz, src.zz)
    copyto!(dst.yz, src.yz)
    copyto!(dst.xz, src.xz)
    copyto!(dst.xy, src.xy)
    copyto!(dst.II, src.II)
    return dst
end

_show_component(x) = x isa Number ? repr(x) : summary(x)

function Base.show(io::IO, A::SymmetricTensor2D)
    return print(
        io, "SymmetricTensor2D(xx=", _show_component(A.xx),
        ", yy=", _show_component(A.yy),
        ", xy=", _show_component(A.xy),
        ", II=", _show_component(A.II), ")"
    )
end

function Base.show(io::IO, A::SymmetricTensor3D)
    return print(
        io, "SymmetricTensor3D(xx=", _show_component(A.xx),
        ", yy=", _show_component(A.yy),
        ", zz=", _show_component(A.zz),
        ", yz=", _show_component(A.yz),
        ", xz=", _show_component(A.xz),
        ", xy=", _show_component(A.xy),
        ", II=", _show_component(A.II), ")"
    )
end

"""
    VoigtTensor2D{T} <: AbstractVoigtTensor{T}
    VoigtTensor2D(xx, yy, xy)

A single two-dimensional symmetric second-order tensor with scalar components
`xx`, `yy`, `xy` in Voigt order. Mixed argument types are promoted to a common
`T`; `VoigtTensor2D{T}(xx, yy, xy)` converts to a chosen `T` instead.
"""
struct VoigtTensor2D{T} <: AbstractVoigtTensor{T}
    xx::T
    yy::T
    xy::T

    VoigtTensor2D{T}(xx, yy, xy) where {T} = new{T}(xx, yy, xy)
end

"""
    VoigtTensor3D{T} <: AbstractVoigtTensor{T}
    VoigtTensor3D(xx, yy, zz, yz, xz, xy)

A single three-dimensional symmetric second-order tensor with scalar components
`xx`, `yy`, `zz`, `yz`, `xz`, `xy` in Voigt order. Promotion follows
[`VoigtTensor2D`](@ref).
"""
struct VoigtTensor3D{T} <: AbstractVoigtTensor{T}
    xx::T
    yy::T
    zz::T
    yz::T
    xz::T
    xy::T

    VoigtTensor3D{T}(xx, yy, zz, yz, xz, xy) where {T} = new{T}(xx, yy, zz, yz, xz, xy)
end

VoigtTensor2D(args::Vararg{Any, 3}) = VoigtTensor2D{promote_type(map(typeof, args)...)}(args...)
VoigtTensor3D(args::Vararg{Any, 6}) = VoigtTensor3D{promote_type(map(typeof, args)...)}(args...)

@inline StaticArrays.SVector(A::VoigtTensor3D) = SA[A.xx, A.yy, A.zz, A.yz, A.xz, A.xy]
@inline StaticArrays.SVector(A::VoigtTensor2D) = SA[A.xx, A.yy, A.xy]
@inline StaticArrays.MVector(A::VoigtTensor3D) = @MVector [A.xx, A.yy, A.zz, A.yz, A.xz, A.xy]
@inline StaticArrays.MVector(A::VoigtTensor2D) = @MVector [A.xx, A.yy, A.xy]

@inline StaticArrays.SMatrix(A::VoigtTensor3D) = @SMatrix [A.xx A.xy A.xz; A.xy A.yy A.yz; A.xz A.yz A.zz]
@inline StaticArrays.SMatrix(A::VoigtTensor2D) = @SMatrix [A.xx A.xy; A.xy A.yy]
@inline StaticArrays.MMatrix(A::VoigtTensor3D) = @MMatrix [A.xx A.xy A.xz; A.xy A.yy A.yz; A.xz A.yz A.zz]
@inline StaticArrays.MMatrix(A::VoigtTensor2D) = @MMatrix [A.xx A.xy; A.xy A.yy]

"""
    VoigtTensor(xx, yy, xy) -> VoigtTensor2D
    VoigtTensor(xx, yy, zz, yz, xz, xy) -> VoigtTensor3D
    VoigtTensor(x::SVector)
    VoigtTensor(A::AbstractSymmetricTensor, i)

Build a [`VoigtTensor2D`](@ref) or [`VoigtTensor3D`](@ref) from three or six
components in Voigt order, from a static vector of that length, or from sample
`i` of a symmetric tensor field. Components are promoted to a common type. Any
other number of components is an `ArgumentError`.
"""
@inline VoigtTensor(::Vararg{Real, N}) where {N} = throw(
    ArgumentError(
        "VoigtTensor only supports 2D (3 components) and 3D (6 components) tensors, but got $N components."
    )
)
@inline VoigtTensor(args::Vararg{Real, 3}) = VoigtTensor2D(args...)
@inline VoigtTensor(args::Vararg{Real, 6}) = VoigtTensor3D(args...)
@inline VoigtTensor(A::AbstractSymmetricTensor, i::Integer) = VoigtTensor(A[i])
@inline VoigtTensor(x::SVector) = VoigtTensor(x...)

for op in (:+, :-)
    @eval begin
        Base.$op(A::AbstractVoigtTensor, B::AbstractVoigtTensor) = VoigtTensor($op(SVector(A), SVector(B)))
    end
end

@inline Base.:(-)(A::AbstractVoigtTensor) = VoigtTensor(-SVector(A))
@inline Base.:(*)(alpha::Number, A::AbstractVoigtTensor) = VoigtTensor(alpha * SVector(A))
@inline Base.:(*)(A::AbstractVoigtTensor, alpha::Number) = alpha * A
@inline Base.:(/)(A::AbstractVoigtTensor, alpha::Number) = VoigtTensor(SVector(A) / alpha)

# The result stays immutable: an `MVector` result is heap-allocated even where it
# does not escape, which rules it out inside GPU kernels.
@inline Base.:(*)(A::AbstractVoigtTensor, x::StaticVector) = SMatrix(A) * SVector(x)
@inline Base.:(*)(A::AbstractVoigtTensor, B::AbstractVoigtTensor) = SMatrix(A) * SMatrix(B)

@inline Base.:(\)(A::AbstractVoigtTensor, x::StaticVector) = SMatrix(A) \ SVector(x)
