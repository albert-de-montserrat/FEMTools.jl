"""
    AbstractVectorField{T}

Supertype for vector fields stored component-wise (struct of arrays). `T` is the
storage type of a single component: a scalar for a vector sampled at one point,
or an array for a whole field of vectors.
"""
abstract type AbstractVectorField{T} end

"""
    VectorField2D{T} <: AbstractVectorField{T}
    VectorField2D(x, y)
    VectorField2D(backend, [T = Float64], nnods)

Two-dimensional vector field storing the components `x` and `y`.

Each component may be a scalar or an array of sample values; both share the
storage type `T`. The `backend` form allocates two zeroed `nnods`-element arrays
on the given `KernelAbstractions` backend.

`A[i]` returns the components at sample `i` as an `SVector`; `A[q, iel]` does the
same for components stored as `nq × nels` matrices of integration-point values.
"""
struct VectorField2D{T} <: AbstractVectorField{T}
    x::T
    y::T

    VectorField2D{T}(x, y) where {T} = new{T}(x, y)
end

"""
    VectorField3D{T} <: AbstractVectorField{T}
    VectorField3D(x, y, z)
    VectorField3D(backend, [T = Float64], nnods)

Three-dimensional vector field storing the components `x`, `y` and `z`.
Component storage and indexing follow [`VectorField2D`](@ref).
"""
struct VectorField3D{T} <: AbstractVectorField{T}
    x::T
    y::T
    z::T

    VectorField3D{T}(x, y, z) where {T} = new{T}(x, y, z)
end

VectorField2D(x::T, y::T) where {T} = VectorField2D{T}(x, y)
VectorField3D(x::T, y::T, z::T) where {T} = VectorField3D{T}(x, y, z)

# `backend` is annotated so these never intersect the component-wise
# constructors above, which share their arity.
function VectorField2D(backend::Backend, ::Type{T}, nnods::Integer) where {T}
    return VectorField2D(ntuple(i -> KernelAbstractions.zeros(backend, T, nnods), Val(2))...)
end
function VectorField3D(backend::Backend, ::Type{T}, nnods::Integer) where {T}
    return VectorField3D(ntuple(i -> KernelAbstractions.zeros(backend, T, nnods), Val(3))...)
end

VectorField2D(backend::Backend, nnods::Integer) = VectorField2D(backend, Float64, nnods)
VectorField3D(backend::Backend, nnods::Integer) = VectorField3D(backend, Float64, nnods)

"""
    VectorField(x, y) -> VectorField2D
    VectorField(x, y, z) -> VectorField3D

Construct a vector field from its components. Any other number of components is
an `ArgumentError`.
"""
@inline VectorField(x, y) = VectorField2D(x, y)
@inline VectorField(x, y, z) = VectorField3D(x, y, z)
@inline VectorField(::Vararg{Any, N}) where {N} = throw(
    ArgumentError(
        "VectorField only supports 2D (2 components) and 3D (3 components) vectors, but got $N components."
    )
)

Base.eltype(::Type{<:AbstractVectorField{T}}) where {T} = eltype(T)
Base.size(A::AbstractVectorField) = size(A.x)
Base.length(A::AbstractVectorField) = length(A.x)
Base.axes(A::AbstractVectorField) = axes(A.x)
Base.eachindex(A::AbstractVectorField) = eachindex(A.x)
Base.firstindex(A::AbstractVectorField) = firstindex(A.x)
Base.lastindex(A::AbstractVectorField) = lastindex(A.x)

@inline Base.getindex(A::VectorField2D, i::Vararg{Integer}) = SA[A.x[i...], A.y[i...]]
@inline Base.getindex(A::VectorField3D, i::Vararg{Integer}) = SA[A.x[i...], A.y[i...], A.z[i...]]

@inline function Base.setindex!(A::VectorField2D, v, i::Vararg{Integer})
    A.x[i...], A.y[i...] = v[1], v[2]
    return v
end
@inline function Base.setindex!(A::VectorField3D, v, i::Vararg{Integer})
    A.x[i...], A.y[i...], A.z[i...] = v[1], v[2], v[3]
    return v
end

# Unpack the components so the struct plugs straight into assemblers that
# dispatch on `NTuple{2}` / `NTuple{3}`.
@inline Base.Tuple(A::VectorField2D) = (A.x, A.y)
@inline Base.Tuple(A::VectorField3D) = (A.x, A.y, A.z)

# In-place component-wise copy, e.g. `copyto!(v_old, v)` to advance a history.
function Base.copyto!(dst::VectorField2D, src::VectorField2D)
    copyto!(dst.x, src.x)
    copyto!(dst.y, src.y)
    return dst
end
function Base.copyto!(dst::VectorField3D, src::VectorField3D)
    copyto!(dst.x, src.x)
    copyto!(dst.y, src.y)
    copyto!(dst.z, src.z)
    return dst
end

function Base.show(io::IO, A::VectorField2D)
    return print(
        io, "VectorField2D(x=", _show_component(A.x),
        ", y=", _show_component(A.y), ")"
    )
end

function Base.show(io::IO, A::VectorField3D)
    return print(
        io, "VectorField3D(x=", _show_component(A.x),
        ", y=", _show_component(A.y),
        ", z=", _show_component(A.z), ")"
    )
end
