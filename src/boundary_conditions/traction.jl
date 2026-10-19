"""
    TractionBoundaryCondition(facets::BoundaryFacets, traction)

Prescribed traction `t` on the boundary edges `facets` of a 2-D Stokes mesh.

`traction` is an `SVector{2}` or `NTuple{2}` for a constant traction, or a
callable `x -> SVector{2}` of the node coordinates `x`. The callable receives
coordinates in the mesh's units and returns stress in the solver's units; on a
GPU backend it must be GPU-compatible (an isbits closure).

The traction enters the momentum residual as the nodal load `∫_Γ Nᵢ t ds`; see
[`FEMTools.assemble_boundary_load`](@ref).
"""
struct TractionBoundaryCondition{F <: BoundaryFacets, L} <: AbstractBoundaryCondition
    facets::F
    traction::L

    function TractionBoundaryCondition(facets::BoundaryFacets, traction)
        load = _traction_load(traction)
        return new{typeof(facets), typeof(load)}(facets, load)
    end
end

"""
    PressureBoundaryCondition(facets::BoundaryFacets, pressure)

Prescribed normal pressure `p` on the boundary edges `facets` of a 2-D Stokes
mesh, i.e. the traction `t = −p n` with `n` the outward unit normal.

`pressure` is a `Number` for a constant load, or a callable `x -> p` of the
node coordinates `x`, with the same unit and GPU conventions as
[`TractionBoundaryCondition`](@ref). `p` carries the sign of the solver
pressure: a positive load compresses the boundary and raises `P` beneath it.

To change the load between time steps, build a new condition from the same
`facets`.
"""
struct PressureBoundaryCondition{F <: BoundaryFacets, L} <: AbstractBoundaryCondition
    facets::F
    pressure::L

    function PressureBoundaryCondition(facets::BoundaryFacets, pressure)
        load = _pressure_load(pressure)
        return new{typeof(facets), typeof(load)}(facets, load)
    end
end

const _TRACTION_ERROR = "traction must be an SVector{2}, an NTuple{2}, or a callable returning an SVector{2}"
const _PRESSURE_ERROR = "pressure must be a Number or a callable returning a Number"

_traction_load(t::SVector{2}) = Returns(t)
_traction_load(t::NTuple{2, Number}) = Returns(SVector(t))
_traction_load(::Union{Number, AbstractArray, Tuple}) = throw(ArgumentError(_TRACTION_ERROR))
_traction_load(t) = t

_pressure_load(p::Number) = Returns(p)
_pressure_load(::Union{AbstractArray, Tuple}) = throw(ArgumentError(_PRESSURE_ERROR))
_pressure_load(p) = p

# Kernel-side views of the conditions: the load callable without the facet
# arrays, with constant loads in the mesh precision `T`.
struct _TractionLoad{L}
    traction::L
end
struct _PressureLoad{L}
    pressure::L
end
_kernel_load(bc::TractionBoundaryCondition, ::Type{T}) where {T} = _TractionLoad(_in_precision(bc.traction, T))
_kernel_load(bc::PressureBoundaryCondition, ::Type{T}) where {T} = _PressureLoad(_in_precision(bc.pressure, T))

_in_precision(load::Returns{<:Number}, ::Type{T}) where {T} = Returns(T(load.value))
_in_precision(load::Returns{<:SVector{2}}, ::Type{T}) where {T} = Returns(SVector{2, T}(load.value))
_in_precision(load, ::Type) = load

# Callable results are converted before any arithmetic, so the kernel stays in
# the precision of `x` and `n`.
@inline _traction_at(load::_TractionLoad, x, n) = SVector{2, eltype(n)}(load.traction(x))
@inline _traction_at(load::_PressureLoad, x, n) = -convert(eltype(n), load.pressure(x)) * n

# Lagrange basis on s ∈ [−1, 1] and its derivative, in edge-path node order:
# (a, b) for linear edges, (a, midpoint, b) for quadratic edges.
@inline _edge_shape(s, ::Val{2}) = (SVector((1 - s) / 2, (1 + s) / 2), SVector(-one(s) / 2, one(s) / 2))
@inline _edge_shape(s, ::Val{3}) =
    (SVector(s * (s - 1) / 2, 1 - s^2, s * (s + 1) / 2), SVector(s - one(s) / 2, -2s, s + one(s) / 2))

# Three-point Gauss-Legendre rule. A pressure load integrates `Nᵢ p x'(s)`,
# exact for linear `p` on a curved quadratic edge; a traction load integrates
# `Nᵢ t |x'(s)|`, exact only on straight edges.
@inline function _edge_gauss_rule(::Type{T}) where {T}
    a = sqrt(T(3) / 5)
    return ((-a, T(5) / 9), (zero(T), T(8) / 9), (a, T(5) / 9))
end

@kernel function boundary_load_kernel!(
        F, @Const(coords), @Const(el2n), @Const(elements), @Const(local_edges), load, ::Val{NV},
    ) where {NV}
    ifacet = @index(Global)
    iel = elements[ifacet]
    paths = _boundary_edge_paths_2d(Val(NV))
    path = paths[local_edges[ifacet]]
    NE = length(path)
    corners = map(first, paths)
    centroid = sum(c -> coords[el2n[c, iel]], corners) / length(corners)
    xs = SVector(map(i -> coords[el2n[i, iel]], path))
    T = eltype(eltype(coords))
    for (s, w) in _edge_gauss_rule(T)
        N, dN = _edge_shape(s, Val(NE))
        x = sum(N .* xs)
        tangent = sum(dN .* xs)
        ds = norm(tangent)
        n = SVector(tangent[2], -tangent[1]) / ds
        dot(n, centroid - x) > 0 && (n = -n)
        t = _traction_at(load, x, n)
        for a in 1:NE
            node = el2n[path[a], iel]
            for c in 1:2
                Atomix.@atomic :monotonic F[c][node] += T(N[a] * t[c] * w * ds)
            end
        end
    end
end

const _TractionLike = Union{TractionBoundaryCondition, PressureBoundaryCondition}

"""
    assemble_boundary_load!(F, bcs, mesh; workgroup=256) -> nothing

Add the nodal load `Fᵢ = ∫_Γ Nᵢ t ds` of `bcs` (one traction or pressure
condition, or a tuple of them) to the two component vectors `F` of the 2-D
`Mesh` or `MixedMesh` `mesh`, using its current coordinates.

The Stokes momentum residual `R` balances the load as `R − F`. Edges are
integrated isoparametrically with three Gauss points, so curved quadratic
edges carry their true normal and length.
"""
function assemble_boundary_load!(
        F::NTuple{2, AbstractVector}, bc::_TractionLike, mesh::Union{Mesh{2}, MixedMesh{2}};
        workgroup = 256,
    )
    bc.facets.nels == mesh.nels || throw(
        DimensionMismatch(
            "facets were built for a mesh with $(bc.facets.nels) elements, not $(mesh.nels)"
        )
    )
    all(f -> length(f) == mesh.nnodes, F) || throw(
        DimensionMismatch("load vectors must have one entry per mesh node ($(mesh.nnodes))")
    )
    backend = KA.get_backend(first(F))
    launch!(
        boundary_load_kernel!, backend, workgroup, length(bc.facets),
        F, mesh.coords, mesh.el2n, bc.facets.elements, bc.facets.local_edges,
        _kernel_load(bc, eltype(eltype(mesh.coords))), Val(size(mesh.el2n, 1)),
    )
    return nothing
end

function assemble_boundary_load!(F::NTuple{2, AbstractVector}, bcs::Tuple, mesh; workgroup = 256)
    foreach(bc -> assemble_boundary_load!(F, bc, mesh; workgroup), bcs)
    return nothing
end

"""
    assemble_boundary_load(bcs, mesh; workgroup=256) -> NTuple{2}

Allocate zeroed load vectors on the mesh backend, with the coordinate element
type, and fill them with [`FEMTools.assemble_boundary_load!`](@ref).
"""
function assemble_boundary_load(bcs, mesh::Union{Mesh{2}, MixedMesh{2}}; workgroup = 256)
    T = eltype(eltype(mesh.coords))
    backend = KA.get_backend(mesh.coords)
    F = ntuple(_ -> KA.zeros(backend, T, mesh.nnodes), Val(2))
    assemble_boundary_load!(F, bcs, mesh; workgroup)
    return F
end
