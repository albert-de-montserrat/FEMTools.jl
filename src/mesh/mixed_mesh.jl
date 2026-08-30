"""
    MixedMesh{nDim, O1, O2, ...}
    MixedMesh(element, elementP, coords, DoFs, el2n, DoFsP, el2nP)

Mixed finite-element mesh storing separate connectivities for two fields.

`O1` is the polynomial order of the primary field (e.g. velocity) and `O2` is
the order of the secondary field (e.g. pressure). Both fields share the same
element geometry (`coords`, `nels`) but have independent node numbering and
degree-of-freedom maps.

Fields:
- `coords`  : vertex coordinates (`AbstractVector{SVector{nDim, T}}`).
- `normals` : outward nodal normals; interior nodes store the zero vector.
- `nels`    : number of elements.
- `DoFs`    : primary-field degree-of-freedom indices.
- `el2n`    : primary element-to-node connectivity (`N1 × nels`).
- `nnodes`  : number of primary-field nodes.
- `DoFsP`   : secondary-field degree-of-freedom indices.
- `el2nP`   : secondary element-to-node connectivity (`N2 × nels`).
- `nnodesP` : number of secondary-field nodes.
"""
struct MixedMesh{nDim, O1, O2, T1, T2, T3, T4, T5, T6} <: AbstractMesh
    coords::T1  # vertex coordinates
    normals::T6 # outward nodal normals
    nels::Int   # number of elements
    # Field 1: Velocity
    DoFs::T2    # degrees of freedom
    el2n::T3    # element-to-node connectivity
    nnodes::Int # number of nodes
    # Field 2: Pressure
    DoFsP::T4    # degrees of freedom
    el2nP::T5    # element-to-node connectivity
    nnodesP::Int # number of nodes

    function MixedMesh{nDim, O1, O2, T1, T2, T3, T4, T5, T6}(
        coords::T1,
        normals::T6,
        DoFs::T2,
        el2n::T3,
        nnodes::Int,
        nels::Int,
        DoFsP::T4,
        el2nP::T5,
        nnodesP::Int,
    ) where {nDim, O1, O2, T1, T2, T3, T4, T5, T6}
        length(normals) == nnodes || throw(ArgumentError("normal vector has the wrong number of nodes"))
        size(el2n, 2) == nels || throw(ArgumentError("velocity connectivity has the wrong number of elements"))
        size(el2nP, 2) == nels || throw(ArgumentError("pressure connectivity has the wrong number of elements"))
        return new{nDim, O1, O2, T1, T2, T3, T4, T5, T6}(
            coords, normals, nels,
            DoFs, el2n, nnodes,
            DoFsP, el2nP, nnodesP,
        )
    end
end
function Base.show(io::IO, mesh::MixedMesh{nDim, O1, O2}) where {nDim, O1, O2}
    print(io, "MixedMesh{", nDim, ", ", O1, ", ", O2, "}(nnodes=", mesh.nnodes,
          ", nnodesP=", mesh.nnodesP, ", nels=", mesh.nels, ")")
end

function _compute_node_normals(coords::AbstractVector{<:SVector{2, FP}}, el2n::AbstractMatrix{<:Integer}) where FP
    normals = fill(zero(SVector{2, FP}), length(coords))
    edge_paths = _boundary_edge_paths_2d(size(el2n, 1))
    corner_lids = unique(first.(edge_paths))

    edge_count = Dict{Tuple{Int32, Int32}, Int}()
    for iel in axes(el2n, 2), path in edge_paths
        a = Int32(el2n[first(path), iel])
        b = Int32(el2n[last(path), iel])
        key = minmax(a, b)
        edge_count[key] = get(edge_count, key, 0) + 1
    end

    for iel in axes(el2n, 2)
        centroid = sum(coords[Int(el2n[lid, iel])] for lid in corner_lids) / length(corner_lids)
        for path in edge_paths
            a = Int32(el2n[first(path), iel])
            b = Int32(el2n[last(path), iel])
            get(edge_count, minmax(a, b), 0) == 1 || continue

            for i in 1:(length(path) - 1)
                ia = Int(el2n[path[i], iel])
                ib = Int(el2n[path[i + 1], iel])
                tangent = coords[ib] - coords[ia]
                normal = SVector{2, FP}(tangent[2], -tangent[1])
                midpoint = (coords[ia] + coords[ib]) / 2
                dot(normal, centroid - midpoint) > 0 && (normal = -normal)
                normals[ia] += normal
                normals[ib] += normal
            end
        end
    end

    return [iszero(norm(n)) ? n : n / norm(n) for n in normals]
end

# Nodal normals are stored for boundary post-processing and are not read by any
# solver path, so three-dimensional meshes carry the zero vector everywhere.
_compute_node_normals(coords::AbstractVector{<:SVector{3, FP}}, ::AbstractMatrix{<:Integer}) where FP =
    fill(zero(SVector{3, FP}), length(coords))


function MixedMesh(
    element::ReferenceElement,
    elementP::ReferenceElement,
    coords::AbstractVector{<:SVector{nDim}},
    DoFs,
    el2n,
    DoFsP,
    el2nP,
) where {nDim}
    coords_cpu = Array(coords)
    el2n_cpu = Array(el2n)
    normals_cpu = _compute_node_normals(coords_cpu, el2n_cpu)
    normals = typeof(coords)(normals_cpu)
    return MixedMesh{
        nDim,
        order(element),
        order(elementP),
        typeof(coords),
        typeof(DoFs),
        typeof(el2n),
        typeof(DoFsP),
        typeof(el2nP),
        typeof(normals),
    }(
        coords,
        normals,
        DoFs,
        el2n,
        length(coords),
        size(el2n, 2),
        DoFsP,
        el2nP,
        Int(maximum(DoFsP)),
    )
end

"""
    MixedMesh(mesh_v, element_P) -> MixedMesh

Construct a mixed velocity–pressure mesh from a pre-built velocity mesh and a
pressure reference element.

The pressure field is treated as **discontinuous linear** (P1-disc): each
triangle gets its own three pressure DoFs, built internally via
`generate_discontinuous_linear_mesh`. `mesh_v` supplies coordinates, velocity
DoF indices, and velocity connectivity; `element_P` supplies the pressure
polynomial order stored in the `MixedMesh` type parameter.

The pressure topology is built from CPU copies, then moved back to the same
array backend as `mesh_v`.
"""
function MixedMesh(mesh_v::Mesh{nDim, O1}, element_P::ReferenceElement) where {nDim, O1}
    coords_cpu = Array(mesh_v.coords)
    el2n_cpu   = Array(mesh_v.el2n)
    el2nP_cpu, DoFsP_cpu, _ = generate_discontinuous_linear_mesh(coords_cpu, el2n_cpu)
    normals_cpu = _compute_node_normals(coords_cpu, el2n_cpu)
    normals = typeof(mesh_v.coords)(normals_cpu)
    DoFsP = typeof(mesh_v.el2n)(DoFsP_cpu)
    el2nP = typeof(mesh_v.el2n)(el2nP_cpu)
    return MixedMesh{
        nDim,
        O1,
        order(element_P),
        typeof(mesh_v.coords),
        typeof(mesh_v.DoFs),
        typeof(mesh_v.el2n),
        typeof(DoFsP),
        typeof(el2nP),
        typeof(normals),
    }(
        mesh_v.coords,
        normals,
        mesh_v.DoFs,
        mesh_v.el2n,
        length(coords_cpu),
        size(el2n_cpu, 2),
        DoFsP,
        el2nP,
        Int(maximum(DoFsP_cpu)),
    )
end

"""
    MixedMeshCache(backend, workgroup, mesh, element_v, element_P)

Precompute geometry for both fields of a 2D mixed mesh at the velocity
integration points.

Both geometry arrays are allocated on `backend`. Coordinates and connectivity
are converted with [`TA`](@ref) before the kernels are launched, so a cache
constructed with `CUDABackend()` contains device-resident `CuArray`s. Use the
same backend for the mesh, cache, solver state, phase arrays, and boundary
arrays; mixing host and device inputs in an assembly kernel is unsupported.

Fields:
- `geo_v`: primary-field shape-function gradients and weighted volumes.
- `geo_P`: secondary-field shape-function gradients and weighted volumes.
- `element_v`, `element_P`: reference elements used to build the cache.
"""
struct MixedMeshCache{GV, GP, EV, EP}
    geo_v::GV
    geo_P::GP
    element_v::EV
    element_P::EP
end

MixedMeshCache(geo_v, geo_P) = MixedMeshCache(geo_v, geo_P, nothing, nothing)

function MixedMeshCache(
    backend,
    workgroup,
    mesh::MixedMesh{2},
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
) where {NV, NP, FP, TV <: AbstractElement{2, NV, FP}, TP <: AbstractElement{2, NP, FP}}
    ip_v = element_v.integration_points
    NQ_v = length(ip_v.ω)

    ξq_v    = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)
    ∂N∂ξq_P = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_v[q]), NQ_v)

    GeoV = NTuple{NQ_v, Tuple{SMatrix{NV, 2, FP, 2NV}, FP}}
    GeoP = NTuple{NQ_v, Tuple{SMatrix{NP, 2, FP, 2NP}, FP}}
    geo_v = KA.allocate(backend, GeoV, mesh.nels)
    geo_P = KA.allocate(backend, GeoP, mesh.nels)

    TDev   = TA(backend)
    coords = TDev(mesh.coords)
    el2n   = TDev(mesh.el2n)
    el2nP  = TDev(mesh.el2nP)

    precompute_geometry_kernel!(backend, workgroup)(
        geo_v, coords, el2n, ∂N∂ξq_v, ip_v.ω, Val(NV);
        ndrange = mesh.nels,
    )
    precompute_geometry_kernel!(backend, workgroup)(
        geo_P, coords, el2nP, ∂N∂ξq_P, ip_v.ω, Val(NP);
        ndrange = mesh.nels,
    )
    KA.synchronize(backend)

    return MixedMeshCache(geo_v, geo_P, element_v, element_P)
end

"""
    MixedMeshCache(backend, workgroup, mesh::MixedMesh{3}, element_v, element_P)

Precompute geometry for a three-dimensional mixed mesh at the velocity
integration points.

The discontinuous pressure basis is defined on the velocity element's reference
cell rather than on a sub-element of its own, so both geometry fields hold the
velocity element's data. Only the quadrature weight of `geo_P` is read by the
pressure residual and pressure scaling.
"""
function MixedMeshCache(
    backend,
    workgroup,
    mesh::MixedMesh{3},
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
) where {NV, NP, FP, TV <: AbstractElement{3, NV, FP}, TP <: AbstractElement{3, NP, FP}}
    ip_v = element_v.integration_points
    NQ_v = length(ip_v.ω)

    ξq_v    = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q], ip_v.ζ[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)

    GeoV  = NTuple{NQ_v, Tuple{SMatrix{NV, 3, FP, 3NV}, FP}}
    geo_v = KA.allocate(backend, GeoV, mesh.nels)

    TDev = TA(backend)
    precompute_geometry_kernel!(backend, workgroup)(
        geo_v, TDev(mesh.coords), TDev(mesh.el2n), ∂N∂ξq_v, ip_v.ω, Val(NV);
        ndrange = mesh.nels,
    )
    KA.synchronize(backend)

    return MixedMeshCache(geo_v, geo_v, element_v, element_P)
end

# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

# Local velocity nodes carrying the discontinuous pressure DoFs. Each row is the
# node sitting at a reference point where the pressure basis is nodal, so that
# gathering a continuous field at these nodes reproduces it exactly under the
# pressure shape functions.
#
# Triangles: the P1 basis of `LinearElement{2, 3}` is nodal at the corners.
# T10/T11: the P1 basis is nodal at the four tetrahedron vertices.
# Hex27: the four-mode basis (1, ξ, η, ζ) is nodal at the cell center and the
# +x, +y and +z face centers.
_pressure_node_rows(::Val{2}, nlocal::Integer) = nlocal >= 3 ? (1, 2, 3) :
    throw(ArgumentError("triangle connectivity needs at least 3 local nodes"))
function _pressure_node_rows(::Val{3}, nlocal::Integer)
    nlocal in (10, 11) && return (1, 2, 3, 4)
    nlocal == 27 && return (27, 23, 24, 26)
    throw(ArgumentError("three-dimensional discontinuous pressure needs T10, T11, or Hex27 connectivity"))
end

"""
    generate_discontinuous_linear_mesh(coords, el2n) -> (p_el2n, p_el2dof, p_dof_coords)

Build the topology and element-to-DoF map for discontinuous linear pressure
elements.

`el2n` may be T3, T6 or T7 triangle connectivity, T10/T11 tetrahedral
connectivity, or Hex27 connectivity. The returned `p_el2n` selects the velocity
nodes at which the pressure basis is nodal: triangle/tetrahedron corners, or the
cell center and +x, +y, +z face centers of a hexahedron. Each element receives
its own `NP` pressure DoFs,
`p_el2dof[:, iel] == NP(iel - 1) .+ (1:NP)`.

Returns `(p_el2n, p_el2dof, p_dof_coords)`, where `p_dof_coords` duplicates the
selected node coordinates per element so a discontinuous nodal pressure field
can be plotted or initialized directly on pressure DoFs.
"""
function generate_discontinuous_linear_mesh(
        coords::AbstractVector{<:SVector{nDim}}, el2n::AbstractMatrix{<:Integer},
    ) where {nDim}
    rows = _pressure_node_rows(Val(nDim), size(el2n, 1))
    NP   = length(rows)

    nels      = size(el2n, 2)
    p_el2n    = Matrix{Int32}(el2n[collect(rows), :])
    p_el2dof  = Matrix{Int32}(undef, NP, nels)
    p_dof_coords = Vector{eltype(coords)}(undef, NP * nels)

    for iel in 1:nels
        base = NP * (iel - 1)
        for a in 1:NP
            dof = base + a
            p_el2dof[a, iel] = Int32(dof)
            p_dof_coords[dof] = coords[p_el2n[a, iel]]
        end
    end

    return p_el2n, p_el2dof, p_dof_coords
end
