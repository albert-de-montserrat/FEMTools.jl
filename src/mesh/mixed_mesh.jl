"""
    MixedMesh{nDim, O1, O2, ...}
    MixedMesh(element, elementP, coords, DoFs, el2n, DoFsP, el2nP; workgroup=256)

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
- `geometry`: [`MixedMeshCache`](@ref) holding both fields' precomputed geometry
  and reference elements, or `nothing` when the mesh was built without a
  primary-field reference element. After moving `coords`, refresh it with
  [`update_geometry!`](@ref).

The geometry is computed on the backend of `coords`.
"""
struct MixedMesh{nDim, O1, O2, T1, T2, T3, T4, T5, T6, G} <: AbstractMesh
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
    geometry::G  # MixedMeshCache or nothing

    function MixedMesh{nDim, O1, O2, T1, T2, T3, T4, T5, T6, G}(
        coords::T1,
        normals::T6,
        DoFs::T2,
        el2n::T3,
        nnodes::Int,
        nels::Int,
        DoFsP::T4,
        el2nP::T5,
        nnodesP::Int,
        geometry,
    ) where {nDim, O1, O2, T1, T2, T3, T4, T5, T6, G}
        length(normals) == nnodes || throw(ArgumentError("normal vector has the wrong number of nodes"))
        size(el2n, 2) == nels || throw(ArgumentError("velocity connectivity has the wrong number of elements"))
        size(el2nP, 2) == nels || throw(ArgumentError("pressure connectivity has the wrong number of elements"))
        return new{nDim, O1, O2, T1, T2, T3, T4, T5, T6, G}(
            coords, normals, nels,
            DoFs, el2n, nnodes,
            DoFsP, el2nP, nnodesP,
            geometry,
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


function MixedMesh(
    element::ReferenceElement,
    elementP::ReferenceElement,
    coords::AbstractVector{<:SVector{nDim}},
    DoFs,
    el2n,
    DoFsP,
    el2nP;
    workgroup = 256,
    geometry_precision = eltype(eltype(coords)),
) where {nDim}
    coords_cpu = Array(coords)
    el2n_cpu = Array(el2n)
    normals_cpu = _compute_node_normals(coords_cpu, el2n_cpu)
    normals = typeof(coords)(normals_cpu)
    mesh = MixedMesh{
        nDim,
        order(element),
        order(elementP),
        typeof(coords),
        typeof(DoFs),
        typeof(el2n),
        typeof(DoFsP),
        typeof(el2nP),
        typeof(normals),
        Nothing,
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
        nothing,
    )
    return _with_geometry(mesh, element, elementP, workgroup, geometry_precision)
end

"""
    MixedMesh(mesh_v, element_P; workgroup=256) -> MixedMesh

Construct a mixed velocity–pressure mesh from a pre-built velocity mesh and a
pressure reference element.

The pressure field is treated as **discontinuous linear** (P1-disc): each
triangle gets its own three pressure DoFs, built internally via
`generate_discontinuous_linear_mesh`. `mesh_v` supplies coordinates, velocity
DoF indices, and velocity connectivity; `element_P` supplies the pressure
polynomial order stored in the `MixedMesh` type parameter.

The pressure topology is built from CPU copies, then moved back to the same
array backend as `mesh_v`. When `mesh_v` stores its reference element, the
geometry of both fields is precomputed into `geometry`; otherwise `geometry`
is `nothing` and the high-level Stokes solvers reject the mesh.
"""
function MixedMesh(
    mesh_v::Mesh{nDim, O1},
    element_P::ReferenceElement;
    workgroup = 256,
    geometry_precision = eltype(eltype(mesh_v.coords)),
) where {nDim, O1}
    coords_cpu = Array(mesh_v.coords)
    el2n_cpu   = Array(mesh_v.el2n)
    el2nP_cpu, DoFsP_cpu, _ = generate_discontinuous_linear_mesh(coords_cpu, el2n_cpu)
    normals_cpu = _compute_node_normals(coords_cpu, el2n_cpu)
    normals = typeof(mesh_v.coords)(normals_cpu)
    DoFsP = typeof(mesh_v.el2n)(DoFsP_cpu)
    el2nP = typeof(mesh_v.el2n)(el2nP_cpu)
    mesh = MixedMesh{
        nDim,
        O1,
        order(element_P),
        typeof(mesh_v.coords),
        typeof(mesh_v.DoFs),
        typeof(mesh_v.el2n),
        typeof(DoFsP),
        typeof(el2nP),
        typeof(normals),
        Nothing,
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
        nothing,
    )
    return _with_geometry(mesh, mesh_v.element, element_P, workgroup, geometry_precision)
end

_with_geometry(mesh::MixedMesh, ::Nothing, element_P, workgroup, geometry_precision) = mesh

function _with_geometry(mesh::MixedMesh{nDim, O1, O2}, element_v::ReferenceElement, element_P, workgroup, geometry_precision) where {nDim, O1, O2}
    geometry = MixedMeshCache(
        KA.get_backend(mesh.coords), workgroup, mesh, element_v, element_P; geometry_precision,
    )
    return MixedMesh{nDim, O1, O2,
                     typeof(mesh.coords), typeof(mesh.DoFs), typeof(mesh.el2n),
                     typeof(mesh.DoFsP), typeof(mesh.el2nP), typeof(mesh.normals),
                     typeof(geometry)}(
        mesh.coords, mesh.normals, mesh.DoFs, mesh.el2n, mesh.nnodes, mesh.nels,
        mesh.DoFsP, mesh.el2nP, mesh.nnodesP, geometry,
    )
end

function _mesh_geometry(mesh::MixedMesh)
    geometry = mesh.geometry
    isnothing(geometry) && throw(ArgumentError(
        "mixed mesh has no geometry; build it from a velocity mesh that stores its reference element, e.g. MixedMesh(Mesh(backend, coords, el2n, element_v), element_P)"))
    isnothing(geometry.element_v) && throw(ArgumentError(
        "mixed mesh geometry has no reference elements; construct it with MixedMeshCache(backend, workgroup, mesh, element_v, element_P)"))
    return geometry
end

"""
    MixedMeshCache(backend, workgroup, mesh, element_v, element_P; geometry_precision=FP)

Precompute geometry for both fields of a 2D mixed mesh at the velocity
integration points. A `MixedMesh` built with its reference elements already
stores one as `mesh.geometry`.

Both geometry arrays are allocated on `backend`. Coordinates and connectivity
are converted with [`TA`](@ref) before the kernels are launched, so a cache
constructed with `CUDABackend()` contains device-resident `CuArray`s. Use the
same backend for the mesh, cache, solver state, phase arrays, and boundary
arrays; mixing host and device inputs in an assembly kernel is unsupported.

Fields:
- `geo_v`: primary-field inverse Jacobians and weighted volumes, as
  [`QuadraturePointGeometry`](@ref). Combine with
  [`shape_function_gradients`](@ref) of `element_v` to obtain physical gradients.
- `geo_P`: weighted volumes of the secondary field at the primary field's
  quadrature points. Only the volumes are stored: the pressure residual and the
  pressure scaling are the only consumers and neither uses secondary-field
  gradients.
- `element_v`, `element_P`: reference elements used to build the cache.

`geometry_precision` sets the element type of both arrays and defaults to the
reference elements'. `Float32` halves them under a `Float64` solve; see
[`precompute_geometry`](@ref) for what that trades.
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
    element_P::ReferenceElement{TP};
    geometry_precision = FP,
) where {NV, NP, FP, TV <: AbstractElement{2, NV, FP}, TP <: AbstractElement{2, NP, FP}}
    _check_geometry_precision(geometry_precision)
    NQ_v = length(element_v.integration_points.ω)
    FPg  = geometry_precision
    geo_v = KA.allocate(backend, NTuple{NQ_v, QuadraturePointGeometry{2, FPg, 4}}, mesh.nels)
    geo_P = KA.allocate(backend, NTuple{NQ_v, FPg}, mesh.nels)
    _fill_mixed_geometry!(geo_v, geo_P, backend, workgroup, mesh, element_v, element_P)
    return MixedMeshCache(geo_v, geo_P, element_v, element_P)
end

function _fill_mixed_geometry!(
    geo_v, geo_P, backend, workgroup, mesh::MixedMesh{2},
    element_v::ReferenceElement{TV}, element_P::ReferenceElement{TP},
) where {NV, NP, TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}}
    ip_v = element_v.integration_points
    ∂N∂ξq_v = shape_function_gradients(element_v, ip_v)
    ∂N∂ξq_P = shape_function_gradients(element_P, ip_v)

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
    return nothing
end

"""
    update_geometry!(mesh::MixedMesh; workgroup=256) -> mesh

Recompute `mesh.geometry` in place from the current `mesh.coords`. Call it
after moving the mesh nodes; the geometry arrays keep their identity, so
references to `mesh.geometry.geo_v` and `mesh.geometry.geo_P` stay valid.
"""
function update_geometry!(mesh::MixedMesh{2}; workgroup = 256)
    geometry = _mesh_geometry(mesh)
    backend = KA.get_backend(geometry.geo_v)
    _fill_mixed_geometry!(
        geometry.geo_v, geometry.geo_P, backend, workgroup, mesh,
        geometry.element_v, geometry.element_P,
    )
    return mesh
end

# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

"""
    generate_discontinuous_linear_mesh(coords, el2n) -> (p_el2n, p_el2dof, p_dof_coords)

Build the linear triangle topology and element-to-DoF map for discontinuous
linear pressure elements.

`el2n` may be either T3 or T6 triangle connectivity. The returned `p_el2n`
uses rows 1:3, i.e. the corner nodes of each triangle. The returned
`p_el2dof` gives each element its own three pressure DoFs:
`p_el2dof[:, iel] == 3(iel - 1) .+ (1:3)`.

Returns `(p_el2n, p_el2dof, p_dof_coords)`, where `p_dof_coords` duplicates
corner coordinates per element so a discontinuous nodal pressure field can be
plotted or initialized directly on pressure DoFs.
"""
function generate_discontinuous_linear_mesh(coords, el2n::AbstractMatrix{<:Integer})
    size(el2n, 1) >= 3 || throw(ArgumentError("triangle connectivity needs at least 3 local nodes"))

    nels      = size(el2n, 2)
    p_el2n    = Matrix{Int32}(el2n[1:3, :])
    p_el2dof  = Matrix{Int32}(undef, 3, nels)
    p_dof_coords = Vector{eltype(coords)}(undef, 3 * nels)

    for iel in 1:nels
        base = 3 * (iel - 1)
        for a in 1:3
            dof = base + a
            p_el2dof[a, iel] = Int32(dof)
            p_dof_coords[dof] = coords[p_el2n[a, iel]]
        end
    end

    return p_el2n, p_el2dof, p_dof_coords
end
