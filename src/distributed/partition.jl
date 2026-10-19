"""
    MeshPartition

Host-side description of one part of a serial nodal mesh split by element
ownership. Built by `partition_mesh`; it holds no communicator and does
not depend on MPI.

Fields:

- `part`: part label in `1:nparts`.
- `nodes`: serial node index of every local node; the `nowned` owned nodes come
  first, followed by ghost nodes owned by another part.
- `elements`: serial index of every element owned by this part, in increasing
  order. Local element `i` is serial element `elements[i]`.
- `nowned`, `nglobal`: owned node count and serial node count.
- `coords`, `el2n`: local coordinates and connectivity; `el2n` indexes `nodes`
  and keeps the serial element node ordering and integer type.
- `Γnodes`: local indices of the serial mesh's physical boundary nodes.
  Interfaces between parts are not physical boundaries and are never added.
- `halo`: local indices of nodes incident to more than one part, owned entries
  first. `halo_ids[i]` is a dense interface-node ID in `1:nshared` that is the
  same on every part, and `halo_owners[i]` is the part owning `halo[i]`.

A part may own no elements, and a part owning elements may own no nodes.
"""
struct MeshPartition{C, E, I}
    part::Int
    nodes::Vector{Int}
    elements::Vector{Int}
    nowned::Int
    nglobal::Int
    coords::C
    el2n::E
    Γnodes::Vector{I}
    halo::Vector{I}
    halo_ids::Vector{Int}
    halo_owners::Vector{Int}
    nshared::Int
end

"""
    partition_mesh(mesh, assignment, nparts) -> Vector{MeshPartition}

Split `mesh` into `nparts` `MeshPartition`s. `assignment[e]` is the part
in `1:nparts` owning serial element `e`, for example from
`partition_elements`.

Every node is owned by the lowest-numbered part among the parts owning an
element incident to it. Nodes referenced by no element are rejected because
they have no owner. The mesh is copied to the host, so device meshes are
accepted.
"""
function partition_mesh(mesh::Mesh, assignment::AbstractVector{<:Integer}, nparts::Integer)
    nparts > 0 || throw(ArgumentError("nparts must be positive, got $nparts"))
    length(assignment) == mesh.nels || throw(
        DimensionMismatch("assignment has $(length(assignment)) entries for $(mesh.nels) elements")
    )
    coords, el2n, Γnodes = Array(mesh.coords), Array(mesh.el2n), Array(mesh.Γnodes)
    nnodes = mesh.nnodes
    all(n -> 1 <= n <= nnodes, el2n) || throw(ArgumentError("connectivity references a node outside 1:$nnodes"))

    elements = [Int[] for _ in 1:nparts]
    for (e, p) in zip(axes(el2n, 2), assignment)
        1 <= p <= nparts || throw(ArgumentError("element $e is assigned to part $p, outside 1:$nparts"))
        push!(elements[p], e)
    end

    # Parts are visited in increasing order, so the first part to touch a node
    # is its lowest incident part.
    owner, nsharers, seen = zeros(Int, nnodes), zeros(Int, nnodes), zeros(Int, nnodes)
    for p in 1:nparts, e in elements[p], n in view(el2n, :, e)
        seen[n] == p && continue
        seen[n] = p
        nsharers[n] += 1
        owner[n] == 0 && (owner[n] = p)
    end
    orphan = findfirst(iszero, owner)
    orphan === nothing || throw(ArgumentError("node $orphan is referenced by no element"))
    shared_id = zeros(Int, nnodes)
    nshared = 0
    for n in 1:nnodes
        nsharers[n] > 1 && (shared_id[n] = (nshared += 1))
    end

    I = eltype(el2n)
    local_id = zeros(Int, nnodes)
    return map(1:nparts) do p
        touched = sort!(unique!(Int.(vec(el2n[:, elements[p]]))))
        nodes = vcat(filter(n -> owner[n] == p, touched), filter(n -> owner[n] != p, touched))
        for (i, n) in enumerate(nodes)
            local_id[n] = i
        end
        local_el2n = similar(el2n, size(el2n, 1), length(elements[p]))
        for (i, e) in enumerate(elements[p]), k in axes(el2n, 1)
            local_el2n[k, i] = local_id[el2n[k, e]]
        end
        local_Γnodes = I[local_id[n] for n in Γnodes if local_id[n] != 0]
        halo = I[i for (i, n) in enumerate(nodes) if shared_id[n] != 0]
        partition = MeshPartition(
            p, nodes, elements[p], count(n -> owner[n] == p, nodes), nnodes,
            coords[nodes], local_el2n, local_Γnodes,
            halo, shared_id[nodes[halo]], owner[nodes[halo]], nshared,
        )
        local_id[nodes] .= 0
        partition
    end
end

"""
    localize_dirichlet(partition, bc) -> DirichletBoundaryCondition

Return the constraints of the serial boundary condition `bc` that fall on nodes
owned by `partition`, indexed locally. Ghost nodes carry no constraint: their
values arrive from the owning part. Repeated serial entries must agree.
"""
function localize_dirichlet(partition::MeshPartition, bc::DirichletBoundaryCondition)
    DoFs, vals = Array(bc.DoFs), Array(bc.vals)
    constraint = Dict{Int, eltype(vals)}()
    for (n, v) in zip(DoFs, vals)
        1 <= n <= partition.nglobal ||
            throw(ArgumentError("boundary node $n is outside 1:$(partition.nglobal)"))
        isequal(get(constraint, n, v), v) ||
            throw(ArgumentError("conflicting boundary values at node $n"))
        constraint[n] = v
    end
    owned = view(partition.nodes, 1:partition.nowned)
    local_DoFs = eltype(DoFs)[i for (i, n) in enumerate(owned) if haskey(constraint, n)]
    return DirichletBoundaryCondition(local_DoFs, [constraint[owned[i]] for i in local_DoFs])
end

"""
    reconstruct_owned(nglobal, ids, values) -> Vector

Assemble a serial nodal array of length `nglobal` from owned pieces: `ids[p]`
lists the serial nodes owned by part `p` and `values[p]` their values. Every
serial node must be supplied exactly once.
"""
function reconstruct_owned(nglobal::Integer, ids, values)
    length(ids) == length(values) || throw(DimensionMismatch("ids and values describe different part counts"))
    result = Vector{mapreduce(eltype, promote_type, values)}(undef, nglobal)
    written = falses(nglobal)
    for (part_ids, part_values) in zip(ids, values)
        length(part_ids) == length(part_values) ||
            throw(DimensionMismatch("$(length(part_ids)) ids for $(length(part_values)) values"))
        for (n, v) in zip(part_ids, Array(part_values))
            written[n] && throw(ArgumentError("node $n is supplied by more than one part"))
            written[n] = true
            result[n] = v
        end
    end
    missing_node = findfirst(!, written)
    missing_node === nothing || throw(ArgumentError("node $missing_node is supplied by no part"))
    return result
end

"""
    local_mesh(partition, element, backend = CPU()) -> Mesh

Build the element-aware `Mesh` of one partition on `backend`. Its boundary nodes
are the partition's inherited physical boundary nodes, not boundary detection
on the local connectivity, which would also mark partition interfaces.
Geometry is precomputed on the host and uploaded with the other arrays.
"""
function local_mesh(
        partition::MeshPartition, element::ReferenceElement{T}, backend = CPU(),
    ) where {nDim, T <: AbstractElement{nDim}}
    TDev = TA(backend)
    geometry = TDev(precompute_geometry(partition.coords, partition.el2n, element))
    coords, el2n = TDev(partition.coords), TDev(partition.el2n)
    DoFs, Γnodes = TDev(Int32.(1:length(partition.nodes))), TDev(partition.Γnodes)
    return Mesh{
        nDim, order(element), Nothing, Nothing, typeof(coords), typeof(DoFs),
        typeof(el2n), typeof(Γnodes), typeof(element), typeof(geometry),
    }(
        nothing, nothing, coords, DoFs, el2n, Γnodes, length(partition.nodes),
        length(partition.elements), element, geometry,
    )
end

"""
    partition_elements(mesh, nparts) -> Vector{Int}

Assign every element of `mesh` to a part in `1:nparts` by partitioning its
face-adjacency graph. At most `nels` parts receive elements; any remaining
parts own none.

The method is supplied by a package extension. Call `using Metis` before using
it.
"""
function partition_elements(args...)
    return throw(
        ArgumentError(
            "partition_elements has no method for these arguments; if Metis is " *
                "not loaded yet, run `using Metis` to enable the partitioning extension"
        )
    )
end
