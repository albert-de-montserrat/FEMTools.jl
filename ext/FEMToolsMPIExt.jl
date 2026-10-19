module FEMToolsMPIExt

using FEMTools
using FEMTools: Mesh, ReferenceElement, DirichletBoundaryCondition, TA
using KernelAbstractions
import KernelAbstractions as KA
using MPI
import PartitionedArrays as PA
using Serialization

"""
    DistributedMesh

Rank-local state returned by `FEMTools.distribute_mesh`: a private
communicator, the root rank, this rank's `MeshPartition` and `local_mesh`, and
halo buffers for the mesh floating-point type and `Int32`. On the root rank
`parts` holds every rank's partition for later field and boundary-condition
scatters; elsewhere it is `nothing`.
"""
struct DistributedMesh{P, M, H, B, R}
    comm::MPI.Comm
    root::Int
    part::P
    mesh::M
    halo::H
    backend::B
    workgroup::Int
    parts::R
end

_comm(comm::MPI.Comm) = comm
_comm(distributed::DistributedMesh) = distributed.comm

function FEMTools.collective(f, target::Union{MPI.Comm, DistributedMesh})
    comm = _comm(target)
    result = nothing
    message = ""
    try
        result = f()
    catch err
        message = first(sprint(showerror, err), 2048)
    end
    failed = MPI.Allreduce(isempty(message) ? typemax(Int) : MPI.Comm_rank(comm), MPI.MIN, comm)
    if failed != typemax(Int)
        message = MPI.bcast(message, comm; root = failed)
        error("MPI rank $failed: $message")
    end
    return result
end

function FEMTools.agree(value, target::Union{MPI.Comm, DistributedMesh})
    reference = MPI.bcast(value, _comm(target); root = 0)
    FEMTools.collective(target) do
        isequal(value, reference) || throw(ArgumentError("ranks supplied different values: $value vs $reference on rank 0"))
    end
    return nothing
end

# Serialize once on root, then agree on receive-buffer allocation before the
# payload moves, so a failed allocation cannot strand root inside Scatterv.
function scatter_objects(objects, comm, root)
    rank = MPI.Comm_rank(comm)
    packed = FEMTools.collective(comm) do
        rank == root || return nothing
        length(objects) == MPI.Comm_size(comm) || throw(DimensionMismatch("one object per rank is required"))
        buffers = map(objects) do object
            io = IOBuffer()
            serialize(io, object)
            take!(io)
        end
        counts = length.(buffers)
        sum(counts) <= typemax(Cint) || throw(ArgumentError("scattered setup data exceeds the MPI count limit"))
        (; data = reduce(vcat, buffers), counts)
    end
    count = MPI.Scatter(rank == root ? Int64.(packed.counts) : nothing, Int64, comm; root)
    receive = FEMTools.collective(comm) do
        Vector{UInt8}(undef, count)
    end
    MPI.Scatterv!(rank == root ? MPI.VBuffer(packed.data, packed.counts) : nothing, receive, comm; root)
    return FEMTools.collective(comm) do
        deserialize(IOBuffer(receive))
    end
end

function halo_buffer(::Type{T}, indices, ranks, backend, n) where {T}
    host = zeros(T, n)
    vector = PA.PVector(map(_ -> host, ranks), indices)
    device = KA.zeros(backend, T, n)
    return (; host, vector, device)
end

function FEMTools.distribute_mesh(
        mesh, assignment, comm::MPI.Comm; root = 0, backend = CPU(), workgroup = 256,
    )
    MPI.Initialized() && !MPI.Finalized() || error("initialize MPI before distribute_mesh")
    FEMTools.agree((root, string(typeof(backend)), workgroup), comm)
    nranks = MPI.Comm_size(comm)
    FEMTools.collective(comm) do
        0 <= root < nranks || throw(ArgumentError("root $root is outside a communicator of size $nranks"))
    end
    private = MPI.Comm_dup(comm)
    try
        rank = MPI.Comm_rank(private)
        parts = FEMTools.collective(private) do
            rank == root || return nothing
            mesh isa Mesh && mesh.element isa ReferenceElement ||
                throw(ArgumentError("root must supply an element-aware Mesh"))
            FEMTools.partition_mesh(
                mesh, isnothing(assignment) ? FEMTools.partition_elements(mesh, nranks) : assignment, nranks,
            )
        end
        payloads = rank == root ? [(; part, element = mesh.element) for part in parts] : nothing
        (; part, element) = scatter_objects(payloads, private, root)
        local_data = FEMTools.collective(private) do
            (; mesh = FEMTools.local_mesh(part, element, backend), indices = TA(backend)(part.halo))
        end
        # PartitionedArrays borrows the private communicator; it is freed by `close`.
        ranks = PA.distribute_with_mpi(LinearIndices((nranks,)); comm = private, duplicate_comm = false)
        indices = map(ranks) do p
            n = count(<=(part.nowned), part.halo)
            PA.OwnAndGhostIndices(
                PA.OwnIndices(part.nshared, p, part.halo_ids[1:n]),
                PA.GhostIndices(part.nshared, part.halo_ids[(n + 1):end], part.halo_owners[(n + 1):end]),
            )
        end
        FP = eltype(eltype(part.coords))
        nhalo = length(part.halo)
        halo = (;
            float = halo_buffer(FP, indices, ranks, backend, nhalo),
            integer = halo_buffer(Int32, indices, ranks, backend, nhalo),
            indices = local_data.indices,
        )
        return DistributedMesh(private, root, part, local_data.mesh, halo, backend, workgroup, parts)
    catch
        MPI.free(private)
        rethrow()
    end
end

function Base.close(distributed::DistributedMesh)
    distributed.comm == MPI.COMM_NULL && return nothing
    KA.synchronize(distributed.backend)
    MPI.Barrier(distributed.comm)
    MPI.free(distributed.comm)
    return nothing
end

@kernel function pack_kernel!(buffer, @Const(field), @Const(indices))
    i = @index(Global)
    buffer[i] = field[indices[i]]
end

@kernel function unpack_kernel!(field, @Const(buffer), @Const(indices))
    i = @index(Global)
    field[indices[i]] = buffer[i]
end

# Host-staged halo exchange: pack shared entries on the device, run the
# PartitionedArrays operation on the host copy, and unpack. `PA.consistent!`
# copies owner values to ghosts; `PA.assemble!` adds ghost values to owners and
# zeroes the ghosts.
function exchange!(distributed::DistributedMesh, field, operation)
    distributed.comm != MPI.COMM_NULL || error("the distributed mesh is closed")
    (; float, integer, indices) = distributed.halo
    buffer = eltype(field) === eltype(float.host) ? float :
        eltype(field) === Int32 ? integer : nothing
    FEMTools.collective(distributed) do
        buffer === nothing && throw(
            ArgumentError("halo exchange supports $(eltype(float.host)) and Int32 fields, got $(eltype(field))")
        )
        length(field) == distributed.mesh.nnodes ||
            throw(DimensionMismatch("field has $(length(field)) entries for $(distributed.mesh.nnodes) local nodes"))
        if !isempty(indices)
            pack_kernel!(distributed.backend, distributed.workgroup)(
                buffer.device, field, indices; ndrange = length(indices),
            )
            KA.synchronize(distributed.backend)
            copyto!(buffer.host, buffer.device)
        end
    end
    wait(operation(buffer.vector))
    FEMTools.collective(distributed) do
        if !isempty(indices)
            copyto!(buffer.device, buffer.host)
            unpack_kernel!(distributed.backend, distributed.workgroup)(
                field, buffer.device, indices; ndrange = length(indices),
            )
            KA.synchronize(distributed.backend)
        end
    end
    return nothing
end

function FEMTools.scatter_field!(local_array, root_array, distributed::DistributedMesh)
    rank = MPI.Comm_rank(distributed.comm)
    pieces = FEMTools.collective(distributed) do
        rank == distributed.root || return nothing
        nglobal = distributed.part.nglobal
        length(root_array) == nglobal ||
            throw(DimensionMismatch("root array has $(length(root_array)) entries for $nglobal nodes"))
        host = Array(root_array)
        [host[p.nodes] for p in distributed.parts]
    end
    values = scatter_objects(pieces, distributed.comm, distributed.root)
    FEMTools.collective(distributed) do
        length(local_array) == length(values) ||
            throw(DimensionMismatch("local array has $(length(local_array)) entries for $(length(values)) local nodes"))
        copyto!(local_array, values)
    end
    return local_array
end

function FEMTools.localize_bc(root_bc, distributed::DistributedMesh)
    rank = MPI.Comm_rank(distributed.comm)
    pieces = FEMTools.collective(distributed) do
        rank == distributed.root || return nothing
        [FEMTools.localize_dirichlet(p, root_bc) for p in distributed.parts]
    end
    bc = scatter_objects(pieces, distributed.comm, distributed.root)
    return FEMTools.collective(distributed) do
        TDev = TA(distributed.backend)
        DirichletBoundaryCondition(TDev(bc.DoFs), TDev(bc.vals))
    end
end

function FEMTools.gather_owned(local_array, distributed::DistributedMesh)
    (; part, comm, root) = distributed
    owned = FEMTools.collective(distributed) do
        length(local_array) == length(part.nodes) ||
            throw(DimensionMismatch("local array has $(length(local_array)) entries for $(length(part.nodes)) local nodes"))
        (; ids = part.nodes[1:part.nowned], values = Array(view(local_array, 1:part.nowned)))
    end
    gathered = MPI.gather(owned, comm; root)
    return FEMTools.collective(distributed) do
        MPI.Comm_rank(comm) == root || return nothing
        FEMTools.reconstruct_owned(part.nglobal, [g.ids for g in gathered], [g.values for g in gathered])
    end
end

# Shared dynamic-relaxation hooks: owner-only updates and reductions, halo
# refresh before reads, and ghost accumulation after assembly.
FEMTools._dr_owned_count(d::DistributedMesh, n) = d.part.nowned
FEMTools._dr_global_count(d::DistributedMesh, n) = d.part.nglobal
FEMTools._dr_owned_values(d::DistributedMesh, a) = view(a, 1:d.part.nowned)
FEMTools._dr_guard(f, d::DistributedMesh) = FEMTools.collective(f, d)
FEMTools._dr_sum(d::DistributedMesh, x) = MPI.Allreduce(x, MPI.SUM, d.comm)
FEMTools._dr_norm(d::DistributedMesh, a) = sqrt(FEMTools._dr_sum(d, sum(abs2, a)))
FEMTools._dr_refresh!(d::DistributedMesh, a) = exchange!(d, a, PA.consistent!)

function FEMTools._dr_finish_update!(d::DistributedMesh, u, rate)
    KA.synchronize(d.backend)
    all(isfinite, FEMTools._dr_owned_values(d, u)) &&
        all(isfinite, FEMTools._dr_owned_values(d, rate)) || error("nonfinite owner update")
    return nothing
end

function FEMTools._dr_accumulate!(d::DistributedMesh, arrays...)
    for a in arrays
        FEMTools.collective(d) do
            all(isfinite, a) || error("nonfinite element contribution")
        end
        exchange!(d, a, PA.assemble!)
    end
    return nothing
end

function FEMTools._dr_spectral_bound(d::DistributedMesh, rows, diagonal, name)
    local_max = FEMTools.collective(d) do
        all(x -> isfinite(x) && x > 0, diagonal) || error("$name preconditioner has invalid owner diagonal")
        all(isfinite, rows) || error("$name Jacobian row sums are nonfinite")
        isempty(diagonal) ? zero(eltype(diagonal)) : maximum(rows ./ diagonal)
    end
    bound = MPI.Allreduce(local_max, MPI.MAX, d.comm)
    isfinite(bound) && bound > 0 || error("$name preconditioner produced invalid global λmax")
    return bound
end

end # module FEMToolsMPIExt
