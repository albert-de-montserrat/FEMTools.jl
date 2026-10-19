# Entry points of the MPI execution layer. Their methods live in
# `FEMToolsMPIExt`, which loads with MPI and PartitionedArrays; the core package
# never initializes MPI.

_missing_mpi_extension(name) = throw(
    ArgumentError(
        "$name has no method for these arguments; if MPI and PartitionedArrays are " *
            "not loaded yet, run `using MPI, PartitionedArrays` to enable the MPI extension"
    )
)

"""
    distribute_mesh(mesh, assignment, comm; root = 0, backend = CPU(), workgroup = 256)

Collectively distribute a serial element-aware `mesh` held by rank `root` over
the MPI communicator `comm`. On `root`, `assignment` is a vector of parts in
`1:nranks` (rank `r` receives part `r + 1`) or `nothing` to call
`partition_elements`; other ranks pass `nothing` for both `mesh` and
`assignment`.

Returns a distributed mesh owning a private duplicate of `comm`, the rank's
`MeshPartition`, its `local_mesh` on `backend`, and cached halo buffers. Pass it
as `execution` to the shared dynamic-relaxation loop. Release it collectively
with `close` before `MPI.Finalize`. A failure on any rank raises the same error
on every rank.
"""
distribute_mesh(args...; kwargs...) = _missing_mpi_extension("distribute_mesh")

"""
    scatter_field!(local_array, root_array, distributed_mesh) -> local_array

Collectively fill the nodal `local_array` (owned and ghost nodes) of every rank
from the serial nodal array `root_array` held by the distributed mesh's root.
Other ranks pass `nothing` for `root_array`.
"""
scatter_field!(args...; kwargs...) = _missing_mpi_extension("scatter_field!")

"""
    localize_bc(root_bc, distributed_mesh) -> DirichletBoundaryCondition

Collectively return each rank's owned constraints of the serial boundary
condition `root_bc` held by the distributed mesh's root, in local indices and on
the mesh backend. Other ranks pass `nothing`.
"""
localize_bc(args...; kwargs...) = _missing_mpi_extension("localize_bc")

"""
    gather_owned(local_array, distributed_mesh)

Collectively reconstruct a serial nodal array from every rank's owned entries.
Returns the host array on the root rank and `nothing` elsewhere.
"""
gather_owned(args...; kwargs...) = _missing_mpi_extension("gather_owned")

"""
    collective(f, comm_or_distributed_mesh)

Call `f()` on every rank and agree on failure: if any rank throws, every rank
throws an `ErrorException` naming the lowest failing rank and its message.
Returns the local result of `f`.
"""
collective(args...; kwargs...) = _missing_mpi_extension("collective")

"""
    agree(value, comm_or_distributed_mesh)

Collectively check that `value` is equal on every rank; otherwise every rank
throws.
"""
agree(args...; kwargs...) = _missing_mpi_extension("agree")
