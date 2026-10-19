module MPIThermal

using FEMTools, KernelAbstractions, StaticArrays, LinearAlgebra, SparseArrays
using MPI, Metis, PartitionedArrays, Serialization
import KernelAbstractions as KA
import PartitionedArrays as PA

# This helper has no top-level solve. The sibling driver owns MPI initialization.
function collective(f, comm)
    result = nothing
    message = ""
    try
        result = f()
    catch err
        message = first(sprint(showerror, err), 2048)
    end
    rank = MPI.Comm_rank(comm)
    failed = MPI.Allreduce(isempty(message) ? typemax(Int) : rank, MPI.MIN, comm)
    if failed != typemax(Int)
        message = MPI.bcast(message, comm; root = failed)
        error("MPI thermal rank $failed: $message")
    end
    return result
end

function agree(value, comm; root = 0)
    reference = MPI.bcast(value, comm; root)
    collective(comm) do
        isequal(value, reference) || throw(ArgumentError("ranks supplied different problem/solver controls"))
    end
    return nothing
end

function element_assignment(mesh, nparts)
    active = min(nparts, mesh.nels)
    active == 1 && return zeros(Int, mesh.nels)
    adjacency = generate_element_adjacency(mesh.el2n; shared = :face, dimension = 2)
    rows, columns = Int[], Int[]
    for (i, neighbors) in enumerate(adjacency), j in neighbors
        push!(rows, i)
        push!(columns, j)
    end
    graph = sparse(rows, columns, ones(Int, length(rows)), mesh.nels, mesh.nels)
    return Int.(Metis.partition(graph, active)) .- 1
end

function partition_problem(problem, nparts; assignment = nothing)
    (; mesh, fields, bc) = problem
    mesh.element isa ReferenceElement{<:LinearElement{2, 3}} ||
        throw(ArgumentError("this example supports conforming 2D T3 meshes"))
    mesh.nels > 0 || throw(ArgumentError("the global mesh is empty"))
    owners_e = isnothing(assignment) ? element_assignment(mesh, nparts) : assignment
    length(owners_e) == mesh.nels || throw(DimensionMismatch("element assignment length"))
    all(r -> 0 <= r < nparts, owners_e) || throw(ArgumentError("element assignment rank outside communicator"))
    all(a -> length(a) == mesh.nnodes, values(fields)) || throw(DimensionMismatch("nodal input length"))
    incidence = [Int[] for _ in 1:mesh.nnodes]
    for e in 1:mesh.nels, node in mesh.el2n[:, e]
        push!(incidence[node], owners_e[e])
    end
    all(x -> !isempty(x), incidence) || throw(ArgumentError("orphan global node"))
    foreach(unique!, incidence)
    owner = minimum.(incidence)
    shared = findall(x -> length(x) > 1, incidence)
    shared_id = Dict(node => i for (i, node) in enumerate(shared))
    constraints = Dict{Int, eltype(bc.vals)}()
    for (node, value) in zip(bc.DoFs, bc.vals)
        1 <= node <= mesh.nnodes || throw(ArgumentError("boundary node outside global mesh"))
        haskey(constraints, node) && constraints[node] != value &&
            throw(ArgumentError("conflicting boundary values at global node $node"))
        constraints[node] = value
    end
    return map(0:(nparts - 1)) do rank
        elements = findall(==(rank), owners_e)
        touched = sort!(unique(vec(mesh.el2n[:, elements])))
        owned = filter(n -> owner[n] == rank, touched)
        ghosts = filter(n -> owner[n] != rank, touched)
        nodes = vcat(owned, ghosts)
        local_id = Dict(node => i for (i, node) in enumerate(nodes))
        connectivity = map(n -> Int32(local_id[n]), mesh.el2n[:, elements])
        boundary = Int32[local_id[n] for n in mesh.Γnodes if haskey(local_id, n)]
        fixed = filter(n -> haskey(constraints, n), owned)
        halo = findall(n -> haskey(shared_id, n), nodes)
        halo_owned = count(i -> i <= length(owned), halo)
        halo_ids = Int[shared_id[nodes[i]] for i in halo]
        halo_owners = Int32[owner[nodes[i]] + 1 for i in halo]
        (; nodes, elements, nowned = length(owned), nglobal = mesh.nnodes,
            coords = mesh.coords[nodes], connectivity, boundary,
            fields = map(a -> a[nodes], fields),
            bc_nodes = Int32[local_id[n] for n in fixed], bc_values = [constraints[n] for n in fixed],
            halo = Int32.(halo), halo_owned, halo_ids, halo_owners, nshared = length(shared))
    end
end

# Serialize once on root, then coordinate receive-buffer allocation before payloads.
function scatter_parts(parts, comm; root = 0)
    rank = MPI.Comm_rank(comm)
    packed = collective(comm) do
        rank == root || return nothing
        buffers = map(parts) do part
            io = IOBuffer()
            serialize(io, part)
            take!(io)
        end
        counts = length.(buffers)
        sum(counts) <= typemax(Cint) || throw(ArgumentError("root setup exceeds the initial MPI count limit"))
        (; data = reduce(vcat, buffers), counts)
    end
    count = MPI.Scatter(rank == root ? Int64.(packed.counts) : nothing, Int64, comm; root)
    receive = collective(comm) do
        Vector{UInt8}(undef, count)
    end
    MPI.Scatterv!(rank == root ? MPI.VBuffer(packed.data, packed.counts) : nothing, receive, comm; root)
    return collective(comm) do
        deserialize(IOBuffer(receive))
    end
end

@kernel function pack_kernel!(buffer, @Const(field), @Const(indices))
    i = @index(Global)
    buffer[i] = field[indices[i]]
end

@kernel function unpack_kernel!(field, @Const(buffer), @Const(indices))
    i = @index(Global)
    field[indices[i]] = buffer[i]
end

struct ThermalContext{P, M, H, B}
    comm::MPI.Comm
    part::P
    mesh::M
    halo::H
    backend::B
    workgroup::Int
end

function halo_buffer(::Type{T}, indices, parts, backend, n) where {T}
    host = zeros(T, n)
    vector = PA.PVector(map(_ -> host, parts), indices)
    device = KA.zeros(backend, T, n)
    return (; host, vector, device)
end

function distribute(problem, comm; backend = CPU(), root = 0, assignment = nothing, workgroup = 128)
    MPI.Initialized() && !MPI.Finalized() || error("initialize MPI before distribution")
    agree((root, string(typeof(backend)), workgroup), comm)
    private = MPI.Comm_dup(comm)
    try
        parts = collective(private) do
            MPI.Comm_rank(private) == root ? partition_problem(problem, MPI.Comm_size(private); assignment) : nothing
        end
        part = scatter_parts(parts, private; root)
        local_data = collective(private) do
            TDev = FEMTools.TA(backend)
            FP = eltype(eltype(part.coords))
            element = ReferenceElement(LinearElement{2, 3, FP})
            coords, connectivity = TDev(part.coords), TDev(part.connectivity)
            dofs, boundary = TDev(Int32.(1:length(part.nodes))), TDev(part.boundary)
            # Carry physical boundaries explicitly: local exterior edges include interfaces.
            geometry = TDev(precompute_geometry(part.coords, part.connectivity, element))
            mesh = Mesh{2, 1, Nothing, Nothing, typeof(coords), typeof(dofs), typeof(connectivity),
                typeof(boundary), typeof(element), typeof(geometry)}(
                nothing, nothing, coords, dofs, connectivity, boundary, length(part.nodes),
                length(part.elements), element, geometry,
            )
            (; mesh, halo_indices = TDev(part.halo), FP)
        end
        # The private communicator is transferred to the PA backend; retain its
        # public MPI handle for explicit cleanup after all cached operations finish.
        distributed_parts = PA.distribute_with_mpi(LinearIndices((MPI.Comm_size(private),));
            comm = private, duplicate_comm = false)
        indices = map(distributed_parts) do p
            n = part.halo_owned
            PA.OwnAndGhostIndices(PA.OwnIndices(part.nshared, p, part.halo_ids[1:n]),
                PA.GhostIndices(part.nshared, part.halo_ids[(n + 1):end], part.halo_owners[(n + 1):end]))
        end
        float = halo_buffer(local_data.FP, indices, distributed_parts, backend, length(part.halo))
        integer = halo_buffer(Int32, indices, distributed_parts, backend, length(part.halo))
        halo = (; float, integer, indices = local_data.halo_indices)
        return ThermalContext(private, part, local_data.mesh, halo, backend, workgroup)
    catch
        MPI.free(private)
        rethrow()
    end
end

function Base.close(context::ThermalContext)
    context.comm == MPI.COMM_NULL && return nothing
    KA.synchronize(context.backend)
    MPI.Barrier(context.comm)
    MPI.free(context.comm)
    return nothing
end

function exchange!(context, field, operation)
    context.comm != MPI.COMM_NULL || error("MPI thermal context is closed")
    buffer = eltype(field) <: Integer ? context.halo.integer : context.halo.float
    collective(context.comm) do
        if !isempty(context.halo.indices)
            pack_kernel!(context.backend, context.workgroup)(buffer.device, field, context.halo.indices;
                ndrange = length(context.halo.indices))
            KA.synchronize(context.backend)
            copyto!(buffer.host, buffer.device)
        end
    end
    wait(operation(buffer.vector))
    collective(context.comm) do
        if !isempty(context.halo.indices)
            copyto!(buffer.device, buffer.host)
            unpack_kernel!(context.backend, context.workgroup)(field, buffer.device, context.halo.indices;
                ndrange = length(context.halo.indices))
            KA.synchronize(context.backend)
        end
    end
    return nothing
end

FEMTools._dr_owned_count(c::ThermalContext, n) = c.part.nowned
FEMTools._dr_global_count(c::ThermalContext, n) = c.part.nglobal
FEMTools._dr_owned_values(c::ThermalContext, a) = view(a, 1:c.part.nowned)
FEMTools._dr_guard(f, c::ThermalContext) = collective(f, c.comm)
FEMTools._dr_sum(c::ThermalContext, x) = MPI.Allreduce(x, MPI.SUM, c.comm)
FEMTools._dr_norm(c::ThermalContext, a) = sqrt(FEMTools._dr_sum(c, sum(abs2, a)))
FEMTools._dr_refresh!(c::ThermalContext, a) = exchange!(c, a, PA.consistent!)

function FEMTools._dr_finish_update!(c::ThermalContext, u, rate)
    KA.synchronize(c.backend)
    all(isfinite, FEMTools._dr_owned_values(c, u)) &&
        all(isfinite, FEMTools._dr_owned_values(c, rate)) || error("nonfinite owner update")
    return nothing
end

function FEMTools._dr_accumulate!(c::ThermalContext, arrays...)
    for a in arrays
        collective(c.comm) do
            all(isfinite, a) || error("nonfinite element contribution")
        end
        exchange!(c, a, PA.assemble!)
    end
end

function FEMTools._dr_spectral_bound(c::ThermalContext, rows, diagonal, name)
    local_max = collective(c.comm) do
        all(x -> isfinite(x) && x > 0, diagonal) || error("$name preconditioner has invalid owner diagonal")
        all(isfinite, rows) || error("$name Jacobian row sums are nonfinite")
        isempty(diagonal) ? zero(eltype(diagonal)) : maximum(rows ./ diagonal)
    end
    bound = MPI.Allreduce(local_max, MPI.MAX, c.comm)
    isfinite(bound) && bound > 0 || error("$name preconditioner produced invalid global λmax")
    return bound
end

function prepare_state(context, material; CFL = 0.9, tolerance = 1.0e-8)
    return collective(context.comm) do
        dr = ThermalDiffusionDR(context.mesh, material; CFL, ϵ = tolerance)
        for key in keys(context.part.fields)
            copyto!(getproperty(dr, key), getproperty(context.part.fields, key))
        end
        TDev = FEMTools.TA(context.backend)
        bc = DirichletBoundaryCondition(TDev(context.part.bc_nodes), TDev(context.part.bc_values))
        (; dr, bc)
    end
end

function solve_step!(dr, context, bc; dt, Tref = eltype(dr.T)(273),
        tolerance = dr.ϵ, max_iterations = 10_000, check_interval = 50,
        collect_history = true, throw_on_failure = true, verbose = true)
    agree((dt, Tref, tolerance, max_iterations, check_interval, collect_history,
        throw_on_failure, dr.CFL, dr.c_fact, dr.k, dr.Cp, dr.ρ0, dr.α, dr.K), context.comm)
    collective(context.comm) do
        isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
        all(i -> 1 <= i <= context.part.nowned, Array(bc.DoFs)) ||
            throw(ArgumentError("BCs must use local owner indices"))
    end
    for a in (dr.T0, dr.P, dr.source, dr.phases)
        FEMTools._dr_refresh!(context, a)
    end
    return solve!(dr, context.mesh, bc; dt, Tref, tolerance, max_iterations, check_interval,
        collect_history, throw_on_failure, workgroup = context.workgroup, execution = context,
        verbose = verbose && MPI.Comm_rank(context.comm) == 0)
end

function gather_field(field, context; root = 0)
    local_data = collective(context.comm) do
        n = context.part.nowned
        (; ids = context.part.nodes[1:n], values = Array(view(field, 1:n)))
    end
    gathered = MPI.gather(local_data, context.comm; root)
    return collective(context.comm) do
        MPI.Comm_rank(context.comm) == root || return nothing
        result = Vector{eltype(field)}(undef, context.part.nglobal)
        for chunk in gathered
            result[chunk.ids] = chunk.values
        end
        result
    end
end

gather_temperature(dr, context; kwargs...) = gather_field(dr.T, context; kwargs...)

end # module
