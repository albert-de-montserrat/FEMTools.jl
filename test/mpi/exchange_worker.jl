using Test, FEMTools, DomainSets, KernelAbstractions, MPI, PartitionedArrays, Metis
using DomainSets: ×

if "--cuda" in ARGS
    using CUDA
    CUDA.allowscalar(false)
end

# Number of parts owning an element incident to each serial node.
function incidence_counts(mesh, assignment)
    parts = [Set{Int}() for _ in 1:mesh.nnodes]
    for e in 1:mesh.nels, n in Array(mesh.el2n)[:, e]
        push!(parts[n], assignment[e])
    end
    return length.(parts)
end

# Halo copy/add, field scatter/gather, and boundary localization on one
# distribution. `assignment(mesh, nranks)` runs on root only.
function check_exchange(comm, mesh_builder, assignment; root = 0, backend = CPU())
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    mesh, parts = FEMTools.collective(comm) do
        rank == root || return (nothing, nothing)
        mesh = mesh_builder()
        (mesh, assignment(mesh, nranks))
    end
    distributed = FEMTools.distribute_mesh(mesh, parts, comm; root, backend, workgroup = 32)
    try
        (; part) = distributed
        FP = eltype(eltype(part.coords))
        nlocal = length(part.nodes)
        on_backend(x) = FEMTools.TA(backend)(x)
        @test distributed.mesh.nnodes == nlocal
        @test MPI.Allreduce(part.nowned, MPI.SUM, distributed.comm) == part.nglobal

        # Scatter fills owned and ghost nodes; gather rebuilds serial order.
        ids = rank == root ? FP.(1:mesh.nnodes) : nothing
        field = FEMTools.scatter_field!(on_backend(zeros(FP, nlocal)), ids, distributed)
        @test Array(field) == FP.(part.nodes)
        gathered = FEMTools.gather_owned(field, distributed)
        rank == root ? (@test gathered == ids) : (@test gathered === nothing)

        for T in (FP, Int32)
            # Copy: owners overwrite every ghost.
            local_ids = T.(part.nodes)
            local_ids[(part.nowned + 1):end] .= -1
            field = on_backend(local_ids)
            FEMTools._dr_refresh!(distributed, field)
            @test Array(field) == T.(part.nodes)

            # Add: owners receive every incident part's contribution exactly
            # once and ghosts are zeroed, also when buffers are reused.
            counts = rank == root ? T.(incidence_counts(mesh, parts)) : nothing
            expected = Array(FEMTools.scatter_field!(on_backend(zeros(T, nlocal)), counts, distributed))
            expected[(part.nowned + 1):end] .= 0
            for _ in 1:2
                field = on_backend(ones(T, nlocal))
                FEMTools._dr_accumulate!(distributed, field)
                @test Array(field) == expected
            end
        end

        bc = rank == root ? DirichletBoundaryCondition(Array(mesh.Γnodes), FP.(Array(mesh.Γnodes))) : nothing
        local_bc = FEMTools.localize_bc(bc, distributed)
        DoFs, vals = Array(local_bc.DoFs), Array(local_bc.vals)
        @test all(<=(part.nowned), DoFs)
        @test FP.(part.nodes[DoFs]) == vals
        nΓ = MPI.bcast(rank == root ? length(mesh.Γnodes) : nothing, distributed.comm; root)
        @test MPI.Allreduce(length(DoFs), MPI.SUM, distributed.comm) == nΓ
    finally
        close(distributed)
    end
    @test distributed.comm == MPI.COMM_NULL
    close(distributed)
    return nothing
end

unit_square(FP, element, nels) = () -> Mesh((FP(0) .. FP(1)) × (FP(0) .. FP(1)), ReferenceElement(element), nels)
metis(mesh, nranks) = FEMTools.partition_elements(mesh, nranks)
round_robin(mesh, nranks) = [mod1(e, nranks) for e in 1:mesh.nels]
# Every element on the last part except one on each other part: uneven and,
# with more parts than elements, empty parts.
uneven(mesh, nranks) = [e < nranks ? e : nranks for e in 1:mesh.nels]

function run_cases(comm, backend)
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    for FP in (Float32, Float64)
        check_exchange(comm, unit_square(FP, LinearElement{2, 3, FP}, (4, 4)), metis; backend)
        check_exchange(comm, unit_square(FP, LinearElement{2, 3, FP}, (4, 4)), round_robin; backend, root = nranks - 1)
        check_exchange(comm, unit_square(FP, LinearElement{2, 3, FP}, (1, 1)), uneven; backend)
    end
    # Q9 cells meeting at the centre vertex: vertex-only and up to four sharers.
    check_exchange(comm, unit_square(Float64, QuadraticElement{2, 9, Float64}, (2, 2)), round_robin; backend)

    # Communicator ranks differ from world ranks.
    reversed = MPI.Comm_split(comm, 0, -rank)
    check_exchange(reversed, unit_square(Float64, LinearElement{2, 3, Float64}, (3, 3)), round_robin; backend)
    MPI.free(reversed)

    # Two live distributed meshes keep their messages apart.
    build = unit_square(Float64, LinearElement{2, 3, Float64}, (3, 3))
    # Root-side Metis partitioning when no assignment is given.
    one = FEMTools.distribute_mesh(rank == 0 ? build() : nothing, nothing, comm; backend)
    other = FEMTools.distribute_mesh(rank == 0 ? build() : nothing, nothing, comm; backend)
    for distributed in (one, other, one)
        field = FEMTools.TA(backend)(Float64.(distributed.part.nodes))
        FEMTools._dr_refresh!(distributed, field)
        @test Array(field) == distributed.part.nodes
    end
    close(other)
    close(one)
    @test_throws "the distributed mesh is closed" FEMTools._dr_refresh!(one, FEMTools.TA(backend)(zeros(3)))

    # Setup and local failures reach every rank as the same error.
    root = nranks - 1
    bad = rank == root ? build() : nothing
    @test_throws "MPI rank $root: ArgumentError: element 1 is assigned to part 0" FEMTools.distribute_mesh(
        bad, rank == root ? zeros(Int, 18) : nothing, comm; root, backend,
    )
    @test_throws "MPI rank 0: ArgumentError: root must supply an element-aware Mesh" FEMTools.distribute_mesh(
        nothing, nothing, comm; backend,
    )
    nranks > 1 && @test_throws "ranks supplied different values" FEMTools.distribute_mesh(
        rank == 0 ? build() : nothing, nothing, comm; backend, workgroup = rank == nranks - 1 ? 64 : 32,
    )
    distributed = FEMTools.distribute_mesh(rank == 0 ? build() : nothing, nothing, comm; backend)
    try
        nlocal = length(distributed.part.nodes)
        wrong = rank == nranks - 1 ? nlocal + 1 : nlocal
        @test_throws "MPI rank $(nranks - 1): DimensionMismatch: local array has" FEMTools.scatter_field!(
            FEMTools.TA(backend)(zeros(wrong)), rank == 0 ? zeros(16) : nothing, distributed,
        )
        @test_throws "halo exchange supports Float64 and Int32 fields, got Float16" FEMTools._dr_refresh!(
            distributed, FEMTools.TA(backend)(zeros(Float16, nlocal)),
        )
    finally
        close(distributed)
    end
    return nothing
end

MPI.Init(threadlevel = :funneled)
try
    backend = CPU()
    if "--cuda" in ARGS
        local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
        local_rank = MPI.Comm_rank(local_comm)
        length(CUDA.devices()) > local_rank || error("CUDA tests require distinct GPUs for local ranks")
        CUDA.device!(local_rank)
        MPI.free(local_comm)
        backend = CUDA.CUDABackend()
    end
    @testset "MPI exchange $(MPI.Comm_size(MPI.COMM_WORLD)) ranks" begin
        run_cases(MPI.COMM_WORLD, backend)
    end
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("PASS MPI exchange worker")
catch err
    showerror(stderr, err, catch_backtrace())
    MPI.Abort(MPI.COMM_WORLD, 1)
finally
    MPI.Finalized() || MPI.Finalize()
end
