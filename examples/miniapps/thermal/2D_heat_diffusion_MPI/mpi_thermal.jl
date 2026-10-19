module MPIThermal

using FEMTools, MPI, Metis, PartitionedArrays

# This helper has no top-level solve. The sibling driver owns MPI initialization.
# Nodal thermal inputs scattered from the root problem to every rank.
const FIELDS = (:T, :T0, :P, :source, :phases)

"""
    distribute(problem, comm; backend = CPU(), root = 0, assignment = nothing, workgroup = 128)

Distribute the T3 mesh of the root-held `problem = (; mesh, bc, fields)`;
other ranks pass `nothing`.
"""
function distribute(problem, comm; backend = CPU(), root = 0, assignment = nothing, workgroup = 128)
    rank = MPI.Comm_rank(comm)
    FEMTools.collective(comm) do
        rank == root || return nothing
        (; mesh, fields) = problem
        mesh.element isa ReferenceElement{<:LinearElement{2, 3}} ||
            throw(ArgumentError("this example supports conforming 2D T3 meshes"))
        mesh.nels > 0 || throw(ArgumentError("the global mesh is empty"))
        all(key -> length(fields[key]) == mesh.nnodes, FIELDS) || throw(DimensionMismatch("nodal input length"))
    end
    return FEMTools.distribute_mesh(
        rank == root ? problem.mesh : nothing, rank == root ? assignment : nothing, comm;
        root, backend, workgroup,
    )
end

function prepare_state(distributed, problem, material; CFL = 0.9, tolerance = 1.0e-8)
    dr = FEMTools.collective(distributed) do
        ThermalDiffusionDR(distributed.mesh, material; CFL, ϵ = tolerance)
    end
    root = MPI.Comm_rank(distributed.comm) == distributed.root
    for key in FIELDS
        FEMTools.scatter_field!(getproperty(dr, key), root ? problem.fields[key] : nothing, distributed)
    end
    bc = FEMTools.localize_bc(root ? problem.bc : nothing, distributed)
    return (; dr, bc)
end

function solve_step!(dr, distributed, bc; dt, Tref = eltype(dr.T)(273),
        tolerance = dr.ϵ, max_iterations = 10_000, check_interval = 50,
        collect_history = true, throw_on_failure = true, verbose = true)
    FEMTools.agree((dt, Tref, tolerance, max_iterations, check_interval, collect_history,
        throw_on_failure, dr.CFL, dr.c_fact, dr.k, dr.Cp, dr.ρ0, dr.α, dr.K), distributed)
    FEMTools.collective(distributed) do
        isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
        all(i -> 1 <= i <= distributed.part.nowned, Array(bc.DoFs)) ||
            throw(ArgumentError("BCs must use local owner indices"))
    end
    for a in (dr.T0, dr.P, dr.source, dr.phases)
        FEMTools._dr_refresh!(distributed, a)
    end
    return solve!(dr, distributed.mesh, bc; dt, Tref, tolerance, max_iterations, check_interval,
        collect_history, throw_on_failure, workgroup = distributed.workgroup, execution = distributed,
        verbose = verbose && MPI.Comm_rank(distributed.comm) == 0)
end

gather_temperature(dr, distributed) = FEMTools.gather_owned(dr.T, distributed)

end # module
