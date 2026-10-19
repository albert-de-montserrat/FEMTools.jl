using DomainSets, KernelAbstractions, FEMTools, MPI, Printf
using DomainSets: ×

include(joinpath(@__DIR__, "mpi_thermal.jl"))
using .MPIThermal

if "--cuda" in ARGS
    using CUDA
end

"""
    main(nels=(32,32); nsteps=5, backend=CPU(), FP=Float64, comm=MPI.COMM_WORLD)

MPI copy of `2D_heat_diffusion_triangles`: the same 20 km × 20 km domain,
material properties, hot bottom/cold top, and backward-Euler DYREL stepping.
Metis assigns owned triangles; PartitionedArrays exchanges shared nodes only.
All ranks participate. Returns gathered temperature on rank zero and `nothing`
on peers, together with global solve statistics and local ownership counts.
"""
function main(nels = (32, 32); nsteps = 5, backend = CPU(), FP = Float64,
        comm = MPI.COMM_WORLD, workgroup = 128, verbose = true)
    FEMTools.agree((nels, nsteps, FP), comm)
    rank = MPI.Comm_rank(comm)
    material = ThermalMaterial(;
        k = (FP(3), FP(2.5)), Cp = (FP(1200), FP(1100)), ρ0 = (FP(3300), FP(2700)),
        α = (FP(3e-5), FP(2e-5)), K = (FP(1e11), FP(8e10)),
    )
    Lx, Ly = FP(10e3), FP(20e3)
    T_bot, T_top, Tref = FP(873), FP(273), FP(273)
    dt = FP(1e3 * 365 * 24 * 3600)
    problem = FEMTools.collective(comm) do
        all(>(0), nels) && nsteps > 0 || throw(ArgumentError("mesh dimensions and nsteps must be positive"))
        rank == 0 || return nothing
        Ω = (-Lx .. Lx) × (-Ly .. zero(FP))
        element = ReferenceElement(LinearElement{2, 3, FP})
        mesh = Mesh(CPU(), Ω, element, nels)
        bottom = findall(p -> p[2] == -Ly, mesh.coords)
        top = findall(p -> p[2] == 0, mesh.coords)
        bc = DirichletBoundaryCondition(Int32.(vcat(bottom, top)),
            vcat(fill(T_bot, length(bottom)), fill(T_top, length(top))))
        T = fill((T_bot + T_top) / 2, mesh.nnodes)
        T[bc.DoFs] = bc.vals
        # Activate both phases, which the original example leaves as an option.
        phases = Int32[p[2] > -Ly / 2 ? 2 : 1 for p in mesh.coords]
        fields = (; T, T0 = copy(T), P = zeros(FP, mesh.nnodes),
            source = zeros(FP, mesh.nnodes), phases)
        (; mesh, bc, fields)
    end
    context = MPIThermal.distribute(problem, comm; backend, workgroup)
    try
        tolerance = FP === Float32 ? FP(1e-5) : FP(1e-8)
        (; dr, bc) = MPIThermal.prepare_state(context, problem, material; tolerance)
        history = NamedTuple[]
        for step in 1:nsteps
            rank == 0 && verbose && @printf("time step %d / %d; MPI ranks = %d\n", step, nsteps, MPI.Comm_size(comm))
            copyto!(dr.T0, dr.T)
            fill!(dr.∂T∂τ, 0)
            stats = MPIThermal.solve_step!(dr, context, bc; dt, Tref, verbose)
            push!(history, stats)
        end
        T = MPIThermal.gather_temperature(dr, context)
        counts = (; owned_nodes = context.part.nowned, ghost_nodes = context.mesh.nnodes - context.part.nowned,
            owned_elements = context.mesh.nels)
        rank == 0 && verbose && println("MPI thermal diffusion complete: temperature range = ", extrema(T))
        return (; T, history, counts)
    finally
        close(context)
    end
end

function run_example()
    MPI.Init(threadlevel = :funneled)
    try
        backend = CPU()
        if "--cuda" in ARGS
            # Selection precedes all device allocations. A scheduler may expose
            # one GPU/rank; otherwise use the rank within the shared-memory node.
            CUDA.allowscalar(false)
            local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
            local_rank = MPI.Comm_rank(local_comm)
            CUDA.device!(length(CUDA.devices()) == 1 ? 0 : local_rank)
            MPI.free(local_comm)
            backend = CUDA.CUDABackend()
        end
        n = parse(Int, get(ENV, "FEMTOOLS_MPI_N", "32"))
        steps = parse(Int, get(ENV, "FEMTOOLS_MPI_STEPS", "5"))
        FP = "--float32" in ARGS ? Float32 : Float64
        return Base.invokelatest(main, (n, n); nsteps = steps, backend, FP)
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        MPI.Abort(MPI.COMM_WORLD, 1)
        rethrow()
    finally
        MPI.Finalized() || MPI.Finalize()
    end
end

run_example()
