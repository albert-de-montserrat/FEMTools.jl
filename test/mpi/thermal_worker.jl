using Test, FEMTools, DomainSets, KernelAbstractions, MPI, LinearAlgebra, SparseArrays
using DomainSets: ×
include(joinpath(@__DIR__, "..", "..", "examples", "miniapps", "thermal", "2D_heat_diffusion_MPI", "mpi_thermal.jl"))
using .MPIThermal

if "--cuda" in ARGS
    using CUDA
    CUDA.allowscalar(false)
end

function problem_data(FP, nels; uniform = false)
    element = ReferenceElement(LinearElement{2, 3, FP})
    mesh = Mesh(CPU(), (FP(0) .. FP(1)) × (FP(0) .. FP(1)), element, nels)
    T = FP[1 + p[1] + 2p[2] + sinpi(p[1]) * sinpi(p[2]) for p in mesh.coords]
    bc = DirichletBoundaryCondition(mesh.Γnodes, FP[1 + mesh.coords[n][1] + 2mesh.coords[n][2] for n in mesh.Γnodes])
    T[bc.DoFs] = bc.vals
    phases = uniform ? ones(Int32, mesh.nnodes) : Int32[p[1] > FP(0.5) ? 2 : 1 for p in mesh.coords]
    fields = (; T, T0 = copy(T), source = FP[1 + p[2] for p in mesh.coords], P = zeros(FP, mesh.nnodes), phases)
    return (; mesh, bc, fields)
end

# Independent affine-triangle consistent-mass residual/Jacobian reference.
function independent_system(problem, dt)
    (; mesh, fields) = problem
    FP = eltype(fields.T)
    M, L = zeros(FP, mesh.nnodes, mesh.nnodes), zeros(FP, mesh.nnodes, mesh.nnodes)
    PC, rows = zeros(FP, mesh.nnodes), zeros(FP, mesh.nnodes)
    for nodes in eachcol(mesh.el2n)
        points = mesh.coords[nodes]
        affine = hcat(ones(FP, 3), first.(points), last.(points))
        gradients = inv(affine)[2:3, :]'
        area = abs(det(affine)) / 2
        mass = area / 12 .* (ones(FP, 3, 3) + Matrix{FP}(I, 3, 3))
        stiffness = area .* (gradients * gradients')
        jacobian = -mass - dt .* stiffness
        M[nodes, nodes] += mass
        L[nodes, nodes] += stiffness
        PC[nodes] += abs.(diag(jacobian))
        rows[nodes] += vec(sum(abs, jacobian; dims = 2))
    end
    R = M * (fields.T0 - fields.T) + dt .* (M * fields.source - L * fields.T)
    A, b = M + dt .* L, M * fields.T0 + dt .* (M * fields.source)
    fixed = Int.(problem.bc.DoFs)
    free = setdiff(1:mesh.nnodes, fixed)
    T = copy(fields.T)
    T[fixed] = problem.bc.vals
    T[free] = A[free, free] \ (b[free] - A[free, fixed] * T[fixed])
    return (; R, PC, rows, T)
end

function run_case(comm, FP, nels; uniform = false, ghost_only = false, backend = CPU())
    rank, size = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    root = size - 1
    material = ThermalMaterial(; k = uniform ? (FP(1),) : (FP(1), FP(3)),
        Cp = FP(1), ρ0 = FP(1), α = FP(0), K = FP(Inf))
    problem = MPIThermal.collective(comm) do
        rank == root ? problem_data(FP, nels; uniform) : nothing
    end
    assignment = MPIThermal.collective(comm) do
        rank == root && ghost_only && size > 1 || return nothing
        counts = zeros(Int, problem.mesh.nnodes)
        for node in problem.mesh.el2n
            counts[node] += 1
        end
        chosen = findfirst(e -> all(n -> counts[n] > 1, problem.mesh.el2n[:, e]), 1:problem.mesh.nels)
        owners = ones(Int, problem.mesh.nels)
        owners[chosen] = size
        owners
    end
    context = MPIThermal.distribute(problem, comm; backend, root, assignment, workgroup = 32)
    try
        tolerance = FP === Float32 ? FP(2e-5) : FP(1e-9)
        (; dr, bc) = MPIThermal.prepare_state(context, material; tolerance)
        if ghost_only && size > 1 && rank == size - 1
            @test context.mesh.nels == 1
            @test context.part.nowned == 0
        end
        @test context.mesh.nels == length(context.part.elements)
        @test MPI.Allreduce(context.part.nowned, MPI.SUM, context.comm) == context.part.nglobal
        @test MPI.Allreduce(context.mesh.nels, MPI.SUM, context.comm) == 2prod(nels)
        dt = FP(0.05)
        # Repeated reverse assembly must clear old ghost contributions.
        for repetition in 1:2
            MPIThermal.collective(context.comm) do
                FEMTools._assemble_thermal!(dr, dt, context.mesh, context.mesh.geometry,
                    context.mesh.element, FP(0), backend, 32, true)
            end
            FEMTools._dr_accumulate!(context, dr.R, dr.∂R∂T, dr.PC)
            obtained = map(a -> MPIThermal.gather_field(a, context; root), (dr.R, dr.∂R∂T, dr.PC))
            MPIThermal.collective(context.comm) do
                rank == root || return nothing
                if uniform
                    exact = independent_system(problem, dt)
                    for (a, b) in zip(obtained, (exact.R, exact.rows, exact.PC))
                        @test a ≈ b rtol = 200eps(FP) atol = 200eps(FP)
                    end
                end
            end
        end
        serial = MPIThermal.collective(context.comm) do
            rank == root || return nothing
            state = ThermalDiffusionDR(problem.mesh, material; CFL = 0.9, ϵ = tolerance)
            for key in keys(problem.fields)
                copyto!(getproperty(state, key), getproperty(problem.fields, key))
            end
            state
        end
        for step in 1:2
            if step > 1
                copyto!(dr.T0, dr.T)
                dr.source .*= FP(1.25)
                fill!(dr.∂T∂τ, 0)
            end
            stats = MPIThermal.solve_step!(dr, context, bc; dt, Tref = FP(0), verbose = false,
                tolerance, check_interval = 5, max_iterations = 5000)
            T = MPIThermal.gather_temperature(dr, context; root)
            @test stats.converged
            @test all(isfinite, dr.T)
            MPIThermal.collective(context.comm) do
                rank == root || return nothing
                if step > 1
                    copyto!(serial.T0, serial.T)
                    serial.source .*= FP(1.25)
                    fill!(serial.∂T∂τ, 0)
                end
                serial_stats = solve!(serial, problem.mesh, problem.bc; dt, Tref = FP(0),
                    workgroup = 32, tolerance, check_interval = 5, max_iterations = 5000, verbose = false)
                @test serial_stats.converged
                @test T ≈ serial.T rtol = 20tolerance atol = 20tolerance
                if uniform && step == 1
                    @test T ≈ independent_system(problem, dt).T rtol = 20tolerance atol = 20tolerance
                end
            end
        end
        failed = MPIThermal.solve_step!(dr, context, bc; dt, Tref = FP(0), tolerance = FP(0),
            max_iterations = 2, check_interval = 5, throw_on_failure = false, verbose = false)
        @test !failed.converged
        @test failed.iterations == 2
        # A local failure becomes the same exception on every rank.
        faulty_rank = MPI.Allreduce(context.part.nowned > 0 ? rank : typemax(Int), MPI.MIN, context.comm)
        rank == faulty_rank && fill!(dr.source, FP(NaN))
        @test_throws ErrorException MPIThermal.solve_step!(dr, context, bc; dt, verbose = false)
    finally
        close(context)
    end
    @test context.comm == MPI.COMM_NULL
    close(context)
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
    @testset "MPI thermal $(MPI.Comm_size(MPI.COMM_WORLD)) ranks" begin
        for FP in (Float32, Float64)
            run_case(MPI.COMM_WORLD, FP, (4, 4); uniform = true, backend)
            run_case(MPI.COMM_WORLD, FP, (4, 4); backend, ghost_only = true)
            run_case(MPI.COMM_WORLD, FP, (1, 1); backend)
        end
        # Communicator ranks intentionally differ from world ranks.
        reversed = MPI.Comm_split(MPI.COMM_WORLD, 0, -MPI.Comm_rank(MPI.COMM_WORLD))
        run_case(reversed, Float64, (3, 3); backend)
        MPI.free(reversed)
    end
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("PASS MPI thermal worker")
catch err
    showerror(stderr, err, catch_backtrace())
    MPI.Abort(MPI.COMM_WORLD, 1)
finally
    MPI.Finalized() || MPI.Finalize()
end
