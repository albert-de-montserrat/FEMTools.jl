using MPI

# Run with --project=test/mpi. Workers are deliberately outside top-level test_*.jl.
# `--ranks=1,2` overrides the default rank counts, e.g. on a machine with one GPU.
project = @__DIR__
workers = joinpath.(@__DIR__, ("exchange_worker.jl", "thermal_worker.jl"))
cuda = "--cuda" in ARGS
ranks_arg = findfirst(startswith("--ranks="), ARGS)
rank_counts = ranks_arg === nothing ? (cuda ? (1, 2) : (1, 2, 4)) :
    parse.(Int, split(ARGS[ranks_arg][(length("--ranks=") + 1):end], ","))
timeout = parse(Float64, get(ENV, "FEMTOOLS_MPI_TEST_TIMEOUT", "300"))

for worker in workers, ranks in rank_counts
    mktemp() do path, io
        args = cuda ? ["--cuda"] : String[]
        cmd = `$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project $worker $args`
        process = run(pipeline(cmd; stdout = io, stderr = io); wait = false)
        status = timedwait(() -> process_exited(process), timeout; pollint = 0.2)
        if status == :timed_out
            if Sys.iswindows()
                run(ignorestatus(`taskkill /PID $(getpid(process)) /T /F`))
            else
                kill(process)
            end
            wait(process)
        end
        flush(io)
        output = read(path, String)
        print(output)
        status == :timed_out && error("MPI $ranks-rank $(basename(worker)) exceeded $timeout seconds")
        success(process) || error("MPI $ranks-rank $(basename(worker)) failed")
    end
end
