using MPI

# Run with --project=examples. Workers are deliberately outside top-level test_*.jl.
project = normpath(joinpath(@__DIR__, "..", "..", "examples"))
worker = joinpath(@__DIR__, "thermal_worker.jl")
cuda = "--cuda" in ARGS
rank_counts = cuda ? (1, 2) : (1, 2, 4)
timeout = parse(Float64, get(ENV, "FEMTOOLS_MPI_TEST_TIMEOUT", "300"))

for ranks in rank_counts
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
        status == :timed_out && error("MPI $ranks-rank test exceeded $timeout seconds")
        success(process) || error("MPI $ranks-rank worker failed")
    end
end
