# Time-step response convergence at fixed geometry, material, scales, and event duration.
# julia --project=examples examples/benchmarks/stokes/chamber_memory/chamber_memory.jl
include(joinpath(@__DIR__, "..", "..", "..", "stokes", "volcano", "chamber_memory_setup2D.jl"))
using .ChamberMemory2D
using LinearAlgebra

function run_chamber_memory_benchmark(; steps_days = (10.0, 5.0, 2.5),
        mesh_options = (;), run_options = (;))
    println("Julia $VERSION; CPU; $(Threads.nthreads()) Julia threads; fixed-mesh plane strain")
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    warmup = ChamberMemory2D.chamber_cycle_history(; cycle_lengths = (20ChamberMemory2D.day,))
    ChamberMemory2D.run_chamber_memory(model, warmup; run_options...)
    rows = NamedTuple[]
    previous = nothing
    for dt in steps_days
        row = try
            history = ChamberMemory2D.chamber_cycle_history(;
                Δt = dt * ChamberMemory2D.day, cycle_lengths = (40ChamberMemory2D.day,))
            elapsed = @elapsed result = ChamberMemory2D.run_chamber_memory(model, history; run_options...)
            change = if isnothing(previous)
                missing
            else
                indices = [only(findall(==(t), result.time)) for t in previous.time]
                norm(previous.uz - result.uz[:, indices]) / norm(result.uz[:, indices])
            end
            previous = result
            (; dt_days = dt, converged = true, elapsed, nels = result.nels,
                iterations = sum(s.iter for s in result.stats), change,
                budget_error = maximum(abs, result.budget) / maximum(abs, result.net_area), failure = "")
        catch err
            err isa InterruptException && rethrow()
            (; dt_days = dt, converged = false, elapsed = missing, nels = model.mesh.nels,
                iterations = missing, change = missing, budget_error = missing, failure = sprint(showerror, err))
        end
        push!(rows, row)
        println(row)
    end
    return rows
end

run_chamber_memory_benchmark()
