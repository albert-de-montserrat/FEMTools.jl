# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/impulse_response3D.jl [out_dir] [nsteps] [case] [backend] [pulse_rate]
# Stage F of StationsPaper/PLAN.md: the station impulse response of the 3-D
# Svartsengi chamber at the 17 GNSS stations, with the same runs and gates as
# impulse_response2D.jl: a one-step unit pulse followed by source-free steps;
# the same pulse with twice the step (time-step gate); and a delayed pulse and
# a random signed history over `ncheck` steps (shift and superposition gates).
# The mesh is output/svartsengi_mesh3D/svartsengi_mesh3D.jls. The reference
# physics has gravity and a Lagrangian mesh, so every response is the
# difference from a source-free run on the same time grid; a "linear_" prefix
# on the case gives the fixed-mesh model without gravity. `backend` is "cuda"
# (default) or "cpu". Writes CSVs, a Julia snapshot, and VTK fields after the
# first step and at the end of the response run.
include(joinpath(@__DIR__, "chamber_memory_setup3D.jl"))
include(joinpath(@__DIR__, "station_kernels.jl"))
using .ChamberMemory3D
using FEMTools
using KernelAbstractions: CPU
using LinearAlgebra
using Printf
using Random
using Serialization
using CUDA

const CM = ChamberMemory3D
const day = CM.day

"""Unit-pulse history: rate `pulse_rate` [m³/s] during step `start`, nothing otherwise, on a uniform grid."""
function pulse_history(Δt, nsteps; start = 1, pulse_rate = 1.0)
    recharge_rate = [n == start ? pulse_rate : 0.0 for n in 1:nsteps]
    return (; time_edges = collect(range(0.0, nsteps * Δt; length = nsteps + 1)), recharge_rate,
        withdrawal_volume = zeros(nsteps))
end

relative_error(a, b) = norm(a - b) / norm(b)

function write_fields(path, output_mesh, model, result, scale)
    write_vtk(path, output_mesh; title = "3-D chamber impulse response, m per (m^3/s)",
        point_data = (; displacement = map(f -> f ./ scale, result.u_field)), cell_data = (; phase = model.cell_phase))
    return path
end

function impulse_response3D_demo(; out_dir = joinpath(@__DIR__, "..", "output", "impulse_response3D"),
        mesh_file = joinpath(@__DIR__, "..", "output", "svartsengi_mesh3D", "svartsengi_mesh3D.jls"),
        Δt = 1day, nsteps = 365, ncheck = 40, pulse_rate = 1.0, seed = 2026, backend = CUDABackend(),
        run_options = (; gravity = true, advection = true))
    mkpath(out_dir)
    model = CM.load_chamber_memory_mesh(mesh_file; backend)
    output_mesh = Mesh(CPU(), model.coords_SI, model.el2n, model.element_v)
    @printf("mesh: %d cells, %d nodes, %d stations; Δt = %.2f days; backend %s\n", model.mesh.nels,
        model.mesh.nnodes, length(model.stations), Δt / day, nameof(typeof(backend)))

    # With gravity or advection the source-free state drifts slightly, so every
    # response is taken relative to a source-free run on the same time grid. The
    # baseline of a grid is a prefix of the longest one with the same step, so
    # one baseline per step length serves every run.
    nonlinear = get(run_options, :gravity, false) || get(run_options, :advection, false)
    baselines = Dict{Float64, NamedTuple}()
    function baseline(step, n)
        cached = get(baselines, step, nothing)
        if isnothing(cached) || length(cached.time) < n + 1
            cached = CM.run_chamber_memory(model, (; time_edges = collect(range(0.0, n * step; length = n + 1)),
                recharge_rate = zeros(n), withdrawal_volume = zeros(n)); run_options...)
            baselines[step] = cached
        end
        return cached
    end
    function respond(history)
        result = CM.run_chamber_memory(model, history; run_options...)
        nonlinear || return result
        n = length(history.recharge_rate)
        reference = baseline(history.time_edges[2] - history.time_edges[1], n)
        prefix(u) = u[:, 1:(n + 1)]
        # Node fields are the final cumulative displacement, so they only match when the baseline has the same length.
        fields = length(reference.time) == n + 1 ? map(-, result.u_field, reference.u_field) : result.u_field
        return merge(result, (; ux = result.ux .- prefix(reference.ux), uy = result.uy .- prefix(reference.uy),
            uz = result.uz .- prefix(reference.uz), u_field = fields))
    end
    # The source-free run comes first, so the response run can reuse it and the timing is per run.
    nonlinear && (seconds_baseline = @elapsed baseline(Δt, nsteps))
    seconds = @elapsed response = respond(pulse_history(Δt, nsteps; pulse_rate))
    @printf("response run: %d steps in %.0f s (%.1f s per step)%s; budget/source %.1e\n", nsteps, seconds,
        seconds / nsteps, nonlinear ? @sprintf(", baseline %.0f s", seconds_baseline) : "",
        maximum(abs, response.budget) / (pulse_rate * Δt))
    H = response.uz[:, 2:end] ./ pulse_rate
    Hx, Hy = response.ux[:, 2:end] ./ pulse_rate, response.uy[:, 2:end] ./ pulse_rate
    write_fields(joinpath(out_dir, @sprintf("impulse_day%04d.vtk", nsteps)), output_mesh, model, response, pulse_rate)
    first_step = respond(pulse_history(Δt, 1; pulse_rate))
    write_fields(joinpath(out_dir, "impulse_day0001.vtk"), output_mesh, model, first_step, pulse_rate)

    # Time-step check: a 2Δt pulse is two consecutive Δt pulses, so it is the two-step bin of H.
    ncoarse = nsteps ÷ 2
    coarse = respond(pulse_history(2Δt, ncoarse; pulse_rate))
    fine_binned = history_jacobian(H, [1:2])[:, 2:2:(2ncoarse), 1]
    time_step_error = relative_error(fine_binned, coarse.uz[:, 2:end] ./ pulse_rate)

    delay = ncheck ÷ 4
    delayed = respond(pulse_history(Δt, ncheck; start = delay + 1, pulse_rate))
    shift_error = relative_error(delayed.uz[:, (delay + 2):end] ./ pulse_rate, H[:, 1:(ncheck - delay)])
    rates = pulse_rate .* (2 .* rand(MersenneTwister(seed), ncheck) .- 0.5)
    signed = (; time_edges = collect(range(0.0, ncheck * Δt; length = ncheck + 1)),
        recharge_rate = max.(rates, 0.0), withdrawal_volume = max.(-rates, 0.0) .* Δt)
    mixed = respond(signed)
    superposition_error = relative_error(displacement_from_rates(H[:, 1:ncheck], rates), mixed.uz[:, 2:end])
    @printf("gates: time step (Δt vs 2Δt) %.2e   shift %.2e   superposition %.2e\n",
        time_step_error, shift_error, superposition_error)

    open(joinpath(out_dir, "impulse_response.csv"), "w") do io
        println(io, "station,east_m,north_m,step,time_s,uz_per_rate_s_per_m2,ux_per_rate_s_per_m2,uy_per_rate_s_per_m2")
        for k in 1:nsteps, s in eachindex(model.stations)
            println(io, join((model.station_ids[s], model.station_xy[s]..., k, k * Δt, H[s, k], Hx[s, k], Hy[s, k]), ','))
        end
    end
    open(joinpath(out_dir, "gates.csv"), "w") do io
        println(io, "check,relative_error")
        println(io, "time_step_dt_vs_2dt,", time_step_error)
        println(io, "shift_invariance,", shift_error)
        println(io, "superposition,", superposition_error)
    end
    summary = (; H, Hx, Hy, station_ids = model.station_ids, station_xy = model.station_xy, Δt, nsteps, pulse_rate,
        gates = (; time_step_error, shift_error, superposition_error), geometry = model.geometry,
        nels = model.mesh.nels, material = response.material, physics = response.physics,
        numerics = response.numerics, seconds)
    serialize(joinpath(out_dir, "impulse_response.jls"), summary)
    println("output: ", abspath(out_dir))
    return summary
end

const CASES = Dict(
    "reference" => (;),
    "elastic_shell" => (; η_shell = 1.0e23),
    # Superposition gate check (PLAN.md resume point 3): a tighter solver tolerance.
    "tight" => (; ϵ_tol = 1.0e-11),
)
const PHYSICS = (; gravity = true, advection = true)

function case_options(name)
    linear = startswith(name, "linear_")
    base = linear ? name[(length("linear_") + 1):end] : name
    return (; run_options = linear ? CASES[base] : merge(PHYSICS, CASES[base]))
end

impulse_response3D_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "..", "output", "impulse_response3D") : first(ARGS),
    nsteps = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 365, case_options(length(ARGS) > 2 ? ARGS[3] : "reference")...,
    backend = length(ARGS) > 3 && ARGS[4] == "cpu" ? CPU() : CUDABackend(),
    pulse_rate = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 1.0)
