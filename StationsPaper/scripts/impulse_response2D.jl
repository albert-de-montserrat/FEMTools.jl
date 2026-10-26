# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. -t 8 StationsPaper/scripts/impulse_response2D.jl [out_dir] [nsteps] [case]
# Stage B/C of StationsPaper/PLAN.md: the station impulse response of the
# plane-strain chamber on the wide domain, and the checks that make it the
# exact Q-history Jacobian. With fixed mechanics and a uniform step, the model
# is linear and time-invariant in the net source rate, so one unit pulse gives
#   K[i, k, j] = ∂u_z(x_i, t_k) / ∂Q_j
# for every station, epoch, and source bin (see OBSERVATION_MODEL.md).
# Runs, in one process (threads help one large run on this machine):
#   1. a one-step pulse followed by `nsteps - 1` source-free steps (the response);
#   2. the same pulse with twice the step, to bound the time-discretization error;
#   3. a delayed pulse and 4. a random history over `ncheck` steps, compared
#      with shifted and superposed copies of run 1 (shift and superposition gates).
# Writes CSVs, a Julia snapshot, and VTK fields after the first step and at
# the end of the response run. Every case has gravity and a Lagrangian mesh
# (RESULTS.md result 15), so every response is the difference from a
# source-free run and the gates measure how far the model is from linear; a
# "linear_" prefix on the case name gives the fixed-mesh model without gravity.
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
include(joinpath(@__DIR__, "station_kernels.jl"))
using .ChamberMemory2D
using FEMTools: write_vtk
using LinearAlgebra
using Printf
using Random
using Serialization

const CM = ChamberMemory2D
const day = CM.day

"""Unit-pulse history: rate `pulse_rate` during step `start`, nothing otherwise, on a uniform grid."""
function pulse_history(Δt, nsteps; start = 1, pulse_rate = 1.0e-2)
    recharge_rate = [n == start ? pulse_rate : 0.0 for n in 1:nsteps]
    return (; time_edges = collect(range(0.0, nsteps * Δt; length = nsteps + 1)), recharge_rate,
        withdrawal_area = zeros(nsteps))
end

relative_error(a, b) = norm(a - b) / norm(b)

function write_fields(path, model, result, scale)
    coords = [c * model.L_c for c in Array(model.mesh.coords)]
    write_vtk(path, model.mesh; coords, title = "chamber impulse response, m per (m^2/s)",
        point_data = (; displacement = (result.ux_field ./ scale, result.uz_field ./ scale)),
        cell_data = (; phase = model.cell_phase))
    return path
end

function impulse_response_demo(; out_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D"),
        Δt = 1day, nsteps = 365, ncheck = 40, pulse_rate = 1.0e-2, seed = 2026,
        mesh_options = WIDE, run_options = (;))
    mkpath(out_dir)
    model = CM.build_chamber_memory_mesh(; mesh_options...)
    @printf("mesh: %d cells, %d stations at x = %.1f..%.1f km; Δt = %.2f days; %d threads\n",
        model.mesh.nels, length(model.station_x), extrema(model.station_x) ./ 1.0e3..., Δt / day, Threads.nthreads())

    # With gravity or advection the source-free state drifts slightly, so every
    # response is taken relative to a source-free run on the same time grid.
    nonlinear = get(run_options, :gravity, false) || get(run_options, :advection, false)
    function respond(history)
        result = CM.run_chamber_memory(model, history; run_options...)
        nonlinear || return result
        baseline = CM.run_chamber_memory(model, (; history.time_edges, recharge_rate = zero(history.recharge_rate),
            withdrawal_area = zero(history.withdrawal_area)); run_options...)
        return merge(result, NamedTuple(name => getfield(result, name) .- getfield(baseline, name)
            for name in (:ux, :uz, :ux_field, :uz_field)))
    end
    seconds = @elapsed response = respond(pulse_history(Δt, nsteps; pulse_rate))
    H = response.uz[:, 2:end] ./ pulse_rate
    Hx = response.ux[:, 2:end] ./ pulse_rate
    @printf("response run: %d steps in %.0f s; budget/source %.1e\n", nsteps, seconds,
        maximum(abs, response.budget) / (pulse_rate * Δt))
    write_fields(joinpath(out_dir, @sprintf("impulse_day%03d.vtk", nsteps)), model, response, pulse_rate)
    first_step = respond(pulse_history(Δt, 1; pulse_rate))
    write_fields(joinpath(out_dir, "impulse_day001.vtk"), model, first_step, pulse_rate)

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
        recharge_rate = max.(rates, 0.0), withdrawal_area = max.(-rates, 0.0) .* Δt)
    mixed = respond(signed)
    superposition_error = relative_error(displacement_from_rates(H[:, 1:ncheck], rates), mixed.uz[:, 2:end])
    @printf("gates: time step (Δt vs 2Δt) %.2e   shift %.2e   superposition %.2e\n",
        time_step_error, shift_error, superposition_error)

    open(joinpath(out_dir, "impulse_response.csv"), "w") do io
        println(io, "station_x_m,step,time_s,uz_per_rate_s_per_m,ux_per_rate_s_per_m")
        for k in 1:nsteps, s in eachindex(model.station_x)
            println(io, join((model.station_x[s], k, k * Δt, H[s, k], Hx[s, k]), ','))
        end
    end
    open(joinpath(out_dir, "gates.csv"), "w") do io
        println(io, "check,relative_error")
        println(io, "time_step_dt_vs_2dt,", time_step_error)
        println(io, "shift_invariance,", shift_error)
        println(io, "superposition,", superposition_error)
    end
    summary = (; H, Hx, station_x = copy(model.station_x), Δt, nsteps, pulse_rate,
        gates = (; time_step_error, shift_error, superposition_error),
        geometry = model.geometry, nels = model.mesh.nels, material = response.material,
        numerics = response.numerics, seconds)
    serialize(joinpath(out_dir, "impulse_response.jls"), summary)
    println("output: ", abspath(out_dir))
    return summary
end

# Mechanics variants for the separability check (RESULTS.md), selected by the optional third argument.
const STATION_X = Tuple([collect(0.0:500.0:12.0e3); [14.0e3, 16.0e3, 18.0e3, 20.0e3]])
const WIDE = (; domain = (640.0e3, 320.0e3), far_growth = 0.2, station_x = STATION_X)
const CASES = Dict(
    "reference" => (; mesh_options = WIDE, run_options = (;)),
    "shallow" => (; mesh_options = merge(WIDE, (; center_depth = 3.0e3)), run_options = (;)),
    "thick_shell" => (; mesh_options = merge(WIDE, (; shell_radius = 3.0e3)), run_options = (;)),
    "soft_shell" => (; mesh_options = WIDE, run_options = (; η_shell = 1.0e17)),
    # Thermal aureole: viscosity rising log-linearly from 1e17 Pa s at the wall to 1e20 Pa s
    # at 3 km, in eight bands, so Maxwell times span 39 days to over a century.
    "aureole" => (; mesh_options = merge(WIDE, (; shell_radius = 3.0e3, shell_bands = 8)),
        run_options = (; η_shell = Tuple(10.0 .^ range(17.0, 20.0; length = 8)))),
)

# Reference physics (RESULTS.md result 15): gravity and a Lagrangian mesh.
const PHYSICS = (; gravity = true, advection = true)

# "eta_<value>" sets the shell viscosity [Pa s] on the reference geometry, for the viscosity sweep.
# A "linear_" prefix runs the case on the fixed mesh without gravity.
function case_options(name)
    linear = startswith(name, "linear_")
    base = linear ? name[(length("linear_") + 1):end] : name
    options = startswith(base, "eta_") ?
        (; mesh_options = WIDE, run_options = (; η_shell = parse(Float64, base[5:end]))) : CASES[base]
    return (; options.mesh_options, run_options = linear ? options.run_options : merge(PHYSICS, options.run_options))
end

impulse_response_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "..", "output", "impulse_response2D") : first(ARGS),
    nsteps = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 365, case_options(length(ARGS) > 2 ? ARGS[3] : "reference")...)
