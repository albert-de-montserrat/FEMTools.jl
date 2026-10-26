# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/plastic_forward2D.jl case history [out_dir] [nsteps]
# Forward truths for plastic_misspecification2D.jl: the reference chamber
# (gravity + Lagrangian mesh, Maxwell shell, wide domain of impulse_response2D.jl)
# made Drucker-Prager plastic with a tensile cap, driven by one supply history
# of the Stage D suite over 720 pre-window plus 365 window days. Plasticity
# makes the model nonlinear, so each history needs its own run instead of the
# impulse response. `case` is one of the keys of CASES; `history` is "none"
# (source-free baseline), "H0", "H1", or "H6". Q_ref comes from the gravity
# impulse response, as in mogi_comparison2D.jl. An optional `nsteps` truncates
# the run (smoke tests). Writes <case>_<history>.jls with station displacements
# and plastic-strain diagnostics.
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
include(joinpath(@__DIR__, "synthetic_recovery2D.jl"))
using .ChamberMemory2D
using Printf
using Serialization

const CM = ChamberMemory2D

const STATION_X = Tuple([collect(0.0:500.0:12.0e3); [14.0e3, 16.0e3, 18.0e3, 20.0e3]])
const WIDE = (; domain = (640.0e3, 320.0e3), far_growth = 0.2, station_x = STATION_X)
# Crust strength of dike_injection2D.jl; η_vp near G Δt for a 1-day step, so the
# plastic return acts without overshooting.
const STRENGTH = (; friction = 30.0, dilation = 0.0, cohesion = 1.0e7, tensile_strength = -5.0e6, η_vp = 3.0e15)
const CASES = Dict(
    "reference" => nothing,
    "tensile_dry" => merge(STRENGTH, (; λ_fluid = 0.0)),
    "tensile_hydrostatic" => merge(STRENGTH, (; λ_fluid = 0.37)),
    "tensile_lithostatic" => merge(STRENGTH, (; λ_fluid = 1.0)),
)

function forward_histories(prewindow, window; stop_day = 180)
    suite = Dict(first(split(name)) => q for (name, q) in synthetic_histories(prewindow, window))
    return Dict("none" => zeros(prewindow + window), "H0" => suite["H0"], "H1" => suite["H1"],
        "H6" => [day <= prewindow + stop_day ? 1.0 : 0.0 for day in 1:(prewindow + window)])
end

function plastic_forward_demo(case, history; out_dir = joinpath(@__DIR__, "..", "output", "plastic_forward2D"),
        impulse_dir = joinpath(@__DIR__, "..", "output", "impulse_response2D_3yr"),
        window = 365, prewindow = 720, uplift = 0.24, nsteps = prewindow + window)
    mkpath(out_dir)
    reference = deserialize(joinpath(impulse_dir, "impulse_response.jls"))
    Q_ref = uplift / sum(reference.H[findfirst(==(0.0), reference.station_x), 1:window])
    q = forward_histories(prewindow, window)[history][1:nsteps]
    model = CM.build_chamber_memory_mesh(; WIDE...)
    plasticity = CASES[case]
    run = (; time_edges = collect(range(0.0, nsteps * CM.day; length = nsteps + 1)), recharge_rate = Q_ref .* q,
        withdrawal_area = zeros(nsteps))
    # A looser tolerance than the impulse runs: plastic steps stall just above 1e-10.
    seconds = @elapsed result = CM.run_chamber_memory(model, run; gravity = true, advection = true, plasticity,
        ϵ_tol = 1.0e-9, iterMax = 300_000)
    @printf("%s %s: %d steps in %.0f s; max plastic γ %.2e, θ %.2e; uz(x=0) end %.4f m\n", case, history, nsteps,
        seconds, maximum(result.plastic_γ; init = 0.0), maximum(result.plastic_θ; init = 0.0), result.uz[1, end])
    path = joinpath(out_dir, "$(case)_$(history).jls")
    serialize(path, (; case, history, plasticity, Q_ref, q, prewindow, window, station_x = result.station_x,
        uz = result.uz, ux = result.ux, pressure = result.pressure, plastic_γ = result.plastic_γ,
        plastic_θ = result.plastic_θ, iterations = [s.iter for s in result.stats], seconds))
    println("output: ", abspath(path))
    return path
end

plastic_forward_demo(ARGS[1], ARGS[2]; out_dir = length(ARGS) > 2 ? ARGS[3] : joinpath(@__DIR__, "..", "output", "plastic_forward2D"),
    (length(ARGS) > 3 ? (; nsteps = parse(Int, ARGS[4])) : (;))...)
