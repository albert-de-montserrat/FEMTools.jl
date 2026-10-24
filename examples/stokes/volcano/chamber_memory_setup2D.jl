# Function-only, CPU plane-strain prototype shared by drivers, benchmarks, and tests.
# Reuses the previous dike/chamber mesher and FEMTools' finite-K Stokes/history paths.
module ChamberMemory2D

using FEMTools
using KernelAbstractions: CPU
using StaticArrays
using LinearAlgebra
using Random
using Serialization

include(joinpath(@__DIR__, "..", "..", "gmsh_meshing.jl"))
include(joinpath(@__DIR__, "..", "..", "reykjanes", "injection_source.jl"))
include(joinpath(@__DIR__, "elastic_dike2D.jl"))

const day = 86_400.0

"""
    build_chamber_memory_mesh(; domain, chamber_radius, shell_radius, center_depth, ...) -> NamedTuple

Build a fixed T7/P1-discontinuous plane-strain chamber/shell section, without solving.
Inputs and station positions are SI. Phase 1 is host, 2 magma, 3 shell. The top is
traction-free; side-normal and base-vertical velocities vanish. Repeated simulations
reuse this mesh but allocate fresh state, so pressure/stress cannot leak between runs.
This CPU prototype has no gravity, heat transport, yielding, or mesh advection.
"""
function build_chamber_memory_mesh(;
        domain = (40.0e3, 20.0e3), chamber_radius = 1.0e3, shell_radius = 2.0e3,
        center_depth = 4.0e3, max_area = 4.0e6, refinement = 8,
        surface_size = 1.0e3, station_x = (-8.0e3, -4.0e3, 0.0, 4.0e3, 8.0e3),
        workgroup = 128,
    )
    Lx, depth = domain
    all(x -> isfinite(x) && x > 0, (Lx, depth, chamber_radius, shell_radius,
        center_depth, max_area, refinement, surface_size)) ||
        throw(ArgumentError("geometry and mesh controls must be finite and positive"))
    chamber_radius < shell_radius || throw(ArgumentError("shell_radius must exceed chamber_radius"))
    !isempty(station_x) && all(x -> isfinite(x) && abs(x) < Lx / 2, station_x) ||
        throw(ArgumentError("stations must be nonempty, finite, and strictly inside the domain width"))
    # Fixed scales across time-step and material sweeps keep absolute residuals comparable.
    L_c, t_c, σ_c = 1.0e3, day, 1.0e7
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    coords, el2n, _, _ = build_gmsh_t7_concentric_disk_mesh(;
        Lx = Lx / L_c, Ly = depth / L_c, cx = Lx / (2L_c), cy = -center_depth / L_c,
        radii = (chamber_radius, shell_radius) ./ L_c, max_area = max_area / L_c^2,
        refinement, surface_size = surface_size / L_c,
        surface_xrange = (max(0.0, Lx / 2 - 3center_depth), min(Lx, Lx / 2 + 3center_depth)) ./ L_c,
    )
    coords = [c - SVector(Lx / (2L_c), 0.0) for c in coords]
    FEMTools.straighten_t7_geometry!(coords, el2n)
    mesh_v = Mesh(CPU(), coords, el2n, element_v; workgroup)
    mesh = MixedMesh(mesh_v, element_P; workgroup)
    cell_phase = map(axes(el2n, 2)) do e
        c = sum(coords[el2n[a, e]] for a in 1:3) / 3
        r = norm(c - SVector(0.0, -center_depth / L_c)) * L_c
        r < chamber_radius ? 2 : r < shell_radius ? 3 : 1
    end
    all(p -> p in cell_phase, 1:3) || throw(ArgumentError("mesh must resolve all three material regions"))
    phases_v = repeat(reshape(cell_phase, 1, :), length(element_v), 1)
    phases_P = repeat(reshape(cell_phase, 1, :), length(element_P), 1)
    tol = sqrt(eps(Float64)) * max(Lx, depth) / L_c
    surface = Int32[n for n in unique(vec(el2n[1:3, :])) if abs(coords[n][2]) < tol]
    sort!(surface; by = n -> coords[n][1])
    stations = [surface[argmin(abs.(getindex.(coords[surface], 1) .- x / L_c))] for x in station_x]
    allunique(stations) || throw(ArgumentError("stations map to the same surface node; refine or separate them"))
    sides = Int32[n for n in eachindex(coords) if abs(abs(coords[n][1]) - Lx / (2L_c)) < tol]
    base = Int32[n for n in eachindex(coords) if abs(coords[n][2] + depth / L_c) < tol]
    bc_vx = DirichletBoundaryCondition(nothing, sides, zeros(length(sides)))
    bc_vy = DirichletBoundaryCondition(nothing, base, zeros(length(base)))
    DoFsP = Array(mesh.DoFsP)
    geo_P = Array(mesh.geometry.geo_P)
    NqP = shape_function_values(element_P, element_v.integration_points)
    chamber_cells = findall(==(2), cell_phase)
    chamber_area = sum(sum(geo_P[e]) for e in chamber_cells) * L_c^2
    source_weights = uniform_pressure_source(1.0, DoFsP, geo_P, chamber_cells)
    return (; mesh, element_v, element_P, phases_v, phases_P, cell_phase, chamber_cells,
        DoFsP, geo_P, NqP, chamber_area, source_weights, bc_vx, bc_vy, stations,
        station_x = [coords[n][1] * L_c for n in stations], L_c, t_c, σ_c, workgroup,
        geometry = (; domain, chamber_radius, shell_radius, center_depth, max_area, refinement, surface_size))
end

"""
    chamber_cycle_history(; Δt=10day, cycle_lengths=(40day,50day,30day), ...) -> NamedTuple

Prescribe positive recharge [m²/s] and finite-duration withdrawal events [m²].
The returned step rates are exact interval averages of one physical schedule:
changing Δt does not change event duration or total withdrawal. Event boundaries
are inserted in the time grid. Withdrawals are positive amounts removed, separate
from recharge; a signed net source is passed to FEMTools. Cycle `j` withdraws
`withdrawal_fraction[j]` of the recharge it received; a scalar applies to every cycle.
"""
function chamber_cycle_history(; Δt = 10day, cycle_lengths = (40day, 50day, 30day),
        withdrawal_duration = 10day, Q = 1.0e-3, withdrawal_fraction = 0.75)
    all(x -> isfinite(x) && x > 0, (Δt, withdrawal_duration, Q, cycle_lengths...)) ||
        throw(ArgumentError("time controls, cycle lengths, and recharge must be finite and positive"))
    !isempty(cycle_lengths) && withdrawal_duration <= minimum(cycle_lengths) ||
        throw(ArgumentError("each cycle must contain its withdrawal interval"))
    fractions = withdrawal_fraction isa Number ? fill(withdrawal_fraction, length(cycle_lengths)) :
        collect(withdrawal_fraction)
    length(fractions) == length(cycle_lengths) ||
        throw(DimensionMismatch("withdrawal_fraction needs one entry per cycle or a single value"))
    all(f -> isfinite(f) && 0 <= f <= 1, fractions) ||
        throw(ArgumentError("withdrawal_fraction must lie in [0, 1]"))
    ends = cumsum(collect(cycle_lengths))
    starts = ends .- withdrawal_duration
    time_edges = sort!(unique!([collect(0.0:Δt:last(ends)); starts; ends]))
    recharge_rate = fill(Q, length(time_edges) - 1)
    withdrawal_area = [sum(eachindex(ends)) do j
        overlap = max(0.0, min(time_edges[n + 1], ends[j]) - max(time_edges[n], starts[j]))
        Q * cycle_lengths[j] * fractions[j] * overlap / withdrawal_duration
    end for n in eachindex(recharge_rate)]
    return (; time_edges, recharge_rate, withdrawal_area)
end

"""
    run_chamber_memory(model, history; η_shell=3e17, ...) -> NamedTuple

Advance the fixed-mesh, linear Maxwell chamber with finite elastic bulk storage.
History has `time_edges` [s], interval-average `recharge_rate` [m²/s], and positive
`withdrawal_area` [m²] removed in each interval. Returns station displacement [m]
and pressure perturbation [Pa] including their zero initial sample, the final-time
cumulative displacement at every mesh node (`ux_field`, `uz_field` [m], for
visualization; not tracked at earlier times), per-step chamber expansion and
compression-storage increments [m²], a signed continuity-budget residual [m²],
and solver statistics. Area changes integrate div(v) on the reference geometry;
they are not advected polygon areas. Stress and pressure are committed once after
an accepted solve, without corotation in this linear small-strain model.
"""
function run_chamber_memory(model, history; η_shell = 3.0e17, η_host = 1.0e23,
        η_magma = 1.0e14, G_host = 3.0e10, G_magma = 1.0e8,
        K_host = 5.0e10, K_magma = 1.0e10,
        ϵ_tol = 1.0e-10, iterMax = 100_000, verbose = false)
    (; time_edges, recharge_rate, withdrawal_area) = history
    nsteps = length(time_edges) - 1
    nsteps > 0 && first(time_edges) == 0 && all(isfinite, time_edges) && all(>(0), diff(time_edges)) ||
        throw(ArgumentError("time_edges must start at zero and increase strictly"))
    length(recharge_rate) == length(withdrawal_area) == nsteps ||
        throw(DimensionMismatch("one recharge rate and withdrawal amount are required per interval"))
    all(x -> isfinite(x) && x >= 0, recharge_rate) && all(x -> isfinite(x) && x >= 0, withdrawal_area) ||
        throw(ArgumentError("recharge and withdrawal must be finite and nonnegative"))
    all(x -> isfinite(x) && x > 0, (η_shell, η_host, η_magma, G_host, G_magma, K_host, K_magma, ϵ_tol)) ||
        throw(ArgumentError("material parameters and tolerance must be finite and positive"))
    iterMax > 0 || throw(ArgumentError("iterMax must be positive"))
    (; mesh, phases_v, phases_P, L_c, t_c, σ_c, workgroup, chamber_cells) = model
    η_c = σ_c * t_c
    material = StokesMaterial(;
        η = (η_host, η_magma, η_shell) ./ η_c,
        # finite_K explicitly selects K; ηb is retained only for compatible scaling paths.
        ηb = (K_host, K_magma, K_host) ./ σ_c,
        G = (G_host, G_magma, G_host) ./ σ_c,
        K = (K_host, K_magma, K_host) ./ σ_c,
        α = (0.0, 0.0, 0.0), ρ0 = (1.0, 1.0, 1.0), g = (0.0, 0.0), Tref = 0.0,
    )
    nq = length(model.element_v.integration_points.ω)
    dr = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP, material;
        stress_size = (nq, mesh.nels), CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9)
    γP = zeros(mesh.nnodesP)
    ux, uz = zeros(length(model.stations), nsteps + 1), zeros(length(model.stations), nsteps + 1)
    ux_field, uz_field = zeros(mesh.nnodes), zeros(mesh.nnodes)
    pressure = zeros(nsteps + 1)
    expansion, storage, budget = zeros(nsteps), zeros(nsteps), zeros(nsteps)
    net_area = recharge_rate .* diff(time_edges) .- withdrawal_area
    stats = NamedTuple[]
    ∂N∂ξ = shape_function_gradients(model.element_v)
    integrate_P(P) = pressure_source_integral(P, model.DoFsP, model.geo_P, model.NqP, chamber_cells)
    area_nd = model.chamber_area / L_c^2
    τ = FEMTools.stress(dr)
    τ_old = FEMTools.stress_old(dr)
    for n in 1:nsteps
        dt = time_edges[n + 1] - time_edges[n]
        Δτ = dt / t_c
        assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh, 100.0, Δτ; phases_v, workgroup)
        step_stats = solve_stokes_dyrel!(dr, mesh, model.bc_vx, model.bc_vy, Δτ, γP;
            phases_v, phases_P, finite_K = true, Qq = model.source_weights,
            Q2D = net_area[n] / L_c^2 / Δτ, ncheck = 50, ϵ_tol,
            iterMax, total_iterMax = iterMax, rel_drop0 = 0.1, verbose, workgroup)
        step_stats.converged || error("chamber step $n did not converge: $(step_stats.err_abs), $(step_stats.err_rel)")
        push!(stats, step_stats)
        divergence_integral = sum(chamber_cells) do e
            nodes = view(mesh.el2n, :, e)
            vx = SVector{7}(dr.v.x[nodes])
            vy = SVector{7}(dr.v.y[nodes])
            sum(element_geometry(mesh.geometry.geo_v, e, ∂N∂ξ)) do (B, dΩ)
                ((B' * vx)[1] + (B' * vy)[2]) * dΩ
            end
        end
        expansion[n] = divergence_integral * Δτ * L_c^2
        storage[n] = integrate_P(dr.P .- dr.P0) / dr.K[2] * L_c^2
        budget[n] = net_area[n] - expansion[n] - storage[n]
        pressure[n + 1] = integrate_P(dr.P) / area_nd * σ_c
        ux[:, n + 1] .= ux[:, n] .+ dr.v.x[model.stations] .* (L_c * Δτ)
        uz[:, n + 1] .= uz[:, n] .+ dr.v.y[model.stations] .* (L_c * Δτ)
        ux_field .+= dr.v.x .* (L_c * Δτ)
        uz_field .+= dr.v.y .* (L_c * Δτ)
        update_stokes_current_stress!(dr, mesh, τ, Δτ; phases_v, workgroup)
        foreach(copyto!, τ_old, τ)
        copyto!(dr.P0, dr.P)
    end
    return (; time = copy(time_edges), station_x = copy(model.station_x), ux, uz, pressure,
        ux_field, uz_field, recharge_area = recharge_rate .* diff(time_edges),
        withdrawal_area = copy(withdrawal_area),
        net_area, expansion, storage, budget, stats, nels = mesh.nels,
        chamber_area = model.chamber_area, geometry = model.geometry,
        material = (; η_shell, η_host, η_magma, G_host, G_magma, K_host, K_magma),
        numerics = (; ϵ_tol, iterMax))
end

"""
    chamber_elastic_reference(result) -> NamedTuple

Existing Mogi line-source response to the measured chamber expansion, in meters.
Compare only elastic-host runs. Finite source size and domain truncation mean this
is an approximate spatial oracle, not an exact chamber-pressure or volume law.
"""
function chamber_elastic_reference(result)
    (; G_host, K_host) = result.material
    ν = (3K_host - 2G_host) / (2 * (3K_host + G_host))
    areas = [0.0; cumsum(result.expansion)]
    depth = result.geometry.center_depth
    ux = [first(line_source_surface(x; area_change = area, depth, ν))
        for x in result.station_x, area in areas]
    uz = [last(line_source_surface(x; area_change = area, depth, ν))
        for x in result.station_x, area in areas]
    reference_norm = norm(ux)^2 + norm(uz)^2
    error = iszero(reference_norm) ? 0.0 :
        sqrt((norm(result.ux - ux)^2 + norm(result.uz - uz)^2) / reference_norm)
    return (; ux, uz, relative_error = error)
end

"""Write SI-valued budget and station-displacement CSVs plus a Julia result snapshot."""
function write_chamber_result(directory, name, result)
    mkpath(directory)
    serialize(joinpath(directory, name * ".jls"), result)
    open(joinpath(directory, name * "_budget.csv"), "w") do io
        println(io, "time_s,recharge_m2,withdrawal_m2,expansion_m2,storage_m2,budget_error_m2,pressure_Pa")
        for n in eachindex(result.net_area)
            println(io, join((result.time[n + 1], result.recharge_area[n], result.withdrawal_area[n],
                result.expansion[n], result.storage[n], result.budget[n], result.pressure[n + 1]), ','))
        end
    end
    open(joinpath(directory, name * "_stations.csv"), "w") do io
        println(io, "station_x_m,time_s,ux_m,uz_m")
        for n in eachindex(result.time), s in eachindex(result.station_x)
            println(io, join((result.station_x[s], result.time[n], result.ux[s, n], result.uz[s, n]), ','))
        end
    end
    return directory
end

const IDENTIFIABILITY_PARAMETERS = (:log_recharge_amplitude, :log_shell_viscosity,
    :log_withdrawal_amplitude, :log_magma_bulk_modulus, :log_recent_supply_ratio)

const DEFAULT_PRIOR_STD = (log_recharge_amplitude = 0.5, log_shell_viscosity = 1.0,
    log_withdrawal_amplitude = 0.5, log_magma_bulk_modulus = 0.5, log_recent_supply_ratio = 0.5)
const DEFAULT_TRUTH_FACTORS = (log_recharge_amplitude = 1.2, log_shell_viscosity = 1.5,
    log_withdrawal_amplitude = 1.3, log_magma_bulk_modulus = 0.8, log_recent_supply_ratio = 0.8)

"""
    pilot_setup(model, history, parameters, target, prior_std, truth_factors; recent_window, η_shell, K_magma, run...)

Validate a parameter selection and return `(; t, prior, θ_true, simulate)`: the
target's index, the selected prior σs, the log truth offsets, and
`simulate(θ)`, a fresh forward run at log offsets `θ` from nominal for the
selected `parameters`. Unselected parameters stay at nominal. The recent-supply
ratio scales recharge in the final `recent_window` by the exact overlap
fraction of each interval.
"""
function pilot_setup(model, history, parameters, target, prior_std, truth_factors;
        recent_window = 30day, η_shell = 3.0e17, K_magma = 1.0e10, run...)
    !isempty(parameters) && allunique(parameters) && all(in(IDENTIFIABILITY_PARAMETERS), parameters) ||
        throw(ArgumentError("parameters must be distinct entries of $IDENTIFIABILITY_PARAMETERS"))
    t = findfirst(==(target), parameters)
    isnothing(t) && throw(ArgumentError("target $target must be one of the selected parameters"))
    all(p -> haskey(prior_std, p) && haskey(truth_factors, p), parameters) ||
        throw(ArgumentError("prior_std and truth_factors need an entry for every selected parameter"))
    prior = [prior_std[p] for p in parameters]
    truth_factor = [truth_factors[p] for p in parameters]
    all(x -> isfinite(x) && x > 0, (η_shell, K_magma, prior..., truth_factor...)) ||
        throw(ArgumentError("material scales, truth factors, and priors must be finite and positive"))
    edges = history.time_edges
    :log_recent_supply_ratio in parameters && !(isfinite(recent_window) &&
        0 < recent_window <= last(edges) - first(edges)) &&
        throw(ArgumentError("recent_window must be positive and no longer than the history"))
    window_start = last(edges) - recent_window
    recent_fraction = [max(0.0, edges[n + 1] - max(edges[n], window_start)) / (edges[n + 1] - edges[n])
        for n in eachindex(history.recharge_rate)]
    function simulate(θ)
        factor(name) = (i = findfirst(==(name), parameters); isnothing(i) ? 1.0 : exp(θ[i]))
        recharge = history.recharge_rate .* factor(:log_recharge_amplitude) .*
            (1 .+ recent_fraction .* (factor(:log_recent_supply_ratio) - 1))
        return run_chamber_memory(model,
            merge(history, (; recharge_rate = recharge,
                withdrawal_area = history.withdrawal_area .* factor(:log_withdrawal_amplitude)));
            η_shell = η_shell * factor(:log_shell_viscosity),
            K_magma = K_magma * factor(:log_magma_bulk_modulus), run...)
    end
    return (; t, prior, θ_true = log.(truth_factor), simulate)
end

"""
    chamber_identifiability_pilot(model, history; parameters, target = :log_recharge_amplitude, ...) -> NamedTuple

Central full-history finite differences in the log-scale `parameters`, distinct
entries of `IDENTIFIABILITY_PARAMETERS`: recharge amplitude, shell viscosity,
withdrawal amplitude (one factor scaling every prescribed withdrawal event),
magma bulk modulus (inverse magma compressibility), and the recent-supply
ratio (recharge during the final `recent_window` relative to before; interval
rates use the exact overlap fraction). The default `parameters` are the first
four. Each run starts from a fresh mechanical state. Compare all epochs with
cycle-end sampling and vertical-only sampling. Uncertainty and noisy recovery
use a local Gaussian tangent, with an independently run nonlinear FEM truth;
`prior_std` and `truth_factors` are keyed by parameter name, and only the
selected entries are used. Parameters left out stay at their nominal value in
every run, including the truth, so they are treated as exactly known. Noise is
independent, equal-component displacement noise; initial state and source
geometry are known.

Per-row statistics (`σ_marginal`, `σ_fixed`, RMSE, coverage) refer to `target`;
"fixed" fixes every other selected parameter at nominal. `correlations` is the
sensitivity cosine between `target` and each of `nuisance_parameters`. The
`2length(parameters) + 2` forward runs are independent and go through
`mapper` (e.g. `Distributed.pmap`, with this module loaded on every worker).
The tangent error is reported; this is a local pilot, not full nonlinear
inference or network optimization.
"""
function chamber_identifiability_pilot(model, history;
        parameters = IDENTIFIABILITY_PARAMETERS[1:4], target = :log_recharge_amplitude,
        recent_window = 30day, η_shell = 3.0e17, K_magma = 1.0e10, fd_step = 0.02, noise = 0.003,
        prior_std = DEFAULT_PRIOR_STD, truth_factors = DEFAULT_TRUTH_FACTORS,
        samples = 100, seed = 2026, mapper = map, run...)
    all(x -> isfinite(x) && x > 0, (fd_step, noise)) ||
        throw(ArgumentError("fd_step and noise must be finite and positive"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    (; t, prior, θ_true, simulate) = pilot_setup(model, history, parameters, target, prior_std,
        truth_factors; recent_window, η_shell, K_magma, run...)
    nθ = length(parameters)
    unit(k, h) = [i == k ? h : 0.0 for i in 1:nθ]
    runs = mapper(simulate, [[zeros(nθ)]; [unit(k, fd_step) for k in 1:nθ];
        [unit(k, -fd_step) for k in 1:nθ]; [θ_true]])
    baseline, plus, minus, truth = runs[1], runs[2:(nθ + 1)], runs[(nθ + 2):(2nθ + 1)], runs[end]
    # A withdrawal is a contiguous block of positive removed areas; take its final epoch.
    event_epochs = [n + 1 for n in eachindex(history.withdrawal_area)
        if history.withdrawal_area[n] > 0 &&
            (n == length(history.withdrawal_area) || history.withdrawal_area[n + 1] == 0)]
    isempty(event_epochs) && throw(ArgumentError("cycle-end comparison needs withdrawal events"))
    others = [k for k in 1:nθ if k != t]
    experiments = ((name = "all_components_all_epochs", epochs = 2:length(baseline.time), vertical = false),
        (name = "all_components_event_ends", epochs = event_epochs, vertical = false),
        (name = "vertical_all_epochs", epochs = 2:length(baseline.time), vertical = true))
    rows = map(experiments) do experiment
        data(r) = experiment.vertical ? vec(r.uz[:, experiment.epochs]) :
            vcat(vec(r.ux[:, experiment.epochs]), vec(r.uz[:, experiment.epochs]))
        J = reduce(hcat, [(data(plus[k]) - data(minus[k])) / (2fd_step) for k in 1:nθ])
        ns, nt = size(baseline.uz, 1), length(baseline.time) - 1
        index = reshape(1:(ns * nt), ns, nt)
        positions = vec(index[:, collect(experiment.epochs) .- 1])
        noise_rows = experiment.vertical ? positions .+ ns * nt : [positions; positions .+ ns * nt]
        norms = norm.(eachcol(J))
        all(>(0), norms) || error("zero sensitivity: this experiment cannot compare all parameters")
        correlations = [dot(J[:, t], J[:, k]) / (norms[t] * norms[k]) for k in others]
        singular_values = svdvals(J * Diagonal(prior) / noise)
        precision = J' * J / noise^2 + Diagonal(1 ./ prior .^ 2)
        covariance = inv(Symmetric(precision))
        σ_marginal = sqrt(covariance[t, t])
        σ_fixed = inv(sqrt(precision[t, t]))
        signal = data(truth) - data(baseline)
        tangent_error = iszero(norm(signal)) ? 0.0 : norm(signal - J * θ_true) / norm(signal)
        rng = MersenneTwister(seed)
        errors_joint, errors_fixed = zeros(samples), zeros(samples)
        for i in 1:samples
            # Subsets see the same realization of the complete station/time record.
            observed = signal + noise * randn(rng, 2ns * nt)[noise_rows]
            estimate = covariance * (J' * observed / noise^2)
            # Every other selected parameter fixed at nominal despite the independently generated truth.
            fixed_estimate = dot(J[:, t], observed) / (noise^2 * precision[t, t])
            errors_joint[i] = estimate[t] - θ_true[t]
            errors_fixed[i] = fixed_estimate - θ_true[t]
        end
        (; experiment.name, ndata = length(signal), correlations, singular_values, J,
            σ_marginal, σ_fixed, tangent_error,
            rmse_joint = sqrt(sum(abs2, errors_joint) / samples),
            rmse_fixed = sqrt(sum(abs2, errors_fixed) / samples),
            coverage_joint = count(e -> abs(e) <= 1.96σ_marginal, errors_joint) / samples,
            coverage_fixed = count(e -> abs(e) <= 1.96σ_fixed, errors_fixed) / samples)
    end
    return (; rows, baseline, truth, target, parameters = Tuple(parameters),
        nuisance_parameters = Tuple(parameters[others]),
        settings = (; η_shell, K_magma, fd_step, noise, recent_window,
            prior_std = prior, truth_factors = exp.(θ_true), samples, seed))
end

"""
    chamber_gauss_newton(model, history; parameters, target = :log_recent_supply_ratio, ...) -> NamedTuple

Gauss–Newton MAP fit of the log-scale `parameters` to the noise-free FEM truth
(all stations, both components, all epochs), with a zero-mean Gaussian prior
`prior_std`. Each iteration re-linearizes at the current estimate with central
finite differences: `2length(parameters) + 1` independent forward runs through
`mapper`. Stops when the largest step falls below `step_tol` or after
`iterations`; `converged` says which.

Noisy recovery takes `samples` independent-noise realizations as one
fixed-Jacobian step from a linearization point, which is exact to first order
in the noise. It is evaluated twice with the same noise: at iteration 1 (the
one-step estimate linearized at nominal, as in `chamber_identifiability_pilot`)
and at the last linearization point, whose step produced the final iterate. Both report the target's bias, RMSE, and 95%
interval coverage. `posterior_σ` is every parameter's marginal σ at the last
linearization point. Unselected parameters stay nominal, as in the pilot.
"""
function chamber_gauss_newton(model, history;
        parameters = IDENTIFIABILITY_PARAMETERS, target = :log_recent_supply_ratio,
        recent_window = 30day, η_shell = 3.0e17, K_magma = 1.0e10, fd_step = 0.02, noise = 0.003,
        prior_std = DEFAULT_PRIOR_STD, truth_factors = DEFAULT_TRUTH_FACTORS,
        iterations = 4, step_tol = 1.0e-3, samples = 100, seed = 2026, mapper = map, run...)
    all(x -> isfinite(x) && x > 0, (fd_step, noise, step_tol)) ||
        throw(ArgumentError("fd_step, noise, and step_tol must be finite and positive"))
    iterations > 0 && samples > 0 || throw(ArgumentError("iterations and samples must be positive"))
    (; t, prior, θ_true, simulate) = pilot_setup(model, history, parameters, target, prior_std,
        truth_factors; recent_window, η_shell, K_magma, run...)
    nθ = length(parameters)
    data(r) = vcat(vec(r.ux[:, 2:end]), vec(r.uz[:, 2:end]))
    observed = data(only(mapper(simulate, [θ_true])))
    prior_precision = Diagonal(1 ./ prior .^ 2)
    unit(k, h) = [i == k ? h : 0.0 for i in 1:nθ]
    θ = zeros(nθ)
    steps = NamedTuple[]
    linearizations = NamedTuple[]
    converged = false
    for iteration in 1:iterations
        runs = mapper(simulate, [[θ]; [θ + unit(k, fd_step) for k in 1:nθ]; [θ - unit(k, fd_step) for k in 1:nθ]])
        predicted = data(runs[1])
        J = reduce(hcat, [(data(runs[1 + k]) - data(runs[1 + nθ + k])) / (2fd_step) for k in 1:nθ])
        precision = Symmetric(J' * J / noise^2 + prior_precision)
        push!(linearizations, (; θ = copy(θ), J, covariance = inv(precision)))
        Δ = precision \ (J' * (observed - predicted) / noise^2 - prior_precision * θ)
        θ += Δ
        push!(steps, (; iteration, max_step = maximum(abs, Δ),
            misfit = sqrt(sum(abs2, observed - predicted) / length(observed)) / noise, θ = copy(θ)))
        if maximum(abs, Δ) < step_tol
            converged = true
            break
        end
    end
    ndata = length(observed)
    # The first-order noisy estimate from a linearization point: noise-free estimate plus C J' ε / σ².
    function recovery(noise_free, lin)
        rng = MersenneTwister(seed)
        errors = [(noise_free + lin.covariance * (lin.J' * (noise * randn(rng, ndata))) / noise^2)[t] - θ_true[t]
            for _ in 1:samples]
        σ = sqrt(lin.covariance[t, t])
        return (; bias = noise_free[t] - θ_true[t], σ, rmse = sqrt(sum(abs2, errors) / samples),
            coverage = count(e -> abs(e) <= 1.96σ, errors) / samples)
    end
    return (; one_step = recovery(steps[1].θ, linearizations[1]), converged_fit = recovery(θ, last(linearizations)),
        θ, θ_true, posterior_σ = sqrt.(diag(last(linearizations).covariance)), converged, steps, target,
        parameters = Tuple(parameters),
        settings = (; η_shell, K_magma, fd_step, noise, recent_window, prior_std = prior,
            truth_factors = exp.(θ_true), iterations, step_tol, samples, seed))
end

end # module
