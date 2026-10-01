# The recharge-and-intrusion cycle of GAP-11 of REYKJANES_PLAN.md, on the machinery of
# `event_stepping.jl` and the rules of `event_protocol.jl`. It runs nothing.
#
# One cycle is: recharge the reservoir in transactional steps until the protocol's detector trips;
# go back to the last state before it tripped and bracket the step at which it first does; take the
# path the detector found; and open a dike on that path by the smallest amount that drives the band's
# closure traction to the closure stress of the path plus the arrest overpressure, balancing the
# injection with a sink in the reservoir. Every search restarts from one accepted state.
#
# Steps 4 and 5 of the plan's protocol — the contact rule, the enthalpy of the transferred magma and
# the full per-event record — are not here: they need the thermal work of M3, which the plan puts
# after the null test of M2 on purpose.

using Printf
using Statistics
using FEMTools

include(joinpath(@__DIR__, "reykjanes_setup.jl"))
include(joinpath(@__DIR__, "event_protocol.jl"))
include(joinpath(@__DIR__, "cycle_output.jl"))

"""
    run_cycles(; nevents=1, max_steps=40, recharge_rate=2e-7, protocol=EventProtocol(), kwargs...) -> NamedTuple

Recharge a Reykjanes-like section until the protocol declares an event, refine the crossing, open a
dike on the failed path, and repeat until `nevents` events or `max_steps` accepted steps.

`recharge_rate` is the magma supply in m² s⁻¹ per unit strike length, spread over the sill with the
quadrature of the continuity residual. `Δt` is the recharge step, retried smaller when a step is
rejected. Keywords that are not listed go to `build_reykjanes_model`.

`spinup_steps` steps of tectonic loading alone run first, so that the reservoir pressure the
thresholds are measured from, `P_baseline`, is an equilibrated state and not the lithostatic guess
the model is warm-started with. The clock starts after them.

Returns the `protocol` it used, the `events` it recorded, the per-step `history`, `P_baseline`, and
the `model`, so that a caller can go on inspecting the state it stopped in. Each event records the
step it was found in, the bracket of the crossing in physical time, the reservoir pressure and its
rise over the baseline (`ΔP_crit`), the path, the closure stress of the path, the opening the
amplitude search settled on and the value it reached, and the outcome of each search. An event whose
crossing or amplitude search did not resolve is recorded with that outcome and stops the run: a
threshold that is not resolved is not a threshold.

`write_output = true` writes a VTK snapshot into `out_dir` after the spin-up, after every accepted
step, and on both sides of every event, together with an `index.txt` saying which file is which. The
snapshots carry the detector's own fields — the failed fraction of each element, the hydraulic
margin, the fixed corridor and the fill of each of its bins, and the band the dike opened — so the
threshold in the log can be looked at instead of taken on trust. See
[`build_cycle_writer`](@ref). It costs one postprocess and one file per snapshot, so it is off by
default in sweeps.
"""
function run_cycles(;
        nevents = 1,
        max_steps = 40,
        spinup_steps = 1,
        recharge_rate = 2.0e-7,
        protocol::EventProtocol = EventProtocol(),
        Δt = 1kyr,
        verbose = true,
        write_output = false,
        out_dir = joinpath(@__DIR__, "output_cycles"),
        backend = CPU(), workgroup = 128,
        model_kwargs...,
    )
    recharge_rate > 0 || throw(ArgumentError("run_cycles needs a positive recharge rate"))
    model = build_reykjanes_model(; Δt, backend, workgroup, model_kwargs...)
    (; L_c, t_c, σ_c, Δτ, Tref, G, cell_phase, coords_cpu, groups) = model
    (; mesh_v, mesh_stokes, el2n_v_cpu, DoFsP_cpu, NqP_cpu, phases_v, phases_P) = model
    (; dr, τ, τ_old, thermal, bc_vx, bc_vy, bc_T, γP, update_pressure_scaling!, plastic) = model

    detector = build_event_detector(model, protocol)
    write_snapshot = write_output ?
        build_cycle_writer(model, detector, protocol; out_dir) : (args...; kwargs...) -> nothing
    geo_P_cpu = Array(mesh_stokes.geometry.geo_P)
    sill_cells = findall(==(2), cell_phase)
    element_G = [G[cell_phase[iel]] for iel in 1:mesh_stokes.nels]
    Q_recharge = uniform_pressure_source(
        recharge_rate * t_c / L_c^2, DoFsP_cpu, geo_P_cpu, sill_cells,
    )
    Δτ_intrusion = protocol.intrusion_Δt / t_c

    solve!(Δτ_step) = solve_coupled_dyrel!(
        thermal, dr, mesh_v, mesh_stokes, bc_T, bc_vx, bc_vy, Δτ_step, γP;
        Tref, phases_v, phases_P, τ_old, plastic, workgroup,
        ncheck = 100, ϵ_tol = 1.0e-3, iterMax = 50_000, total_iterMax = 75_000, rel_drop0 = 1.0e-2,
        verbose = false, verbose_inner = false,
    )
    # An accepted step commits its own history, so that a restored trial undoes it with the rest.
    function commit_step!(Δτ_step)
        stats = solve!(Δτ_step)
        update_stokes_current_stress!(
            dr, mesh_stokes, τ, Δτ_step; phases_v, τ_old, plastic, workgroup,
        )
        foreach(copyto!, τ_old, τ)
        return stats
    end

    function recharge_step!(Δτ_step)
        copyto!(dr.P0, dr.P)
        copyto!(thermal.T0, thermal.T)
        copyto!(dr.Q, Q_recharge)
        update_pressure_scaling!(Δτ_step)
        return commit_step!(Δτ_step)
    end

    # The intrusion: the eigenstrain of GAP-11 on the failed path, drained from the reservoir, over a
    # step short against the Maxwell time so the response is elastic.
    function intrusion_step!(band_cells)
        return function (opening, Δτ_step)
            copyto!(dr.P0, dr.P)
            copyto!(thermal.T0, thermal.T)
            τ_old_cpu = map(Array, τ_old)
            Q_cpu = zeros(mesh_stokes.nnodesP)
            apply_dike_opening!(
                τ_old_cpu, Q_cpu, DoFsP_cpu, element_G, band_cells,
                opening / L_c, protocol.band_width / L_c, protocol.dike_normal, Δτ_step,
            )
            balance_dike_source!(Q_cpu, DoFsP_cpu, geo_P_cpu, NqP_cpu, band_cells, sill_cells)
            foreach(copyto!, τ_old, τ_old_cpu)
            copyto!(dr.Q, Q_cpu)
            update_pressure_scaling!(Δτ_step)
            return commit_step!(Δτ_step)
        end
    end

    # What the amplitude search watches. `solve_amplitude!` searches for a quantity that grows with
    # the opening, so the draining reservoir is watched through its negative, and its target with it.
    function arrest_observation(band_cells)
        if protocol.arrest_target === :band_traction
            return function (_, _)
                τ_nn = normal_deviatoric_stress(map(x -> Array(x) .* σ_c, τ), protocol.dike_normal)
                state = detector.detect()
                return band_closure_traction(τ_nn, state.P_ip, detector.areas, band_cells)
            end
        end
        return function (_, _)
            state = detector.detect()
            return -magma_pressure(state.P_reservoir, detector.head, detector.areas, band_cells)
        end
    end
    arrest_sign = protocol.arrest_target === :band_traction ? 1 : -1

    objects = (; dr, thermal, γP)
    events = NamedTuple[]
    history = NamedTuple[]
    time = 0.0
    Δτ_step = Δτ
    stopped = :max_steps

    # Spin-up: steps under tectonic loading alone, so that the threshold is measured from an
    # equilibrated state and not from the lithostatic guess the model is warm-started with. The
    # clock starts after them.
    spinup_history = NamedTuple[]
    for ispin in 1:spinup_steps
        report = attempt_step!(
            Δτ_spin -> (
                copyto!(dr.P0, dr.P); copyto!(thermal.T0, thermal.T);
                fill!(dr.Q, 0); update_pressure_scaling!(Δτ_spin); commit_step!(Δτ_spin)
            ),
            objects; Δτ = Δτ_step, max_attempts = protocol.max_step_attempts, Δτ_min = Δτ / 64,
        )
        if !report.accepted
            verbose && @warn "spin-up step rejected" step_failure_message(report)
            return (;
                protocol, events, history, stopped = report.outcome, model, detector, time,
                P_baseline = NaN, spinup_steps, spinup_history,
            )
        end
        # The baseline every threshold is measured against is whatever this loop leaves behind, so
        # its own trajectory is evidence: a state still moving after the last step is not a state to
        # measure a pressure rise from.
        P_spin = mean(Array(dr.P)[model.sill_P_dofs]) * σ_c
        push!(spinup_history, (; step = ispin, P_reservoir = P_spin, Δt = report.Δτ * t_c))
        verbose && @printf(
            "spin-up step %2d  P_res = %8.4f MPa%s\n", ispin, P_spin / 1.0e6,
            ispin == 1 ? "" : @sprintf("   Δ = %+7.4f MPa", (P_spin - spinup_history[end - 1].P_reservoir) / 1.0e6),
        )
    end
    baseline = detector.detect()
    P_baseline = baseline.P_reservoir
    write_snapshot("spinup", baseline; time = 0.0)
    verbose && @printf(
        "spin-up %d × %.3f kyr  P_res = %7.2f MPa  failed %4d  path %3d  corridor %.2f (%d/%d bins, %d els)\n",
        spinup_steps, Δτ_step * t_c / kyr, P_baseline / 1.0e6,
        count(baseline.failed), length(baseline.path), baseline.fraction,
        count(>=(protocol.corridor_fill), baseline.fractions), length(baseline.fractions),
        length(detector.corridor.cells),
    )

    for istep in 1:max_steps
        # The state this step starts from, which a crossing search returns to.
        trials = capture_trials(objects; rebuild! = () -> update_pressure_scaling!(Δτ_step))
        report = attempt_step!(
            recharge_step!, objects;
            Δτ = Δτ_step, max_attempts = protocol.max_step_attempts, Δτ_min = Δτ / 64,
        )
        if !report.accepted
            stopped = report.outcome
            verbose && @warn "step rejected" istep step_failure_message(report)
            break
        end
        Δτ_step = report.Δτ
        time += Δτ_step * t_c
        state = detector.detect()
        push!(history, (;
            istep, time, Δt = Δτ_step * t_c, state.P_reservoir,
            failed = count(state.failed), component = length(state.component),
            path = length(state.path), fraction = state.fraction,
            filled = count(>=(protocol.corridor_fill), state.fractions), attempts = report.attempts,
        ))
        verbose && @printf(
            "step %3d  t = %7.2f kyr  Δt = %6.3f kyr  P_res = %7.2f MPa  failed %4d  component %4d  path %3d  corridor %.2f (%d/%d bins)\n",
            istep, time / kyr, Δτ_step * t_c / kyr, state.P_reservoir / 1.0e6,
            count(state.failed), length(state.component), length(state.path), state.fraction,
            count(>=(protocol.corridor_fill), state.fractions), length(state.fractions),
        )
        write_snapshot("step", state; time)
        state.tripped || continue

        # An event. Go back before the step and find where the criterion first trips.
        reset_trial!(trials)
        crossing = bracket_crossing!(
            (_, _) -> detector.detect().tripped, trials, recharge_step!;
            Δτ = Δτ_step, samples = protocol.crossing_samples,
            tolerance = protocol.crossing_tolerance * Δτ_step, max_trials = protocol.max_trials,
        )
        if crossing.outcome !== :crossing_bracketed
            push!(events, (; istep, outcome = crossing.outcome, crossing, intrusion = nothing))
            stopped = crossing.outcome
            verbose && @warn "the crossing did not resolve" istep crossing.outcome
            break
        end
        time = time - Δτ_step * t_c + crossing.hi * t_c
        at_crossing = detector.detect()
        path = at_crossing.path
        s_n = closure_stress(at_crossing.diagnostics.s3, detector.areas, path)
        target = s_n + protocol.arrest_overpressure

        # The intrusion, from the state in which the criterion has just tripped.
        intrusion_trials = capture_trials(objects; rebuild! = () -> update_pressure_scaling!(Δτ_step))
        path_length = sqrt(sum(detector.areas[iel] for iel in path))
        overpressure = protocol.arrest_target === :band_traction ?
            protocol.arrest_overpressure :
            magma_pressure(at_crossing.P_reservoir, detector.head, detector.areas, path) - target
        write_snapshot("event_pre", at_crossing; band = path, time)
        intrusion = solve_amplitude!(
            arrest_observation(path), intrusion_trials, intrusion_step!(path);
            Δτ = Δτ_intrusion, target = arrest_sign * target,
            # The opening a crack of this length needs to change its pressure by that much.
            amplitude = clamp(
                2 * abs(overpressure) * path_length / (G[1] * σ_c), 1.0e-3, protocol.max_opening,
            ),
            amplitude_max = protocol.max_opening, rtol = protocol.amplitude_rtol,
            max_trials = protocol.max_trials,
        )
        push!(events, (;
            istep, outcome = intrusion.outcome, time, crossing_lo = crossing.lo * t_c,
            crossing_hi = crossing.hi * t_c, at_crossing.P_reservoir, path = collect(path),
            path_length, closure_stress = s_n, target, opening = intrusion.amplitude,
            detector_rule = protocol.detector, fractions = copy(at_crossing.fractions),
            at_crossing.fraction,
            ΔP_crit = at_crossing.P_reservoir - P_baseline,
            reached = intrusion.observation === nothing ? nothing : arrest_sign * intrusion.observation,
            crossing, intrusion,
        ))
        verbose && @printf(
            "  event %d at %.3f kyr  path %3d els (%.0f m)  s_n = %6.2f MPa  target = %6.2f MPa  opening = %.4f m  reached = %s  [%s]\n",
            length(events), time / kyr, length(path), path_length, s_n / 1.0e6, target / 1.0e6,
            intrusion.amplitude,
            intrusion.observation === nothing ? "-" :
                string(round(arrest_sign * intrusion.observation / 1.0e6; digits = 2), " MPa"),
            intrusion.outcome,
        )
        write_snapshot("event_post", detector.detect(); band = path, time)
        if intrusion.outcome !== :amplitude_bracketed
            stopped = intrusion.outcome
            break
        end
        time += protocol.intrusion_Δt
        length(events) >= nevents && (stopped = :nevents; break)
    end

    return (;
        protocol, events, history, stopped, model, detector, time, P_baseline, spinup_steps,
        spinup_history,
    )
end
