#
# First-threshold refinement study: is `ΔP_crit^(1)` a property of the model, or of the discretisation
# and the protocol?
#
# Gate G1 of REYKJANES_PLAN.md asks for the first failure threshold to change by less than 5% under
# refinement, and the plan's error budget asks for the numerical uncertainty to be small against the
# smallest memory signal worth reporting. Nothing downstream — repeated intrusion, `M_n`, the
# factorial arms — means anything until that holds, so this sweep runs `run_cycles` to its *first*
# event only (`examples/reykjanes/event_cycles.jl`) and reports, per run:
#
#   - the threshold: the reservoir pressure at the crossing and its rise over the spun-up baseline,
#     `ΔP_crit`, which is the quantity the gate is about;
#   - when it happened, and how wide the bracket around it was left;
#   - what failed: the length of the path, its closure stress, and the opening the dike needed;
#   - whether every search resolved. A run whose crossing or amplitude search did not resolve has no
#     threshold, and its row must not be read as one.
#
# The sweeps separate the two kinds of cause deliberately, because they are not the same question:
#
#   - `mesh`, `time_step`, `crossing`, `solver` and `domain` are *numerical*. `ΔP_crit` must not
#     depend on them. This is the gate.
#   - `strength`, `reach`, `arrest` and `band` are *protocol*. `ΔP_crit` is expected to depend on
#     them, and by how much is the sensitivity that has to be reported with any threshold, because
#     none of these is fixed by data yet.
#
# Each run is a handful of coupled solves plus two bracketed searches, so a full sweep is tens of
# minutes. Run one family at a time.
#
#     julia --project=examples
#     include("examples/benchmarks/stokes/first_threshold/first_threshold.jl")
#     rows = run_first_threshold_benchmark(; sweeps = (:mesh,))
#     rows = run_first_threshold_benchmark()
#

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "reykjanes", "event_cycles.jl"))

"""
    first_threshold_sweeps() -> NamedTuple

The settings of each sweep, as keyword arguments of `run_cycles`.

Numerical: `mesh` (element size and sill refinement together), `sill` (the sill refinement alone,
at fixed element size, which is where the event pressure actually moves), `time_step` (the recharge
step, which also sets how far the crossing search has to reach back), `crossing` (the samples and
the bracket width of the crossing search), `solver` (`spinup_steps`, which changes the state the
threshold is measured from), `baseline` (that spin-up crossed with the sill refinement, which is how
the drift of the baseline was separated from the mesh) and `domain` (the width and depth of the
section, which is independent of mesh refinement).

Protocol: `strength` (`T₀`), `reach` (how close to the surface the failure must come), `arrest` (the
arrest overpressure), `band` (the numerical band width of the dike), `detector` (which of the three
rules of Gate G1 declares the event) and `corridor` (the geometry and fill threshold of the
mesh-independent corridor). Their spread is sensitivity to be quoted, not convergence error.
"""
function first_threshold_sweeps()
    return (;
        mesh = [
            (; max_area, refinement) for (max_area, refinement) in (
                (1.6e7, 3), (8.0e6, 3), (4.0e6, 4), (2.0e6, 4),
            )
        ],
        # The recharge step and the spin-up are not independent: `spinup_steps` steps of `Δt` is
        # `spinup_steps * Δt` of spin-up, and the `baseline` family showed the baseline is still
        # drifting there. Halving `Δt` at fixed `spinup_steps` therefore halves the spin-up and moves
        # the state the threshold is measured from, which is not a time-step error. This family holds
        # the spun-up *time* at 2 kyr and refines the step inside it, so a spread is the step alone.
        time_step = [
            (; Δt, spinup_steps) for (Δt, spinup_steps) in
                ((2kyr, 1), (1kyr, 2), (0.5kyr, 4), (0.25kyr, 8))
        ],
        # The `mesh` family moves the element size and the sill boundary discretisation together, so
        # it cannot say which of them a spread belongs to. This one moves the sill alone: the
        # elliptical reservoir is a polygon, and its refinement sets how well the geometry the
        # reservoir pressure is read on is resolved, at almost no cost in elements.
        sill = [(; max_area = 8.0e6, refinement) for refinement in (3, 4, 6, 8)],
        crossing = [
            (; protocol = EventProtocol(; crossing_samples, crossing_tolerance))
                for (crossing_samples, crossing_tolerance) in ((2, 0.05), (4, 0.02), (8, 0.005))
        ],
        solver = [(; spinup_steps) for spinup_steps in (1, 2, 4)],
        domain = [
            (; domain) for domain in
                ((40.0e3, 20.0e3), (60.0e3, 20.0e3), (40.0e3, 30.0e3), (80.0e3, 40.0e3))
        ],
        strength = [
            (; protocol = EventProtocol(; tensile_strength)) for tensile_strength in
                (2.5e6, 5.0e6, 7.5e6, 1.0e7)
        ],
        # The corridor runs from the reservoir crest, 4 km down in this section, to the target
        # depth, so a reach depth of 4 km would leave it no height at all. That is an error, not a
        # row: the sweep stays above the crest.
        reach = [(; protocol = EventProtocol(; reach_depth)) for reach_depth in (1.0e3, 2.0e3, 3.0e3)],
        arrest = [
            (; protocol = EventProtocol(; arrest_overpressure)) for arrest_overpressure in
                (5.0e5, 1.0e6, 2.0e6)
        ],
        band = [(; protocol = EventProtocol(; band_width)) for band_width in (50.0, 100.0, 200.0)],
        detector = [
            (; protocol = EventProtocol(; detector)) for detector in
                (:corridor_column, :corridor_fraction, :graph_path)
        ],
        # Gate G1s residual is the conditioning of the threshold, not the detector: across the mesh
        # sweep the event pressure moves 0.48% and the spun-up baseline 1.56%, while their difference
        # moves 45.6%. This family crosses the spin-up with the sill refinement the jump belongs to,
        # so `P_baseline` can be read against `spinup_steps` at fixed refinement and against
        # refinement at fixed spin-up. Its spread line mixes both and means nothing; read the rows.
        baseline = vec(
            [
                (; max_area, refinement, spinup_steps)
                    for spinup_steps in (1, 2, 4), (max_area, refinement) in ((1.6e7, 3), (4.0e6, 4))
            ]
        ),
        corridor = [
            (; protocol = EventProtocol(; corridor_width, corridor_bins, corridor_fill))
                for (corridor_width, corridor_bins, corridor_fill) in (
                    (1.0e3, 4, 0.5), (2.0e3, 4, 0.5), (4.0e3, 4, 0.5),
                    (2.0e3, 2, 0.5), (2.0e3, 8, 0.5),
                    (2.0e3, 4, 0.25), (2.0e3, 4, 0.75),
                )
        ],
    )
end

"""
    first_threshold_case(; kwargs...) -> NamedTuple

Run `run_cycles` to its first event with the given keyword arguments and reduce it to the scalars the
study tracks, plus the wall time in seconds.

A run that throws, or that reaches `max_steps` without an event, yields a row with `missing` numbers
and its `outcome`, so a sweep records it and continues. `resolved` is true only when the event was
reached and both of its searches resolved.
"""
function first_threshold_case(; max_steps = 12, recharge_rate = 2.0e-7, kwargs...)
    return try
        time = @elapsed result = run_cycles(;
            nevents = 1, max_steps, recharge_rate, verbose = false, kwargs...,
        )
        event = isempty(result.events) ? nothing : first(result.events)
        if event === nothing || !haskey(event, :ΔP_crit)
            (;
                nels = result.model.mesh_stokes.nels, ΔP_crit = missing, P_event = missing,
                t_event = missing, bracket = missing, path = missing, path_length = missing,
                closure = missing, fraction = missing, filled = missing, opening = missing,
                P_baseline = result.P_baseline,
                steps = length(result.history),
                outcome = result.stopped, resolved = false, time,
            )
        else
            (;
                nels = result.model.mesh_stokes.nels, event.ΔP_crit, P_event = event.P_reservoir,
                t_event = event.time, bracket = event.crossing_hi - event.crossing_lo,
                result.P_baseline,
                path = length(event.path), event.path_length, closure = event.closure_stress,
                event.fraction, filled = count(>=(result.protocol.corridor_fill), event.fractions),
                event.opening, steps = length(result.history), outcome = event.outcome,
                resolved = event.outcome === :amplitude_bracketed, time,
            )
        end
    catch err
        err isa InterruptException && rethrow()
        (;
            nels = missing, ΔP_crit = missing, P_event = missing, t_event = missing,
            bracket = missing, path = missing, path_length = missing, closure = missing,
            fraction = missing, filled = missing, opening = missing, P_baseline = missing,
            steps = missing, outcome = Symbol(first(split(sprint(showerror, err), '\n'))),
            resolved = false, time = missing,
        )
    end
end

function print_first_threshold_row(sweep, setting, row)
    label = join(("$k = $(_threshold_label(v))" for (k, v) in pairs(setting)), ", ")
    if ismissing(row.ΔP_crit)
        @printf("%-10s %-38s no threshold: %s\n", sweep, label, row.outcome)
        return
    end
    @printf(
        "%-10s %-38s %5d els  ΔP_crit %6.3f MPa  base %7.2f  P %7.2f  t %6.3f kyr  ±%.3f  path %2d (%4.0f m)  corridor %.2f (%d bins)  s_n %6.2f  w %.4f m  %5.0f s  %s\n",
        sweep, label, row.nels, row.ΔP_crit / 1.0e6, row.P_baseline / 1.0e6,
        row.P_event / 1.0e6, row.t_event / kyr,
        row.bracket / kyr, row.path, row.path_length, row.fraction, row.filled,
        row.closure / 1.0e6, row.opening, row.time,
        row.resolved ? "" : "UNRESOLVED",
    )
    return
end

# A protocol prints as the fields that differ from the default, so a row says what was varied.
function _threshold_label(protocol::EventProtocol)
    default = EventProtocol()
    changed = [
        "$f = $(getfield(protocol, f))" for f in fieldnames(EventProtocol)
            if getfield(protocol, f) != getfield(default, f)
    ]
    return isempty(changed) ? "default" : join(changed, ", ")
end
_threshold_label(x) = x

"""
    run_first_threshold_benchmark(; sweeps = keys(first_threshold_sweeps()), kwargs...) -> Vector{NamedTuple}

Run the requested sweeps, print one line per run and return the rows, each tagged with its `sweep` and
`setting`, followed by the spread of both thresholds within each sweep.

Two numbers are reported per sweep, because they are not equally conditioned. `ΔP_crit` is the rise
over the spun-up baseline, and it is a difference of two pressures of order 126 MPa, so it carries an
amplification of `P / ΔP ≈ 25` to `30`: a 0.2% move of either pressure is a 5% move of the
difference, and the `baseline` family showed the subtracted reference is the part that has not
converged. `P_event` is the absolute reservoir pressure at failure, and it is the invariant — it
moves 0.05 to 0.5 MPa across the same meshes. Read the numerical sweeps on `P_event` against an
absolute tolerance, and `ΔP_crit` only once the baseline is equilibrated.

The spread of the numerical sweeps is what Gate G1 is about: it must be small against the smallest
memory signal worth reporting. The spread of the protocol sweeps is not a numerical error at all —
it is the sensitivity that has to be quoted with any threshold, since none of those rules is fixed by
data yet.
"""
function run_first_threshold_benchmark(; sweeps = keys(first_threshold_sweeps()), kwargs...)
    settings = first_threshold_sweeps()
    rows = NamedTuple[]
    for sweep in sweeps
        for setting in settings[sweep]
            row = first_threshold_case(; setting..., kwargs...)
            print_first_threshold_row(sweep, setting, row)
            push!(rows, merge((; sweep, setting), row))
        end
        print_first_threshold_spread(sweep, [r for r in rows if r.sweep === sweep && r.resolved])
    end
    return rows
end

# The spread of `ΔP_crit` is the gate as originally written; the spread of `P_event` is the quantity
# it should be written on, and the absolute move in MPa is what the error budget compares to `M_min`.
function print_first_threshold_spread(sweep, resolved)
    length(resolved) > 1 || return
    for (name, values) in
            (("ΔP_crit", [r.ΔP_crit for r in resolved]), ("P_event", [r.P_event for r in resolved]))
        lo, hi = extrema(values)
        spread = (hi - lo) / (sum(values) / length(values))
        @printf(
            "%-10s %d resolved runs: %-7s from %8.3f to %8.3f MPa, spread %6.3f MPa = %5.2f%%\n",
            sweep, length(resolved), name, lo / 1.0e6, hi / 1.0e6, (hi - lo) / 1.0e6, 100spread,
        )
    end
    return
end
