# Transactional stepping for the event protocol, milestone GAP-11 of REYKJANES_PLAN.md, built on the
# snapshots of GAP-12. Bisection of a failure threshold and the root-finding on an intrusion
# amplitude both discard many trials, so a step is a trial until it has passed the acceptance checks:
# it must leave nothing behind when it is rejected. `attempt_step!` captures the state, runs the step,
# accepts it, or restores the state and retries with a smaller step, and reports the outcome in a
# structured form instead of throwing. It knows nothing about the physics: the caller passes the step
# and the checks.

using FEMTools

include(joinpath(@__DIR__, "state_snapshot.jl"))

"""
Outcomes a rejected step reports, from the failure contract of GAP-11: the solver did not converge
(`:solver_failed`), a field is not usable (`:invalid_state`), or the geometry is not (`:invalid_geometry`).
An accepted step reports `:accepted`.
"""
const STEP_FAILURE_OUTCOMES = (:solver_failed, :invalid_state, :invalid_geometry)

"""
    StepRejection(outcome, detail)

Exception with which a step function rejects its own trial, for a condition it alone can see, such as
a mesh movement that would invert an element. `outcome` is one of [`STEP_FAILURE_OUTCOMES`](@ref) and
`detail` says what failed. [`attempt_step!`](@ref) turns it into a rejected attempt; every other
exception propagates, because it is a bug and not a physical rejection.
"""
struct StepRejection <: Exception
    outcome::Symbol
    detail::String
    function StepRejection(outcome::Symbol, detail::AbstractString)
        outcome in STEP_FAILURE_OUTCOMES ||
            throw(ArgumentError("step rejection outcome must be one of $STEP_FAILURE_OUTCOMES, got :$outcome"))
        return new(outcome, String(detail))
    end
end

Base.showerror(io::IO, e::StepRejection) = print(io, "step rejected (:", e.outcome, "): ", e.detail)

"""
    StepCheck(name, outcome, test)

One acceptance check of a step. `test(objects, stats)` returns `nothing` when the step passes and a
string saying what failed when it does not; `outcome` is the [`STEP_FAILURE_OUTCOMES`](@ref) entry
the failure reports. `objects` is what was given to [`attempt_step!`](@ref) and `stats` is what the
step function returned.
"""
struct StepCheck{F}
    name::Symbol
    outcome::Symbol
    test::F
    function StepCheck(name::Symbol, outcome::Symbol, test::F) where {F}
        outcome in STEP_FAILURE_OUTCOMES ||
            throw(ArgumentError("step check outcome must be one of $STEP_FAILURE_OUTCOMES, got :$outcome"))
        return new{F}(name, outcome, test)
    end
end

"""
    solver_converged_check() -> StepCheck

Check that the step returned `converged = true`. A step that ends on its iteration cap reports
`converged = false`, so this is the check that keeps an unconverged state out of the detector.
"""
solver_converged_check() = StepCheck(:solver_converged, :solver_failed, _converged_detail)

function _converged_detail(_, stats)
    hasproperty(stats, :converged) || return "the step returned no `converged` flag: $(typeof(stats))"
    stats.converged === true && return nothing
    return string("the solver did not converge", _residual_detail(stats))
end

function _residual_detail(stats)
    parts = [string(f, " = ", getproperty(stats, f)) for f in (:err, :err_T, :iter) if hasproperty(stats, f)]
    return isempty(parts) ? "" : string(" (", join(parts, ", "), ")")
end

"""
    finite_state_check() -> StepCheck

Check that every state array of `objects` is finite, so that a diverged solve is caught even when it
reported convergence. It walks the same arrays as [`physical_state`](@ref) and names the first
offending one.
"""
finite_state_check() = StepCheck(:finite_state, :invalid_state, _finite_state_detail)

_finite_state_detail(objects, _) = _nonfinite_detail(physical_state(objects), "")

"""The acceptance checks every step runs: convergence and finite state."""
default_step_checks() = (solver_converged_check(), finite_state_check())

_finite_entry(x::Number) = isfinite(x)
_finite_entry(x) = all(_finite_entry, x)

_nonfinite_detail(::Nothing, ::AbstractString) = nothing
_nonfinite_detail(x::AbstractArray, path) = all(_finite_entry, x) ? nothing : "$path holds a non-finite value"

function _nonfinite_detail(x::Union{Tuple, NamedTuple}, path)
    for (key, value) in pairs(x)
        detail = _nonfinite_detail(value, isempty(path) ? string(key) : string(path, ".", key))
        detail === nothing || return detail
    end
    return nothing
end

"""
    attempt_step!(step!, objects::NamedTuple; Δτ, kwargs...) -> NamedTuple

Run `step!(Δτ)` as a transaction on the state of `objects` and return its report.

The state of `objects` is captured first, as in [`capture_state`](@ref). The step is accepted when it
passes every entry of `checks`, and the state of `objects` is then the state it left. Otherwise the
capture is restored, `rebuild!` is called, and the step is retried with `Δτ` multiplied by `shrink`,
until `max_attempts` is reached or the next step would fall below `Δτ_min`. A rejected step therefore
leaves the state exactly as it was, which is what makes a discarded bisection or amplitude trial safe.

`step!` returns whatever the checks read, normally the solver statistics; it may reject its own trial
by throwing a [`StepRejection`](@ref). Anything the step changes outside `objects` must be rebuilt by
`rebuild!`, which runs after every restore: the mesh geometry, the pressure scaling and the sources
are derived from the state, not part of it.

Keywords:
- `values`: the non-array state, as in [`capture_state`](@ref) (physical time, counters).
- `checks`: the acceptance checks, [`default_step_checks`](@ref) by default.
- `max_attempts`, `shrink`, `Δτ_min`: the retry budget. The default makes one attempt, and `Δτ_min`
  defaults to the smallest step `max_attempts` allows, so that the attempt count alone limits the
  retries until a physically smallest step is given.
- `rebuild!`: called after each restore, with no arguments.

The report holds `outcome` (`:accepted` or the [`STEP_FAILURE_OUTCOMES`](@ref) entry of the last
failure), `accepted`, the `Δτ` and `attempts` of the last attempt, `limit` (`:none`, `:attempts` or
`:step_size`, which of the two budgets ended the retries), `stats` (the accepted step's return value,
`nothing` when it failed), `failures` (one entry per rejected attempt, with its `attempt`, `Δτ`,
`check`, `outcome` and `detail`), and `values`, the captured non-array state that a caller adopts
after a failure to undo what its trial counted.
"""
function attempt_step!(
        step!, objects::NamedTuple;
        Δτ, values = (;), checks = default_step_checks(),
        max_attempts::Integer = 1, shrink = 0.5, Δτ_min = Δτ * shrink^(max_attempts - 1),
        rebuild! = nothing,
    )
    Δτ > 0 || throw(ArgumentError("Δτ must be positive"))
    0 < Δτ_min <= Δτ || throw(ArgumentError("Δτ_min must be positive and at most Δτ"))
    max_attempts >= 1 || throw(ArgumentError("max_attempts must be at least 1"))
    max_attempts == 1 || 0 < shrink < 1 ||
        throw(ArgumentError("shrink must be in (0, 1) to retry with a smaller step"))

    snapshot = capture_state(objects; values...)
    failures = NamedTuple[]
    Δτ_attempt = Δτ

    for attempt in 1:max_attempts
        stats, failure = _run_attempt(step!, objects, Δτ_attempt, checks)
        if failure === nothing
            return (;
                outcome = :accepted, accepted = true, Δτ = Δτ_attempt, attempts = attempt,
                limit = :none, stats, failures, values = deepcopy(snapshot.values),
            )
        end
        push!(failures, (; attempt, Δτ = Δτ_attempt, failure...))
        restored = restore_state!(objects, snapshot)
        rebuild! === nothing || rebuild!()

        attempt == max_attempts && return _rejected_report(failures, :attempts, restored)
        Δτ_attempt *= shrink
        Δτ_attempt < Δτ_min && return _rejected_report(failures, :step_size, restored)
    end
end

function _run_attempt(step!, objects, Δτ, checks)
    stats = try
        step!(Δτ)
    catch err
        err isa StepRejection || rethrow()
        return nothing, (; check = :step_rejected, err.outcome, err.detail)
    end
    for check in checks
        detail = check.test(objects, stats)
        detail === nothing && continue
        return stats, (; check = check.name, check.outcome, detail = String(detail))
    end
    return stats, nothing
end

function _rejected_report(failures, limit, values)
    last_failure = last(failures)
    return (;
        outcome = last_failure.outcome, accepted = false, Δτ = last_failure.Δτ,
        attempts = length(failures), limit, stats = nothing, failures, values,
    )
end

"""
    step_failure_message(report) -> String

One line per rejected attempt of `report`, for a log or an error message.
"""
function step_failure_message(report)
    lines = ["attempt $(f.attempt) at Δτ = $(f.Δτ) rejected by $(f.check) (:$(f.outcome)): $(f.detail)"
             for f in report.failures]
    return join(lines, "\n")
end

# --------------------------------------------------------------------------------------------------
# Repeated trials from one accepted state: the crossing refinement of GAP-11 step 1, and the
# amplitude solve of step 3, both restart every trial from the same left state.
# --------------------------------------------------------------------------------------------------

"""
    TrialSet

One captured state and the settings every trial from it shares: see [`capture_trials`](@ref).
"""
struct TrialSet{O, S, R, C}
    objects::O
    snapshot::S
    rebuild!::R
    checks::C
end

"""
    capture_trials(objects::NamedTuple; rebuild! = nothing, checks = default_step_checks(), values...) -> TrialSet

Capture the state of `objects` as the left state of a sequence of trials, each of which
[`run_trial!`](@ref) restores. `values`, `checks` and `rebuild!` mean what they mean in
[`capture_state`](@ref) and [`attempt_step!`](@ref).
"""
function capture_trials(objects::NamedTuple; rebuild! = nothing, checks = default_step_checks(), values...)
    return TrialSet(objects, capture_state(objects; values...), rebuild!, checks)
end

"""
    reset_trial!(trials::TrialSet) -> NamedTuple

Restore the captured state and rebuild what is derived from it, and return the captured values.
"""
function reset_trial!(trials::TrialSet)
    values = restore_state!(trials.objects, trials.snapshot)
    trials.rebuild! === nothing || trials.rebuild!()
    return values
end

"""
    run_trial!(observe, trials::TrialSet, step!; Δτ, kwargs...) -> NamedTuple

Run `step!(Δτ)` from the captured state of `trials`, observe the result, and restore the state
whatever happened, so that the next trial starts from the same left state.

`observe(objects, stats)` runs only on a step that was accepted, and the returned `observation` is
`nothing` otherwise, which is how a caller tells a measured trial from a failed one. `kwargs` go to
[`attempt_step!`](@ref), which supplies the acceptance checks of `trials`.
"""
function run_trial!(observe, trials::TrialSet, step!; Δτ, kwargs...)
    report = attempt_step!(step!, trials.objects; Δτ, checks = trials.checks, kwargs...)
    observation = report.accepted ? observe(trials.objects, report.stats) : nothing
    reset_trial!(trials)
    return (; report, observation)
end

"""
    bracket_crossing!(detector, trials::TrialSet, step!; Δτ, kwargs...) -> NamedTuple

Find the earliest step size in `(0, Δτ]` at which `detector` first becomes true, starting every
trial from the captured state of `trials`.

The caller has just seen `detector` become true after a step of `Δτ` from that state and `detector`
false at the state itself. The interval is first sampled at `samples` equally spaced step sizes,
because a connected detector need not be monotone under healing, stress redistribution or a change
of path: bisection alone would return whichever crossing it happened to fall into. The earliest
bracket the sampling finds is then bisected until it is shorter than `tolerance`. The threshold
precision this sets must be well below the smallest memory signal of interest, or a discrete
detector on a continuous field leaves a jitter of one step.

Keywords: `samples = 4` probes, including `Δτ` itself, which also checks that the trip is
reproducible from the restored state; `tolerance = Δτ / 100`; `max_trials = 20` trials in all;
`commit = true` to re-run and keep the step at `hi` once the bracket is resolved, so that the run
continues from the state in which the criterion has just tripped. Anything else goes to
[`attempt_step!`](@ref).

The report holds `outcome`, `lo` and `hi` (the bracket, `lo` not tripped and `hi` tripped),
`tolerance`, `toggles` (how often the sampled detector changed, more than one meaning several
crossings), `probes` (one entry per trial) and `committed` (the accepted step at `hi`, or `nothing`).
The outcomes are `:crossing_bracketed`; `:unresolved_crossing`, when the sampling saw several
crossings or the trial budget ran out before the tolerance; `:no_crossing`, when no probe tripped,
which means the trip was not reproduced; and `:solver_failed`. Only `:crossing_bracketed` leaves the
state after the crossing step, and only when `commit`; every other outcome leaves the left state.
"""
function bracket_crossing!(
        detector, trials::TrialSet, step!;
        Δτ, samples::Integer = 4, tolerance = Δτ / 100, max_trials::Integer = 20, commit = true,
        kwargs...,
    )
    Δτ > 0 || throw(ArgumentError("Δτ must be positive"))
    samples >= 1 || throw(ArgumentError("samples must be at least 1"))
    0 < tolerance <= Δτ || throw(ArgumentError("tolerance must be positive and at most Δτ"))
    max_trials >= samples ||
        throw(ArgumentError("max_trials must be at least samples, the trials the sampling needs"))

    probes = NamedTuple[]
    report(outcome; lo = zero(Δτ), hi = Δτ, toggles = 0, committed = nothing) =
        (; outcome, lo, hi, tolerance, toggles, samples, probes, committed)

    function probe!(Δτ_trial)
        trial = run_trial!(detector, trials, step!; Δτ = Δτ_trial, kwargs...)
        push!(probes, (; Δτ = Δτ_trial, tripped = trial.observation, outcome = trial.report.outcome))
        return trial.observation
    end

    # Sample the whole interval before bisecting any part of it.
    tripped = Bool[]
    for k in 1:samples
        sampled = probe!(k * Δτ / samples)
        sampled === nothing && return report(:solver_failed)
        push!(tripped, sampled)
    end
    any(tripped) || return report(:no_crossing)

    crossing = findfirst(tripped)
    toggles = count(k -> tripped[k] != (k == 1 ? false : tripped[k - 1]), 1:samples)
    lo = (crossing - 1) * Δτ / samples
    hi = crossing * Δτ / samples

    while hi - lo > tolerance && length(probes) < max_trials
        mid = (lo + hi) / 2
        sampled = probe!(mid)
        sampled === nothing && return report(:solver_failed; lo, hi, toggles)
        sampled ? (hi = mid) : (lo = mid)
    end

    if toggles > 1 || hi - lo > tolerance
        return report(:unresolved_crossing; lo, hi, toggles)
    end

    # The crossing step was run from this same state during the search, so re-running it is the
    # shortest way to leave the run in the state the event starts from.
    committed = commit ? attempt_step!(step!, trials.objects; Δτ = hi, checks = trials.checks, kwargs...) : nothing
    if committed !== nothing && !committed.accepted
        reset_trial!(trials)
        return report(:solver_failed; lo, hi, toggles)
    end
    return report(:crossing_bracketed; lo, hi, toggles, committed)
end

"""
    solve_amplitude!(observe, trials::TrialSet, step!; Δτ, target, amplitude, kwargs...) -> NamedTuple

Find the smallest nonnegative amplitude at which `observe` reaches `target`, starting every trial
from the captured state of `trials`.

`step!(amplitude, Δτ)` injects that amplitude and takes one step; `observe(objects, stats)` returns
the quantity the amplitude drives, for the intrusion of GAP-11 the closure traction of the band, for
which the amplitude is the dike opening. The search assumes `observe` grows with the amplitude, and
safeguards that assumption rather than trusting it: it measures the state at amplitude zero, then
doubles `amplitude` until the target is passed or `amplitude_max` is reached, and only then bisects.
A search that never passes the target is `:no_arrest_bracket` — an outcome, not a forced intrusion.

Keywords: `amplitude`, the first guess, which must be positive; `amplitude_max = 1024 amplitude`;
`rtol = 1e-3`, the width of the final bracket relative to its upper end; `max_trials = 30` trials in
all; `check_zero = true`, the trial at amplitude zero, which a caller that already knows the state is
below the target can skip; `commit = true` to re-run and keep the step at the bracket's upper end,
the smallest amplitude that reaches the target. Anything else goes to [`attempt_step!`](@ref).

The report holds `outcome`, `amplitude` (the committed upper end), `lo` and `hi`, `rtol`,
`observation` (the value at `hi`), `probes` (one entry per trial) and `committed`. The outcomes are
`:amplitude_bracketed`; `:no_intrusion_needed`, when the target is already met at zero;
`:no_arrest_bracket`, when no amplitude up to `amplitude_max` reaches it; `:unresolved_amplitude`,
when the trial budget ran out with a bracket in hand; and `:solver_failed`. Only
`:amplitude_bracketed` leaves the state after the injected step, and only when `commit`; every other
outcome leaves the left state.
"""
function solve_amplitude!(
        observe, trials::TrialSet, step!;
        Δτ, target, amplitude, amplitude_max = 1024 * amplitude, rtol = 1.0e-3,
        max_trials::Integer = 30, check_zero = true, commit = true, kwargs...,
    )
    amplitude > 0 || throw(ArgumentError("the first amplitude must be positive"))
    amplitude <= amplitude_max || throw(ArgumentError("amplitude_max must be at least the first amplitude"))
    0 < rtol < 1 || throw(ArgumentError("rtol must be in (0, 1)"))
    max_trials >= 2 || throw(ArgumentError("max_trials must leave room for a bracket and a bisection"))

    probes = NamedTuple[]
    report(outcome; lo = zero(amplitude), hi = amplitude, observation = nothing, committed = nothing) =
        (; outcome, amplitude = hi, lo, hi, rtol, observation, probes, committed)

    function probe!(trial_amplitude)
        trial = run_trial!(observe, trials, Δτ_trial -> step!(trial_amplitude, Δτ_trial); Δτ, kwargs...)
        push!(probes, (; amplitude = trial_amplitude, trial.observation, outcome = trial.report.outcome))
        return trial.observation
    end

    if check_zero
        at_rest = probe!(zero(amplitude))
        at_rest === nothing && return report(:solver_failed)
        at_rest >= target && return report(
            :no_intrusion_needed; lo = zero(amplitude), hi = zero(amplitude), observation = at_rest,
        )
    end

    # Double until the target is passed: the bracket is what makes the bisection below safe.
    lo, hi = zero(amplitude), amplitude
    local observation
    while true
        observation = probe!(hi)
        observation === nothing && return report(:solver_failed; lo, hi)
        observation >= target && break
        lo = hi
        hi >= amplitude_max && return report(:no_arrest_bracket; lo, hi)
        hi = min(2hi, amplitude_max)
        length(probes) >= max_trials && return report(:no_arrest_bracket; lo, hi)
    end

    while hi - lo > rtol * hi && length(probes) < max_trials
        mid = (lo + hi) / 2
        value = probe!(mid)
        value === nothing && return report(:solver_failed; lo, hi, observation)
        value >= target ? ((hi, observation) = (mid, value)) : (lo = mid)
    end
    hi - lo <= rtol * hi || return report(:unresolved_amplitude; lo, hi, observation)

    committed = commit ?
        attempt_step!(Δτ_step -> step!(hi, Δτ_step), trials.objects; Δτ, checks = trials.checks, kwargs...) :
        nothing
    if committed !== nothing && !committed.accepted
        reset_trial!(trials)
        return report(:solver_failed; lo, hi, observation)
    end
    return report(:amplitude_bracketed; lo, hi, observation, committed)
end
