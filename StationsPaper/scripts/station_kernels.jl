# Function-only helpers that turn a station impulse response into Q-history
# Jacobians and displacements (OBSERVATION_MODEL.md). `H` is stations × steps:
# the displacement at the end of step `n` per unit rate during step 1 only.

"""
    displacement_from_rates(H, rates) -> Matrix

Displacement at the end of every step for per-step net rates `rates`, by
superposing shifted copies of the unit-rate response `H`.
"""
function displacement_from_rates(H, rates)
    u = zeros(size(H, 1), length(rates))
    for k in eachindex(rates), s in 1:k
        u[:, k] .+= rates[s] .* view(H, :, k - s + 1)
    end
    return u
end

"""
    history_jacobian(H, bin_steps, nt = size(H, 2)) -> Array{Float64, 3}

Stations × epochs × bins Jacobian of displacement with respect to the
interval-average rate in each source bin, over the first `nt` epochs. Bin `j`
covers the consecutive steps `bin_steps[j]`; a piecewise-constant rate in a bin
is the sum of its unit-step pulses, so each column is a sum of shifted copies of `H`.
"""
function history_jacobian(H, bin_steps, nt = size(H, 2))
    K = zeros(size(H, 1), nt, length(bin_steps))
    for (j, steps) in enumerate(bin_steps), s in steps, k in s:nt
        K[:, k, j] .+= view(H, :, k - s + 1)
    end
    return K
end

"""Bins of `width` steps tiling `first_step:(first_step + nsteps - 1)` backwards from its end."""
tile_back(first_step, nsteps, width) = reverse([max(first_step, last_step - width + 1):last_step
    for last_step in (first_step + nsteps - 1):(-width):first_step])

"""
    window_jacobian(H, stations; window, width, prewindow, prewidth, events) -> (; K, nwindow, centres, bins)

Stations × epochs × bins Jacobian of the displacement during the observation
window, relative to its start, with respect to the window bin rates (first
`nwindow` bins) and the pre-window bin rates. Window bins of `width` steps tile
`window` backwards from its end, so the most recent bin is complete and only
the earliest may be shorter; pre-window bins of `prewidth` steps tile the
`prewindow` steps before it the same way. A source in the pre-window moves the
stations only through relaxation inside the window. Each step in `events`
(counted from the window start) is taken out of its window bin and becomes a
one-step bin of its own, placed last. `bins` are the step ranges counted from
the start of the pre-window; `centres` are the window-bin midpoints in steps
from the window start.
"""
function window_jacobian(H, stations; window = 365, width = 1, prewindow = 0, prewidth = 30, events = Int[])
    size(H, 2) >= prewindow + window || throw(ArgumentError("impulse response shorter than pre-window + window"))
    window_bins = tile_back(prewindow + 1, window, width)
    centres = [(first(b) + last(b) - 1) / 2 - prewindow for b in window_bins]
    if !isempty(events)
        event_steps = prewindow .+ events
        window_bins = [setdiff(b, event_steps) for b in window_bins]
    end
    pre_bins = prewindow == 0 ? UnitRange{Int}[] : tile_back(1, prewindow, prewidth)
    bins = [window_bins; pre_bins; [(prewindow + e):(prewindow + e) for e in events]]
    full = history_jacobian(H[stations, :], bins, prewindow + window)
    reference = prewindow == 0 ? 0.0 : full[:, prewindow:prewindow, :]
    K = full[:, (prewindow + 1):end, :] .- reference
    return (; K, nwindow = length(window_bins), centres, bins)
end

"""Displacement during the window, relative to its start, for per-step rates over pre-window + window (station fastest)."""
window_displacement(H, rates, prewindow) =
    (u = displacement_from_rates(H, rates); vec(u[:, (prewindow + 1):end] .- (prewindow == 0 ? 0.0 : u[:, prewindow])))
