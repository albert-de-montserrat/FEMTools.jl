# Recharge-and-intrusion cycles of a Reykjanes-like section: the event protocol of GAP-11 of
# REYKJANES_PLAN.md end to end.
#
# Runs `run_cycles` from `event_cycles.jl` on a coarse section: recharge the sill until a connected
# failed path reaches the shallow crust, bracket the step at which it first does, and open a dike on
# that path by the amount that drains the magma pressure to the closure stress of the path plus the
# arrest overpressure. Every rule of that sentence is a field of `EventProtocol` in
# `event_protocol.jl`, and every one is a placeholder until Phase 0 of the science plan fixes it.
#
# The geometry, the geotherm, the material values, the recharge rate and the whole protocol are
# illustrative. This is the machinery working end to end, not a Reykjanes result.

include(joinpath(@__DIR__, "event_cycles.jl"))

"""
    main(; kwargs...) -> NamedTuple

Run two events on the coarse section and return the `run_cycles` result. Keyword arguments go to
`run_cycles`.

Writes VTK into `examples/reykjanes/output_cycles/` by default: one snapshot after the spin-up, one
per accepted step, and one on each side of every event, with `index.txt` naming them. Open the
series in ParaView and colour by `failed_fraction`, `corridor_bin_fill` or `in_band` to see what the
protocol did; `write_output = false` turns it off.

`max_steps` has to cover the first event, which the corridor detector places later than the
adjacency one did — about 3.4 kyr on this section, so roughly 4 steps of 1 kyr for the first event
and about twice that for the second.
"""
main(; kwargs...) = run_cycles(;
    nevents = 2, max_steps = 30, recharge_rate = 2.0e-7,
    Δt = 1kyr, refinement = 8, max_area = 4.0e6, write_output = true, kwargs...,
)

main()
