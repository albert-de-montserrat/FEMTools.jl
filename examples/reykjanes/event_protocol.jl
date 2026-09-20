# The event protocol of GAP-11 of REYKJANES_PLAN.md: the choices that turn the stress state of a step
# into an event, and the event into an intrusion. It runs nothing; `reykjanes_thermal_stokes.jl` and
# the cycle runner build on it.
#
# Every number in `EventProtocol` is a protocol choice, not a solver setting, and every one of them
# is a placeholder until Phase 0 of the science plan fixes it from data. They are gathered in one
# versioned object, and named in the event record, precisely because the memory result depends on
# them: a threshold is only as meaningful as the rule that declared it.

using Statistics
using FEMTools

include(joinpath(@__DIR__, "event_stepping.jl"))
include(joinpath(@__DIR__, "injection_source.jl"))
include(joinpath(@__DIR__, "dike_functions", "dike_eigenstrain.jl"))

"""
    EventProtocol(; kwargs...)

The rules that define an event and its intrusion, in SI. Held together so that a run records the
protocol it used, and so that the sensitivity of a threshold to a rule can be measured by changing
one field.

Detection:
- `criterion = :tensile`: which integration points count as failed, one of `:tensile` (the hydraulic
  margin `P_f − s₃ − T₀ ≥ 0`), `:shear` (Drucker--Prager `F > yield_tolerance`), `:either` or
  `:both`. The science plan does not fix the Boolean rule; this names it.
- `tensile_strength = 5e6`: `T₀`. **The detector is unusable at `T₀ = 0`**: with a magma less dense
  than the crust, a connected hydraulic path from the sill to the surface exists in the undisturbed
  state, so the run would declare an event on its first step.
- `magmastatic_head = true`: the fluid pressure at a point is the reservoir pressure less the weight
  of the magma column up to it, `P_f(y) = P_res − ρ_m g (y − y_res)`. With the reservoir pressure
  applied everywhere instead, the criterion compares a 4.5 km sill pressure with near-surface stress
  and is met almost everywhere.
- `magma_density = 2800`: `ρ_m` of that column, and `reach_depth = 2e3`: how close to the surface a
  connected path must reach to count as an event.
- `require_path = true`: an event needs failed rock reaching that depth, not merely a failed element
  somewhere. What "reaching" means is `detector`.
- `detector = :corridor_column`: the rule that turns the failed elements into a verdict, and the
  subject of Gate G1. `:graph_path` is the original rule — a face-connected chain of failed elements
  from the reservoir boundary to the target depth — and it **is mesh dependent by construction**:
  the chain lives on the element graph, so refining changes when it closes even where the stress
  field has converged. The first-threshold mesh sweep measured a 77.7% spread of `ΔP_crit` against
  the gate's 5% for that reason. The two corridor rules ask the same physical question of a fixed
  geometric strip instead, `|x − x_sill| ≤ corridor_width/2` from the reservoir crest up to
  `reach_depth`, cut into `corridor_bins` bins of equal height: `:corridor_column` trips when every
  bin is filled to `corridor_fill` by area, which is the geometric statement of a spanning path, and
  `:corridor_fraction` trips when the corridor as a whole is that full, which is smoother but does
  not require the failure to reach the shallow end. Only the mesh enters, through the area fraction,
  and that converges under refinement. `:graph_path` is kept so the sweep can quote all three.
- `corridor_width = 2e3`, `corridor_bins = 4`, `corridor_fill = 0.5`: the corridor's geometry and
  fill threshold, in metres and as a fraction. They are protocol choices like `reach_depth`, not
  numerical settings: the width should be wide against the element size so that a bin samples many
  elements, and narrow against the sill, and `corridor_bins` fixes the vertical resolution of the
  detector independently of the mesh. A bin that no element falls in makes the corridor
  `under_resolved`, and an under-resolved corridor never trips.

Crossing refinement: `crossing_samples = 4` and `crossing_tolerance = 0.02`, the width of the final
bracket as a fraction of the step, passed to [`bracket_crossing!`](@ref).

Intrusion:
- `arrest_target = :reservoir_pressure`: what the opening is driven to `s_n + ΔP_arrest`. The plan's
  rule drains the reservoir until the magma pressure *at the path*, the reservoir pressure less the
  magmastatic head up to it, falls to that value: the dike opens until the reservoir can no longer
  push it. The alternative `:band_traction` drives the band's own closure traction up to it instead,
  and **opens almost nothing**: the ambient traction across the dike normal is the closure stress
  already, within 0.5 MPa on the coarse section, so the opening is set by `ΔP_arrest` against the
  path's compliance alone and the criterion re-trips at once. It is kept as a comparison, not a
  default.
- `arrest_overpressure = 1e6`: `ΔP_arrest`, how far above the closure stress of the path the magma
  pressure is left when the dike is taken to have arrested.
- `band_width = 100.0`: the numerical width `h` of the opened band. The crack benchmark shows the
  answer does not depend on it, and it is kept separate from the opening and from any regularisation
  length, as the plan requires.
- `dike_normal = (1.0, 0.0)`: the opening direction. A rift-normal, vertical dike in this section.
- `intrusion_Δt = 3.15576e7` (one year): a step short against the Maxwell time, so the intrusion is
  elastic. It is bounded from *below* as well, and not by physics: the model's characteristic time is
  its tectonic one (2 Myr here), and an intrusion step of an hour is `Δτ ≈ 6e-11` in those units,
  where the relaxation does not converge at all. Measured on the coarse section, one year and ten
  years give the same band traction to 0.3% and cost 1 900 and 1 700 iterations, while a century
  differs by 2.4% because `Δt/t_M ≈ 1` there and the intrusion is no longer elastic.
- `max_opening = 20.0`: the largest opening the amplitude search may try, in metres. Reaching it is
  a `:no_arrest_bracket` outcome, not a forced intrusion.
- `amplitude_rtol = 0.02`: the width of the final amplitude bracket, relative to its upper end.

Budgets: `max_step_attempts = 3` retries per step and `max_trials = 20` trials per search.
"""
struct EventProtocol{FP}
    criterion::Symbol
    tensile_strength::FP
    yield_tolerance::FP
    magmastatic_head::Bool
    magma_density::FP
    reach_depth::FP
    require_path::Bool
    detector::Symbol
    corridor_width::FP
    corridor_bins::Int
    corridor_fill::FP
    crossing_samples::Int
    crossing_tolerance::FP
    arrest_target::Symbol
    arrest_overpressure::FP
    band_width::FP
    dike_normal::NTuple{2, FP}
    intrusion_Δt::FP
    max_opening::FP
    amplitude_rtol::FP
    max_step_attempts::Int
    max_trials::Int
end

const EVENT_CRITERIA = (:tensile, :shear, :either, :both)
const ARREST_TARGETS = (:reservoir_pressure, :band_traction)
const DETECTOR_RULES = (:corridor_column, :corridor_fraction, :graph_path)

function EventProtocol(;
        criterion::Symbol = :tensile,
        tensile_strength = 5.0e6,
        yield_tolerance = 0.0,
        magmastatic_head = true,
        magma_density = 2800.0,
        reach_depth = 2.0e3,
        require_path = true,
        detector::Symbol = :corridor_column,
        corridor_width = 2.0e3,
        corridor_bins = 4,
        corridor_fill = 0.5,
        crossing_samples = 4,
        crossing_tolerance = 0.02,
        arrest_target::Symbol = :reservoir_pressure,
        arrest_overpressure = 1.0e6,
        band_width = 100.0,
        dike_normal = (1.0, 0.0),
        intrusion_Δt = 3.15576e7,   # one year; see the docstring for the window this sits in
        max_opening = 20.0,
        amplitude_rtol = 0.02,
        max_step_attempts = 3,
        max_trials = 20,
    )
    criterion in EVENT_CRITERIA ||
        throw(ArgumentError("criterion must be one of $EVENT_CRITERIA, got :$criterion"))
    tensile_strength >= 0 || throw(ArgumentError("tensile strength must be nonnegative"))
    magma_density > 0 || throw(ArgumentError("magma density must be positive"))
    reach_depth >= 0 || throw(ArgumentError("reach depth must be nonnegative"))
    detector in DETECTOR_RULES ||
        throw(ArgumentError("detector must be one of $DETECTOR_RULES, got :$detector"))
    corridor_width > 0 || throw(ArgumentError("the corridor width must be positive"))
    corridor_bins >= 1 || throw(ArgumentError("the corridor needs at least one bin"))
    0 < corridor_fill <= 1 || throw(ArgumentError("corridor_fill must be in (0, 1]"))
    crossing_samples >= 1 || throw(ArgumentError("crossing_samples must be at least one"))
    0 < crossing_tolerance <= 1 || throw(ArgumentError("crossing_tolerance must be in (0, 1]"))
    arrest_target in ARREST_TARGETS ||
        throw(ArgumentError("arrest_target must be one of $ARREST_TARGETS, got :$arrest_target"))
    arrest_overpressure >= 0 || throw(ArgumentError("arrest overpressure must be nonnegative"))
    band_width > 0 || throw(ArgumentError("band width must be positive"))
    intrusion_Δt > 0 || throw(ArgumentError("the intrusion time step must be positive"))
    max_opening > 0 || throw(ArgumentError("max opening must be positive"))
    0 < amplitude_rtol < 1 || throw(ArgumentError("amplitude_rtol must be in (0, 1)"))
    hypot(dike_normal...) > 0 || throw(ArgumentError("the dike normal must be nonzero"))

    FP = promote_type(
        typeof(tensile_strength), typeof(yield_tolerance), typeof(magma_density), typeof(reach_depth),
        typeof(corridor_width), typeof(corridor_fill),
        typeof(crossing_tolerance), typeof(arrest_overpressure), typeof(band_width),
        typeof(intrusion_Δt), typeof(max_opening), typeof(amplitude_rtol), eltype(dike_normal),
    )
    return EventProtocol{FP}(
        criterion, tensile_strength, yield_tolerance, magmastatic_head, magma_density, reach_depth,
        require_path, detector, FP(corridor_width), Int(corridor_bins), FP(corridor_fill),
        crossing_samples, crossing_tolerance, arrest_target, arrest_overpressure, band_width,
        map(FP, Tuple(dike_normal)), intrusion_Δt, max_opening, amplitude_rtol,
        max_step_attempts, max_trials,
    )
end

"""
    failed_points(diagnostics, criterion) -> Matrix{Bool}

The per-integration-point failure flags of [`dike_failure_diagnostics`](@ref) selected by
`criterion`, `:both` being the same point failing in shear and in tension.

This is the field the corridor detector measures an area with. `failed_elements` reduces it to one
flag per element, which is what a graph search and a band of whole elements need.
"""
function failed_points(diagnostics, criterion::Symbol)
    return criterion === :tensile ? diagnostics.tensile_failed :
        criterion === :shear ? diagnostics.shear_failed :
        criterion === :either ? diagnostics.failed :
        diagnostics.shear_failed .& diagnostics.tensile_failed
end

"""
    failed_elements(diagnostics, criterion) -> Vector{Bool}

Reduce the per-integration-point flags of [`dike_failure_diagnostics`](@ref) to one flag per element
under `criterion`: an element fails when any of its integration points does.
"""
function failed_elements(diagnostics, criterion::Symbol)
    return vec(any(failed_points(diagnostics, criterion); dims = 1))
end

"""
    closure_stress(s3, areas, elements) -> Real

The area-weighted compression-positive closure stress of `elements`: the mean over them of the least
compressive principal stress `s₃`, which is the stress a dike opening perpendicular to it must
exceed. `s3` is the `nq × nels` array of [`dike_failure_diagnostics`](@ref) and `areas` is one area
per element.

This is the `s_n` of the intrusion, and the first protocol freezes it before the intrusion starts:
recomputing it while the band opens would make it part of the root function of the amplitude search.
"""
function closure_stress(s3, areas, elements)
    isempty(elements) && throw(ArgumentError("the closure stress needs at least one element"))
    total = sum(areas[iel] for iel in elements)
    total > 0 || throw(ArgumentError("the path elements must have positive area"))
    return sum(areas[iel] * mean(@view s3[:, iel]) for iel in elements) / total
end

"""
    band_closure_traction(τ_nn, P_ip, areas, elements) -> Real

The area-weighted traction that would close the opened band, `P − τ_nn` across the dike normal,
averaged over `elements`.

The crack benchmark (`dike_crack_setup.jl`) shows that this, and not the band's mean pressure, is the
pressure of the crack the eigenstrain band represents: the band is held in compression along its
length as well as across it, and its mean pressure is about 20% below the crack pressure and moves
with the band's aspect. The arrest condition of the intrusion is therefore written on this.
"""
function band_closure_traction(τ_nn, P_ip, areas, elements)
    isempty(elements) && throw(ArgumentError("the band traction needs at least one element"))
    total = sum(areas[iel] for iel in elements)
    total > 0 || throw(ArgumentError("the band elements must have positive area"))
    return sum(
        areas[iel] * mean(P_ip[q, iel] - τ_nn[q, iel] for q in axes(P_ip, 1)) for iel in elements
    ) / total
end

"""
    normal_deviatoric_stress(τ, n) -> Matrix

The deviatoric normal stress `n ⋅ τ ⋅ n` at every integration point, from the `(xx, yy, xy)` host
arrays of the 2-D stress history and a unit dike normal `n`.
"""
function normal_deviatoric_stress(τ, n)
    nx, ny = Tuple(n) ./ hypot(Tuple(n)...)
    return @. nx^2 * τ[1] + ny^2 * τ[2] + 2nx * ny * τ[3]
end

"""
    build_event_detector(model, protocol; τ = model.τ) -> NamedTuple

Return `detect()`, which reads the current state of `model` and reports what has failed, under
`protocol`. Everything that does not change with the state — the element graph, the reservoir seeds,
the target elements, the fixed detection corridor, the strength of each element and the element
areas — is built once here.

`detect()` returns the per-integration-point `diagnostics`, the element `failed` flags, the
face-connected `component` reachable from the reservoir boundary, the `path`, the corridor's per-bin
failed-area `fractions` and its total `fraction`, `under_resolved`, the reservoir pressure
`P_reservoir`, the interpolated `P_ip`, and `tripped`, the protocol's Boolean verdict. All of it is
in SI.

`path` is the material the intrusion opens. Under `:graph_path` it is the shortest face-connected
chain of failed elements from the reservoir to the target depth; under the corridor rules it is the
failed rock of the corridor bins that count, which is the same kind of set chosen by geometry rather
than by the element graph. The corridor fractions and the connected `component` are reported whatever
the rule is, so that one run can be read against both detectors.
"""
function build_event_detector(model, protocol::EventProtocol; τ = model.τ)
    (; mesh_stokes, el2n_v_cpu, DoFsP_cpu, NqP_cpu, coords_cpu, cell_phase, groups) = model
    (; L_c, σ_c, C_crust, C_sill, ϕ, dr, sill_P_dofs) = model
    nels = mesh_stokes.nels

    adjacency = generate_element_adjacency(el2n_v_cpu; shared = :face)
    node2element = generate_node2element(el2n_v_cpu, length(coords_cpu))
    seeds = generate_boundary_elements(groups.sill, node2element)
    centroids = [
        (
            mean(coords_cpu[el2n_v_cpu[a, iel]][1] for a in 1:3) * L_c,
            mean(coords_cpu[el2n_v_cpu[a, iel]][2] for a in 1:3) * L_c,
        ) for iel in 1:nels
    ]
    targets = Int32[iel for iel in 1:nels if centroids[iel][2] >= -protocol.reach_depth]
    reservoir_y = mean(coords_cpu[n][2] for n in groups.sill) * L_c

    strength = [cell_phase[iel] == 1 ? C_crust : C_sill for iel in 1:nels] .* σ_c
    friction = fill(ϕ, nels)
    areas = [
        begin
            p, q, r = (coords_cpu[el2n_v_cpu[a, iel]] .* L_c for a in 1:3)
            abs((q[1] - p[1]) * (r[2] - p[2]) - (r[1] - p[1]) * (q[2] - p[2])) / 2
        end for iel in 1:nels
    ]

    # The fixed detection corridor of Gate G1: a vertical strip of host rock over the reservoir,
    # from its crest up to the target depth, binned in geometry. It is built from the sill's own
    # position rather than from the mesh, and the reservoir's elements are kept out of it: they are
    # magma, not rock that can fail and carry a dike.
    sill_x = mean(coords_cpu[n][1] for n in groups.sill) * L_c
    reservoir_crest = maximum(coords_cpu[n][2] for n in groups.sill) * L_c
    -protocol.reach_depth > reservoir_crest || throw(
        ArgumentError(
            "the corridor is empty: reach_depth = $(protocol.reach_depth) m puts the target at " *
                "$(-protocol.reach_depth) m, at or below the reservoir crest at $reservoir_crest m",
        ),
    )
    corridor = build_dike_corridor(
        centroids, areas;
        x_center = sill_x, half_width = protocol.corridor_width / 2,
        y_bottom = reservoir_crest, y_top = -protocol.reach_depth,
        nbins = protocol.corridor_bins, eligible = cell_phase .== 1,
    )

    # The fluid pressure at an integration point is the reservoir pressure less the magma column up
    # to it, so the head is a fixed offset per point and only the reservoir pressure changes.
    head = [
        protocol.magmastatic_head ?
            -protocol.magma_density * 9.81 *
            (sum(NqP_cpu[q][a] * coords_cpu[el2n_v_cpu[a, iel]][2] for a in 1:3) * L_c - reservoir_y) :
            0.0 for q in 1:model.NQ_v, iel in 1:nels
    ]

    function detect()
        P_cpu = Array(dr.P) .* σ_c
        P_ip = [
            sum(NqP_cpu[q][a] * P_cpu[DoFsP_cpu[a, iel]] for a in axes(DoFsP_cpu, 1))
                for q in 1:model.NQ_v, iel in 1:nels
        ]
        P_reservoir = mean(P_cpu[sill_P_dofs])
        diagnostics = dike_failure_diagnostics(
            map(x -> Array(x) .* σ_c, τ), P_ip, strength, friction,
            P_reservoir .+ head, protocol.tensile_strength;
            yield_tolerance = protocol.yield_tolerance * σ_c,
        )
        points = failed_points(diagnostics, protocol.criterion)
        failed = vec(any(points; dims = 1))
        component = dike_connected_component(adjacency, failed, seeds)
        # The corridor measures an area, so it weighs each element by the quadrature of it that
        # failed; the graph and the band take whole elements, as they must.
        under_resolved = any(iszero, corridor.bin_area)
        if protocol.detector === :graph_path
            path = dike_shortest_path(adjacency, failed, seeds, targets)
            spans = !isempty(path)
            fractions = corridor_bin_fractions(corridor, points, areas)
            fraction = corridor_failed_fraction(corridor, points, areas)
        else
            verdict = corridor_verdict(
                corridor, points, areas, protocol.detector, protocol.corridor_fill,
            )
            path = verdict.cells
            spans = verdict.tripped
            fractions = verdict.fractions
            fraction = verdict.fraction
        end
        tripped = protocol.require_path ? spans : any(failed)
        return (;
            tripped, diagnostics, failed, component, path, fractions, fraction, under_resolved,
            P_reservoir, P_ip, areas,
        )
    end

    return (;
        detect, adjacency, seeds, targets, corridor, areas, strength, friction, reservoir_y, head,
    )
end

"""
    magma_pressure(P_reservoir, head, areas, elements) -> Real

The magma pressure on `elements`: the reservoir pressure less the magmastatic head up to them,
area-weighted the same way as [`closure_stress`](@ref) so that the two can be compared. `head` is the
per-integration-point offset built by [`build_event_detector`](@ref), and is zero when the protocol
does not use a magmastatic head.

This is the quantity the intrusion drains: the dike opens until it falls to the closure stress of the
path plus the arrest overpressure.
"""
function magma_pressure(P_reservoir, head, areas, elements)
    isempty(elements) && throw(ArgumentError("the magma pressure needs at least one element"))
    total = sum(areas[iel] for iel in elements)
    total > 0 || throw(ArgumentError("the path elements must have positive area"))
    return P_reservoir + sum(areas[iel] * mean(@view head[:, iel]) for iel in elements) / total
end
