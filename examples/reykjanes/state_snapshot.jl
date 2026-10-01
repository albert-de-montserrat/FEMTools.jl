# In-memory snapshot and restore of the physical state of a run, milestone GAP-12 of REYKJANES_PLAN.md.
# Bisection of a threshold, root-finding on an intrusion amplitude and the branches of the factorial
# arms all restart from one accepted state, so a snapshot must restore it exactly and as often as
# needed. It is not an on-disk checkpoint.

using FEMTools

"""
Fields of a `StokesDR` that are solver scratch, not state: `solve_stokes_dyrel!` zeroes `∂v∂τ` and
`Rv0` when it starts and recomputes the others from the current state on every assembly.
"""
const STOKES_SCRATCH_FIELDS = (:∂v∂τ, :Rv, :Rv0, :∂Rv∂v, :PC_v, :∂P∂τ, :RP, :RP0, :Pnum)

"""
Fields of a `ThermalDiffusionDR` that are solver scratch, not state: the thermal solve zeroes `∂T∂τ` and
`R0` when it starts and recomputes the others from the current state on every assembly.
"""
const THERMAL_SCRATCH_FIELDS = (:R, :R0, :∂R∂T, :PC, :∂T∂τ)

"""
    physical_state(object) -> Tuple, NamedTuple or AbstractArray

Return the arrays of `object` that carry the physical state of a run, aliasing the live arrays.

Arrays pass through, tuples and named tuples map over their entries, and a `StokesDR`, a
`ThermalDiffusionDR`, an `IntegrationPointPlasticHistory` or a `CapPlasticHistory` gives a named tuple
of its state arrays.
Solver scratch (`STOKES_SCRATCH_FIELDS`, `THERMAL_SCRATCH_FIELDS`) is left out, and so are the immutable
per-phase parameters. `M_P` is derived from the phases, the time step and the geometry: it is restored
with them, and a caller that changes one of the three rebuilds it.
"""
physical_state(x::AbstractArray) = x
physical_state(x::Union{Tuple, NamedTuple}) = map(physical_state, x)
physical_state(::Nothing) = nothing
physical_state(history::FEMTools.IntegrationPointPlasticHistory) = (; history.λ, history.εpl, history.D)
physical_state(history::FEMTools.CapPlasticHistory) = (; history.γ, history.θ)
physical_state(thermal::ThermalDiffusionDR) = (; thermal.T, thermal.T0, thermal.P, thermal.source, thermal.phases)

function physical_state(dr::StokesDR)
    return (;
        v = Tuple(dr.v), τ = physical_state_tensor(dr.τ), τ_old = physical_state_tensor(dr.τ_old),
        dr.P, dr.P0, dr.T, dr.T0, dr.Q, dr.M_P, dr.phases_v, dr.phases_P,
        plastic_history = physical_state(dr.plastic_history),
    )
end

physical_state_tensor(::Nothing) = nothing
physical_state_tensor(τ) = Tuple(τ)

"""
    StateSnapshot

Deep copies of the arrays returned by [`physical_state`](@ref) and of the immutable `values` given to
[`capture_state`](@ref). It never aliases a live array, so it can be restored any number of times.
"""
struct StateSnapshot{A, V}
    arrays::A
    values::V
end

"""
    capture_state(objects::NamedTuple; values...) -> StateSnapshot

Copy the physical state of `objects`, for example `(; dr, thermal, history, coords)`, together with
`values`, the scalars and small tuples that are not arrays: physical time, cumulative injected volume,
event counters. Anything that changes length between capture and restore must go through `values`,
because arrays are restored in place.
"""
function capture_state(objects::NamedTuple; values...)
    return StateSnapshot(_copy_arrays(physical_state(objects)), deepcopy(NamedTuple(values)))
end

_copy_arrays(x::AbstractArray) = copy(x)
_copy_arrays(x::Union{Tuple, NamedTuple}) = map(_copy_arrays, x)
_copy_arrays(::Nothing) = nothing

"""
    restore_state!(objects::NamedTuple, snapshot::StateSnapshot) -> NamedTuple

Copy the captured arrays back into `objects`, which must have the structure it had at capture, and
return a copy of the captured `values`. Scratch is untouched; a size or element-type mismatch throws.
"""
function restore_state!(objects::NamedTuple, snapshot::StateSnapshot)
    _restore!(physical_state(objects), snapshot.arrays)
    return deepcopy(snapshot.values)
end

function _restore!(live::AbstractArray, saved::AbstractArray)
    axes(live) == axes(saved) && eltype(live) == eltype(saved) || throw(
        DimensionMismatch(
            "state array changed since capture: $(eltype(live)) $(size(live)) against $(eltype(saved)) $(size(saved))",
        ),
    )
    copyto!(live, saved)
    return nothing
end

function _restore!(live::Union{Tuple, NamedTuple}, saved::Union{Tuple, NamedTuple})
    keys(live) == keys(saved) || throw(ArgumentError("state structure changed since capture"))
    foreach(_restore!, live, saved)
    return nothing
end

_restore!(::Nothing, ::Nothing) = nothing
_restore!(live, saved) = throw(ArgumentError("state structure changed since capture: $(typeof(live)) against $(typeof(saved))"))
