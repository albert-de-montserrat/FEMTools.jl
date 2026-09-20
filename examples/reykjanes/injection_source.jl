# Volumetric sources for the continuity equation on a P1-disc pressure field, milestone GAP-10 of
# REYKJANES_PLAN.md. `Q` is indexed by pressure DoF (`mesh.DoFsP`) and `geo_P` holds the pressure
# quadrature weights at the velocity integration points, so both helpers integrate the way the
# continuity residual does. They run on host copies and are shared by the recharge, the dike sink and
# the tests.

"""
    pressure_source_integral(Q, DoFsP, geo_P, NqP, elements = axes(DoFsP, 2)) -> Real

Integrate the pressure-DoF source `Q` over `elements` with the quadrature of the continuity residual,
`Σₑ Σ_q dΩ_q Σₐ Nₐ(q) Q[DoFsP[a, e]]`.

`DoFsP` and `geo_P` are host copies of `mesh.DoFsP` and `mesh.geometry.geo_P`, and `NqP` is
`shape_function_values(element_P, element_v.integration_points)`. The result carries the units of `Q`
times those of `geo_P`, so it is the injected area per unit time when `geo_P` is an area.
"""
function pressure_source_integral(Q, DoFsP, geo_P, NqP, elements = axes(DoFsP, 2))
    total = zero(promote_type(eltype(Q), eltype(eltype(geo_P))))
    for iel in elements
        weights = geo_P[iel]
        for q in eachindex(weights)
            source = zero(total)
            for a in axes(DoFsP, 1)
                source += NqP[q][a] * Q[DoFsP[a, iel]]
            end
            total += weights[q] * source
        end
    end
    return total
end

"""
    uniform_pressure_source(rate, DoFsP, geo_P, elements) -> Vector

Return the pressure-DoF source that is constant on `elements` and zero elsewhere, scaled so that
[`pressure_source_integral`](@ref) over `elements` equals `rate`.

`rate` is a volume per unit time, an area per second in 2-D, in the units of `geo_P`; a positive value
injects. In characteristic units pass `V̇ t_c / L_c²`. A negative `rate` is the sink that closes a
transfer: give it minus the integral of the source it removes and the two integrate to zero over the
mesh, which is cancellation of the source terms and not a mass budget.
"""
function uniform_pressure_source(rate, DoFsP, geo_P, elements)
    isfinite(rate) || throw(ArgumentError("source rate must be finite"))
    region = unique(elements)
    isempty(region) && throw(ArgumentError("source region has no elements"))
    area = sum(sum(geo_P[iel]) for iel in region)
    area > 0 || throw(ArgumentError("source region must have positive area"))

    Q = zeros(promote_type(typeof(rate), eltype(eltype(geo_P))), maximum(DoFsP))
    for iel in region, a in axes(DoFsP, 1)
        Q[DoFsP[a, iel]] = rate / area
    end
    return Q
end

"""
    balance_dike_source!(Q, DoFsP, geo_P, NqP, source_elements, sink_elements) -> Real

Add to `Q` the uniform sink over `sink_elements` that cancels the integral of the existing source over
`source_elements`, and return the integral it cancelled.

This is the reservoir sink of a dike transfer: the band's source and the reservoir's sink then
integrate to zero over the mesh with the quadrature of the continuity residual, to roundoff. That is
cancellation of the source terms, not a budget of magma mass: reservoir and band storage,
deformation, density, thermal expansion and boundary flux are separate quantities.

The two element sets must not overlap, or the sink would be added on top of the source it is meant to
remove.
"""
function balance_dike_source!(Q, DoFsP, geo_P, NqP, source_elements, sink_elements)
    source = unique(source_elements)
    sink = unique(sink_elements)
    isdisjoint(source, sink) || throw(ArgumentError("the sink must not overlap the source it balances"))
    isempty(sink) && throw(ArgumentError("the sink region has no elements"))

    injected = pressure_source_integral(Q, DoFsP, geo_P, NqP, source)
    area = sum(sum(geo_P[iel]) for iel in sink)
    area > 0 || throw(ArgumentError("the sink region must have positive area"))
    for iel in sink, a in axes(DoFsP, 1)
        Q[DoFsP[a, iel]] -= injected / area
    end
    return injected
end
