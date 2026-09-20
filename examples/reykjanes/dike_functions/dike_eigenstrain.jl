"""Return the plane-strain eigenstrain increment for a dike opening.

The opening `w` is distributed across a numerical band of width `h` with
normal `n`.  The returned tensor includes the out-of-plane component so that
the deviatoric part is traceless in three dimensions, while the in-plane
components use the `(xx, yy, xy)` ordering of the 2-D Stokes stress history.
"""
function dike_eigenstrain_increment(w, h, n)
    isfinite(w) && isfinite(h) && w >= 0 ||
        throw(ArgumentError("dike opening must be finite and nonnegative"))
    isfinite(h) && h > 0 || throw(ArgumentError("dike band width must be finite and positive"))
    length(n) == 2 || throw(ArgumentError("a 2-D dike normal must have two components"))
    nx, ny = n
    isfinite(nx) && isfinite(ny) || throw(ArgumentError("dike normal must be finite"))
    norm_n = hypot(nx, ny)
    norm_n > 0 || throw(ArgumentError("dike normal must be nonzero"))

    f = w / h
    nx /= norm_n
    ny /= norm_n
    one_third = one(f) / 3
    return (
        xx = f * (nx * nx - one_third),
        yy = f * (ny * ny - one_third),
        zz = -f * one_third,
        xy = f * nx * ny,
        volumetric = f,
    )
end

"""Apply a dike's deviatoric eigenstrain to selected 2-D stress-history cells.

`τ_old` must be `(xx, yy, xy)` with one `Nq × nels` array per component.
`elements` is an iterable of 1-based element indices.  `G` may be a scalar,
an element vector, or an `Nq × nels` array; the latter is preferred when the
constitutive shear modulus is quadrature-point dependent.  The volumetric
part is intentionally not applied here: callers add `w / h / Δt` to the
continuity source using their physical pressure quadrature.
"""
function apply_dike_eigenstrain!(τ_old, G, strain, elements)
    length(τ_old) == 3 || throw(ArgumentError("2-D stress history needs (xx, yy, xy)"))
    size(τ_old[1]) == size(τ_old[2]) == size(τ_old[3]) ||
        throw(DimensionMismatch("stress-history components must have equal sizes"))
    size(τ_old[1], 1) > 0 || throw(ArgumentError("stress history has no quadrature points"))
    nels = size(τ_old[1], 2)
    all(isfinite, (strain.xx, strain.yy, strain.xy)) ||
        throw(ArgumentError("eigenstrain components must be finite"))

    for iel in elements
        1 <= iel <= nels || throw(BoundsError(τ_old[1], (:, iel)))
        for q in axes(τ_old[1], 1)
            Gq = G isa Number ? G : G isa AbstractVector ? G[iel] : G[q, iel]
            isfinite(Gq) && Gq >= 0 || throw(ArgumentError("shear modulus must be finite and nonnegative"))
            factor = 2 * Gq
            τ_old[1][q, iel] -= factor * strain.xx
            τ_old[2][q, iel] -= factor * strain.yy
            τ_old[3][q, iel] -= factor * strain.xy
        end
    end
    return τ_old
end

"""Add the volumetric dike source to cell-local pressure DoFs.

For the discontinuous P1 pressure layout used by the Reykjanes harness, a
constant source on an element is represented exactly by assigning the same
value to its three local pressure DoFs.  `Q` is modified in place and is
expected to be a host array at this workflow boundary.
"""
function add_dike_source!(Q, DoFsP, elements, volumetric_rate)
    isfinite(volumetric_rate) || throw(ArgumentError("dike source must be finite"))
    for iel in elements
        1 <= iel <= size(DoFsP, 2) || throw(BoundsError(DoFsP, (:, iel)))
        for a in axes(DoFsP, 1)
            Q[DoFsP[a, iel]] += volumetric_rate
        end
    end
    return Q
end

"""Apply one 2-D dike opening to stress history and the continuity source."""
function apply_dike_opening!(τ_old, Q, DoFsP, G, elements, w, h, n, Δt)
    isfinite(Δt) && Δt > 0 || throw(ArgumentError("dike time step must be finite and positive"))
    strain = dike_eigenstrain_increment(w, h, n)
    apply_dike_eigenstrain!(τ_old, G, strain, elements)
    add_dike_source!(Q, DoFsP, elements, strain.volumetric / Δt)
    return strain
end

@inline _dike_ip_value(x::Number, _q, _iel) = x
@inline _dike_ip_value(x::AbstractVector, _q, iel) = x[iel]
@inline _dike_ip_value(x::AbstractMatrix, q, iel) = x[q, iel]

"""Compute plane-strain shear and tensile diagnostics at every integration point.

The stored deviatoric stress is `(τxx, τyy, τxy)` and `τzz` is reconstructed
as `-(τxx + τyy)`.  Compression-positive stress is `S = P I - τ`.  The yield
residual uses the current FEMTools Drucker--Prager convention,
`F = τII - (C*cos(ϕ) + P*sin(ϕ))`.  `Pf` and `T0` define the independent
hydraulic tensile margin `Pf - s3 - T0`.

`C`, `ϕ`, `Pf`, and `T0` may be scalars, element vectors, or `nq × nels`
arrays.  `yield_tolerance` controls only the returned Boolean flag; the raw
residuals are always returned for threshold studies.
"""
function dike_failure_diagnostics(
        τ_ip, P_ip, C, ϕ, Pf, T0 = zero(eltype(τ_ip[1])); yield_tolerance = zero(eltype(τ_ip[1])),
    )
    length(τ_ip) == 3 || throw(ArgumentError("2-D stress input needs (τxx, τyy, τxy)"))
    size(τ_ip[1]) == size(τ_ip[2]) == size(τ_ip[3]) == size(P_ip) ||
        throw(DimensionMismatch("stress and pressure IP arrays must have equal sizes"))
    nq, nels = size(P_ip)
    FP = promote_type(eltype(τ_ip[1]), eltype(P_ip), typeof(yield_tolerance))
    s3 = zeros(FP, nq, nels)
    F = zeros(FP, nq, nels)
    hydraulic_margin = zeros(FP, nq, nels)
    orientation = fill(zero(FP), nq, nels)
    degenerate = falses(nq, nels)
    in_plane = falses(nq, nels)
    shear_failed = falses(nq, nels)
    tensile_failed = falses(nq, nels)

    for iel in 1:nels, q in 1:nq
        τxx, τyy, τxy = τ_ip[1][q, iel], τ_ip[2][q, iel], τ_ip[3][q, iel]
        Pq = P_ip[q, iel]
        τzz = -(τxx + τyy)
        τII = sqrt((τxx^2 + τyy^2 + τzz^2) / 2 + τxy^2)
        Cq = _dike_ip_value(C, q, iel)
        ϕq = _dike_ip_value(ϕ, q, iel)
        Pfq = _dike_ip_value(Pf, q, iel)
        T0q = _dike_ip_value(T0, q, iel)
        all(isfinite, (τxx, τyy, τxy, Pq, Cq, ϕq, Pfq, T0q)) ||
            throw(ArgumentError("dike failure inputs must be finite"))

        a = Pq - τxx
        d = Pq - τyy
        b = -τxy
        mid = (a + d) / 2
        radius = hypot((a - d) / 2, b)
        λin = mid - radius
        λout = Pq - τzz
        tol = sqrt(eps(FP)) * max(one(FP), abs(mid), abs(λout), abs(radius))
        s3q = min(λin, λout)
        degenerate[q, iel] = abs(λin - λout) <= tol || radius <= tol
        in_plane[q, iel] = λin <= λout + tol
        if in_plane[q, iel]
            # Eigenvector of the smaller in-plane eigenvalue, modulo π.
            vx, vy = b, λin - a
            if abs(vx) <= tol && abs(vy) <= tol
                vx, vy = λin - d, b
            end
            θ = atan(vy, vx)
            orientation[q, iel] = θ >= π / 2 ? θ - π : θ < -π / 2 ? θ + π : θ
        end

        Fq = τII - (Cq * cos(ϕq) + Pq * sin(ϕq))
        margin = Pfq - s3q - T0q
        s3[q, iel] = s3q
        F[q, iel] = Fq
        hydraulic_margin[q, iel] = margin
        shear_failed[q, iel] = Fq > yield_tolerance
        tensile_failed[q, iel] = margin >= zero(FP)
    end
    return (; s3, F, hydraulic_margin, orientation, degenerate, in_plane,
        shear_failed, tensile_failed, failed = shear_failed .| tensile_failed)
end

function _validate_dike_graph(adjacency, active)
    length(adjacency) == length(active) ||
        throw(DimensionMismatch("adjacency and active flags must have equal lengths"))
    nels = length(adjacency)
    for (iel, neighbors) in enumerate(adjacency), jel in neighbors
        1 <= jel <= nels || throw(BoundsError(adjacency, jel))
    end
    return nels
end

"""Return the active connected component reached from `seeds` in BFS order."""
function dike_connected_component(adjacency, active, seeds)
    nels = _validate_dike_graph(adjacency, active)
    queue = Int32[]
    visited = falses(nels)
    for seed in sort!(unique!(Int.(collect(seeds))))
        1 <= seed <= nels || throw(BoundsError(adjacency, seed))
        active[seed] || continue
        visited[seed] = true
        push!(queue, Int32(seed))
    end
    head = 1
    while head <= length(queue)
        iel = queue[head]
        head += 1
        for jel in adjacency[iel]
            active[jel] && !visited[jel] || continue
            visited[jel] = true
            push!(queue, Int32(jel))
        end
    end
    return queue
end

"""Return a deterministic shortest active path from seeds to any target."""
function dike_shortest_path(adjacency, active, seeds, targets)
    nels = _validate_dike_graph(adjacency, active)
    target_set = Set(Int.(collect(targets)))
    all(1 <= target <= nels for target in target_set) ||
        throw(BoundsError(adjacency, first(target_set)))
    parent = zeros(Int32, nels)
    queue = Int32[]
    for seed in sort!(unique!(Int.(collect(seeds))))
        1 <= seed <= nels || throw(BoundsError(adjacency, seed))
        active[seed] || continue
        parent[seed] = -1
        push!(queue, Int32(seed))
    end
    head = 1
    found = 0
    while head <= length(queue)
        iel = queue[head]
        head += 1
        if iel in target_set
            found = iel
            break
        end
        for jel in adjacency[iel]
            active[jel] && parent[jel] == 0 || continue
            parent[jel] = iel
            push!(queue, Int32(jel))
        end
    end
    found == 0 && return Int32[]
    path = Int32[]
    while found != -1
        push!(path, Int32(found))
        found = parent[found]
    end
    return reverse!(path)
end

# ---------------------------------------------------------------------------------------------
# The mesh-independent detector of GAP-11 / Gate G1.
#
# The adjacency detector asks whether a face-connected chain of failed elements spans from the
# reservoir to the target depth. That is a property of the element graph: refining changes which
# elements exist, so it changes when a chain closes even where the stress field has converged. The
# first-threshold mesh sweep measured 77.7% against a 5% gate for exactly this reason.
#
# The replacement asks the same physical question — is there failed host rock all the way from the
# reservoir crest to the target depth — of a *fixed geometric* corridor instead. The corridor and its
# bins are defined in metres and do not move with the mesh; the mesh only enters through the area
# fraction of failed material in each bin, which converges under refinement.

"""
    build_dike_corridor(centroids, areas; x_center, half_width, y_bottom, y_top, nbins, eligible = nothing)

Bin the elements of a mesh into a fixed vertical corridor, once, so that a detector can be evaluated
on geometry rather than on the element graph.

The corridor is the strip `|x − x_center| ≤ half_width` between `y_bottom` and `y_top`, cut into
`nbins` bins of equal height. An element joins the bin its centroid falls in, and `eligible`, a
per-element `Bool` vector when given, keeps material out of it altogether — the reservoir itself, for
instance, whose elements are not host rock. All lengths are in metres.

Returns the bin `edges`, the per-element bin index `bin` (`0` outside), the `members` and sampled
`bin_area` of each bin, the corridor `cells` and their total `area`, and `geometric_area`, the area
the corridor would have if the mesh tiled it exactly. A bin with no members is not an error here:
it is reported through `counts`, and the detector treats it as unfilled and under-resolved, because
a corridor the mesh cannot sample must not declare an event.
"""
function build_dike_corridor(
        centroids, areas;
        x_center, half_width, y_bottom, y_top, nbins, eligible = nothing,
    )
    nels = length(areas)
    length(centroids) == nels ||
        throw(DimensionMismatch("one centroid per element is needed, got $(length(centroids)) for $nels"))
    eligible === nothing || length(eligible) == nels ||
        throw(DimensionMismatch("one eligibility flag per element is needed"))
    half_width > 0 || throw(ArgumentError("the corridor half width must be positive"))
    y_top > y_bottom || throw(ArgumentError("the corridor must have positive height"))
    nbins >= 1 || throw(ArgumentError("the corridor needs at least one bin"))
    all(isfinite, (x_center, half_width, y_bottom, y_top)) ||
        throw(ArgumentError("the corridor geometry must be finite"))

    FP = promote_type(typeof(float(x_center)), typeof(float(y_bottom)), eltype(areas))
    edges = collect(range(FP(y_bottom), FP(y_top); length = nbins + 1))
    Δy = (FP(y_top) - FP(y_bottom)) / nbins

    bin = zeros(Int, nels)
    members = [Int32[] for _ in 1:nbins]
    for iel in 1:nels
        (eligible === nothing || eligible[iel]) || continue
        x, y = centroids[iel]
        abs(x - x_center) <= half_width || continue
        y_bottom <= y <= y_top || continue
        k = clamp(floor(Int, (y - y_bottom) / Δy) + 1, 1, nbins)
        bin[iel] = k
        push!(members[k], Int32(iel))
    end

    bin_area = FP[isempty(m) ? zero(FP) : sum(areas[iel] for iel in m) for m in members]
    cells = sort!(reduce(vcat, members; init = Int32[]))
    return (;
        x_center = FP(x_center), half_width = FP(half_width), y_bottom = FP(y_bottom),
        y_top = FP(y_top), edges, bin, members, bin_area, counts = length.(members), cells,
        area = sum(bin_area), geometric_area = 2 * FP(half_width) * (FP(y_top) - FP(y_bottom)),
    )
end

"""
    corridor_failed_weight(failed, iel) -> Real

How much of element `iel` has failed, in [0, 1]. `failed` is either one flag per element, which gives
0 or 1, or the `nq × nels` flags of the integration points, which gives the fraction of the element's
quadrature that failed.

The point form is what the detector uses. Reducing an element to a single flag before measuring an
area is a coarsening on top of the mesh: at the resolutions this section is run at, a corridor bin
holds a handful of elements, and a Boolean per element would leave the fraction as quantised as the
adjacency rule it replaces.
"""
corridor_failed_weight(failed::AbstractVector, iel) = failed[iel] ? 1.0 : 0.0
function corridor_failed_weight(failed::AbstractMatrix, iel)
    nq = size(failed, 1)
    return count(@view failed[:, iel]) / nq
end

"""
    corridor_bin_fractions(corridor, failed, areas) -> Vector

The failed-area fraction of each bin of `corridor`: the failed area in it over the area it samples,
with each element contributing [`corridor_failed_weight`](@ref) of its own area. The denominator is
the area the mesh actually put in the bin, not the bin's geometric area, so the fraction is unbiased
at any resolution and converges to the true fraction under refinement. An unsampled bin gives zero.
"""
function corridor_bin_fractions(corridor, failed, areas)
    _check_corridor_flags(failed, areas)
    FP = eltype(corridor.bin_area)
    fractions = zeros(FP, length(corridor.members))
    for k in eachindex(corridor.members)
        iszero(corridor.bin_area[k]) && continue
        failed_area = sum(
            areas[iel] * corridor_failed_weight(failed, iel) for iel in corridor.members[k];
            init = zero(FP),
        )
        fractions[k] = failed_area / corridor.bin_area[k]
    end
    return fractions
end

"""
    corridor_failed_fraction(corridor, failed, areas) -> Real

The failed-area fraction of the whole corridor, weighted as in [`corridor_bin_fractions`](@ref). Zero
when the corridor samples nothing.
"""
function corridor_failed_fraction(corridor, failed, areas)
    _check_corridor_flags(failed, areas)
    FP = eltype(corridor.bin_area)
    iszero(corridor.area) && return zero(FP)
    failed_area = sum(
        areas[iel] * corridor_failed_weight(failed, iel) for iel in corridor.cells; init = zero(FP)
    )
    return FP(failed_area) / corridor.area
end

function _check_corridor_flags(failed, areas)
    nels = failed isa AbstractMatrix ? size(failed, 2) : length(failed)
    nels == length(areas) ||
        throw(DimensionMismatch("one failure flag per element or per integration point is needed"))
    return nels
end

"""
    corridor_cells(corridor, failed; bins = eachindex(corridor.members)) -> Vector{Int32}

The failed elements of the given bins of `corridor`, in element order: the elements with any failed
quadrature in them, which is the same reduction [`failed_elements`](@ref) makes. This is the band the
intrusion opens, and it is defined by the corridor's geometry and the failure field, not by a graph
search.
"""
function corridor_cells(corridor, failed; bins = eachindex(corridor.members))
    cells = Int32[]
    for k in bins, iel in corridor.members[k]
        corridor_failed_weight(failed, iel) > 0 && push!(cells, iel)
    end
    return sort!(cells)
end

"""
    corridor_verdict(corridor, failed, areas, rule, fill_fraction) -> NamedTuple

Evaluate the mesh-independent detector on `corridor`.

`rule` is `:corridor_column`, which asks that *every* bin from the reservoir crest to the target
depth be filled to `fill_fraction` — the geometric statement of "failed rock spans the corridor" —
or `:corridor_fraction`, which asks only that the corridor as a whole be that full. The column rule
keeps the depth-spanning meaning of the adjacency detector; the fraction rule is the smoother
measure, and is kept so that the sweep can quote both.

Returns `tripped`, the per-bin `fractions`, the corridor `fraction`, the failed `cells` of the bins
that count, and `under_resolved`, true when a bin has no elements at all. An under-resolved corridor
never trips: a bin the mesh cannot sample is a resolution failure, not an absence of failure.
"""
function corridor_verdict(corridor, failed, areas, rule::Symbol, fill_fraction)
    0 < fill_fraction <= 1 || throw(ArgumentError("the corridor fill fraction must be in (0, 1]"))
    fractions = corridor_bin_fractions(corridor, failed, areas)
    fraction = corridor_failed_fraction(corridor, failed, areas)
    under_resolved = any(iszero, corridor.bin_area)
    if rule === :corridor_column
        filled = findall(>=(fill_fraction), fractions)
        tripped = !under_resolved && length(filled) == length(fractions)
        cells = corridor_cells(corridor, failed; bins = filled)
    elseif rule === :corridor_fraction
        tripped = !under_resolved && fraction >= fill_fraction
        cells = corridor_cells(corridor, failed)
    else
        throw(ArgumentError("unknown corridor rule :$rule"))
    end
    return (; tripped, fractions, fraction, cells, under_resolved)
end
