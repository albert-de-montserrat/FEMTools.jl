#
# Memory-footprint benchmark for the Stokes DYREL solvers.
#
# Reports where solver memory goes, component by component and normalised per
# element, alongside the wall time, iteration count, and host allocation volume
# of a forward solve on the same state. The three are reported together because
# a change that trades one for another has to be visible: a smaller footprint
# bought with more arithmetic per iteration is a different result from a smaller
# footprint at equal speed, and only the side-by-side table distinguishes them.
#
# Footprints are `sizeof(eltype(A)) * length(A)` summed over the arrays the
# solver holds. That counts payload bytes only, so a host run and a device run of
# the same case report identical numbers and the array header is not mistaken for
# solver state.
#
# Run from a warm session, after a discarded warm-up: the quantities of interest
# are steady-state footprint and throughput, not time-to-first-solve.
#
#     julia --project=.
#     include("examples/benchmarks/memory_footprint.jl")
#     results = run_memory_benchmark()
#     write_memory_baseline(results, joinpath(@__DIR__, "memory_baseline.jl"))
#
# From a session running modified code, compare against that record:
#
#     base = read_memory_baseline(joinpath(@__DIR__, "memory_baseline.jl"))
#     compare_memory_footprints(base, run_memory_benchmark())
#

using Printf
using Statistics
using StaticArrays
using DomainSets
using KernelAbstractions
import KernelAbstractions as KA
using FEMTools
import FEMTools

"""
    device_bytes(x) -> Int

Payload bytes of `x`: the element size times the length of every array reachable
from it, ignoring array headers and host-side wrappers.

Arrays are measured the same way whether they live on the host or on a device, so
a footprint taken on `CPU()` is directly comparable to one taken on a GPU
backend. `nothing` contributes nothing, which lets optional solver state be
summed without special-casing.
"""
device_bytes(A::AbstractArray) = sizeof(eltype(A)) * length(A)
device_bytes(::Nothing) = 0
device_bytes(x::Union{Tuple, NamedTuple}) = sum(device_bytes, x; init = 0)
device_bytes(x::Union{FEMTools.AbstractVectorField, FEMTools.AbstractSymmetricTensor}) =
    sum(f -> device_bytes(getfield(x, f)), fieldnames(typeof(x)); init = 0)

# Deviatoric stress history and per-node phase tags are tracked as their own
# components: both are candidates for removal or narrowing, so they must not be
# hidden inside the nodal total.
const STRESS_FIELDS = (:τ, :τ_old)
const PHASE_FIELDS = (:phases_v, :phases_P)

"""
    stokes_dr_bytes(dr) -> NamedTuple

Split the bytes held by a `StokesDR` into `stress`, `phase`, and `nodal`.
"""
function stokes_dr_bytes(dr)
    stress = sum(f -> device_bytes(getfield(dr, f)), STRESS_FIELDS; init = 0)
    phase = sum(f -> device_bytes(getfield(dr, f)), PHASE_FIELDS; init = 0)
    nodal = 0
    for f in fieldnames(typeof(dr))
        f in STRESS_FIELDS && continue
        f in PHASE_FIELDS && continue
        # Scalar solver parameters and the per-phase material tuples are not
        # bulk state; only the field arrays scale with the mesh.
        v = getfield(dr, f)
        v isa Union{AbstractArray, FEMTools.AbstractVectorField} && (nodal += device_bytes(v))
    end
    return (; stress, phase, nodal)
end

"""
    mixed_stokes_footprint(mesh, dr) -> NamedTuple

Byte counts of one 2-D mixed Stokes state, by component, plus the total and the
total per element.

`geo_v` and `geo_P` are the element geometry precomputed in `mesh.geometry`, `topology` the
coordinates and both connectivities and degree-of-freedom maps, and `normals` the
outward nodal normals carried by the mixed mesh.
"""
function mixed_stokes_footprint(mesh, dr)
    parts = stokes_dr_bytes(dr)
    geo_v = device_bytes(mesh.geometry.geo_v)
    geo_P = device_bytes(mesh.geometry.geo_P)
    topology = device_bytes((mesh.coords, mesh.el2n, mesh.DoFs, mesh.el2nP, mesh.DoFsP))
    normals = device_bytes(mesh.normals)
    total = geo_v + geo_P + topology + normals + parts.stress + parts.phase + parts.nodal
    return (;
        geo_v, geo_P,
        stress = parts.stress, nodal = parts.nodal, phase = parts.phase,
        topology, normals,
        total, bytes_per_element = total / mesh.nels,
    )
end

"""
    structured_stokes_case(; nx, ny, backend, workgroup, η_incl, ρ_incl) -> NamedTuple

Build a structured T7/P1-disc sinking-block state on the unit square: free-slip
walls, a centred square inclusion of viscosity `η_incl` and density `ρ_incl`
against a unit-viscosity, unit-density matrix.

Returns the mesh with its geometry, solver state, and everything
`solve_stokes_dyrel!` needs, so the footprint and the timed solve are measured on
one and the same state. The mesh is structured rather than generated, which keeps
the benchmark independent of the mesher and reproducible across backends.
"""
function structured_stokes_case(;
        nx = 64, ny = nx, backend = CPU(), workgroup = 128,
        η_incl = 1.0e2, ρ_incl = 2.0, half_width = 0.1, γfact = 20.0,
        stress_history = true,
    )
    Lx = Ly = 1.0
    Ω = DomainSets.:(×)(0.0 .. Lx, 0.0 .. Ly)

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})

    mesh_v = Mesh(backend, Ω, element_v, (nx, ny); workgroup)
    mesh = MixedMesh(mesh_v, element_P; workgroup)

    η = (1.0, η_incl)
    ρ0 = (1.0, ρ_incl)
    material = StokesMaterial(;
        η, ηb = (Inf, Inf), G = (Inf, Inf), α = (0.0, 0.0),
        ρ0, K = (Inf, Inf), g = (0.0, -1.0), Tref = 0.0,
    )
    # The case is purely viscous (G = Inf), so its stress history is never read.
    # `stress_history = false` measures what a model that admits that costs.
    NQ_v = length(element_v.integration_points.ω)
    dr = StokesDR(
        backend, mesh.nnodes, mesh.nnodesP, material;
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9,
        stress_size = stress_history ? (NQ_v, mesh.nels) : :none,
    )

    # Phases and boundary-node lists are built on the host from host copies of the
    # mesh, then moved to the solver's backend: every array a kernel touches has
    # to live where the kernel runs.
    TDev = FEMTools.TA(backend)
    coords = Array(mesh.coords)
    el2nP = Array(mesh.el2nP)
    in_incl(c) = abs(c[1] - Lx / 2) ≤ half_width && abs(c[2] - Ly / 2) ≤ half_width
    cell_phase = Int[
        in_incl(sum(coords[el2nP[a, iel]] for a in 1:3) / 3) ? 2 : 1
        for iel in 1:mesh.nels
    ]
    phases = TDev(reshape(cell_phase, 1, :))

    # Free slip: zero normal velocity on each wall, tangential velocity free.
    tol = max(Lx, Ly) * eps(Float64) * 32
    Γnodes = Array(mesh_v.Γnodes)
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]
    bc_vx = DirichletBoundaryCondition(nothing, TDev(vx_nodes), KA.zeros(backend, Float64, length(vx_nodes)))
    bc_vy = DirichletBoundaryCondition(nothing, TDev(vy_nodes), KA.zeros(backend, Float64, length(vy_nodes)))
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)

    # Hydrostatic initial pressure. The Powell-Hestenes iteration starts from the
    # supplied pressure, so a matrix-hydrostatic guess is what makes the outer
    # loop advance at a rate representative of a production run.
    DoFsP = Array(mesh.DoFsP)
    P_hydro = zeros(Float64, mesh.nnodesP)
    for iel in 1:mesh.nels, a in 1:3
        P_hydro[DoFsP[a, iel]] = ρ0[1] * abs(material.g[2]) * (Ly - coords[el2nP[a, iel]][2])
    end
    copyto!(dr.P, P_hydro)
    copyto!(dr.P0, P_hydro)

    P_hydro_dev = TDev(P_hydro)

    ηγP = ntuple(_ -> mean(η), Val(length(η)))
    γP = KA.zeros(backend, Float64, mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, γfact, 1.0; workgroup, phases_v = phases, η = ηγP,
    )

    τ_old = stress_history ? (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy) : nothing

    # Restore the exact state the solver was first handed. Timing and allocation
    # are compared across repeated solves, and those only measure the same work
    # if every run starts from the same iterate: pressure left over from a
    # previous solve changes the Powell-Hestenes path and with it how often the
    # Jacobian and the spectral estimates are rebuilt.
    function reset!()
        for f in (dr.v.x, dr.v.y, dr.∂v∂τ.x, dr.∂v∂τ.y, dr.∂P∂τ,
                  dr.Rv.x, dr.Rv.y, dr.Rv0.x, dr.Rv0.y, dr.RP, dr.RP0, dr.Pnum)
            fill!(f, 0)
        end
        copyto!(dr.P, P_hydro_dev)
        copyto!(dr.P0, P_hydro_dev)
        apply_bc!(dr.v.x, bc_vx)
        apply_bc!(dr.v.y, bc_vy)
        return nothing
    end

    return (; mesh, dr, phases, bc_vx, bc_vy, γP, τ_old, workgroup, reset!)
end

"""
    memory_footprint_case(; nx, ny, solve, iter_budget, kwargs...) -> NamedTuple

One benchmark row: the component footprint of a structured T7/P1-disc state and,
with `solve = true`, the cost of solving it.

`iter_budget` selects between the two ways of timing this solver. Left as
`nothing`, the solve runs to `ϵ_tol` and its iteration count is the quantity that
must not move: a memory change that alters it has altered the arithmetic. Given a
budget, the solve instead runs exactly that many inner iterations, so two runs
perform identical work and their wall times and allocation volumes compare
directly.

`alloc_MiB` is host allocation volume over the whole solve, also reported per
inner iteration: transient churn inside the iteration is what scales, and over a
long run it is what drives allocator pressure.
"""
function memory_footprint_case(; nx = 64, ny = nx, solve = true, iter_budget = nothing,
        ϵ_tol = 1.0e-6, ncheck = 50, total_iterMax = 50_000, kwargs...)
    case = structured_stokes_case(; nx, ny, kwargs...)
    fp = mixed_stokes_footprint(case.mesh, case.dr)

    row(extra) = (;
        nx, ny, nels = case.mesh.nels, nnodes = case.mesh.nnodes,
        nnodesP = case.mesh.nnodesP, iter_budget, fp..., extra...,
    )

    solve || return row((;
        time = missing, iter = missing, itPH = missing,
        converged = missing, alloc_MiB = missing, alloc_KiB_per_iter = missing,
    ))

    # A budgeted run must not stop early, so no residual can satisfy its
    # tolerance: the iteration count is the control variable, not the residual.
    tol = iter_budget === nothing ? ϵ_tol : 0.0
    budget = iter_budget === nothing ? total_iterMax : iter_budget

    args = (case.dr, case.mesh, case.bc_vx, case.bc_vy, 1.0, case.γP)
    opts = (;
        phases_v = case.phases, phases_P = case.phases, τ_old = case.τ_old,
        plastic = nothing, workgroup = case.workgroup,
        ncheck, ϵ_tol = tol, total_iterMax = budget, rel_drop0 = 1.0e-1,
        verbose = false, verbose_inner = false,
    )

    backend = KA.get_backend(case.dr.v.x)

    stats = solve_stokes_dyrel!(args...; opts...)
    KA.synchronize(backend)

    case.reset!()
    t = @elapsed begin
        stats = solve_stokes_dyrel!(args...; opts...)
        KA.synchronize(backend)
    end

    case.reset!()
    alloc = @allocated solve_stokes_dyrel!(args...; opts...)

    return row((;
        time = t, iter = stats.iter, itPH = stats.itPH, converged = stats.converged,
        alloc_MiB = alloc / 2^20, alloc_KiB_per_iter = alloc / stats.iter / 2^10,
    ))
end

"""
    geometry_footprint_case(element, nels; backend, workgroup) -> NamedTuple

Bytes held by the precomputed geometry of a structured single-field mesh.

This is the 3-D probe. A Hex27 forward solve is far too expensive to run on every
comparison, but the geometry array is the term that dominates a 3-D footprint, so
building the mesh alone measures what the plan moves.
"""
_element_label(::ReferenceElement{T}) where {nDim, N, T <: AbstractElement{nDim, N}} =
    string(nameof(T), "{", nDim, ",", N, "}")

function geometry_footprint_case(element, nels; backend = CPU(), workgroup = 128)
    nDim = length(nels)
    Ω = nDim == 2 ?
        DomainSets.:(×)(0.0 .. 1.0, 0.0 .. 1.0) :
        DomainSets.:(×)(0.0 .. 1.0, 0.0 .. 1.0, 0.0 .. 1.0)
    mesh = Mesh(backend, Ω, element, nels; workgroup)
    geometry = device_bytes(mesh.geometry)
    topology = device_bytes((mesh.coords, mesh.el2n, mesh.DoFs, mesh.Γnodes))
    return (;
        element = _element_label(element), nels = mesh.nels,
        nnodes = mesh.nnodes, geometry, topology,
        geometry_per_element = geometry / mesh.nels,
        geometry_share = geometry / (geometry + topology),
    )
end

"""
    run_memory_benchmark(; footprint_resolutions, converge_resolution, budget_resolution,
                         iter_budget, geometry_cases, backend, workgroup, warmup) -> NamedTuple

Sweep the 2-D Stokes cases and the single-field geometry probes, print one table
each, and return all three sets of rows.

The three Stokes sets answer different questions. `footprint` rows are built but
not solved, so the memory map can be taken at a resolution whose dynamic
relaxation would be too slow to iterate to convergence. The `converged` row
carries the iteration count that a memory change must leave alone. The `budgeted`
row runs a fixed number of inner iterations, giving wall-time and allocation
numbers over identical arithmetic.

The returned value is the unit of comparison: record it with
[`write_memory_baseline`](@ref) before a change and pass it to
[`compare_memory_footprints`](@ref) after.
"""
function run_memory_benchmark(;
        footprint_resolutions = (32, 64),
        converge_resolution = 16,
        budget_resolution = 32,
        iter_budget = 20_000,
        geometry_cases = (
            (ReferenceElement(QuadraticElement{2, 7, Float64}), (64, 64)),
            (ReferenceElement(LinearElement{3, 8, Float64}), (32, 32, 32)),
            (ReferenceElement(QuadraticElement{3, 27, Float64}), (16, 16, 16)),
        ),
        backend = CPU(), workgroup = 128, warmup = true, kwargs...,
    )
    warmup && memory_footprint_case(; nx = 8, backend, workgroup, iter_budget = 200, kwargs...)

    footprint = [memory_footprint_case(; nx, backend, workgroup, solve = false, kwargs...)
                 for nx in footprint_resolutions]
    converged = memory_footprint_case(; nx = converge_resolution, backend, workgroup, kwargs...)
    budgeted = memory_footprint_case(; nx = budget_resolution, backend, workgroup, iter_budget, kwargs...)
    geometry = [geometry_footprint_case(el, n; backend, workgroup) for (el, n) in geometry_cases]

    results = (;
        footprint, converged, budgeted, geometry,
        environment = (;
            nthreads = Threads.nthreads(),
            julia = string(VERSION),
            backend = string(nameof(typeof(backend))),
        ),
    )
    print_memory_table(results)
    return results
end

_mib(b) = b / 2^20

function print_memory_table(results)
    env = results.environment
    @printf("\nJulia %s, %d thread(s), %s backend\n", env.julia, env.nthreads, env.backend)
    println("\n2-D Stokes T7/P1-disc — footprint (MiB)")
    println("  nels    geo_v   geo_P  stress   nodal   phase    topo    norm   total     B/el")
    for r in results.footprint
        @printf("%6d  %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %8.1f\n",
            r.nels, _mib(r.geo_v), _mib(r.geo_P), _mib(r.stress), _mib(r.nodal),
            _mib(r.phase), _mib(r.topology), _mib(r.normals), _mib(r.total),
            r.bytes_per_element)
    end

    println("\nForward solve")
    println("  case         nels     iter  itPH  converged     time s   alloc MiB   KiB/iter")
    for (label, r) in (("converged", results.converged), ("budgeted", results.budgeted))
        @printf("  %-10s %6d %8d %5d  %-9s %10.3f %11.1f %10.2f\n",
            label, r.nels, r.iter, r.itPH, string(r.converged),
            r.time, r.alloc_MiB, r.alloc_KiB_per_iter)
    end

    println("\nSingle-field geometry probe")
    println("  element                    nels   geometry MiB      B/el   share")
    for r in results.geometry
        @printf("  %-24s %7d   %12.3f %9.1f  %5.1f%%\n",
            r.element, r.nels, _mib(r.geometry), r.geometry_per_element,
            100 * r.geometry_share)
    end
    return nothing
end

"""
    residual_fingerprint(path; nx, kwargs...)

Write the momentum residual at one fixed, deterministic state to `path`, as raw
`Float64`.

This is how a memory change is shown to be arithmetically equivalent. Iteration
counts cannot do it: the assemblers scatter with atomics, so the order in which
element contributions are summed depends on thread scheduling, and two runs of
the *same* code reach convergence at different iteration counts. Run this under a
single thread before and after a change and compare the files byte for byte —
under one thread the summation order is fixed, so equal bytes mean the element
mathematics did not move.

    julia --project=. -t 1 -e 'include("examples/benchmarks/memory_footprint.jl"); residual_fingerprint("before.bin")'

then, after the change, the same with `"after.bin"` and `cmp before.bin after.bin`.
"""
function residual_fingerprint(path; nx = 16, kwargs...)
    Threads.nthreads() == 1 ||
        @warn "atomic scatter makes the fingerprint thread-order dependent; run with -t 1" nthreads = Threads.nthreads()
    case = structured_stokes_case(; nx, kwargs...)
    dr, mesh = case.dr, case.mesh
    geometry = mesh.geometry
    for i in eachindex(dr.v.x)
        dr.v.x[i] = sin(3.0i)
        dr.v.y[i] = cos(2.0i)
    end
    for i in eachindex(dr.P)
        dr.P[i] = 0.5 * sin(1.7i)
        dr.T[i] = 0.0
    end
    Rx, Ry = similar(dr.v.x), similar(dr.v.y)
    FEMTools.assemble_momentum_residual_matrices_atomix!(
        Rx, Ry, dr.v.x, dr.v.y, dr.P, dr.T, nothing,
        mesh.el2n, mesh.DoFsP, geometry.geo_v, mesh.nels,
        geometry.element_v, geometry.element_P, case.phases,
        nothing, nothing, nothing,
        dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, 1.0,
        KA.get_backend(dr.v.x), case.workgroup,
    )
    open(path, "w") do io
        write(io, Array(Rx))
        write(io, Array(Ry))
    end
    return path
end

"""
    write_memory_baseline(results, path)

Write `results` to `path` as a Julia expression, so a baseline taken before a
change can be read back by a session running the changed code.

Comparison across a code change cannot happen in one session, which is why the
record goes to disk rather than staying in memory.
"""
function write_memory_baseline(results, path)
    open(path, "w") do io
        println(io, "# Memory-footprint baseline written by examples/benchmarks/memory_footprint.jl.")
        println(io, "# Read it back with `read_memory_baseline(path)`.")
        print(io, repr(results))
        println(io)
    end
    return path
end

"""
    read_memory_baseline(path) -> NamedTuple

Read a record written by [`write_memory_baseline`](@ref).
"""
read_memory_baseline(path) = include(abspath(path))

_ratio(a, b) = (a === missing || b === missing || iszero(b)) ? missing : a / b

function _fmt_ratio(r)
    r === missing && return "     -"
    return @sprintf("%5.2fx", r)
end

"""
    compare_memory_footprints(baseline, candidate)

Print baseline-to-candidate ratios for every tracked quantity and return the rows.

Ratios are stated as `baseline / candidate`, so a value above 1 is an
improvement — less memory, less time, fewer allocated bytes — and a value below 1
is a regression. Footprint rows are matched by element count; a case present in
one record and not the other is reported rather than silently dropped.

The converged row is the correctness check of the whole comparison. A memory
change is meant to leave the arithmetic alone, so its iteration ratio must be
exactly 1; anything else is reported as a defect.
"""
function compare_memory_footprints(baseline, candidate)
    b, c = baseline.environment, candidate.environment
    b == c || throw(ArgumentError(
        "baseline and candidate ran under different conditions ($b vs $c); " *
        "wall time and allocation volume are not comparable across them"))

    base = Dict(r.nels => r for r in baseline.footprint)
    cand = Dict(r.nels => r for r in candidate.footprint)

    only_base = setdiff(keys(base), keys(cand))
    only_cand = setdiff(keys(cand), keys(base))
    isempty(only_base) || @warn "cases only in the baseline" nels = sort!(collect(only_base))
    isempty(only_cand) || @warn "cases only in the candidate" nels = sort!(collect(only_cand))

    println("\n2-D Stokes footprint — baseline / candidate (>1 is an improvement)")
    println("  nels    geo_v   geo_P  stress   nodal   phase    topo    norm   total    B/el")
    footprint = NamedTuple[]
    for nels in sort!(collect(intersect(keys(base), keys(cand))))
        b, c = base[nels], cand[nels]
        row = (; nels,
            geo_v = _ratio(b.geo_v, c.geo_v),
            geo_P = _ratio(b.geo_P, c.geo_P),
            stress = _ratio(b.stress, c.stress),
            nodal = _ratio(b.nodal, c.nodal),
            phase = _ratio(b.phase, c.phase),
            topology = _ratio(b.topology, c.topology),
            normals = _ratio(b.normals, c.normals),
            total = _ratio(b.total, c.total),
            bytes_per_element = _ratio(b.bytes_per_element, c.bytes_per_element))
        push!(footprint, row)
        @printf("%6d  %s  %s  %s  %s  %s  %s  %s  %s  %s\n", nels,
            _fmt_ratio(row.geo_v), _fmt_ratio(row.geo_P), _fmt_ratio(row.stress),
            _fmt_ratio(row.nodal), _fmt_ratio(row.phase), _fmt_ratio(row.topology),
            _fmt_ratio(row.normals), _fmt_ratio(row.total),
            _fmt_ratio(row.bytes_per_element))
    end

    println("\nForward solve — baseline / candidate")
    println("  case         nels     iter    time   alloc")
    solve = NamedTuple[]
    for (label, b, c) in (("converged", baseline.converged, candidate.converged),
                          ("budgeted", baseline.budgeted, candidate.budgeted))
        b.nels == c.nels ||
            throw(ArgumentError("$label case ran at $(b.nels) elements in the baseline and $(c.nels) in the candidate"))
        row = (; label, b.nels,
            iter = _ratio(b.iter, c.iter),
            time = _ratio(b.time, c.time),
            alloc = _ratio(b.alloc_MiB, c.alloc_MiB))
        push!(solve, row)
        @printf("  %-10s %6d  %s  %s  %s\n", label, row.nels,
            _fmt_ratio(row.iter), _fmt_ratio(row.time), _fmt_ratio(row.alloc))
    end

    if baseline.converged.iter != candidate.converged.iter
        @info("converged iteration count moved; check it against the run-to-run spread of the " *
              "unchanged code before reading anything into it, and settle equivalence with " *
              "`residual_fingerprint` under one thread",
            baseline = baseline.converged.iter, candidate = candidate.converged.iter)
    end

    gbase = Dict((r.element, r.nels) => r for r in baseline.geometry)
    println("\nSingle-field geometry — baseline / candidate")
    println("  element                    nels   geometry    B/el")
    geometry = NamedTuple[]
    for c in candidate.geometry
        b = get(gbase, (c.element, c.nels), nothing)
        b === nothing && continue
        row = (; c.element, c.nels,
            geometry = _ratio(b.geometry, c.geometry),
            geometry_per_element = _ratio(b.geometry_per_element, c.geometry_per_element))
        push!(geometry, row)
        @printf("  %-24s %7d  %s  %s\n", row.element, row.nels,
            _fmt_ratio(row.geometry), _fmt_ratio(row.geometry_per_element))
    end

    return (; footprint, solve, geometry)
end
