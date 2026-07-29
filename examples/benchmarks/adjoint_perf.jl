#
# Adjoint solver benchmark for the sinking-block problem.
#
# Sweeps mesh refinement and viscosity contrast, reporting the cost of the
# adjoint solve both in wall time and in iterations. The two are reported
# separately on purpose: a change that reduces the cost of one iteration and a
# change that reduces the number of iterations are different kinds of
# improvement, and a change that trades one for the other has to be visible.
#
# The forward numbers are included as the reference scale. The adjoint runs at
# the converged forward state, so its cost is only meaningful relative to the
# solve that produced that state.
#
# Run from a warm session: the numbers of interest are steady-state solver
# throughput, not time-to-first-solve, so compilation must be amortised by the
# warm-up run rather than folded into the measurement.
#
#     julia --project=examples
#     include("examples/benchmarks/adjoint_perf.jl")
#     results = run_adjoint_benchmark()
#

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "stokes", "sinking_block", "sinking_block_adj.jl"))

"""
    adjoint_benchmark_case(; max_area, η_incl, kwargs...) -> NamedTuple

Run one sinking-block forward and adjoint solve and reduce it to the scalars the
benchmark tracks.

`per_iter_ms` is the adjoint wall time divided by its cumulative inner iteration
count, which is the quantity Phase 1 and Phase 2 of the performance work move.
`itPH` and `iter` are the quantities Phase 3 moves. `converged` must be true for
a row to mean anything: an unconverged solve reports the cost of hitting an
iteration budget, not the cost of solving the problem.
"""
function adjoint_benchmark_case(; max_area, η_incl, kwargs...)
    out = main(;
        max_area, η_incl,
        show_plot = false,
        verbose = false,
        adjoint_verbose = false,
        adjoint_verbose_inner = false,
        kwargs...,
    )

    fwd, adj = out.solve_stats, out.adjoint_stats
    return (;
        max_area,
        η_incl,
        nels = out.mesh_stokes.nels,
        nnodes = out.mesh_stokes.nnodes,
        fwd_iter = fwd.iter,
        fwd_itPH = fwd.itPH,
        fwd_time = out.t_forward,
        fwd_converged = fwd.converged,
        adj_iter = adj.iter,
        adj_itPH = adj.itPH,
        adj_time = out.t_adjoint,
        adj_converged = adj.converged,
        adj_err = adj.err,
        per_iter_ms = 1.0e3 * out.t_adjoint / max(adj.iter, 1),
        adjoint_over_forward = out.t_adjoint / out.t_forward,
        density_gradient = out.density_gradient_by_phase,
        viscosity_gradient = out.viscosity_gradient_by_phase,
    )
end

"""
    run_adjoint_benchmark(; max_areas, η_contrasts, warmup=true, kwargs...) -> Vector{NamedTuple}

Sweep the benchmark over mesh refinements and viscosity contrasts, printing one
row per case and returning the rows.

The default `η_contrasts` deliberately spans four orders of magnitude. At unit
contrast the Stokes operator is trivially conditioned and the iteration counts
say nothing about how the solver behaves on the problems it exists for, so a
convergence result measured only at `η_incl = 1` is not a result.

The warm-up runs the smallest case once and discards it, so that compilation
does not land in the first measured row.
"""
function run_adjoint_benchmark(;
        max_areas = (1 / 32^2, 1 / 64^2, 1 / 128^2),
        η_contrasts = (1.0, 1.0e1, 1.0e2, 1.0e3),
        warmup = true,
        kwargs...,
    )
    if warmup
        @info "Warm-up run (discarded)"
        adjoint_benchmark_case(; max_area = 1 / 16^2, η_incl = 1.0, kwargs...)
    end

    @printf("%10s %8s %8s | %8s %10s | %6s %8s %10s %10s %8s\n",
        "max_area", "η_incl", "nels",
        "fwd_iter", "fwd_time", "PH", "adj_iter", "adj_time", "per_it_ms", "adj/fwd")
    println("-"^100)

    results = NamedTuple[]
    for max_area in max_areas, η_incl in η_contrasts
        r = adjoint_benchmark_case(; max_area, η_incl, kwargs...)
        push!(results, r)
        flag = (r.fwd_converged && r.adj_converged) ? "" : "  <- NOT CONVERGED"
        @printf("%10.3e %8.0e %8d | %8d %10.3f | %6d %8d %10.3f %10.4f %8.2f%s\n",
            r.max_area, r.η_incl, r.nels,
            r.fwd_iter, r.fwd_time,
            r.adj_itPH, r.adj_iter, r.adj_time, r.per_iter_ms, r.adjoint_over_forward,
            flag)
    end
    return results
end

"""
    compare_adjoint_benchmarks(baseline, candidate)

Print the per-case change between two `run_adjoint_benchmark` result vectors,
matched on `(max_area, η_incl)`.

Iteration count and wall time are reported as separate ratios so that the two
kinds of improvement stay distinguishable.
"""
function compare_adjoint_benchmarks(baseline, candidate)
    key(r) = (r.max_area, r.η_incl)
    lookup = Dict(key(r) => r for r in baseline)

    @printf("%10s %8s | %12s %12s %12s\n",
        "max_area", "η_incl", "iter ratio", "time ratio", "per_it ratio")
    println("-"^62)
    for c in candidate
        b = get(lookup, key(c), nothing)
        b === nothing && continue
        @printf("%10.3e %8.0e | %12.3f %12.3f %12.3f\n",
            c.max_area, c.η_incl,
            c.adj_iter / b.adj_iter,
            c.adj_time / b.adj_time,
            c.per_iter_ms / b.per_iter_ms)
    end
    return nothing
end
