import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Printf

include(joinpath(@__DIR__, "stokes_2D_compressible_inclusion.jl"))

"""
    convergence_sweep(; resolutions, curved, kwargs...) -> Vector{NamedTuple}

Solve the compressible inclusion problem at each resolution and report the
observed L2 convergence rates.

The rate, not the error at any single resolution, is what identifies the
discretisation as correct: an error that is small but stagnant under
refinement indicates a consistency error, which a single mesh cannot
distinguish from a well-resolved solution.
"""
function convergence_sweep(;
        resolutions = (16, 24, 32, 48),
        curved = true,
        ϵ_tol = 1.0e-10,
        verbose = false,
        kwargs...,
    )
    results = NamedTuple[]
    for n in resolutions
        out = main(; max_area = 1 / n^2, curved, ϵ_tol, verbose, kwargs...)
        push!(results, (; n, out.h, out.nels, out.rel_v, out.rel_p))
        @printf("n=%3d  nels=%6d  h=%.5f  ‖Δv‖/‖v‖=%.4e  ‖ΔP‖/‖P‖=%.4e\n",
            n, out.nels, out.h, out.rel_v, out.rel_p)
    end

    println("\nObserved rates (successive pairs, rate = log(e₁/e₂)/log(h₁/h₂)):")
    for i in 2:length(results)
        a, b = results[i - 1], results[i]
        rv = log(a.rel_v / b.rel_v) / log(a.h / b.h)
        rp = log(a.rel_p / b.rel_p) / log(a.h / b.h)
        @printf("  n %3d -> %3d :  velocity %.2f   pressure %.2f\n", a.n, b.n, rv, rp)
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== curved interface (isoparametric) ===")
    convergence_sweep(; curved = true)
    println("\n=== straight-sided interface ===")
    convergence_sweep(; curved = false)
end
