# Pressurised elliptical cavity in an elastic plane: milestone M0 of REYKJANES_PLAN.md.
#
# Runs the default case of `solve_elliptical_cavity` from `elliptical_cavity_setup.jl`: a soft,
# compressible elliptical inclusion injected through the volumetric source `Q` inside a Maxwell host,
# one elastic step, compared with the closed-form plane-strain solution. The setup is function-only so
# that `test/test_elliptical_cavity.jl` and the sweeps in
# `examples/benchmarks/stokes/elliptical_cavity/elliptical_cavity.jl` share it.

include(joinpath(@__DIR__, "elliptical_cavity_setup.jl"))

"""
    main(; kwargs...) -> NamedTuple

Solve the default cavity case with `verbose = true`, print the comparison with the closed form and
return the `solve_elliptical_cavity` result. Keyword arguments go to `solve_elliptical_cavity`.
"""
main(; kwargs...) = print_cavity_report(solve_elliptical_cavity(; verbose = true, kwargs...))

main()
