# Eigenstrain dike against a pressurised crack: the dike-opening row of the benchmark ladder of
# REYKJANES_PLAN.md and the validation decision D2 asks for before the intrusion protocol of GAP-11
# is trusted.
#
# Runs the default case of `solve_dike_crack` from `dike_crack_setup.jl`: a flat elliptical band of
# ordinary host material is opened by the eigenstrain of GAP-11 (`τ_old` and `Q` alone, no hole and
# nothing soft), one elastic step, compared with the closed-form pressurised elliptical hole whose
# `b → 0` limit is Sneddon's crack. The setup is function-only so that `test/test_dike_functions.jl`
# and the sweeps in `examples/benchmarks/stokes/dike_crack/dike_crack.jl` share it.

include(joinpath(@__DIR__, "dike_crack_setup.jl"))

"""
    main(; kwargs...) -> NamedTuple

Solve the default dike case with `verbose = true`, print the comparison with the closed form and
return the `solve_dike_crack` result. Keyword arguments go to `solve_dike_crack`.
"""
main(; kwargs...) = print_dike_crack_report(solve_dike_crack(; verbose = true, kwargs...))

main()
