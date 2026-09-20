#
# Eigenstrain dike benchmark: does a band of stress-free strain open like a pressurised crack?
#
# A flat elliptical band of ordinary host material is opened by the eigenstrain of GAP-11 — the
# stress history of its integration points shifted by `-2G Δε*` and the same volumetric increment in
# the continuity source — and one elastic step is solved (`solve_dike_crack` in
# `examples/reykjanes/dike_crack_setup.jl`), then compared with the closed-form pressurised
# elliptical hole of the same semi-axes, whose `b → 0` limit is Sneddon's crack. Each sweep changes
# one setting from the defaults (1 164 elements, band aspect `b/a = 0.05`, `Δt` of one day,
# `ϵ_tol = 1e-6`) and reports, per run:
#
#   - the mechanism: the opening profile on the upper face away from the tips, the centre opening
#     and the displacement error in the host, all against the closed form;
#   - hydraulic consistency: the band's closure traction `−σ_yy = P − τ_yy` against the crack
#     pressure. The band's mean pressure `P` is reported beside it and is *not* that traction: the
#     band is also held in compression along its length, and `P/p` is near 0.8 and moves with the
#     band aspect, so a protocol that reads a dike pressure off `P` is wrong by about 20%;
#   - conservation: `balance`, the band's own continuity identity `ΔA/A + P/K` over the injected
#     eigenstrain, which is independent of the closed form;
#   - cost: inner iterations and wall time.
#
# `ΔA/ΔA_ref` is near one by construction: the eigenstrain is chosen by a fixed point on the solved
# band pressure so that the band delivers the area change of the closed form. It is a check that the
# fixed point converged, not an independent measurement. `storage_iterations = 0` turns it into one.
#
# `converged` must be true for a row to mean anything. A run that stops on the iteration cap reports
# the cost of the budget, not of solving the problem.
#
# Run from a warm session:
#
#     julia --project=examples
#     include("examples/benchmarks/stokes/dike_crack/dike_crack.jl")
#     rows = run_dike_crack_benchmark()
#     rows = run_dike_crack_benchmark(; sweeps = (:mesh, :aspect))
#

using Printf

include(joinpath(@__DIR__, "..", "..", "..", "reykjanes", "dike_crack_setup.jl"))

"""
    dike_crack_sweeps() -> NamedTuple

The settings of each sweep, as keyword arguments of `solve_dike_crack`: `mesh` (element size and
refinement), `aspect` (the half-thickness `b` of the numerical band, against a half-length of
2.5 km), `time_step` (from an hour to a century, against a host Maxwell time of 106 yr, over which
the elastic reference must hold), `storage` (the fixed-point passes on the band's own elastic
storage), `tolerance` (`ϵ_tol`) and `domain` (the radius at which the analytic displacement is
imposed).
"""
function dike_crack_sweeps()
    year = 365.25 * 86_400
    return (;
        mesh = [
            (; max_area, refinement) for (max_area, refinement) in (
                (3.125e6, 3), (1.5625e6, 4), (1.0e6, 6), (1.0e6, 8), (2.5e5, 8),
            )
        ],
        aspect = [(; b) for b in (250.0, 125.0, 62.5, 31.25)],
        time_step = [(; Δt) for Δt in (3600.0, 86_400.0, 30 * 86_400.0, year, 10year, 100year)],
        storage = [(; storage_iterations) for storage_iterations in (0, 1, 2)],
        tolerance = [(; ϵ_tol) for ϵ_tol in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)],
        domain = [(; radius) for radius in (1.0e4, 1.5e4, 2.5e4, 5.0e4)],
    )
end

"""
    dike_crack_benchmark_case(; kwargs...) -> NamedTuple

Run one dike solve with the keyword arguments of `solve_dike_crack` and reduce it to the scalars the
benchmark tracks, plus the wall time in seconds.

A case that throws, for example when the relaxation diverges, yields a row with `converged = false`,
`missing` numbers and the message in `failure`, so a sweep records the failure and continues instead
of aborting on it.
"""
function dike_crack_benchmark_case(; kwargs...)
    return try
        time = @elapsed result = solve_dike_crack(; kwargs...)
        (;
            nels = result.nels, closure_ratio = result.closure_stress / result.P_reference,
            pressure_ratio = result.P_band / result.P_reference,
            area_ratio = result.ΔA / result.ΔA_reference, opening_ratio = result.opening_ratio,
            profile_error = result.profile_error, error_L2 = result.error_L2, balance = result.balance,
            iter = result.iter, err_rel = result.err_rel, converged = result.converged, time, failure = "",
        )
    catch err
        err isa InterruptException && rethrow()
        (;
            nels = missing, closure_ratio = missing, pressure_ratio = missing, area_ratio = missing,
            opening_ratio = missing, profile_error = missing, error_L2 = missing, balance = missing,
            iter = missing, err_rel = missing, converged = false, time = missing,
            failure = sprint(showerror, err),
        )
    end
end

function print_dike_crack_row(sweep, setting, row)
    label = join(("$k = $v" for (k, v) in pairs(setting)), ", ")
    if ismissing(row.iter)
        @printf("%-12s %-30s FAILED: %s\n", sweep, label, first(split(row.failure, '\n')))
        return
    end
    @printf(
        "%-12s %-30s %6d els  σn %.5f  P %.4f  ΔA %.5f  opening %.4f  profile %.1e  L2 %.1e  balance %.6f  iter %6d  %6.1f s  %s\n",
        sweep, label, row.nels, row.closure_ratio, row.pressure_ratio, row.area_ratio,
        row.opening_ratio, row.profile_error, row.error_L2, row.balance, row.iter, row.time,
        row.converged ? "" : "NOT CONVERGED",
    )
    return
end

"""
    run_dike_crack_benchmark(; sweeps = keys(dike_crack_sweeps()), warmup = true) -> Vector{NamedTuple}

Run the requested sweeps, print one line per run and return the rows, each tagged with its `sweep`
and `setting`. `σn`, `P`, `ΔA` and `opening` in the printout are ratios to the closed form,
`profile` is the largest relative error of the opening profile, and `balance` is the band's
continuity identity, which closes to solver accuracy whatever the row says about the closed form.
"""
function run_dike_crack_benchmark(; sweeps = keys(dike_crack_sweeps()), warmup = true)
    warmup && dike_crack_benchmark_case(; max_area = 3.125e6, refinement = 3)
    settings = dike_crack_sweeps()
    rows = NamedTuple[]
    for sweep in sweeps
        for setting in settings[sweep]
            row = dike_crack_benchmark_case(; setting...)
            print_dike_crack_row(sweep, setting, row)
            push!(rows, merge((; sweep, setting), row))
        end
    end
    return rows
end
