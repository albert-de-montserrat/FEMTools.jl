# Function-only setup of the eigenstrain dike against a pressurised crack, the dike-opening row of the
# benchmark ladder of REYKJANES_PLAN.md and the validation decision D2 asks for before the intrusion
# protocol is trusted. It runs nothing and is shared by the driver `dike_crack.jl`,
# `test/test_dike_functions.jl` and `examples/benchmarks/stokes/dike_crack/dike_crack.jl`.
#
# A flat elliptical band of ordinary host material is opened by the eigenstrain of GAP-11: the stress
# history of its integration points is shifted by `-2G Δε*` and the same volumetric increment is put
# into the continuity source. Nothing in the band is soft, and no hole is meshed; the opening is the
# stress-free strain alone. One elastic step is solved on a disc whose outer circle carries the
# displacement of the pressurised elliptical hole of the same semi-axes, and the result is compared
# with that closed form (`cavity_analytic.jl`), whose `b → 0` limit is Sneddon's crack.
#
# A band of semi-axes `(a, b)` is `2b √(1 − (x/a)²)` thick, the shape of the opening of a uniformly
# pressurised crack, so one uniform eigenstrain `f = w / (2b)` across it opens it into that profile
# with centre opening `w` and volume `π a w / 2`. That is why a uniform band eigenstrain can be
# compared with a uniform crack pressure at all.

using Printf
using Statistics
using LinearAlgebra: norm
using StaticArrays
using KernelAbstractions
using FEMTools

include(joinpath(@__DIR__, "..", "triangulate_meshing.jl"))
include(joinpath(@__DIR__, "cavity_analytic.jl"))
include(joinpath(@__DIR__, "injection_source.jl"))
include(joinpath(@__DIR__, "elliptical_cavity_setup.jl"))
include(joinpath(@__DIR__, "dike_functions", "dike_eigenstrain.jl"))

"""
    solve_dike_crack(; a=2.5e3, b=1.25e2, p_target=1e7, ...) -> NamedTuple

Open a flat elliptical band by eigenstrain in an elastic host and compare it with the pressurised
elliptical hole of the same semi-axes.

The band is host material marked as a second phase; `a` and `b` are its semi-axes along `x` and `y`,
and it is opened across `y`. Its uniform eigenstrain is chosen to deliver the area change
`ΔA = C(a, b) p_target` of the closed form, plus the part the band stores in its own elastic
compression, so that a faithful mechanism reproduces `p_target` in the band and the analytic
displacement around it. The outer circle of radius `radius` carries that analytic displacement, so
the result does not depend on the truncation. One Maxwell step from rest is elastic with
`G / (1 + Δt G / η)`, which the reference uses, so `Δt` is free.

Everything is SI on the way in and out; the solve runs in the characteristic units of
[`solve_elliptical_cavity`](@ref), `L_c = a`, `σ_c` and `t_c = G Δt / σ_c`.

Returns the band pressure and area change against the closed form, the opening at the band's centre,
the displacement error in the host and the solver statistics.
"""
function solve_dike_crack(;
        a = 2.5e3, b = 1.25e2, radius = 6a,
        G = 3.0e10, K = 5.0e10, η = 1.0e20,
        Δt = 86_400.0, p_target = 1.0e7, σ_c = p_target, storage_iterations = 1,
        max_area = a^2 / 6.25, refinement = 8,
        γfact = 100.0, ϵ_tol = 1.0e-6, ncheck = 100, iterMax = 50_000, total_iterMax = 50_000,
        rel_drop0 = 1.0e-2, verbose = false,
        backend = CPU(), workgroup = 128,
    )
    b < a || throw(ArgumentError("the dike band must be flatter than it is long"))
    G_effective = G / (1 + G * Δt / η)
    ν = (3K - 2G_effective) / (2 * (3K + G_effective))
    compliance = elliptical_cavity_area_compliance(a, b, G_effective, ν)
    ΔA_reference = compliance * p_target

    L_c = a
    t_c = G * Δt / σ_c
    η_c = σ_c * t_c
    Δτ = Δt / t_c

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    coords_si, el2n_cpu, groups = build_triangulate_t7_cavity_mesh(;
        radius, cavity_radii = (a, b), max_area, refinement,
    )
    coords = coords_si ./ L_c
    mesh_v = Mesh(backend, coords, el2n_cpu, element_v; workgroup)
    mesh_stokes = MixedMesh(mesh_v, element_P; workgroup)
    NV, NP = length(element_v), length(element_P)
    NQ_v = length(element_v.integration_points.ω)

    TDev = FEMTools.TA(backend)
    cell_phase = groups.phase
    phases_v = TDev(repeat(reshape(cell_phase, 1, :), NV, 1))
    phases_P = TDev(repeat(reshape(cell_phase, 1, :), NP, 1))
    DoFsP_cpu = Array(mesh_stokes.DoFsP)
    geo_P_cpu = Array(mesh_stokes.geometry.geo_P)
    band_cells = findall(==(2), cell_phase)

    # Both phases are the same host material: the band is opened by its eigenstrain, not by being soft.
    material = StokesMaterial(;
        η = (η, η) ./ η_c, ηb = (K, K) ./ σ_c, G = (G, G) ./ σ_c,
        α = (0.0, 0.0), ρ0 = (1.0, 1.0), K = (K, K) ./ σ_c, g = (0.0, 0.0), Tref = 0.0,
    )
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9, stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)

    # The band area comes from the pressure quadrature, so that the injected volume is the one the
    # continuity residual integrates, not the one the polygon would suggest.
    area_band = sum(sum(geo_P_cpu[iel]) for iel in band_cells) * L_c^2
    NqP_cpu = shape_function_values(element_P, element_v.integration_points)

    analytic(x, y) = elliptical_cavity_displacement(x, y; a, b, p = p_target, G = G_effective, ν)
    outer_displacement = [analytic((coords_si[n])...) for n in groups.Γnodes]
    velocity_scale = 1 / (L_c * Δτ)
    bc_vx = DirichletBoundaryCondition(
        nothing, TDev(groups.Γnodes), TDev([u[1] * velocity_scale for u in outer_displacement]),
    )
    bc_vy = DirichletBoundaryCondition(
        nothing, TDev(groups.Γnodes), TDev([u[2] * velocity_scale for u in outer_displacement]),
    )
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh_stokes, γfact, Δτ; workgroup, phases_v)

    τ = (dr.τ.xx, dr.τ.yy, dr.τ.xy)
    function inject_and_solve!(eigenstrain)
        foreach(x -> fill!(x, 0), (dr.v.x, dr.v.y, dr.P, dr.P0, τ..., τ_old..., dr.Q))
        τ_old_cpu = map(Array, τ_old)
        Q_cpu = zeros(mesh_stokes.nnodesP)
        apply_dike_opening!(
            τ_old_cpu, Q_cpu, DoFsP_cpu, G / σ_c, band_cells,
            eigenstrain * 2b / L_c, 2b / L_c, (0.0, 1.0), Δτ,
        )
        foreach(copyto!, τ_old, τ_old_cpu)
        copyto!(dr.Q, Q_cpu)
        apply_bc!(dr.v.x, bc_vx)
        apply_bc!(dr.v.y, bc_vy)

        stats = solve_stokes_dyrel!(
            dr, mesh_stokes, bc_vx, bc_vy, Δτ, γP;
            phases_v, phases_P, τ_old, workgroup,
            ncheck, ϵ_tol, iterMax, total_iterMax, rel_drop0, verbose, verbose_inner = false,
        )
        update_stokes_current_stress!(
            dr, mesh_stokes, mesh_stokes.geometry, τ, Δτ; phases_v, τ_old, workgroup,
        )
        return stats
    end

    # ---------------------------------------------------------------------------------------
    # Measurement on the host, in SI
    # ---------------------------------------------------------------------------------------
    corner_area(iel) = begin
        p, q, r = coords_si[el2n_cpu[1, iel]], coords_si[el2n_cpu[2, iel]], coords_si[el2n_cpu[3, iel]]
        ((q[1] - p[1]) * (r[2] - p[2]) - (r[1] - p[1]) * (q[2] - p[2])) / 2
    end
    band_areas = corner_area.(band_cells)
    interface_edges = cavity_interface_edges(el2n_cpu, cell_phase)
    # The opening profile is read on the upper face away from the tips, where the analytic
    # displacement is a small difference of large terms.
    upper_face = [n for n in groups.cavity if coords_si[n][2] > 0 && abs(coords_si[n][1]) <= 0.8a]
    top = argmin(n -> norm(coords_si[n] - SVector(0.0, b)), groups.cavity)
    band_nodes = Set(vec(el2n_cpu[:, band_cells]))
    host_nodes = [
        n for n in unique(vec(el2n_cpu[:, findall(==(1), cell_phase)]))
            if !(n in band_nodes) && !(n in groups.Γnodes)
    ]
    host_reference = [analytic((coords_si[n])...) for n in host_nodes]

    function measure()
        P = Array(dr.P) .* σ_c
        displacement_scale = L_c * Δτ
        ux = Array(dr.v.x) .* displacement_scale
        uy = Array(dr.v.y) .* displacement_scale
        τyy = Array(dr.τ.yy) .* σ_c

        cell_P = [mean(P[DoFsP_cpu[:, iel]]) for iel in band_cells]
        P_band = sum(band_areas .* cell_P) / sum(band_areas)
        # The stress that would close the band, `−σ_yy = P − τ_yy`, is what a fluid at the crack
        # pressure would have to push against; the mean pressure alone is not that traction, because
        # the band is also held in compression along its length.
        closure_stress = sum(band_areas[i] * mean(
                P[DoFsP_cpu[:, iel]]' * NqP_cpu[q] - τyy[q, iel] for q in axes(τyy, 1)
            ) for (i, iel) in pairs(band_cells)) / sum(band_areas)

        ΔA = sum(interface_edges) do (p, m, q)
            Δx, Δy = coords_si[q] - coords_si[p]
            ((ux[p] + 4ux[m] + ux[q]) * Δy - (uy[p] + 4uy[m] + uy[q]) * Δx) / 6
        end
        profile_ratios = [uy[n] / analytic((coords_si[n])...)[2] for n in upper_face]
        error_L2 = sqrt(
            sum(norm(SVector(ux[n], uy[n]) - SVector(u))^2 for (n, u) in zip(host_nodes, host_reference)) /
                sum(norm(SVector(u))^2 for u in host_reference),
        )
        return (;
            P_band, closure_stress, ΔA, error_L2,
            opening_ratio = uy[top] / analytic((coords_si[top])...)[2],
            profile_error = isempty(profile_ratios) ? NaN : maximum(abs, profile_ratios .- 1),
            n_profile = length(profile_ratios),
        )
    end

    # The band stores part of the injection in its own elastic compression, `P / K` of it, so the
    # eigenstrain that delivers the area change of the closed form is not known before the pressure
    # is. A fixed-point pass on the solved band pressure settles it; the first guess uses `p_target`,
    # and `storage_iterations = 0` shows what that guess alone is worth.
    eigenstrain = ΔA_reference / area_band + p_target / K
    local solve_stats, measured
    for iteration in 0:storage_iterations
        solve_stats = inject_and_solve!(eigenstrain)
        measured = measure()
        iteration < storage_iterations && (eigenstrain = ΔA_reference / area_band + measured.P_band / K)
    end

    return (;
        nels = mesh_stokes.nels, nnodes = mesh_stokes.nnodes, aspect = b / a,
        measured..., P_reference = p_target, ΔA_reference,
        # What the band injected, minus what it stored, is what it delivered: the discrete continuity
        # balance of the band, independent of the closed form.
        balance = (measured.ΔA / area_band + measured.P_band / K) / eigenstrain,
        iter = solve_stats.iter, itPH = solve_stats.itPH, err = solve_stats.err,
        err_abs = solve_stats.err_abs, err_rel = solve_stats.err_rel, converged = solve_stats.converged,
        eigenstrain, opening = eigenstrain * 2b, area_band, Δτ, σ_c, t_c,
    )
end

"""
    print_dike_crack_report(result) -> result

Print the ratios of the band pressure, area change and centre opening to the closed form, the largest
relative error of the opening profile, the displacement error and the solver statistics of one
[`solve_dike_crack`](@ref) result.
"""
function print_dike_crack_report(result)
    @printf(
        "%6d els  b/a = %.3f  σn/p = %.5f  P/p = %.4f  ΔA/ΔA_ref = %.5f  opening = %.4f  profile = %.2e (%3d)  L2 = %.2e  balance = %.6f  iter = %6d  err = %.1e  %s\n",
        result.nels, result.aspect, result.closure_stress / result.P_reference,
        result.P_band / result.P_reference, result.ΔA / result.ΔA_reference,
        result.opening_ratio, result.profile_error, result.n_profile, result.error_L2, result.balance,
        result.iter, result.err, result.converged ? "converged" : "NOT CONVERGED",
    )
    return result
end
