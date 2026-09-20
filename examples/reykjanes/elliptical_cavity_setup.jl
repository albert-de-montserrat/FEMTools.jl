# Function-only setup of the pressurised elliptical cavity, milestone M0 of REYKJANES_PLAN.md.
# It runs nothing and is shared by the driver `elliptical_cavity.jl`, `test/test_elliptical_cavity.jl`
# and `examples/benchmarks/stokes/elliptical_cavity/elliptical_cavity.jl`.
#
# A soft, compressible inclusion (the "magma body") is injected with a volumetric source Q inside a
# stiff visco-elastic host, in the elastic-dominated regime Δt ≪ η/G. The host is a disc that
# truncates an infinite plane; its outer circle carries the analytic displacement of the infinite
# problem, so the result does not depend on the disc radius. The solution is compared with
# Muskhelishvili's closed form (`cavity_analytic.jl`). Only existing library features are used.

using Printf
using Statistics
using LinearAlgebra: norm
using StaticArrays
using KernelAbstractions
using FEMTools

include(joinpath(@__DIR__, "..", "triangulate_meshing.jl"))
include(joinpath(@__DIR__, "cavity_analytic.jl"))

"""
    cavity_interface_edges(el2n, phase) -> Vector{NTuple{3, Int32}}

Nodes `(corner, mid-edge, corner)` of every edge shared by an inclusion element (`phase == 2`) and a
host element, ordered counter-clockwise around the inclusion so that the outward normal of the
inclusion times the edge length is `(Δy, −Δx)`.
"""
function cavity_interface_edges(el2n, phase)
    edges = Dict{Tuple{Int32, Int32}, Vector{Tuple{Int, NTuple{3, Int32}}}}()
    for iel in axes(el2n, 2), (i, j, mid) in ((1, 2, 4), (2, 3, 5), (3, 1, 6))
        p, q = el2n[i, iel], el2n[j, iel]
        push!(get!(edges, minmax(p, q), Tuple{Int, NTuple{3, Int32}}[]), (phase[iel], (p, el2n[mid, iel], q)))
    end
    return [
        only(triple for (ph, triple) in sides if ph == 2)
            for sides in values(edges) if length(sides) == 2 && first(sides[1]) != first(sides[2])
    ]
end

"""
    solve_elliptical_cavity(; a=2.5e3, b=0.5e3, Δt=86_400.0, p_target=1e7, ...) -> NamedTuple

Solve one elastic step of the pressurised elliptical cavity and compare it with the analytic solution.

The inclusion has semi-axes `a` (along `x`) and `b`, shear modulus `G_cavity_ratio * G`, bulk modulus
`K_cavity` and pressure storage `ηb = K`. It receives the volumetric strain `ε_inj = Q Δt` chosen so
that the analytic overpressure of a fluid-filled hole is `p_target`. The reference solution uses the
shear modulus of one Maxwell step, `G / (1 + Δt η⁻¹ G)`, so it holds for any `Δt`, not only `Δt ≪ η/G`;
a nonzero `G_cavity_ratio` makes the inclusion slightly stiffer than the hole the reference assumes.

Everything is SI on the way in and out. The solve runs in characteristic units with `L_c = a`, stress
scale `σ_c` (default `p_target`) and time scale `t_c = G Δt / σ_c`, which makes the solution of order
one. The convergence test of the dynamic-relaxation solver takes the smaller of an absolute and a
relative residual, so a scaling that leaves the solution far below `ϵ_tol` stops early; `σ_c` is
exposed to demonstrate this.

Returns the numerical and analytic pressure and area change of the inclusion, the displacement error
outside it and the solver statistics.
"""
function solve_elliptical_cavity(;
        a = 2.5e3, b = 0.5e3, radius = 6a,
        G = 3.0e10, K = 5.0e10, η = 1.0e20,
        G_cavity_ratio = 1.0e-3, K_cavity = 1.0e10, η_cavity = 1.0e16,
        Δt = 86_400.0, p_target = 1.0e7, σ_c = p_target,
        max_area = a^2 / 6.25, refinement = 8,
        γfact = 100.0, ϵ_tol = 1.0e-6, ncheck = 100, iterMax = 50_000, total_iterMax = 50_000,
        rel_drop0 = 1.0e-2, verbose = false,
        backend = CPU(), workgroup = 128,
    )
    # One Maxwell step from rest is elastic with `G / (1 + Δt / t_M)`, while the bulk response stays `K`.
    G_effective = G / (1 + G * Δt / η)
    ν = (3K - 2G_effective) / (2 * (3K + G_effective))
    G_cavity = G_cavity_ratio * G
    area_cavity = π * a * b
    compliance = elliptical_cavity_area_compliance(a, b, G_effective, ν)
    # A fluid-filled hole balances p = K_cavity (ε_inj − ΔA / A) with ΔA = compliance p.
    ε_inj = p_target * (1 + K_cavity * compliance / area_cavity) / K_cavity

    # Characteristic units
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
    cavity_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, findall(==(2), cell_phase)])))

    material = StokesMaterial(;
        η = (η, η_cavity) ./ η_c, ηb = (K, K_cavity) ./ σ_c, G = (G, G_cavity) ./ σ_c,
        α = (0.0, 0.0), ρ0 = (1.0, 1.0), K = (K, K_cavity) ./ σ_c, g = (0.0, 0.0), Tref = 0.0,
    )
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9, stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)

    # Injection, uniform over the inclusion's pressure DoFs.
    Q = zeros(mesh_stokes.nnodesP)
    Q[cavity_P_dofs] .= ε_inj / Δτ
    copyto!(dr.Q, Q)

    # Outer circle: the infinite-plane displacement, as a velocity over the step.
    analytic(x, y) = elliptical_cavity_displacement(x, y; a, b, p = p_target, G = G_effective, ν)
    outer_displacement = [analytic((coords_si[n])...) for n in groups.Γnodes]
    velocity_scale = 1 / (L_c * Δτ)
    bc_vx = DirichletBoundaryCondition(nothing, TDev(groups.Γnodes), TDev([u[1] * velocity_scale for u in outer_displacement]))
    bc_vy = DirichletBoundaryCondition(nothing, TDev(groups.Γnodes), TDev([u[2] * velocity_scale for u in outer_displacement]))
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh_stokes, γfact, Δτ; workgroup, phases_v)

    solve_stats = solve_stokes_dyrel!(
        dr, mesh_stokes, bc_vx, bc_vy, Δτ, γP;
        phases_v, phases_P, τ_old, workgroup,
        ncheck, ϵ_tol, iterMax, total_iterMax, rel_drop0, verbose, verbose_inner = false,
    )

    # ---------------------------------------------------------------------------------------
    # Post-processing on the host, in SI
    # ---------------------------------------------------------------------------------------
    P = Array(dr.P) .* σ_c
    displacement_scale = L_c * Δτ                      # ṽ → u = v Δt
    ux = Array(dr.v.x) .* displacement_scale
    uy = Array(dr.v.y) .* displacement_scale

    corner_area(iel) = begin
        p, q, r = coords_si[el2n_cpu[1, iel]], coords_si[el2n_cpu[2, iel]], coords_si[el2n_cpu[3, iel]]
        ((q[1] - p[1]) * (r[2] - p[2]) - (r[1] - p[1]) * (q[2] - p[2])) / 2
    end
    cavity_cells = findall(==(2), cell_phase)
    areas = corner_area.(cavity_cells)
    area_polygon = sum(areas)
    cell_P = [mean(P[DoFsP_cpu[:, iel]]) for iel in cavity_cells]
    P_cavity = sum(areas .* cell_P) / area_polygon

    ΔA = sum(cavity_interface_edges(el2n_cpu, cell_phase)) do (p, m, q)
        Δx, Δy = coords_si[q] - coords_si[p]
        ((ux[p] + 4ux[m] + ux[q]) * Δy - (uy[p] + 4uy[m] + uy[q]) * Δx) / 6
    end

    host_nodes = setdiff(unique(vec(el2n_cpu[:, findall(==(1), cell_phase)])), groups.Γnodes)
    reference = [analytic((coords_si[n])...) for n in host_nodes]
    error_L2 = sqrt(sum(norm(SVector(ux[n], uy[n]) - SVector(u)) ^ 2 for (n, u) in zip(host_nodes, reference)) /
        sum(norm(SVector(u)) ^ 2 for u in reference))
    # The opening at the top of the cavity is its largest displacement; the tip displacement is a
    # small difference of large terms and too noisy to compare.
    top = argmin(n -> norm(coords_si[n] - SVector(0.0, b)), host_nodes)
    opening_reference = analytic((coords_si[top])...)[2]

    ΔA_reference = compliance * p_target
    return (;
        nels = mesh_stokes.nels, nnodes = mesh_stokes.nnodes,
        P_cavity, P_reference = p_target,
        P_balance = K_cavity * (ε_inj - ΔA / area_polygon),
        ΔA, ΔA_reference,
        error_L2, opening_ratio = uy[top] / opening_reference,
        iter = solve_stats.iter, itPH = solve_stats.itPH, err = solve_stats.err,
        err_abs = solve_stats.err_abs, err_rel = solve_stats.err_rel, converged = solve_stats.converged,
        ε_inj, Δτ, σ_c, t_c,
    )
end

"""
    print_cavity_report(result) -> result

Print the ratios of the numerical pressure, area change and opening to the closed form, the continuity
balance, the displacement error and the solver statistics of one `solve_elliptical_cavity` result.
"""
function print_cavity_report(result)
    @printf(
        "%6d els  P/P_ref = %.5f   ΔA/ΔA_ref = %.5f   P/P_balance = %.6f   L2 error = %.2e   opening = %.4f   iter = %6d   err = %.1e (abs %.1e, rel %.1e)  %s\n",
        result.nels, result.P_cavity / result.P_reference, result.ΔA / result.ΔA_reference,
        result.P_cavity / result.P_balance, result.error_L2, result.opening_ratio, result.iter,
        result.err, result.err_abs, result.err_rel, result.converged ? "converged" : "NOT CONVERGED",
    )
    return result
end
