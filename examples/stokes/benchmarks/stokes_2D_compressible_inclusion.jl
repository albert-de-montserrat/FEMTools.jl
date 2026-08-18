import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Printf
using StaticArrays
using LinearAlgebra
using KernelAbstractions
using FEMTools
using ExactFieldSolutions: Stokes2D_Moutzouris_circle

include(joinpath(@__DIR__, "gmsh_inclusion_mesher.jl"))

const backend   = CPU()
const workgroup = 128

"""
    moutzouris_params(; ηm, ηi, ξm, ξi, r, ε̇, γ̇, ζ̇) -> NamedTuple

Assemble the parameter tuple for `Stokes2D_Moutzouris_circle`.

`ε̇zz` is held at zero: the analytical solution is then a plane-strain state,
which is the only regime the two-dimensional FEMTools discretisation can
represent. `t` and `α` describe the elliptical inclusion and are inert here.
"""
moutzouris_params(; ηm, ηi, ξm, ξi, r, ε̇, γ̇, ζ̇) =
    (ηm = ηm, ηi = ηi, ξm = ξm, ξi = ξi, ri = r, t = 2.0, α = 0.0,
        ε̇ = ε̇, γ̇ = γ̇, ζ̇ = ζ̇, ε̇zz = 0.0)

"""
    analytic_at(x, centre, params) -> NamedTuple

Evaluate the analytical solution at physical position `x`, whose inclusion is
centred on `centre`. The analytical solution places the inclusion at the
origin, so the coordinates are shifted before evaluation.
"""
@inline analytic_at(x, centre, params) =
    Stokes2D_Moutzouris_circle(SVector(x[1] - centre[1], x[2] - centre[2]); params)

"""
    l2_errors(dr, mesh_stokes, cache, element_v, element_P, centre, params)
        -> (; ev, ep, nv, np)

Integrate the L2 errors of velocity and pressure against the analytical
solution, together with the L2 norms of the analytical fields themselves.

Both fields are sampled at the velocity quadrature points, where
`cache.geo_v` supplies the quadrature weight `dΩ` already scaled by the
Jacobian determinant. This is the discretisation's own quadrature, so a
curved element contributes its true area.
"""
function l2_errors(dr, mesh_stokes, cache, element_v, element_P, centre, params)
    Nv_q = shape_function_values(element_v)
    NP_q = shape_function_values(element_P, element_v.integration_points)
    NQ = length(element_v.integration_points.ω)
    NV = length(element_v)
    NP = length(element_P)

    coords = Array(mesh_stokes.coords)
    el2n = Array(mesh_stokes.el2n)
    # dr.P is indexed by DoFsP: el2nP is the continuous vertex connectivity,
    # while DoFsP carries the discontinuous per-element pressure DoFs the
    # assembly kernels gather from.
    dofsP = Array(mesh_stokes.DoFsP)
    geo_v = Array(cache.geo_v)
    vx, vy, P = Array(dr.vx), Array(dr.vy), Array(dr.P)

    ev2 = 0.0; nv2 = 0.0; ep2 = 0.0; np2 = 0.0
    for iel in axes(el2n, 2)
        nodes = @view el2n[:, iel]
        nodesP = @view dofsP[:, iel]
        for q in 1:NQ
            N = Nv_q[q]
            NPq = NP_q[q]
            _, dΩ = geo_v[iel][q]

            x = zero(SVector{2, Float64})
            vhx = 0.0; vhy = 0.0
            for a in 1:NV
                n = nodes[a]
                x += N[a] * coords[n]
                vhx += N[a] * vx[n]
                vhy += N[a] * vy[n]
            end
            Ph = 0.0
            for a in 1:NP
                Ph += NPq[a] * P[nodesP[a]]
            end

            sol = analytic_at(x, centre, params)
            ev2 += ((vhx - sol.V[1])^2 + (vhy - sol.V[2])^2) * dΩ
            nv2 += (sol.V[1]^2 + sol.V[2]^2) * dΩ
            ep2 += (Ph - sol.p)^2 * dΩ
            np2 += sol.p^2 * dΩ
        end
    end
    return (; ev = sqrt(ev2), nv = sqrt(nv2), ep = sqrt(ep2), np = sqrt(np2))
end

"""
    main(; kwargs...) -> NamedTuple

Solve the compressible host-inclusion problem on an unstructured Gmsh T7/P1-disc
mesh and measure the error against the Moutzouris & Duretz (2026) solution.

The analytical solution satisfies `p = -ξ ∇⋅v`, while the FEMTools pressure
residual is `RP = ∫N(-∇⋅v - (P-P0)/(ηb Δt))dΩ`. The two coincide when
`ηb Δt = ξ` and `P0 = 0`, so the bulk viscosities are passed as `ηb` and a
single step of unit `Δt` is taken from a zero initial pressure. `G = Inf`
removes the elastic term, leaving a purely viscous compressible problem.

Velocity is prescribed on the whole outer boundary from the analytical field.
The inclusion perturbation decays as `1/r` and is not negligible at a finite
box, so imposing the far-field affine flow instead would introduce an error
that does not vanish under refinement.
"""
function main(;
        Lx = 1.0, Ly = 1.0,
        r = 0.2,
        ηm = 1.0, ηi = 0.1,
        ξm = 1.0, ξi = 10.0,
        ε̇ = 1.0, γ̇ = 0.0, ζ̇ = 0.5,
        max_area = 1 / 32^2,
        curved = true,
        interface_refine_factor = 4.0,
        ncheck = 100,
        ϵ_tol = 1.0e-10,
        iterMax = 100_000,
        total_iterMax = 100_000,
        γfact = 1.0,
        verbose = true,
    )
    cx, cy = Lx / 2, Ly / 2
    centre = SVector(cx, cy)
    params = moutzouris_params(; ηm, ηi, ξm, ξi, r, ε̇, γ̇, ζ̇)
    Δt = 1.0

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, interface_nodes, cell_phase =
        build_gmsh_t7_inclusion_mesh(;
        Lx, Ly, cx, cy, r, max_area, interface_refine_factor, curved,
    )

    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)

    NV = length(element_v)
    NP = length(element_P)
    NQ_v = length(element_v.integration_points.ω)

    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)

    material = StokesMaterial(;
        η = (ηm, ηi), ηb = (ξm, ξi), G = (Inf, Inf), α = (0.0, 0.0),
        ρ0 = (1.0, 1.0), K = (Inf, Inf), g = (0.0, 0.0), Tref = 0.0,
    )
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    phases_v_cpu = repeat(reshape(Int.(cell_phase), 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(Int.(cell_phase), 1, :), NP, 1)

    verbose && @info "Gmsh mixed mesh (T7/P1-disc)" nnodes_v = mesh_stokes.nnodes nnodes_P =
        mesh_stokes.nnodesP nels = mesh_stokes.nels curved n_outer = length(outer_nodes) n_interface =
        length(interface_nodes) n_incl_els = count(==(2), cell_phase)

    # Dirichlet data for both components on every outer node.
    coords = Array(mesh_v.coords)
    bc_nodes = Int32.(outer_nodes)
    bc_vx_vals = [analytic_at(coords[n], centre, params).V[1] for n in bc_nodes]
    bc_vy_vals = [analytic_at(coords[n], centre, params).V[2] for n in bc_nodes]
    bc_vx = DirichletBoundaryCondition(nothing, bc_nodes, bc_vx_vals)
    bc_vy = DirichletBoundaryCondition(nothing, bc_nodes, bc_vy_vals)

    # Seed the interior with the analytical field so the iteration starts close
    # to the solution; boundary nodes are pinned by apply_bc! regardless.
    coords_v = Array(mesh_stokes.coords)
    copyto!(dr.vx, [analytic_at(c, centre, params).V[1] for c in coords_v])
    copyto!(dr.vy, [analytic_at(c, centre, params).V[2] for c in coords_v])
    apply_bc!(dr.vx, bc_vx)
    apply_bc!(dr.vy, bc_vy)

    # P0 stays zero: the compressible term -(P-P0)/(ηb Δt) is the steady
    # constitutive relation p = -ξ ∇⋅v only when measured from zero pressure.
    fill!(dr.P, 0)
    fill!(dr.P0, 0)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, cache, γfact, Δt; workgroup, phases_v = phases_v_cpu,
    )

    stats = solve_stokes_dyrel!(
        dr, mesh_stokes, cache, bc_vx, bc_vy, Δt, γP;
        phases_v = phases_v_cpu, phases_P = phases_P_cpu,
        τ_old, plastic = nothing, workgroup,
        ncheck, ϵ_tol, iterMax, total_iterMax, verbose,
    )

    err = l2_errors(dr, mesh_stokes, cache, element_v, element_P, centre, params)
    h = sqrt(2 * max_area)
    rel_v = err.ev / err.nv
    rel_p = err.ep / err.np
    verbose && @printf(
        "h=%.5f  nels=%6d  ‖Δv‖/‖v‖=%.6e  ‖ΔP‖/‖P‖=%.6e\n",
        h, mesh_stokes.nels, rel_v, rel_p
    )

    return (; h, nels = mesh_stokes.nels, nnodes_v = mesh_stokes.nnodes,
        rel_v, rel_p, err, stats, dr, mesh_stokes, cache,
        element_v, element_P, centre, params, curved)
end
