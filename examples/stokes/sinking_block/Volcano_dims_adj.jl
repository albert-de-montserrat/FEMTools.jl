using FEMTools

# The adjoint is solved with the production PH/DYREL solver
# `solve_stokes_adjoint_dyrel!`: a velocity DYREL loop alternating with
# Arrow-Hurwicz pressure-adjoint updates. The γP-scaled pressure step is what
# keeps the pressure adjoint advancing at the SI magnitudes used here; the
# experimental coupled solver's mass-preconditioned λP update stalls on
# dimensional problems.

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using ForwardDiff
using LinearAlgebra
using Printf
using Statistics
using StaticArrays
using KernelAbstractions
using CUDA
using Atomix
using Triangulate
using WriteVTK  # triggers FEMToolsWriteVTKExt, providing FEMTools.write_vtu
# Loading GLMakie activates a display backend and triggers FEMToolsMakieExt,
# which provides FEMTools.plot_summary and the other visualization helpers.
using GLMakie

const default_backend = CUDABackend()
const workgroup = 128

include("mesher.jl")

corner_mesh_hmin(coords, el2n) = minimum(
    norm(coords[Int(el2n[b, e])] - coords[Int(el2n[a, e])])
    for e in axes(el2n, 2), (a, b) in ((1, 2), (2, 3), (3, 1)))

corner_max_speed(vx, vy, el2n) = maximum(
    hypot(vx[Int(el2n[a, e])], vy[Int(el2n[a, e])])
    for e in axes(el2n, 2), a in 1:3)

function advect_t7_mesh!(coords, el2n, vx, vy, dt)
    visited = falses(length(coords))
    @inbounds for e in axes(el2n, 2), a in 1:3
        n = Int(el2n[a, e])
        visited[n] || (coords[n] += dt * SVector(vx[n], vy[n]); visited[n] = true)
    end
    @inbounds for e in axes(el2n, 2)
        n1, n2, n3 = Int.(el2n[1:3, e])
        coords[Int(el2n[4, e])] = (coords[n1] + coords[n2]) / 2
        coords[Int(el2n[5, e])] = (coords[n2] + coords[n3]) / 2
        coords[Int(el2n[6, e])] = (coords[n3] + coords[n1]) / 2
        coords[Int(el2n[7, e])] = (coords[n1] + coords[n2] + coords[n3]) / 3
    end
    return coords
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
    objective_vy_kernel!(dJdvy, coords, el2n, geo, Nq,
                         xmin, xmax, ymin, ymax, Val(N))

Assemble the finite-element derivative of

    J(vᵧ) = -∫_Ωₒᵇₛ vᵧ dΩ,

where `Ωₒᵇₛ = (xmin, xmax) × (ymin, ymax)`.

For `vᵧʰ = Σᵢ Nᵢ Vᵧᵢ`, the derivative with respect to a global
velocity degree of freedom is the consistently assembled load vector

    ∂J/∂Vᵧᵢ = -∫_Ωₒᵇₛ Nᵢ dΩ.

Each kernel invocation handles one element. Integration points outside the
observation box contribute nothing. `geo[iel][q][2]` is the physical
quadrature measure `|det(J)|ωq` computed by `precompute_geometry!`.

Neighbouring elements share velocity nodes, so their element load vectors are
scattered with atomic additions. This makes the assembly safe for both CPU and
accelerator backends.
"""
@kernel function objective_vy_kernel!(
    dJdvy,
    @Const(coords),
    @Const(el2n),
    @Const(geo),
    @Const(Nq),
    xmin,
    xmax,
    ymin,
    ymax,
    ::Val{N},
) where {N}
    iel = @index(Global)

    # Gather the element's global velocity-node numbers once. T7 has seven
    # velocity basis functions, including its interior bubble function.
    local_nodes = SVector{N, Int}(ntuple(i -> el2n[i, iel], Val(N)))

    # Accumulate the element contribution locally, reducing the number of
    # global atomic operations from N×NQ to N per element.
    element_load = zero(Nq[1])
    for q in eachindex(Nq)
        N_at_q = Nq[q]

        # Isoparametric map from reference coordinates to the physical
        # quadrature point: xq = Σᵢ Nᵢ(ξq) xᵢ.
        xq = zero(coords[local_nodes[1]])
        for i in 1:N
            xq += N_at_q[i] * coords[local_nodes[i]]
        end

        # The strict bounds deliberately match the FD objective definition.
        # For a discontinuous indicator, quadrature naturally approximates the
        # fraction of a cut element lying inside the observation box.
        if xmin < xq[1] < xmax && ymin < xq[2] < ymax
            dΩ = geo[iel][q][2]
            element_load -= N_at_q * dΩ
        end
    end

    # Consistent FE assembly of ∂J/∂Vy. This replaces the old mesh-dependent
    # operation that assigned -1 independently to every selected node.
    for i in 1:N
        Atomix.@atomic :monotonic dJdvy[local_nodes[i]] += element_load[i]
    end
end

"""Assemble `∂J/∂Vy` for the observation-box integral objective."""
function assemble_objective_vy!(
    dJdvy, mesh, geo, element::ReferenceElement{E},
    xmin, xmax, ymin, ymax, workgroup,
) where {E <: AbstractElement{2, N}} where {N}
    backend = KernelAbstractions.get_backend(dJdvy)
    # Assembly is additive, so always clear the destination before scattering
    # element contributions into it.
    fill!(dJdvy, 0)

    # Use the same T7 quadrature rule as the momentum residual. This keeps the
    # objective and residual discretizations consistent.
    Nq = shape_function_values(element)
    objective_vy_kernel!(backend, workgroup)(
        dJdvy, mesh.coords, mesh.el2n, geo, Nq,
        xmin, xmax, ymin, ymax, Val(N);
        ndrange = mesh.nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    objective_vx_kernel!(dJdvx, coords, el2n, geo, Nq,
                         xmin, xmax, ymin, ymax, Val(N))

Assemble the finite-element derivative of

    J(vₓ) = -∫_Ωₒᵇₛ vₓ dΩ,

the horizontal counterpart of `objective_vy_kernel!`, over the same
observation box `Ωₒᵇₛ = (xmin, xmax) × (ymin, ymax)`. See that kernel's
docstring for the assembly details; only the differentiated velocity
component (and hence the target array) differs.
"""
@kernel function objective_vx_kernel!(
    dJdvx,
    @Const(coords),
    @Const(el2n),
    @Const(geo),
    @Const(Nq),
    xmin,
    xmax,
    ymin,
    ymax,
    ::Val{N},
) where {N}
    iel = @index(Global)

    local_nodes = SVector{N, Int}(ntuple(i -> el2n[i, iel], Val(N)))

    element_load = zero(Nq[1])
    for q in eachindex(Nq)
        N_at_q = Nq[q]

        xq = zero(coords[local_nodes[1]])
        for i in 1:N
            xq += N_at_q[i] * coords[local_nodes[i]]
        end

        if xmin < xq[1] < xmax && ymin < xq[2] < ymax
            dΩ = geo[iel][q][2]
            element_load -= N_at_q * dΩ
        end
    end

    for i in 1:N
        Atomix.@atomic :monotonic dJdvx[local_nodes[i]] += element_load[i]
    end
end

"""Assemble `∂J/∂Vx` for the observation-box integral objective."""
function assemble_objective_vx!(
    dJdvx, mesh, geo, element::ReferenceElement{E},
    xmin, xmax, ymin, ymax, workgroup,
) where {E <: AbstractElement{2, N}} where {N}
    backend = KernelAbstractions.get_backend(dJdvx)
    fill!(dJdvx, 0)

    Nq = shape_function_values(element)
    objective_vx_kernel!(backend, workgroup)(
        dJdvx, mesh.coords, mesh.el2n, geo, Nq,
        xmin, xmax, ymin, ymax, Val(N);
        ndrange = mesh.nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

struct MaterialSensitivityTag end

@kernel function material_sensitivity_kernel!(
    viscosity_sensitivity,
    density_sensitivity,
    G_sensitivity,
    K_sensitivity,
    Q_sensitivity,
    @Const(vx), @Const(vy), @Const(P), @Const(P0), @Const(T), @Const(T0),
    @Const(λvx), @Const(λvy), @Const(λP),
    @Const(el2n), @Const(dofsP), @Const(geo), @Const(geo_P), @Const(phases),
    @Const(τxx_old), @Const(τyy_old), @Const(τxy_old),
    η_element, ρ_element, G_element, K_element, Q_element,
    @Const(α), @Const(g), Tref, Δt,
    @Const(Nq), @Const(NqP), ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    velocity_nodes = SVector{NV, Int}(ntuple(i -> el2n[i, iel], Val(NV)))
    pressure_dofs  = SVector{NP, Int}(ntuple(i -> dofsP[i, iel], Val(NP)))
    phase = Int(phases[1, iel])
    vx_e = SVector{NV}(ntuple(i -> vx[velocity_nodes[i]], Val(NV)))
    vy_e = SVector{NV}(ntuple(i -> vy[velocity_nodes[i]], Val(NV)))
    P_e  = SVector{NP}(ntuple(i -> P[pressure_dofs[i]], Val(NP)))
    P0_e = SVector{NP}(ntuple(i -> P0[pressure_dofs[i]], Val(NP)))
    T_e  = SVector{NP}(ntuple(i -> T[pressure_dofs[i]], Val(NP)))
    T0_e = SVector{NP}(ntuple(i -> T0[pressure_dofs[i]], Val(NP)))
    λx_e = SVector{NV}(ntuple(i -> λvx[velocity_nodes[i]], Val(NV)))
    λy_e = SVector{NV}(ntuple(i -> λvy[velocity_nodes[i]], Val(NV)))
    λP_e = SVector{NP}(ntuple(i -> λP[pressure_dofs[i]], Val(NP)))
    # Each element owns five independent material unknowns. Seed them as the
    # five partial directions of one Dual value so all sensitivities are
    # evaluated together in this GPU thread.
    η_e = ForwardDiff.Dual{MaterialSensitivityTag}(η_element[iel], 1.0, 0.0, 0.0, 0.0, 0.0)
    ρ_e = ForwardDiff.Dual{MaterialSensitivityTag}(ρ_element[iel], 0.0, 1.0, 0.0, 0.0, 0.0)
    G_e = ForwardDiff.Dual{MaterialSensitivityTag}(G_element[iel], 0.0, 0.0, 1.0, 0.0, 0.0)
    K_e = ForwardDiff.Dual{MaterialSensitivityTag}(K_element[iel], 0.0, 0.0, 0.0, 1.0, 0.0)
    Q_e_scalar = ForwardDiff.Dual{MaterialSensitivityTag}(Q_element[iel], 0.0, 0.0, 0.0, 0.0, 1.0)
    Q_e = SVector{NP}(ntuple(_ -> Q_e_scalar, Val(NP)))
    # The material sensitivity is taken with respect to this element's own η and ρ,
    # so the residual is evaluated with single-entry material tuples and a phase
    # vector that indexes that lone entry (all ones). Differentiating the five
    # dual-seeded values gives the per-element sensitivities while thermal
    # expansivity is still read from the element's true phase.
    phase_e = SVector{NV, Int}(ntuple(_ -> 1, Val(NV)))
    τ_old_e = FEMTools.IntegrationPointStress(
        SVector(ntuple(q -> τxx_old[q, iel], length(Nq))),
        SVector(ntuple(q -> τyy_old[q, iel], length(Nq))),
        SVector(ntuple(q -> τxy_old[q, iel], length(Nq))),
    )
    Re_x, Re_y = FEMTools.integrate_momentum_residual(
        (vx_e, vy_e), P_e, nothing, T_e, geo[iel], phase_e,
        (η_e,), (G_e,), (α[phase],),
        (ρ_e,), (K_e,), g, Tref, Δt,
        τ_old_e, nothing, Nq, NqP,
    )
    phase_P = SVector{NP, Int}(ntuple(_ -> 1, Val(NP)))
    RP_e = FEMTools.integrate_PH_pressure_residual(
        (vx_e, vy_e), P_e, P0_e, T_e, T0_e, Q_e, geo[iel], geo_P[iel],
        phase_P, (α[phase],), (K_e,), Δt, NqP,
    )
    partials = ForwardDiff.partials(
        dot(λx_e, Re_x) + dot(λy_e, Re_y) + dot(λP_e, RP_e))
    viscosity_sensitivity[iel] = partials[1]
    density_sensitivity[iel] = partials[2]
    G_sensitivity[iel] = partials[3]
    K_sensitivity[iel] = partials[4]
    Q_sensitivity[iel] = partials[5]
end

"""
    material_sensitivities(dr, mesh_stokes, λvx, λvy, geo_v, phases, τ_old,
                           element_v, element_P, η, ρ0, cell_phase,
                           G, α, K, g, Tref, Δt, Val(NV), Val(NP),
                           backend, workgroup) -> (..., Q_sensitivity, Q_element)

Assemble the per-element material sensitivities `sᵉ(m) = λᵉᵀ ∂Rᵉ/∂m` of the
converged adjoint state with a five-direction dual-number kernel, differentiating
with respect to the element density, viscosity, elastic moduli, and uniform
element volumetric source fields in one backend launch.

`cell_phase[iel]` selects the phase whose `η`/`ρ0` value seeds element `iel`.
Returns the completed element fields as CPU arrays; summing the entries of one
phase gives the derivative with respect to that phase's global material value.

The element material fields and sensitivity outputs are allocated on `backend`.
Mesh connectivity, geometry, phases, forward fields,
and adjoint fields must be on that same backend. Only the completed sensitivity
vectors are transferred to the CPU for reduction and plotting.
"""
function material_sensitivities(
    dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases, τ_old,
    element_v, element_P, η, ρ0, cell_phase,
    G, α, K, g, Tref, Δt, ::Val{NV}, ::Val{NP}, backend, workgroup,
) where {NV, NP}
    η_element = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nels)
    ρ_element = similar(η_element)
    G_element = similar(η_element)
    K_element = similar(η_element)
    Q_element = similar(η_element)
    # Indexing an `NTuple` with a vector returns another tuple.  Copying that
    # tuple directly to a GPU array selects Base's scalar `copyto!` fallback,
    # which is both unsupported by GPUArrays and needlessly serial.  Materialise
    # the phase gathers as ordinary host vectors so the backend can perform one
    # bulk host-to-device transfer.
    copyto!(η_element, collect(η[cell_phase]))
    copyto!(ρ_element, collect(ρ0[cell_phase]))
    copyto!(G_element, collect(G[cell_phase]))
    copyto!(K_element, collect(K[cell_phase]))
    Q_cpu = Array(dr.Q)
    dofsP_cpu = Array(mesh_stokes.DoFsP)
    copyto!(Q_element, [sum(Q_cpu[dofsP_cpu[:, iel]]) / NP for iel in 1:mesh_stokes.nels])
    viscosity_sensitivity_backend = zero(η_element)
    density_sensitivity_backend = zero(ρ_element)
    G_sensitivity_backend = zero(G_element)
    K_sensitivity_backend = zero(K_element)
    Q_sensitivity_backend = zero(Q_element)
    Nq_v = shape_function_values(element_v)
    Nq_P = shape_function_values(element_P, element_v.integration_points)
    material_sensitivity_kernel!(backend, workgroup)(
        viscosity_sensitivity_backend, density_sensitivity_backend,
        G_sensitivity_backend, K_sensitivity_backend, Q_sensitivity_backend,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0, λvx, λvy, λP,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, phases,
        τ_old[1], τ_old[2], τ_old[3],
        η_element, ρ_element, G_element, K_element, Q_element,
        α, g, Tref, Δt, Nq_v, Nq_P, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KernelAbstractions.synchronize(backend)

    # Plotting and phase-wise reductions are CPU-side, so transfer only the
    # completed element fields after the backend FEM integration has finished.
    return (Array(density_sensitivity_backend), Array(viscosity_sensitivity_backend),
            Array(G_sensitivity_backend), Array(K_sensitivity_backend),
            Array(Q_sensitivity_backend), Array(Q_element))
end

"""
    solve_objective_adjoint(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                            phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
                            vx_nodes, vy_nodes, objective_vx, objective_vy,
                            η, ρ0, cell_phase, G, α, K, g, Tref, Val(NV), Val(NP),
                            backend, workgroup; ncheck, adjoint_tol, adjoint_iterMax,
                            adjoint_verbose) -> NamedTuple

Solve the discrete adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` for one objective's load pair
`(objective_vx, objective_vy)` — one of them is the zero array for a
component the objective does not depend on — and contract the converged
adjoint state into per-element material sensitivities.

`J` is evaluated generically as `sum(objective_vx .* dr.vx + objective_vy .*
dr.vy)`: whichever load is zero contributes nothing, so the same expression
gives the right scalar objective for a vx-only or a vy-only functional without
the caller needing to say which. Returns a `NamedTuple` with the adjoint
fields (`λvx`, `λvy`, `λP`), the solver `stats`, the objective value `J` and
its `J_scale` normalisation, the raw per-element sensitivities and their
phase-summed gradients, and the dimensionless logarithmic sensitivities and
gradients used for plotting/output.
"""
function solve_objective_adjoint(
    dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
    phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
    vx_nodes, vy_nodes, objective_vx, objective_vy,
    η, ρ0, cell_phase, G, α, K, g, Tref, ::Val{NV}, ::Val{NP},
    backend, workgroup;
    ncheck, adjoint_tol, adjoint_iterMax, adjoint_verbose,
) where {NV, NP}
    λvx = zero(dr.vx)
    λvy = zero(dr.vy)
    λP  = zero(dr.P)

    stats = solve_stokes_adjoint_dyrel!(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_solve, phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
        objective_vx, objective_vy, λvx, λvy, λP,
        backend, workgroup;
        vx_nodes, vy_nodes, ncheck, adjoint_tol,
        iterMax = adjoint_iterMax, total_iterMax = adjoint_iterMax,
        verbose = adjoint_verbose, collect_history = true,
    )

    density_sensitivity, viscosity_sensitivity, G_sensitivity, K_sensitivity,
        Q_sensitivity, Q_element = material_sensitivities(
        dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases_solve, τ_old,
        element_v, element_P, η, ρ0, cell_phase,
        G, α, K, g, Tref, dt_step, Val(NV), Val(NP), backend, workgroup,
    )

    density_gradient_by_phase = ntuple(
        p -> sum(density_sensitivity[cell_phase .== p]), length(ρ0))
    viscosity_gradient_by_phase = ntuple(
        p -> sum(viscosity_sensitivity[cell_phase .== p]), length(η))
    G_gradient_by_phase = ntuple(p -> sum(G_sensitivity[cell_phase .== p]), length(G))
    K_gradient_by_phase = ntuple(p -> sum(K_sensitivity[cell_phase .== p]), length(K))
    Q_gradient_by_phase = ntuple(p -> sum(Q_sensitivity[cell_phase .== p]), length(K))

    objective_terms = Array(objective_vx) .* Array(dr.vx) .+ Array(objective_vy) .* Array(dr.vy)
    J = sum(objective_terms)
    J_scale = max(abs(J), sum(abs, objective_terms), eps(Float64))
    density_log_sensitivity = density_sensitivity .* [ρ0[p] for p in cell_phase] ./ J_scale
    viscosity_log_sensitivity = viscosity_sensitivity .* [η[p] for p in cell_phase] ./ J_scale
    G_log_sensitivity = G_sensitivity .* [G[p] for p in cell_phase] ./ J_scale
    K_log_sensitivity = K_sensitivity .* [K[p] for p in cell_phase] ./ J_scale
    Q_log_sensitivity = Q_sensitivity .* Q_element ./ J_scale
    density_log_gradient_by_phase = ntuple(
        p -> ρ0[p] * density_gradient_by_phase[p] / J_scale, length(ρ0))
    viscosity_log_gradient_by_phase = ntuple(
        p -> η[p] * viscosity_gradient_by_phase[p] / J_scale, length(η))
    G_log_gradient_by_phase = ntuple(
        p -> G[p] * G_gradient_by_phase[p] / J_scale, length(G))
    K_log_gradient_by_phase = ntuple(
        p -> K[p] * K_gradient_by_phase[p] / J_scale, length(K))
    Q_log_gradient_by_phase = ntuple(
        p -> sum(Q_log_sensitivity[cell_phase .== p]), length(K))

    return (;
        λvx, λvy, λP, stats, J, J_scale,
        density_sensitivity, viscosity_sensitivity, G_sensitivity, K_sensitivity,
        Q_sensitivity, Q_element,
        density_gradient_by_phase, viscosity_gradient_by_phase,
        G_gradient_by_phase, K_gradient_by_phase, Q_gradient_by_phase,
        density_log_sensitivity, viscosity_log_sensitivity,
        G_log_sensitivity, K_log_sensitivity, Q_log_sensitivity,
        density_log_gradient_by_phase, viscosity_log_gradient_by_phase,
        G_log_gradient_by_phase, K_log_gradient_by_phase, Q_log_gradient_by_phase,
    )
end


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; backend=CUDABackend(), max_area=(20e3/64)^2, Δt=100yr, show_plot=true, kwargs...)

Solve one unstructured T7/P1-disc sinking-block Stokes problem and its discrete
adjoint with the production PH/DYREL adjoint solver, then assemble the material
sensitivities of two observation-box velocity objectives over the same box,
`J_vy(v) = -∫_Ωₒᵇₛ vᵧ dΩ` and `J_vx(v) = -∫_Ωₒᵇₛ vₓ dΩ`, each solved with its
own adjoint state since a single combined adjoint would blend their gradients.

The model builds a square domain with a rectangular density/viscosity inclusion,
applies a pure-shear far-field velocity superimposed on free-slip side/bottom
boundary conditions, initialises the pressure from a lithostatic solve, and
runs the Powell-Hestenes/DYREL forward solve. It then
solves the discrete adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` on the same discretisation and
contracts `λᵀ ∂R/∂m` with dual-number AD to obtain per-element density and viscosity
sensitivities.

`backend` selects the KernelAbstractions compute backend. The Triangle mesh is
built on the host, after which mesh connectivity, mixed pressure topology,
geometry caches, phases, boundary arrays, and solver state are placed on that
backend. Load CUDA before passing `CUDABackend()`; use `show_plot=false` for
headless runs.

`nsteps` defaults to one. `max_area` sets the Triangle element-area bound on
the chamber, the conduit, and the volcanic edifice; away from those features
the bound grows with distance — the target edge length increases at rate
`mesh_grading` per unit distance — up to `coarsening_ratio * max_area` on the
lateral sides. `Δt` is the maximum (visco)elastic time step in seconds (100
years by default). The actual step is capped by
`mesh_cfl*hmin/vmax`; after each solve the T7 mesh is advected, straightened,
and its geometry caches are rebuilt. `q_magma` is the positive volumetric
source rate applied to the chamber (phase 3) pressure DoFs. With
`write_vtk_output=true`, each step
writes paired forward and adjoint-sensitivity VTK files to `out_dir`.
`show_plot` toggles the GLMakie summary figure. `ε̇_bg` is the background
pure-shear strain rate [s⁻¹] applied at the side and bottom Dirichlet nodes,
superimposed on the buoyancy-driven flow. The remaining keyword arguments
(`ncheck`, `ϵ_tol`, `iterMax`, `total_iterMax`, and their `adjoint_*`
counterparts) tune the forward and adjoint solver tolerances and iteration
budgets.

Each step calls `solve_objective_adjoint` twice, once per objective, returning
`adj_vy` and `adj_vx` — `NamedTuple`s of the adjoint fields, solver `stats`,
raw and dimensionless-logarithmic per-element sensitivities, and their
phase-summed gradients. Both are written to the per-step adjoint VTK file with
`_Jvy`/`_Jvx`-suffixed field names; the interactive summary plot (when
`show_plot=true`) shows only the `J_vy` (vertical-velocity) objective.
"""
function main(;
    backend = default_backend,
    nsteps = 1,
    max_area = (20.0e3 / 64)^2,
    coarsening_ratio = 25.0,
    mesh_grading = 0.3,
    Δt = 100 * 365.25 * 24 * 3600,
    mesh_cfl = 0.75,
    q_magma = 1.0e-13,
    show_plot = true,
    write_vtk_output = true,
    out_dir = joinpath(@__DIR__, "Volcano_dims_output_stokes_adjoint"),
    ncheck = 5,
    ϵ_tol = 1.0e-6,
    iterMax = 50_000,
    total_iterMax = 50_000,
    adjoint_tol = 1.0e-6,
    adjoint_iterMax = 50_000,
    adjoint_verbose = true,
    # Powell-Hestenes augmentation strength and DYREL Chebyshev damping. A
    # stronger augmentation (γfact) and lighter damping (c_fact) than the historical
    # 20/0.9 cut the forward iteration count by ~15% on this problem without
    # affecting the converged solution.
    γfact = 40.0,
    CFL_v = 0.9,
    c_fact = 0.7,
    # Background pure-shear strain rate superimposed on the buoyancy-driven
    # flow: vx = +ε̇_bg*x, vy = -ε̇_bg*y (domain is centered on (0,0) after the
    # shift below), applied at the same free-slip/free-surface nodes as the
    # zero far-field condition it replaces. 1e-15 s⁻¹ is a typical tectonic
    # background rate.
    ε̇_bg = 1.0e-15,
)
    # Domain
    Lx, Ly = 100.0e3, 20.0e3

    # Material (3 phases: lower background, upper background, inclusion). The
    # background is split by the horizontal line y = cy through the inclusion
    # centre; the two halves start with identical properties and can be tuned
    # independently.
    # SI properties: lower crust, upper crust, silicic magma chamber.
    # Phase 3 is an effective crystal-rich magma/mush rheology. A pure-melt
    # viscosity (~10⁷ Pa s) creates a 10¹³ contrast that the Stokes DR
    # preconditioner cannot resolve on this mesh.
    η     = (1.0e23, 1.0e22, 1.0e18)      # shear viscosity [Pa s]
    α     = (3.0e-5, 3.0e-5, 5.0e-5)      # thermal expansivity [K⁻¹]
    ρ0    = (2700.0, 2800.0, 2400.0)      # reference density [kg m⁻³]
    K     = (60.0e9, 50.0e9, 10.0e9)      # bulk modulus [Pa]
    Cp    = (1000.0, 1000.0, 1200.0)      # heat capacity [J kg⁻¹ K⁻¹]
    k     = (2.5, 2.5, 1.5)               # conductivity [W m⁻¹ K⁻¹]
    ηb    = K                             # pressure storage modulus [Pa]
    G     = (30.0e9, 30.0e9, 5.0e9)       # shear modulus [Pa]
    G_stokes = G
    # plastic = nothing
    C     = 10e6                           # cohesion C
    η_reg = 1e20
    plastic = DruckerPrager(
        (π/6, π/6, π/6),                   # friction angle ϕ = 30° [rad]
        (0.0, 0.0, 0e0),                   # dilation angle Ψ = 0°  [rad] (non-associated)
        (C, C, Inf),                       # cohesion C [same for both phases]
        (η_reg, η_reg, η_reg),             # plastic regularisation viscosity η_reg
        K,                                 # Kb (passed separately from elastic K)
    )
    g     = (0.0, -9.81)                   # gravity [m s⁻²]
    Tref  = 288.15                         # 15 °C [K]

    # DR solver
    # Inclusion geometry. The Triangle PSLG uses this circle as an internal
    # constrained boundary, so no element crosses the material interface.
    chamber_half_width = 5.7e3
    chamber_half_height = 0.5e3
    chamber_depth = 7.5e3
    volcano_rim_radius = 300.0
    volcano_half_span = 20.0e3
    volcano_peak_height = 3400.0
    conduit_half_width = volcano_rim_radius
    cx = 0.0
    cy = Ly / 2 - chamber_depth
    # Observation box on the volcano's right flank (post-shift frame, so x is
    # already the distance from the volcano's axis): the middle half of the
    # slope, between 25% and 75% of the way from the rim (d=volcano_rim_radius)
    # to the base (d=volcano_half_span). The y-range is bounded above by the
    # local surface height at the rim-side edge and below by 0.75e3 beneath the
    # surface height at the base-side edge; quadrature points above the actual,
    # continuously-varying topography between those edges don't exist in the
    # mesh and so contribute nothing, letting the box track the slope without
    # a piecewise-linear y bound.
    flank_d0 = volcano_rim_radius + 0.25 * (volcano_half_span - volcano_rim_radius)
    flank_d1 = volcano_rim_radius + 0.75 * (volcano_half_span - volcano_rim_radius)
    flank_h0 = volcano_profile(
        flank_d0; volcano_cx = 0.0, volcano_half_span, volcano_peak_height, volcano_rim_radius)
    flank_h1 = volcano_profile(
        flank_d1; volcano_cx = 0.0, volcano_half_span, volcano_peak_height, volcano_rim_radius)
    objective_bounds = (flank_d0, flank_d1, Ly / 2 + flank_h1 - 0.75e3, Ly / 2 + flank_h0)

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    # Volcano top surface centred over the inclusion: a flat base, linear flanks,
    # and a flat summit plateau (no crater). Heights are measured above the flat
    # top y=0 in the mesher's frame, so profile(0)=profile(Lx)=0 keeps the side
    # walls vertical.
    volcano_topography(x) = volcano_profile(
        x; volcano_cx = Lx / 2, volcano_half_span,
        volcano_peak_height, volcano_rim_radius,
    )
    # Graded resolution (mesher frame: x ∈ [0, Lx], flat top at y = 0): the
    # target edge length is √(2 max_area) on the chamber, the conduit, and the
    # volcanic edifice (including the near-surface objective layer), and grows
    # linearly with distance from those features up to a far-field cap, so the
    # lateral sides carry far fewer elements than the region of interest.
    h_fine = sqrt(2 * max_area)
    h_coarse = sqrt(2 * coarsening_ratio * max_area)
    box_distance(x, y, x1, x2, y1, y2) =
        hypot(max(x1 - x, x - x2, 0.0), max(y1 - y, y - y2, 0.0))
    max_area_at = (x, y) -> begin
        d = min(
            # chamber (bounding box of the ellipse)
            box_distance(x, y, Lx / 2 - chamber_half_width, Lx / 2 + chamber_half_width,
                -chamber_depth - chamber_half_height, -chamber_depth + chamber_half_height),
            # conduit, from the chamber centre up to the summit
            box_distance(x, y, Lx / 2 - conduit_half_width, Lx / 2 + conduit_half_width,
                -chamber_depth, volcano_peak_height),
            # edifice and the near-surface objective layer beneath it
            box_distance(x, y, Lx / 2 - volcano_half_span, Lx / 2 + volcano_half_span,
                -0.75e3, volcano_peak_height),
        )
        h = min(h_fine + mesh_grading * d, h_coarse)
        return h^2 / 2
    end
    coords_v_cpu, el2n_v_cpu, outer_nodes, interface_nodes = build_triangle_t7_inclusion_mesh(;
        Lx, Ly,
        cx = Lx / 2, cy = -chamber_depth,
        ellipse_half_width = chamber_half_width,
        ellipse_half_height = chamber_half_height,
        conduit_half_width,
        max_area_at, split_background = true, topography = volcano_topography,
    )
    shift = SVector(-Lx / 2, Ly / 2)
    coords_v_cpu = [c + shift for c in coords_v_cpu]
    # Triangle builds the mesh on the host. The backend-aware constructor
    # uploads coordinates, connectivity, DoFs, and detected boundary nodes in
    # one place, so every array read by a subsequent assembly kernel lives on
    # the same device as the solver fields.
    mesh_v = FEMTools.Mesh(backend, coords_v_cpu, el2n_v_cpu; order = 2)
    mesh_stokes = MixedMesh(mesh_v, element_P)

    # Device array constructor for uploading host-built index/BC/phase arrays to
    # the compute backend (`TA(CPU()) === Array`, so this is a no-op on the CPU).
    TDev = FEMTools.TA(backend)

    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels chamber_half_width chamber_half_height chamber_depth conduit_half_width max_area n_interface_nodes=length(interface_nodes)

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)
    ξq_v = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)
    ∂N∂ξq_P = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_v[q]), NQ_v)

    # Allocate and fill both geometry caches on the selected backend. This also
    # evaluates the discontinuous-pressure geometry at the velocity quadrature
    # points, matching the forward and adjoint assemblers.
    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    geo_v, geo_P = cache.geo_v, cache.geo_P

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    dr = StokesDR(
        backend,
        mesh_stokes.nnodes,
        mesh_stokes.nnodesP,
        η, ηb, α;
        ρ0,
        K,
        g,
        Tref,
        CFL_v, CFL_P = 0.9, c_fact,
        stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ     = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — rectangular inclusion
    # ---------------------------------------------------------------------------

    # A point lies in the circular inclusion within a small tolerance; the
    # constrained circle boundary keeps every element wholly inside or outside.
    incl_tol = 1.0e-6
    chamber_conduit_junction = cy + chamber_half_height *
        sqrt(1 - (conduit_half_width / chamber_half_width)^2)
    in_chamber(c) = hypot(
        (c[1] - cx) / chamber_half_width,
        (c[2] - cy) / chamber_half_height,
    ) ≤ 1 + incl_tol
    in_conduit(c) = abs(c[1] - cx) ≤ conduit_half_width + incl_tol &&
        c[2] ≥ chamber_conduit_junction - incl_tol
    in_incl(c) = in_chamber(c) || in_conduit(c)
    # Phase 3 is the inclusion; the background splits at the horizontal line
    # y = cy into a lower half (phase 1) and an upper half (phase 2).
    phase_of(c) = in_incl(c) ? 3 : (c[2] < cy ? 1 : 2)

    coords_v     = Array(mesh_stokes.coords)

    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    # An element is the inclusion only when all three corners lie in the disk:
    # the centroid alone can fall in the thin gap between a boundary chord and
    # the arc. The background half then follows the element-centroid height.
    cell_phase = Int[
        let c1 = coords_v[el2nP_cpu[1, i]], c2 = coords_v[el2nP_cpu[2, i]], c3 = coords_v[el2nP_cpu[3, i]]
            (in_incl(c1) && in_incl(c2) && in_incl(c3)) ? 3 :
                ((c1[2] + c2[2] + c3[2]) / 3 < cy ? 1 : 2)
        end
        for i in 1:mesh_stokes.nels
    ]
    # `cell_phase` stays on the host for the reductions and material-field
    # gathers below; the solver reads phases inside kernels, so upload a device
    # copy for the assembly path.
    phases_solve = TDev(reshape(cell_phase, 1, :))

    # Apply a small, constant volumetric source only inside the magmatic
    # chamber. P1-disc pressure DoFs belong to exactly one element, so this
    # assignment introduces no ambiguity at the chamber boundary.
    Q_cpu = zeros(Float64, mesh_stokes.nnodesP)
    @inbounds for iel in eachindex(cell_phase)
        all(a -> in_chamber(coords_v[el2nP_cpu[a, iel]]), 1:3) &&
            (Q_cpu[DoFsP_cpu[:, iel]] .= q_magma)
    end
    copyto!(dr.Q, Q_cpu)
    @info "Magmatic chamber source" q_magma n_source_dofs=count(x -> !iszero(x), Q_cpu)


    corner_nodes = sort!(unique(vec(el2nP_cpu)))
    corner_id = Dict{Int32, Int32}(old => Int32(i) for (i, old) in enumerate(corner_nodes))
    coords_litho = coords_v[Int.(corner_nodes)]
    el2n_litho = Matrix{Int32}(undef, NP, mesh_stokes.nels)
    @inbounds for iel in 1:mesh_stokes.nels, a in 1:3
        el2n_litho[a, iel] = corner_id[Int32(el2nP_cpu[a, iel])]
    end
    mesh_litho = FEMTools.Mesh(backend, coords_litho, el2n_litho)
   
    geo_litho = FEMTools.precompute_geometry(backend, workgroup, mesh_litho, element_P)

    lp_dr = LithostaticPressureDR(backend, mesh_litho.nnodes, ρ0, α, K; CFL = 0.9, ϵ = 1e-2)
    copyto!(lp_dr.phases, Int[phase_of(c) for c in coords_litho])
    P0_litho = Float64[ρ0[1] * abs(g[2]) * (Ly / 2 - c[2]) for c in coords_litho]
    copyto!(lp_dr.P, P0_litho)
    litho_tol = max(Lx, Ly) * sqrt(eps(Float64))
    top_nodes_litho = Int32[
        corner_id[Int32(n)] for n in outer_nodes
        if haskey(corner_id, Int32(n)) &&
           abs(abs(coords_v[n][1]) - Lx / 2) > litho_tol &&
           abs(coords_v[n][2] + Ly / 2) > litho_tol
    ]
    top_nodes_dev = TDev(top_nodes_litho)
    top_zero = KernelAbstractions.zeros(backend, Float64, length(top_nodes_litho))
    solver!(lp_dr, mesh_litho, geo_litho, element_P, top_nodes_dev, top_zero, top_zero,
        backend, workgroup; ncheck = 50, verbose = false, Tref = Tref, g = g)

    P_litho = Array(lp_dr.P)
    P_hydro = zeros(Float64, mesh_stokes.nnodesP)
    @inbounds for iel in 1:mesh_stokes.nels, a in 1:3
        P_hydro[DoFsP_cpu[a, iel]] = P_litho[corner_id[Int32(el2nP_cpu[a, iel])]]
    end
    copyto!(dr.P, P_hydro)
    copyto!(dr.P0, P_hydro)

    @info "Phases" n_incl=count(==(3), cell_phase) n_lower=count(==(1), cell_phase) n_upper=count(==(2), cell_phase)

    # ---------------------------------------------------------------------------
    # Initial temperature field: a linear background between the free surface
    # (z=0, T=T_top) and the domain base (z=1, T=T_bottom), overridden to the
    # constant T_inclusion inside the density/viscosity inclusion.
    # ---------------------------------------------------------------------------

    geothermal_gradient = 25.0 / 1.0e3     # 25 K/km
    T_top       = 288.15                    # 15 °C
    T_bottom    = T_top + geothermal_gradient * Ly
    T_inclusion = 1173.15                   # 900 °C silicic magma
    T_init = Vector{Float64}(undef, mesh_stokes.nnodesP)
    @inbounds for iel in 1:mesh_stokes.nels, a in 1:3
        dof = DoFsP_cpu[a, iel]
        if cell_phase[iel] == 3
            T_init[dof] = T_inclusion
        else
            z = (Ly / 2 - coords_v[el2nP_cpu[a, iel]][2]) / Ly
            T_init[dof] = T_top + (T_bottom - T_top) * z
        end
    end
    copyto!(dr.T, T_init)
    copyto!(dr.T0, T_init)

    @info "Temperature field" T_top T_bottom T_inclusion

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear superimposed on free slip/free surface:
    #   vx = +ε̇_bg * x,   vy = -ε̇_bg * y   (domain centered on (0,0))
    # The sides (x=±Lx/2) prescribe the normal (vx) component and leave the
    # tangential (vy) component free — vertical free slip. The bottom
    # (y=-Ly/2) prescribes the normal (vy) component and leaves vx free —
    # horizontal free slip. The top has no Dirichlet condition on either
    # component, so it — including the volcano's flanks and crater — is a
    # true traction-free surface, free to move with the material during mesh
    # advection.
    # ---------------------------------------------------------------------------

    # Classify the constrained wall nodes on the host (needs scalar coordinate
    # access), then upload the index and value arrays that the Dirichlet kernels
    # consume to the backend.
    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * sqrt(eps(Float64))
    vx_node_ids = Int32[n for n in Γnodes if abs(abs(coords[n][1]) - Lx / 2) ≤ tol]
    vy_node_ids = Int32[n for n in Γnodes if abs(coords[n][2] + Ly / 2) ≤ tol]
    free_surface_nodes = Int32[
        n for n in Γnodes
        if abs(abs(coords[n][1]) - Lx / 2) > tol &&
           abs(coords[n][2] + Ly / 2) > tol
    ]
    vx_nodes = TDev(vx_node_ids)
    vy_nodes = TDev(vy_node_ids)

    bc_vx_vals = TDev(Float64[ ε̇_bg * coords[n][1] for n in vx_node_ids])
    bc_vy_vals = TDev(Float64[-ε̇_bg * coords[n][2] for n in vy_node_ids])

    # Seed the interior with the analytical pure-shear field so the first
    # step's solve starts from a good initial guess; boundary nodes are
    # overwritten by apply_bc! below, giving the identical result there.
    copyto!(dr.vx, Float64[ ε̇_bg * c[1] for c in coords_v])
    copyto!(dr.vy, Float64[-ε̇_bg * c[2] for c in coords_v])

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) n_free_surface = length(free_surface_nodes) max_vx = maximum(abs, bc_vx_vals) max_vy = maximum(abs, bc_vy_vals)

    # Thermal diffusion uses the continuous P1 corner mesh. Temperatures are
    # transferred to the discontinuous Stokes pressure/temperature DOFs after
    # every thermal solve, with the same dimensional phase properties as Stokes.
    thermal_dr = ThermalDiffusionDR(
        backend, mesh_litho.nnodes, k, Cp, ρ0, α, K;
        CFL = 0.9, ϵ = 1.0e-6)
    copyto!(thermal_dr.phases, Int[phase_of(c) for c in coords_litho])
    thermal_T_init = Float64[
        in_incl(c) ? T_inclusion :
        T_top + (T_bottom - T_top) * (Ly / 2 - c[2]) / Ly
        for c in coords_litho
    ]
    copyto!(thermal_dr.T, thermal_T_init)
    copyto!(thermal_dr.T0, thermal_T_init)

    thermal_top = Int32[
        corner_id[Int32(n)] for n in Γnodes
        if haskey(corner_id, Int32(n)) &&
           abs(abs(coords[n][1]) - Lx / 2) > tol &&
           abs(coords[n][2] + Ly / 2) > tol
    ]
    thermal_bottom = Int32[
        corner_id[Int32(n)] for n in Γnodes
        if haskey(corner_id, Int32(n)) && abs(coords[n][2] + Ly / 2) ≤ tol
    ]
    thermal_dofs = TDev(vcat(thermal_top, thermal_bottom))
    thermal_vals = TDev(vcat(fill(T_top, length(thermal_top)),
                            fill(T_bottom, length(thermal_bottom))))
    thermal_zero = zero(thermal_vals)
    FEMTools.apply_dirichlet!(
        thermal_dr.T, thermal_dofs, thermal_vals, backend, workgroup)
    copyto!(thermal_dr.T0, thermal_dr.T)

    # FEM pressure residuals are assembled in weak form:
    #
    #     RP_i = ∫ N_i (-∇⋅v) dΩ
    #
    # The Arrow-Hurwicz pressure step and numerical pressure
    # correction are calibrated for that pointwise residual.  If we feed the weak
    # residual directly into Pnum or P += γP*RP/M_P, the update is scaled by element
    # volume and pressure convergence stalls/refines incorrectly.
    #
    # Use the helper to assemble both:
    #   dr.M_P = ∫ N_i dΩ
    #   γP      = phase-local viscosity-weighted pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction without
    # losing the local scaling across viscosity contrasts.
    γP = assemble_viscosity_weighted_pressure_scaling(
        dr, mesh_stokes, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup;
        phases_v = phases_solve, η,
    )

    rel_drop0     = 1e-1     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" Δt iterMax total_iterMax ncheck ϵ_tol

    el2n_v_cpu = Array(mesh_stokes.el2n)
    write_vtk_output && mkpath(out_dir)

    # ---------------------------------------------------------------------------
    # Visualisation setup — plotting-triangle areas and the inclusion/observation
    # overlay boxes passed to the Makie helpers in FEMToolsMakieExt.
    # ---------------------------------------------------------------------------

    element_area = FEMTools.element_triangle_areas(coords_v, el2nP_cpu)
    incl_bounds = (;
        cx, cy, rx = chamber_half_width, ry = chamber_half_height,
        conduit_half_width, conduit_bottom = chamber_conduit_junction,
        conduit_top = Ly / 2 + 3400.0,
    )
    obs_bounds  = objective_bounds

    # ---------------------------------------------------------------------------
    # Adjoint solve: (dR/du)' λ + dJ/du = 0
    # ---------------------------------------------------------------------------
    #
    # The adjoint is assembled on the same T7/P1-disc spaces, quadrature, and
    # element operators as the forward problem, and transposed exactly. This is a
    # requirement, not a convenience:
    #   * Transpose consistency. λ solves the transpose of the *discrete* forward
    #     Jacobian, so λᵀ ∂R/∂m is the exact gradient of the discrete objective
    #     (it matches a finite-difference check of that objective to machine
    #     precision). A cheaper/mismatched adjoint discretisation would make the
    #     gradient inconsistent and degrade any optimisation built on it.
    #   * Inf-sup stability. The adjoint is itself a Stokes saddle point; the T7
    #     bubble is what keeps it LBB-stable. A reduced pressure-unstable pair
    #     would reintroduce checkerboard modes into the adjoint fields.
    # The adjoint solve is still cheaper than the forward one, but through solver
    # effort (a single linear solve at the frozen forward state, reusing the
    # forward Jacobian and preconditioner), never through a coarser discretisation.

    # Two independent objectives share the same observation box: J_vy measures
    # net vertical (uplift/subsidence) motion there, J_vx net horizontal
    # (spreading) motion. Each gets its own adjoint solve below — a linear
    # combination would blend their sensitivities into one field — so the
    # "other" component's load is the zero array for that solve.
    objective_vx = zero(dr.Rv_x)
    objective_vy = zero(dr.Rv_y)
    zero_load_vx = zero(dr.Rv_x)
    zero_load_vy = zero(dr.Rv_y)
    assemble_objective_vx!(
        objective_vx, mesh_stokes, geo_v, element_v,
        objective_bounds..., workgroup,
    )
    assemble_objective_vy!(
        objective_vy, mesh_stokes, geo_v, element_v,
        objective_bounds..., workgroup,
    )

    solve_stats_by_step = NamedTuple[]
    adjoint_stats_by_step = NamedTuple[]
    vtk_paths = NamedTuple[]
    solve_stats = nothing
    adj_vy = adj_vx = nothing
    t = 0.0

    for istep in 1:nsteps
        vx_previous, vy_previous = Array(dr.vx), Array(dr.vy)
        vmax = corner_max_speed(vx_previous, vy_previous, el2n_v_cpu)
        hmin = corner_mesh_hmin(coords_v, el2n_v_cpu)
        dt_step = iszero(vmax) ? Δt : min(Δt, mesh_cfl * hmin / vmax)
        t += dt_step
        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh_stokes, geo_P, element_v, element_P,
            γfact, dt_step, backend, workgroup; phases_v = phases_solve, η)
        @info "Physical time step" istep nsteps t dt_step hmin vmax

        solve_stats = solve_stokes_dyrel!(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_solve, phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
            Γnodes, bc_vx_vals, bc_vy_vals, backend, workgroup;
            ncheck, ϵ_tol, iterMax, total_iterMax, rel_drop0,
            verbose = verbose_PH, verbose_inner = verbose_DR,
            vx_nodes, vy_nodes, collect_history = true,
        )
        solve_stats.converged || @warn "Forward solve did not reach tolerance" istep solve_stats
        push!(solve_stats_by_step, solve_stats)

        # Two separate adjoint solves — one per objective — each contracted into
        # its own material sensitivities via `solve_objective_adjoint`.
        adj_vy = solve_objective_adjoint(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
            vx_nodes, vy_nodes, zero_load_vx, objective_vy,
            η, ρ0, cell_phase, G_stokes, α, K, g, Tref, Val(NV), Val(NP),
            backend, workgroup;
            ncheck, adjoint_tol, adjoint_iterMax, adjoint_verbose,
        )
        adj_vx = solve_objective_adjoint(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
            vx_nodes, vy_nodes, objective_vx, zero_load_vy,
            η, ρ0, cell_phase, G_stokes, α, K, g, Tref, Val(NV), Val(NP),
            backend, workgroup;
            ncheck, adjoint_tol, adjoint_iterMax, adjoint_verbose,
        )
        push!(adjoint_stats_by_step, (; Jvy = adj_vy.stats, Jvx = adj_vx.stats))
        adj_vy.stats.converged || @warn "Adjoint solve (J_vy) did not reach tolerance" istep adj_vy.stats
        adj_vx.stats.converged || @warn "Adjoint solve (J_vx) did not reach tolerance" istep adj_vx.stats
        adjoint_verbose && @info "Adjoint solves complete" istep adj_vy.stats adj_vx.stats

        sensitivity_area = copy(element_area)
        @info "Dimensionless logarithmic material sensitivities (J_vy)" J=adj_vy.J J_scale=adj_vy.J_scale density_log_gradient_by_phase=adj_vy.density_log_gradient_by_phase viscosity_log_gradient_by_phase=adj_vy.viscosity_log_gradient_by_phase G_log_gradient_by_phase=adj_vy.G_log_gradient_by_phase K_log_gradient_by_phase=adj_vy.K_log_gradient_by_phase Q_gradient_by_phase=adj_vy.Q_gradient_by_phase Q_log_gradient_by_phase=adj_vy.Q_log_gradient_by_phase
        @info "Dimensionless logarithmic material sensitivities (J_vx)" J=adj_vx.J J_scale=adj_vx.J_scale density_log_gradient_by_phase=adj_vx.density_log_gradient_by_phase viscosity_log_gradient_by_phase=adj_vx.viscosity_log_gradient_by_phase G_log_gradient_by_phase=adj_vx.G_log_gradient_by_phase K_log_gradient_by_phase=adj_vx.K_log_gradient_by_phase Q_gradient_by_phase=adj_vx.Q_gradient_by_phase Q_log_gradient_by_phase=adj_vx.Q_log_gradient_by_phase

        update_stokes_current_stress!(
            dr, mesh_stokes, geo_v, element_v, element_P,
            phases_solve, τ_old, plastic, τ, G_stokes, dt_step, backend, workgroup)
        post = write_vtk_output ? compute_strain_rate_stress_postprocess(
            Array(dr.vx), Array(dr.vy), el2n_v_cpu, Array(geo_v), map(Array, τ), element_v,
        ) : nothing

        P_cpu = Array(dr.P)
        copyto!(thermal_dr.T0, thermal_dr.T)
        solver!(thermal_dr, dt_step, mesh_litho, geo_litho, element_P,
            thermal_dofs, thermal_zero, thermal_vals, backend, workgroup;
            ncheck = 100, iterMax = 50_000, verbose = false, Tref = Tref)
        thermal_T = Array(thermal_dr.T)
        T_cpu = similar(P_cpu)
        @inbounds for iel in 1:mesh_stokes.nels, a in 1:NP
            T_cpu[DoFsP_cpu[a, iel]] = thermal_T[corner_id[Int32(el2nP_cpu[a, iel])]]
        end
        copyto!(dr.T, T_cpu)

        rotate_stress!(dr, mesh_stokes, geo_v, element_v, dt_step)

        vx_cpu, vy_cpu = Array(dr.vx), Array(dr.vy)
        advect_t7_mesh!(coords_v, el2n_v_cpu, vx_cpu, vy_cpu, dt_step)
        element_area = FEMTools.element_triangle_areas(coords_v, el2nP_cpu)
        copyto!(mesh_v.coords, coords_v)
        copyto!(mesh_stokes.coords, coords_v)
        copyto!(mesh_litho.coords, coords_v[Int.(corner_nodes)])
        geo_litho = FEMTools.precompute_geometry(
            backend, workgroup, mesh_litho, element_P)
        FEMTools.precompute_stokes_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n,
            ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels, backend, workgroup)
        FEMTools.precompute_stokes_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP,
            ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels, backend, workgroup)

        if write_vtk_output
            step = lpad(istep, 4, '0')
            P_cpu = Array(dr.P)
            T_cpu = Array(dr.T)
            λP_vy_cpu, λP_vx_cpu = Array(adj_vy.λP), Array(adj_vx.λP)
            λvx_vy_cpu, λvy_vy_cpu = Array(adj_vy.λvx), Array(adj_vy.λvy)
            λvx_vx_cpu, λvy_vx_cpu = Array(adj_vx.λvx), Array(adj_vx.λvy)
            P_cell = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
            T_cell = [mean(T_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
            λP_vy_cell = [mean(λP_vy_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
            λP_vx_cell = [mean(λP_vx_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
            forward_path = joinpath(out_dir,     "volcano_dims_forward_$step")
            sensitivity_path = joinpath(out_dir, "volcano_dims_adjoint_$step")
            FEMTools.write_vtu(forward_path, mesh_stokes;
                point_data = (; velocity = SVector.(vx_cpu, vy_cpu), Vx = vx_cpu, Vy = vy_cpu),
                cell_data = (; P = P_cell, T = T_cell,
                    viscosity = [η[p] for p in cell_phase], phase = cell_phase,
                    strain_xx = post.εxx, strain_yy = post.εyy, strain_zz = post.εzz,
                    strain_xy = post.εxy, strain_II = post.εII,
                    tau_xx = post.τxx, tau_yy = post.τyy, tau_zz = post.τzz,
                    tau_xy = post.τxy, tau_II = post.tauII))
            # Both objectives' adjoint fields and sensitivities are saved side
            # by side, suffixed `_Jvy`/`_Jvx`, so the two gradients can be
            # compared directly in the same VTK file.
            FEMTools.write_vtu(sensitivity_path, mesh_stokes;
                point_data = (;
                    adjoint_velocity_Jvy = SVector.(λvx_vy_cpu, λvy_vy_cpu),
                    lambda_vx_Jvy = λvx_vy_cpu, lambda_vy_Jvy = λvy_vy_cpu,
                    adjoint_velocity_Jvx = SVector.(λvx_vx_cpu, λvy_vx_cpu),
                    lambda_vx_Jvx = λvx_vx_cpu, lambda_vy_Jvx = λvy_vx_cpu),
                cell_data = (;
                    lambda_P_Jvy = λP_vy_cell, lambda_P_Jvx = λP_vx_cell,
                    dlnJvy_dlnrho = adj_vy.density_log_sensitivity ./ sensitivity_area,
                    dlnJvy_dlneta = adj_vy.viscosity_log_sensitivity ./ sensitivity_area,
                    dlnJvy_dlnG = adj_vy.G_log_sensitivity ./ sensitivity_area,
                    dlnJvy_dlnK = adj_vy.K_log_sensitivity ./ sensitivity_area,
                    dJvy_dQ = adj_vy.Q_sensitivity ./ sensitivity_area,
                    dlnJvy_dlnQ = adj_vy.Q_log_sensitivity ./ sensitivity_area,
                    dlnJvx_dlnrho = adj_vx.density_log_sensitivity ./ sensitivity_area,
                    dlnJvx_dlneta = adj_vx.viscosity_log_sensitivity ./ sensitivity_area,
                    dlnJvx_dlnG = adj_vx.G_log_sensitivity ./ sensitivity_area,
                    dlnJvx_dlnK = adj_vx.K_log_sensitivity ./ sensitivity_area,
                    dJvx_dQ = adj_vx.Q_sensitivity ./ sensitivity_area,
                    dlnJvx_dlnQ = adj_vx.Q_log_sensitivity ./ sensitivity_area,
                    phase = cell_phase))
            push!(vtk_paths, (; forward = forward_path, adjoint = sensitivity_path))
            @info "Wrote forward and adjoint VTK files" istep forward_path sensitivity_path
        end

        show_plot && istep == nsteps && FEMTools.plot_summary(
            adj_vy.density_log_sensitivity, adj_vy.viscosity_log_sensitivity,
            adj_vy.G_log_sensitivity, adj_vy.K_log_sensitivity,
            solve_stats.history, adj_vy.stats.history,
            coords_v, el2nP_cpu, sensitivity_area, incl_bounds, obs_bounds;
            log_scaled = true, Q_sensitivity = adj_vy.Q_log_sensitivity)

    end

    # return (;
    #     dr, mesh_stokes, solve_stats, adj_vy, adj_vx, objective_vx, objective_vy,
    #     solve_stats_by_step, adjoint_stats_by_step, vtk_paths,
    # )
    nothing
end

main(;
    max_area = (20.0e3 / 128)^2,
    coarsening_ratio = 25.0,
    mesh_grading = 0.3,
    nsteps = 100,
    ε̇_bg = 1.0e-15,
)
