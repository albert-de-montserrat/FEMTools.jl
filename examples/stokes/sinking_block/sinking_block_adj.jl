using FEMTools

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Enzyme
using LinearAlgebra
using Printf
using Statistics
using StaticArrays
using KernelAbstractions
using Atomix
using Triangulate
using GLMakie: Figure, Axis, Colorbar, poly!, arrows2d!, lines!, Point2f, DataAspect, axislegend

const default_backend = CPU()
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

@kernel function material_contraction_kernel!(
    contracted_residual,
    @Const(vx), @Const(vy), @Const(P), @Const(P0), @Const(T), @Const(T0),
    @Const(λvx), @Const(λvy), @Const(λP),
    @Const(el2n), @Const(dofsP), @Const(geo), @Const(geo_P), @Const(phases),
    @Const(τ_old), η_element, ρ_element, G_element, K_element,
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
    # The material sensitivity is taken with respect to this element's own η and ρ,
    # so the residual is evaluated with single-entry material tuples and a phase
    # vector that indexes that lone entry (all ones). Differentiating w.r.t.
    # `η_element[iel]`/`ρ_element[iel]` then gives the per-element sensitivity while
    # the phase-indexed `G`, `α`, `K` tuples are still read at the true `phase`.
    phase_e = SVector{NV, Int}(ntuple(_ -> 1, Val(NV)))
    τ_old_e = FEMTools.IntegrationPointStress(
        SVector(ntuple(q -> τ_old[1][q, iel], length(Nq))),
        SVector(ntuple(q -> τ_old[2][q, iel], length(Nq))),
        SVector(ntuple(q -> τ_old[3][q, iel], length(Nq))),
    )
    Re_x, Re_y = FEMTools.integrate_momentum_residual(
        (vx_e, vy_e), P_e, nothing, T_e, geo[iel], phase_e,
        (η_element[iel],), (G_element[iel],), (α[phase],),
        (ρ_element[iel],), (K_element[iel],), g, Tref, Δt,
        τ_old_e, nothing, Nq, NqP,
    )
    phase_P = SVector{NP, Int}(ntuple(_ -> 1, Val(NP)))
    RP_e = FEMTools.integrate_PH_pressure_residual(
        (vx_e, vy_e), P_e, P0_e, T_e, T0_e, geo[iel], geo_P[iel],
        phase_P, (α[phase],), (K_element[iel],), Δt, NqP,
    )
    contracted_residual[iel] = dot(λx_e, Re_x) + dot(λy_e, Re_y) + dot(λP_e, RP_e)
end

function launch_material_contraction!(out, vx, vy, P, P0, T, T0, λvx, λvy, λP,
    el2n, dofsP, geo, geo_P, phases, τ_old, η_element, ρ_element, G_element, K_element,
    α, g, Tref, Δt, Nq, NqP, ::Val{NV}, ::Val{NP}, workgroup,
) where {NV, NP}
    fill!(out, 0)
    backend = KernelAbstractions.get_backend(out)
    material_contraction_kernel!(backend, workgroup)(out, vx, vy, P, P0, T, T0, λvx, λvy, λP,
        el2n, dofsP, geo, geo_P, phases, τ_old, η_element, ρ_element, G_element, K_element,
        α, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP); ndrange = size(el2n, 2))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    material_sensitivities(dr, mesh_stokes, λvx, λvy, geo_v, phases, τ_old,
                           element_v, element_P, η, ρ0, cell_phase,
                           G, α, K, g, Tref, Δt, Val(NV), Val(NP),
                           backend, workgroup) -> (density_sensitivity, viscosity_sensitivity)

Assemble the per-element material sensitivities `sᵉ(m) = λᵉᵀ ∂Rᵉ/∂m` of the
converged adjoint state by reverse-differentiating `launch_material_contraction!`
with respect to the element density and viscosity fields.

`cell_phase[iel]` selects the phase whose `η`/`ρ0` value seeds element `iel`.
Returns the two element fields as CPU arrays; summing the entries of one phase
gives the derivative with respect to that phase's global material value.

The contracted residual, element material fields, and their Enzyme shadows are
allocated on `backend`. Mesh connectivity, geometry, phases, forward fields,
and adjoint fields must be on that same backend. Only the completed sensitivity
vectors are transferred to the CPU for reduction and plotting.
"""
function material_sensitivities(
    dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases, τ_old,
    element_v, element_P, η, ρ0, cell_phase,
    G, α, K, g, Tref, Δt, ::Val{NV}, ::Val{NP}, backend, workgroup,
) where {NV, NP}
    contracted_residual = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nels)
    contracted_seed = KernelAbstractions.ones(backend, Float64, mesh_stokes.nels)
    η_element = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nels)
    ρ_element = similar(η_element)
    G_element = similar(η_element)
    K_element = similar(η_element)
    copyto!(η_element, η[cell_phase])
    copyto!(ρ_element, ρ0[cell_phase])
    copyto!(G_element, collect(G[cell_phase]))
    copyto!(K_element, collect(K[cell_phase]))
    viscosity_sensitivity_backend = zero(η_element)
    density_sensitivity_backend = zero(ρ_element)
    G_sensitivity_backend = zero(G_element)
    K_sensitivity_backend = zero(K_element)
    Nq_v = shape_function_values(element_v)
    Nq_P = shape_function_values(element_P, element_v.integration_points)

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(launch_material_contraction!), Enzyme.Const,
        Enzyme.Duplicated(contracted_residual, contracted_seed),
        Enzyme.Const(dr.vx), Enzyme.Const(dr.vy), Enzyme.Const(dr.P), Enzyme.Const(dr.P0),
        Enzyme.Const(dr.T), Enzyme.Const(dr.T0), Enzyme.Const(λvx), Enzyme.Const(λvy), Enzyme.Const(λP),
        Enzyme.Const(mesh_stokes.el2n), Enzyme.Const(mesh_stokes.DoFsP),
        Enzyme.Const(geo_v), Enzyme.Const(geo_P), Enzyme.Const(phases), Enzyme.Const(τ_old),
        Enzyme.Duplicated(η_element, viscosity_sensitivity_backend),
        Enzyme.Duplicated(ρ_element, density_sensitivity_backend),
        Enzyme.Duplicated(G_element, G_sensitivity_backend),
        Enzyme.Duplicated(K_element, K_sensitivity_backend),
        Enzyme.Const(α), Enzyme.Const(g),
        Enzyme.Const(Tref), Enzyme.Const(Δt), Enzyme.Const(Nq_v), Enzyme.Const(Nq_P),
        Enzyme.Const(Val(NV)), Enzyme.Const(Val(NP)), Enzyme.Const(workgroup),
    )

    # Plotting and phase-wise reductions are CPU-side, so transfer only the two
    # completed element fields after the backend FEM integration has finished.
    return (Array(density_sensitivity_backend), Array(viscosity_sensitivity_backend),
            Array(G_sensitivity_backend), Array(K_sensitivity_backend))
end


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; backend=CPU(), max_area=1/64^2, Δt=1, show_plot=true, kwargs...) -> NamedTuple

Solve one unstructured T7/P1-disc sinking-block Stokes problem and its discrete
adjoint, then assemble the material sensitivities of the observation-box velocity
objective `J(vᵧ) = -∫_Ωₒᵇₛ vᵧ dΩ`.

The model builds a square domain with a rectangular density/viscosity inclusion,
applies free-slip boundary conditions, initialises the pressure from a
lithostatic solve, and runs the Powell-Hestenes/DYREL forward solve. It then
solves the discrete adjoint `(∂R/∂u)ᵀλ = -∂J/∂u` on the same discretisation and
contracts `λᵀ ∂R/∂m` with Enzyme to obtain per-element density and viscosity
sensitivities.

`backend` selects the KernelAbstractions compute backend. The Triangle mesh is
built on the host, after which mesh connectivity, mixed pressure topology,
geometry caches, phases, boundary arrays, and solver state are placed on that
backend. Load CUDA before passing `CUDABackend()`; use `show_plot=false` for
headless runs.

`nsteps` defaults to five, `max_area` sets the Triangle mesh refinement, and
`Δt` is the maximum (visco)elastic time step. The actual step is capped by
`mesh_cfl*hmin/vmax`; after each solve the T7 mesh is advected, straightened,
and its geometry caches are rebuilt. With `write_vtk_output=true`, each step
writes paired forward and adjoint-sensitivity VTK files to `out_dir`.
`show_plot` toggles the GLMakie summary figure. The remaining
keyword arguments (`ncheck`, `ϵ_tol`, `iterMax`, `total_iterMax`, and their
`adjoint_*` counterparts) tune the forward and adjoint solver tolerances and
iteration budgets.

Returns a `NamedTuple` with the solver state (`dr`, `mesh_stokes`), forward and
adjoint statistics (`solve_stats`, `adjoint_stats`), the adjoint fields
(`λvx`, `λvy`, `λP`), the objective load `objective_vy`, the per-element
`density_sensitivity`, `viscosity_sensitivity`, `G_sensitivity`, and
`K_sensitivity`, together with their phase-summed gradients, per-step solver
statistics, and the generated `vtk_paths`.
"""
function main(;
    backend = default_backend,
    nsteps = 5,
    max_area = 1 / 64^2,
    Δt = 0.1,
    mesh_cfl = 0.25,
    show_plot = true,
    write_vtk_output = true,
    out_dir = joinpath(@__DIR__, "output_stokes"),
    ncheck = 50,
    ϵ_tol = 1.0e-6,
    iterMax = 50_000,
    total_iterMax = 50_000,
    adjoint_tol = 1.0e-6,
    adjoint_rel_drop = 0.1, # 5.0e-3,
    adjoint_iterMax = 50_000,
    adjoint_total_iterMax = 50_000,
    adjoint_max_ph_iterations = 100,
    adjoint_verbose = true,
    adjoint_verbose_inner = true,
    # Powell-Hestenes augmentation strength and DYREL Chebyshev damping. A
    # stronger augmentation (γfact) and lighter damping (c_fact) than the historical
    # 20/0.9 cut the forward iteration count by ~15% on this problem without
    # affecting the converged solution.
    γfact = 40.0,
    CFL_v = 0.9,
    c_fact = 0.7,
)
    # Domain
    Lx, Ly = 1.0, 1.0

    # Material (2 phases: matrix + inclusion)
    η     = (1.0,     1e0)   # shear viscosity
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     2e0)   # reference density
    K     = (4e0,     4e0)   # bulk modulus  (Inf → incompressible)
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (1e0,     1e0)   # Shear modulus
    G_stokes = G
    plastic = nothing
    g     = (0.0,     -1.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    # Inclusion geometry. The Triangle PSLG uses this rectangle as an internal
    # constrained boundary, so no element crosses the material interface.
    half_width = 0.1
    cx         = 0.0
    cy         = 0.0
    objective_bounds = (-0.2, 0.2, 0.4, 0.5)

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, interface_nodes = build_triangle_t7_inclusion_mesh(;
        Lx, Ly,
        cx = Lx / 2, cy = -Ly / 2, half_width,
        max_area,
    )
    shift = SVector(-Lx / 2, Ly / 2)
    coords_v_cpu = [c + shift for c in coords_v_cpu]
    # Triangle builds the mesh on the host. The backend-aware constructor
    # uploads coordinates, connectivity, DoFs, and detected boundary nodes in
    # one place, so every array read by a subsequent assembly kernel lives on
    # the same device as the solver fields.
    mesh_v = Mesh(backend, coords_v_cpu, el2n_v_cpu; order = 2)
    mesh_stokes = MixedMesh(mesh_v, element_P)

    # Device array constructor for uploading host-built index/BC/phase arrays to
    # the compute backend (`TA(CPU()) === Array`, so this is a no-op on the CPU).
    TDev = FEMTools.TA(backend)

    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels half_width max_area n_interface_nodes=length(interface_nodes)

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
    M_P    = pressure_mass(dr)
    τ     = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — rectangular inclusion
    # ---------------------------------------------------------------------------

    in_incl(c) = abs(c[1] - cx) ≤ half_width && abs(c[2] - cy) ≤ half_width

    coords_v     = Array(mesh_stokes.coords)

    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    cell_phase = Int[
        in_incl((coords_v[el2nP_cpu[1, i]] + coords_v[el2nP_cpu[2, i]] + coords_v[el2nP_cpu[3, i]]) / 3) ? 2 : 1
        for i in 1:mesh_stokes.nels
    ]
    # `cell_phase` stays on the host for the reductions and material-field
    # gathers below; the solver reads phases inside kernels, so upload a device
    # copy for the assembly path.
    phases_solve = TDev(reshape(cell_phase, 1, :))


    corner_nodes = sort!(unique(vec(el2nP_cpu)))
    corner_id = Dict{Int32, Int32}(old => Int32(i) for (i, old) in enumerate(corner_nodes))
    coords_litho = coords_v[Int.(corner_nodes)]
    el2n_litho = Matrix{Int32}(undef, NP, mesh_stokes.nels)
    @inbounds for iel in 1:mesh_stokes.nels, a in 1:3
        el2n_litho[a, iel] = corner_id[Int32(el2nP_cpu[a, iel])]
    end
    mesh_litho = Mesh(backend, coords_litho, el2n_litho)
    ip_litho = element_P.integration_points
    NQ_litho = length(ip_litho.ω)
    ξq_litho = ntuple(q -> SVector(ip_litho.ξ[q], ip_litho.η[q]), NQ_litho)
    ∂N∂ξq_litho = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_litho[q]), NQ_litho)
    GeoLitho = NTuple{NQ_litho, Tuple{SMatrix{NP, 2, Float64, 2NP}, Float64}}
    geo_litho = KernelAbstractions.allocate(backend, GeoLitho, mesh_litho.nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(
        geo_litho, mesh_litho.coords, mesh_litho.el2n,
        ∂N∂ξq_litho, ip_litho.ω, Val(NP);
        ndrange = mesh_litho.nels,
    )
    KernelAbstractions.synchronize(backend)

    lp_dr = LithostaticPressureDR(backend, mesh_litho.nnodes, ρ0, α, K; CFL = 0.9, ϵ = 1e-2)
    copyto!(lp_dr.phases, Int[in_incl(c) ? 2 : 1 for c in coords_litho])
    P0_litho = Float64[ρ0[1] * abs(g[2]) * (Ly / 2 - c[2]) for c in coords_litho]
    copyto!(lp_dr.P, P0_litho)
    litho_tol = max(Lx, Ly) * eps(Float64) * 32
    top_nodes_litho = Int32[i for i in eachindex(coords_litho) if abs(coords_litho[i][2] - Ly / 2) ≤ litho_tol]
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

    @info "Phases" n_incl=count(==(2), cell_phase)

    # ---------------------------------------------------------------------------
    # Boundary conditions: free slip on the side and bottom walls, with a true
    # traction-free top surface. The top has no velocity Dirichlet condition,
    # so its nodes move with the material during mesh advection.
    # ---------------------------------------------------------------------------

    # Classify the constrained wall nodes on the host (needs scalar coordinate
    # access), then upload the index and value arrays that the Dirichlet kernels
    # consume to the backend.
    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = TDev(Int32[n for n in Γnodes if abs(abs(coords[n][1]) - Lx / 2) ≤ tol])
    vy_nodes = TDev(Int32[n for n in Γnodes if abs(coords[n][2] + Ly / 2) ≤ tol])
    free_surface_nodes = Int32[n for n in Γnodes if abs(coords[n][2] - Ly / 2) ≤ tol]

    bc_vx_vals = KernelAbstractions.zeros(backend, Float64, length(vx_nodes))
    bc_vy_vals = KernelAbstractions.zeros(backend, Float64, length(vy_nodes))

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) n_free_surface = length(free_surface_nodes) max_vx = maximum(abs, bc_vx_vals) max_vy = maximum(abs, bc_vy_vals)

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
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, geo_P, element_v, element_P,
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
    # Visualisation helper — per-element average of nodal fields on the P triangles
    # ---------------------------------------------------------------------------

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    # Element areas of the (straight-sided) plotting triangles, equal to ∫dΩ over
    # each element. The raw material sensitivities are un-normalized element
    # integrals sᵉ = λᵉᵀ ∂Rᵉ/∂m ≈ areaᵉ·(sensitivity density); dividing by areaᵉ
    # recovers the mesh-independent sensitivity density used for plotting, while
    # the phase-wise gradients keep summing the raw integrals.
    element_area = [
        let a = coords_v[el2nP_cpu[1, i]], b = coords_v[el2nP_cpu[2, i]], c = coords_v[el2nP_cpu[3, i]]
            abs((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
        end
        for i in 1:mesh_stokes.nels
    ]
    xlo, xhi = cx - half_width, cx + half_width
    ylo, yhi = cy - half_width, cy + half_width
    obs_xlo, obs_xhi, obs_ylo, obs_yhi = objective_bounds

    # Draw both pieces of geometry that matter when interpreting the fields:
    # the material inclusion (white dashed box) and the objective/observation
    # region (yellow solid box). Keeping this in one helper guarantees that the
    # standalone forward figure and the final sensitivity figure use identical
    # overlays.
    function draw_geometry_boxes!(ax)
        lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo];
            color = :white, linewidth = 1.5, linestyle = :dash)
        lines!(ax,
            [obs_xlo, obs_xhi, obs_xhi, obs_xlo, obs_xlo],
            [obs_ylo, obs_ylo, obs_yhi, obs_yhi, obs_ylo];
            color = :yellow, linewidth = 2)
        return nothing
    end

    function plot_fields(vx_dofs, vy_dofs, P_dofs, titles)
        el_vx = [mean(vx_dofs[el2nP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
        el_vy = [mean(vy_dofs[el2nP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
        el_P  = [mean(P_dofs[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
        fig = Figure(size = (1400, 440))
        for (col, title, values, colormap) in (
                (1, titles[1], el_vx, :vik),
                (3, titles[2], el_vy, :vik),
                (5, titles[3], el_P,  :glasgow),
            )
            limits = extrema(values)
            ax = Axis(fig[1, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
            poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
            Colorbar(fig[1, col + 1]; colormap, limits, width = 15, tellheight = false)
            draw_geometry_boxes!(ax)
        end
        display(fig)
        return fig
    end

    # Material-sensitivity maps and the forward + adjoint residual trace.
    # The sensitivities are already element fields, but they are raw element
    # integrals; dividing by `element_area` yields the sensitivity density, which
    # is mesh-independent and free of the per-element speckle that the
    # area-weighted integrals show on an irregular mesh.
    function plot_summary(density_sensitivity, viscosity_sensitivity,
                          G_sensitivity, K_sensitivity, fwd_hist, adj_hist)
        current_pts = [Point2f(c) for c in coords_v]
        current_polys = [[current_pts[el2nP_cpu[1, i]], current_pts[el2nP_cpu[2, i]], current_pts[el2nP_cpu[3, i]]]
                         for i in 1:mesh_stokes.nels]
        fig = Figure(size = (1100, 900))
        for (row, col, title, values, colormap) in (
                (1, 1, "Density sensitivity (per area)", density_sensitivity ./ element_area, :vik),
                (1, 3, "Viscosity sensitivity (per area)", viscosity_sensitivity ./ element_area, :vik),
                (2, 1, "dJ/dG sensitivity (per area)", G_sensitivity ./ element_area, :vik),
                (2, 3, "dJ/dK sensitivity (per area)", K_sensitivity ./ element_area, :vik),
            )
            limits = extrema(values)
            ax = Axis(fig[row, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
            poly!(ax, current_polys; color = values, colormap, colorrange = limits, strokewidth = 0)
            Colorbar(fig[row, col + 1]; colormap, limits, width = 15, tellheight = false)
            draw_geometry_boxes!(ax)
        end

        ax = Axis(fig[3, 1:4]; xlabel = "iteration", ylabel = "log10 residual",
            title = "Residual evolution")
        logres(v) = log10.(max.(v, eps()))
        if !isempty(fwd_hist)
            it = Float64[h.iter for h in fwd_hist]
            lines!(ax, it, logres([h.err_v for h in fwd_hist]); label = "forward V")
            lines!(ax, it, logres([h.err_P for h in fwd_hist]); label = "forward P")
        end
        if !isempty(adj_hist)
            it = Float64[h.iter for h in adj_hist]
            lines!(ax, it, logres([h.err_v for h in adj_hist]); label = "adjoint V", linestyle = :dash)
            lines!(ax, it, logres([h.err_P for h in adj_hist]); label = "adjoint P", linestyle = :dash)
        end
        (isempty(fwd_hist) && isempty(adj_hist)) || axislegend(ax; position = :rt)

        display(fig)
        return fig
    end

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

    objective_vx = zero(dr.Rv_x)
    objective_vy = zero(dr.Rv_y)
    assemble_objective_vy!(
        objective_vy, mesh_stokes, geo_v, element_v,
        objective_bounds..., workgroup,
    )

    solve_stats_by_step = NamedTuple[]
    adjoint_stats_by_step = NamedTuple[]
    vtk_paths = NamedTuple[]
    solve_stats = adjoint_stats = nothing
    λvx = λvy = λP = nothing
    density_sensitivity = viscosity_sensitivity = G_sensitivity = K_sensitivity = nothing
    density_gradient_by_phase = viscosity_gradient_by_phase = nothing
    G_gradient_by_phase = K_gradient_by_phase = nothing
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

        λvx = zero(dr.vx)
        λvy = zero(dr.vy)
        λP  = zero(dr.P)

        adjoint_stats = solve_stokes_adjoint_dyrel!(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_solve, phases_solve, τ_old, plastic, G_stokes, dt_step, γP,
        objective_vx, objective_vy, λvx, λvy, λP,
        backend, workgroup;
        vx_nodes,
        vy_nodes,
        ncheck,
        adjoint_tol,
        rel_drop = adjoint_rel_drop,
        iterMax = adjoint_iterMax,
        total_iterMax = adjoint_total_iterMax,
        max_ph_iterations = adjoint_max_ph_iterations,
        verbose = adjoint_verbose,
        verbose_inner = adjoint_verbose_inner,
        collect_history = true,
    )
        adjoint_history = adjoint_stats.history
        push!(adjoint_stats_by_step, adjoint_stats)
        adjoint_verbose && @info "Adjoint solve complete" istep adjoint_stats

    # -----------------------------------------------------------------------
    # Material sensitivities
    # -----------------------------------------------------------------------
    #
    # With λ solving (∂R/∂u)ᵀλ = -∂J/∂u, the gradient of the discrete objective
    # with respect to a material parameter m and element momentum residual Rᵉ is
    #
    #     sᵉ(m) = λᵉᵀ (∂Rᵉ/∂m).
    #
    # Each entry is the contribution from one element. Summing entries of one
    # phase gives the derivative with respect to that phase's single global
    # material value. Keeping the contributions element-wise gives the spatial
    # sensitivity maps analogous to `dρc` and `η_sens` in the FD script.
    density_sensitivity, viscosity_sensitivity, G_sensitivity, K_sensitivity = material_sensitivities(
        dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases_solve, τ_old,
        element_v, element_P, η, ρ0, cell_phase,
        G_stokes, α, K, g, Tref, dt_step, Val(NV), Val(NP), backend, workgroup,
    )

    density_gradient_by_phase = ntuple(
        p -> sum(density_sensitivity[cell_phase .== p]), length(ρ0),
    )
    viscosity_gradient_by_phase = ntuple(
        p -> sum(viscosity_sensitivity[cell_phase .== p]), length(η),
    )
    G_gradient_by_phase = ntuple(p -> sum(G_sensitivity[cell_phase .== p]), length(G))
    K_gradient_by_phase = ntuple(p -> sum(K_sensitivity[cell_phase .== p]), length(K))
    @info "Material sensitivities assembled" density_gradient_by_phase viscosity_gradient_by_phase G_gradient_by_phase K_gradient_by_phase

        update_stokes_current_stress!(
            dr, mesh_stokes, geo_v, element_v, element_P,
            phases_solve, τ_old, plastic, τ, G_stokes, dt_step, backend, workgroup)
        rotate_stress!(dr, mesh_stokes, geo_v, element_v, dt_step)

        vx_cpu, vy_cpu = Array(dr.vx), Array(dr.vy)
        advect_t7_mesh!(coords_v, el2n_v_cpu, vx_cpu, vy_cpu, dt_step)
        @inbounds for i in eachindex(element_area)
            a, b, c = coords_v[el2nP_cpu[1, i]], coords_v[el2nP_cpu[2, i]], coords_v[el2nP_cpu[3, i]]
            element_area[i] = abs((b[1] - a[1]) * (c[2] - a[2]) -
                                  (c[1] - a[1]) * (b[2] - a[2])) / 2
        end
        copyto!(mesh_v.coords, coords_v)
        copyto!(mesh_stokes.coords, coords_v)
        FEMTools.precompute_stokes_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n,
            ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels, backend, workgroup)
        FEMTools.precompute_stokes_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP,
            ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels, backend, workgroup)

        if write_vtk_output
            step = lpad(istep, 4, '0')
            P_cpu, λP_cpu = Array(dr.P), Array(λP)
            λvx_cpu, λvy_cpu = Array(λvx), Array(λvy)
            P_cell = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
            λP_cell = [mean(λP_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]
            forward_path = joinpath(out_dir, "sinking_block_forward_$step.vtk")
            sensitivity_path = joinpath(out_dir, "sinking_block_adjoint_$step.vtk")
            write_vtk(forward_path, mesh_stokes;
                point_data = (; velocity = SVector.(vx_cpu, vy_cpu), Vx = vx_cpu, Vy = vy_cpu),
                cell_data = (; P = P_cell, phase = cell_phase),
                title = "Sinking block forward solution, step $istep")
            write_vtk(sensitivity_path, mesh_stokes;
                point_data = (; adjoint_velocity = SVector.(λvx_cpu, λvy_cpu),
                    lambda_vx = λvx_cpu, lambda_vy = λvy_cpu),
                cell_data = (; lambda_P = λP_cell, dJ_drho = density_sensitivity,
                    dJ_deta = viscosity_sensitivity, dJ_dG = G_sensitivity,
                    dJ_dK = K_sensitivity, phase = cell_phase),
                title = "Sinking block adjoint sensitivities, step $istep")
            push!(vtk_paths, (; forward = forward_path, adjoint = sensitivity_path))
            @info "Wrote forward and adjoint VTK files" istep forward_path sensitivity_path
        end

        show_plot && istep == nsteps && plot_summary(
            density_sensitivity, viscosity_sensitivity, G_sensitivity, K_sensitivity,
            solve_stats.history, adjoint_history)

    end

    # return (;
    #     dr, mesh_stokes, solve_stats, adjoint_stats, λvx, λvy, λP, objective_vy,
    #     solve_stats_by_step, adjoint_stats_by_step, vtk_paths,
    #     density_sensitivity, viscosity_sensitivity, G_sensitivity, K_sensitivity,
    #     density_gradient_by_phase, viscosity_gradient_by_phase,
    #     G_gradient_by_phase, K_gradient_by_phase,
    # )
    nothing
end

# if abspath(PROGRAM_FILE) == @__FILE__
main(;
    nsteps = 50,
)
# end
