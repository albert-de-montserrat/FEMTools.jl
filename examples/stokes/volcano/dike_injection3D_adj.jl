# Coupled thermal-Stokes model of magma injected into a dipping dike beneath a
# flat surface, in three dimensions, with the adjoint log-sensitivities of the
# uplift above the dike to the bulk modulus and the injection rate.

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using ForwardDiff
using KernelAbstractions
using FEMTools
using GLMakie: Figure, Axis, Colorbar, Label, poly!, lines!, Point2f, DataAspect, save

const hr = 3600.0

# Keep the default invocation CPU-friendly while allowing the script itself to
# be launched on CUDA by setting `_use_cuda = true`. Calling `main` directly
# also accepts any KernelAbstractions backend.
const _use_cuda = false
@static if _use_cuda
    using CUDA
    const _default_backend = CUDA.CUDABackend()
else
    const _default_backend = CPU()
end

include(joinpath(@__DIR__, "volcano_mesh_3D.jl"))

# ---------------------------------------------------------------------------
# Parameter sensitivities of the adjoint
#
# `solve_stokes_adjoint_dyrel!` transposes the Powell-Hestenes augmented system
# [Rv + W·RP; RP], where W = ∂Rv/∂Pnum · γP/M_P is element-local. Its pressure
# multiplier λP is therefore related to the multiplier of the plain residual
# R = [Rv; RP] by λP_plain = λP + Wᵀλv, and for a parameter m
#
#     dJ/dm = λvᵀ ∂Rv/∂m + λP_plainᵀ ∂RP/∂m,
#
# contracted element by element, with the stress, pressure and temperature
# history of the step held fixed.
# ---------------------------------------------------------------------------

"""
    element_state(data, iel, Val(NV), Val(NP)) -> NamedTuple

Element-local converged state, frozen history, and geometry of element `iel`,
gathered the way the forward solve's element residuals gather them.
"""
function element_state(data, iel, ::Val{NV}, ::Val{NP}) where {NV, NP}
    nodes_v = FEMTools.local_nodes_of(data.el2n, iel, Val(NV))
    nodes_P = FEMTools.local_nodes_of(data.DoFsP, iel, Val(NP))
    geo_v = element_geometry(data.geo_v, iel, data.∂N∂ξ)
    NQ = FEMTools.quadrature_points_val(geo_v)
    gather_P(field) = FEMTools._gather_local(field, nodes_P, Val(NP))
    return (;
        nodes_v, nodes_P, geo_v, geo_P = element_geometry(data.geo_P, iel),
        v = map(vc -> FEMTools._gather_local(vc, nodes_v, Val(NV)), data.v),
        P = gather_P(data.P), T = gather_P(data.T), T0 = gather_P(data.T0),
        P0 = FEMTools._gather_old_pressure(data.P0, nodes_P, iel, Val(NP), NQ),
        Pf = gather_P(data.Pf), Q = gather_P(data.Q),
        τ_old = FEMTools._gather_old_stress(data.τ_old, nodes_v, iel, Val(NV), NQ),
        phase_v = FEMTools._gather_phase(data.phases_v, nodes_v, iel, Val(NV)),
        phase_P = FEMTools._gather_phase(data.phases_P, nodes_P, iel, Val(NP)),
        Γ = gather_P(data.γP) ./ gather_P(data.M_P),
    )
end

# Momentum residual of one element, stacked by component; `Pnum` is the
# Powell-Hestenes pressure, zero at the converged forward state.
element_momentum(e, data, K, plastic, Pnum) = vcat(
    FEMTools.integrate_momentum_residual(
        e.v, e.P, Pnum, e.T, e.geo_v, e.phase_v,
        data.η, data.G, data.α, data.ρ0, K, data.g, data.Tref, data.Δt,
        data.Nq, data.NqP, e.τ_old, plastic, nothing, nothing, nothing, nothing, e.Pf,
    )...,
)

element_continuity(e, data, ηb, Q_scale) = FEMTools.integrate_PH_pressure_residual(
    e.v, e.P, e.P0, e.T, e.T0, Q_scale * e.Q, e.geo_v, e.geo_P, e.phase_P,
    data.α, ηb, data.Δt, data.NqP,
)

# Bulk modulus of the volumetric plastic correction, scaled by `factor`.
scale_bulk_modulus(::Nothing, _) = nothing
scale_bulk_modulus(p::DruckerPrager{n}, factor) where {n} = DruckerPrager{n, typeof(factor)}(
    p.cosϕ, p.sinϕ, p.sinΨ, p.C, p.η_reg, p.Kb .* factor, p.damage,
)

"""
    box_mean_load(coords, el2n, geo_v, ∂N∂ξ, Nq, box, Val(NV)) -> (load, volume)

Consistent load `∂J/∂vᵢ = ∫_box Nᵢ dΩ / |box|` of the box-mean objective
`J = ∫_box v dΩ / |box|`, and the box volume. Membership is decided per
quadrature point, so a cut element contributes its quadrature-weighted share.
"""
function box_mean_load(coords, el2n, geo_v, ∂N∂ξ, Nq, box, ::Val{NV}) where {NV}
    xmin, xmax, ymin, ymax, zmin, zmax = box
    load = zeros(length(coords))
    volume = 0.0
    for iel in axes(el2n, 2)
        nodes = FEMTools.local_nodes_of(el2n, iel, Val(NV))
        geo_el = element_geometry(geo_v, iel, ∂N∂ξ)
        for q in eachindex(geo_el)
            _, dΩ = geo_el[q]
            xq = sum(Nq[q][a] * coords[nodes[a]] for a in 1:NV)
            (xmin < xq[1] < xmax && ymin < xq[2] < ymax && zmin < xq[3] < zmax) || continue
            for a in 1:NV
                load[nodes[a]] += Nq[q][a] * dΩ
            end
            volume += dΩ
        end
    end
    volume > 0 || throw(ArgumentError("the observation box contains no quadrature point"))
    return load ./ volume, volume
end

"""
    log_sensitivities(data, λv, λP, Val(NV), Val(NP)) -> (dJ_dlnK, dJ_dlnQ, λP_plain)

Per-element contractions `λᵀ ∂R/∂ln K` and `λᵀ ∂R/∂ln Q` of the multipliers
`λv`, `λP` returned by `solve_stokes_adjoint_dyrel!`, and the pressure
multiplier `λP_plain` of the plain (unaugmented) residual. `K` scales the
material bulk modulus, the pressure storage modulus `ηb`, and the plastic
correction's bulk modulus together; `Q` scales the injection source.
"""
function log_sensitivities(data, λv, λP, ::Val{NV}, ::Val{NP}) where {NV, NP}
    data.ηb == data.K ||
        throw(ArgumentError("the bulk-modulus sensitivity assumes the pressure storage modulus ηb equals K"))
    nels = size(data.el2n, 2)
    dJ_dlnK = zeros(nels)
    dJ_dlnQ = zeros(nels)
    λP_plain = similar(λP)
    for iel in 1:nels
        e = element_state(data, iel, Val(NV), Val(NP))
        λv_e = vcat(map(λ -> FEMTools._gather_local(λ, e.nodes_v, Val(NV)), λv)...)
        Bnum = ForwardDiff.jacobian(
            Pnum -> element_momentum(e, data, data.K, data.plastic, Pnum), zero(e.P),
        )
        λP_e = FEMTools._gather_local(λP, e.nodes_P, Val(NP)) + e.Γ .* (transpose(Bnum) * λv_e)
        λP_plain[e.nodes_P] = λP_e
        dJ_dlnK[iel] = ForwardDiff.derivative(0.0) do s
            factor = exp(s)
            dot(λv_e, element_momentum(e, data, data.K .* factor, scale_bulk_modulus(data.plastic, factor), nothing)) +
                dot(λP_e, element_continuity(e, data, data.ηb .* factor, 1.0))
        end
        dJ_dlnQ[iel] = ForwardDiff.derivative(s -> dot(λP_e, element_continuity(e, data, data.ηb, exp(s))), 0.0)
    end
    return dJ_dlnK, dJ_dlnQ, λP_plain
end

"""
    tet_section(coords, el2n, axis, value) -> (polygons, elements)

Polygons where the plane `x[axis] = value` cuts the straight-sided tetrahedra
spanned by the first four nodes of each element, projected onto the two other
axes, and the indices of the cut elements.
"""
function tet_section(coords, el2n, axis, value)
    keep = filter(!=(axis), (1, 2, 3))
    polygons = Vector{Vector{Point2f}}()
    elements = Int[]
    for iel in axes(el2n, 2)
        corners = ntuple(a -> coords[el2n[a, iel]], 4)
        distance = map(c -> c[axis] - value, corners)
        points = Point2f[]
        for (a, b) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
            (distance[a] < 0) == (distance[b] < 0) && continue
            p = corners[a] + distance[a] / (distance[a] - distance[b]) * (corners[b] - corners[a])
            push!(points, Point2f(p[keep[1]], p[keep[2]]))
        end
        isempty(points) && continue
        center = sum(points) / length(points)
        sort!(points; by = p -> atan(p[2] - center[2], p[1] - center[1]))
        push!(polygons, points)
        push!(elements, iel)
    end
    return polygons, elements
end

"""
    main(; nsteps=24, Δt=1hr, Q_dike=1000, dip=90, obs_side=8km, fd_check=false, ...) -> NamedTuple

Run the coupled thermal--Stokes model of magma injection into a dike on an
unstructured dike-conforming T11/P1-discontinuous mesh, and compute the adjoint
log-sensitivities of the uplift above the dike at the last step.

The objective is the mean vertical velocity `J = ∫_box vz dΩ / |box|` over a box
with a square footprint of side `obs_side`, centred above the top of the dike,
extending from the depth of the dike top to the free surface. Its discrete
adjoint at the converged last step, with that step's stress, pressure and
temperature history held fixed, gives the log-sensitivities
`∂ln J/∂ln K` to the bulk modulus `K` (also the pressure storage modulus `ηb`
and the plastic correction's bulk modulus), as a per-element map and summed
per phase, and `∂ln J/∂ln Q_dike`. With `fd_check = true` the latter is compared
with a central finite difference of two extra Stokes solves of the last step.
The multiplier, the sensitivity map, and the vertical velocity are plotted on
sections across the dike and through the box. The last VTK file carries, per
element, both log-sensitivities (summing to the totals) and their densities per
km³, the corner-averaged adjoint velocities and element-averaged adjoint
pressure, and a mask of the observation box.

A 40 × 40 × 20 km crustal block with a flat, traction-free surface holds a dike
`dike_width` thick and `dike_length` long along strike (`y`), spanning depths
`dike_bottom` to `dike_top` and dipping at `dip` degrees from the horizontal: 90
is vertical, a smaller angle dips toward `+x` and a larger one toward `−x`. The
dike is a conforming material volume, and the mesh is graded from
`mesh_size / refinement` on its faces to `mesh_size` at `refinement_distance`
away. The crust is visco-elasto-plastic (Drucker-Prager, cohesion 10 MPa,
friction angle 30°) and the dike is visco-elastic magma held at `T_magma`. The
yield model sees the effective pressure `P − Pf`, with a hydrostatic pore
pressure `Pf = λ_fluid P_litho` refreshed from the lithostatic pressure every
step.

Magma enters through the continuity source `Q`: a uniform weight on the dike's
pressure DoFs, rescaled by the solver so that the injected volume rate is
`Q_dike` [m³ s⁻¹]. The defaults follow the 2023–2024 Reykjanes dike injections
(Sigmundsson et al., 2024; Parks et al., 2025), whose later events peaked at
800–1700 m³ s⁻¹, delivered over hours, so `Δt` is an hour. On that time scale
the crust, whose Maxwell time `η/G` is about a kyr, responds elastically.

The walls carry no normal velocity and the base no vertical velocity;
temperature is fixed at the surface and the base and insulating on the sides.
The mesh is Lagrangian: after every step all of its nodes move with the
velocity, so the surface deforms freely. The stress history is carried over
without rotation, which over hour-long steps is negligible. At the end of every
step the lithostatic pressure of the current density field and geometry is
recomputed, and the overpressure `P − P_litho` is written to the VTK output.

The velocity space is the 11-node bubble-enriched quadratic tetrahedron and the
pressure space is discontinuous linear on the four tetrahedron vertices, as in
`volcano_thermal_stokes_3D.jl`. The tensile cap and shear heating of the
two-dimensional `dike_injection.jl` are plane strain only in FEMTools and are
not used here.

All arguments and returned fields are SI; the solve itself runs in the
characteristic units defined below.

Returns the time series of mean and maximum second stress invariant, mean dike
pressure and maximum surface uplift rate, the last post-processed field set,
per-step solver statistics, and the adjoint results. Fields are written as VTK
for inspection in an external viewer.
"""
function main(;
        nsteps = 24,
        Δt = 1hr,
        Q_dike = 1000.0,
        dike_width = 500.0,
        dike_length = 8.0e3,
        dike_top = -2.0e3,
        dike_bottom = -8.0e3,
        dip = 90.0,
        mesh_size = 4.0e3,
        refinement = 8.0,
        refinement_distance = 3.0e3,
        # The Drucker-Prager return scales as 2 ηve / (ηve + η_vp): η_vp must be
        # of the order of the crust's viscoelastic viscosity ηve ≈ G Δt for the
        # plastic correction to act, and not below it, or it overshoots.
        η_vp = 3.0e10 * Δt,
        # Pore-fluid pressure as a fraction of the lithostatic pressure; the
        # hydrostatic ratio ρ_water / ρ_crust ≈ 1000 / 2700.
        λ_fluid = 0.37,
        γfact = 100.0,
        T_magma = 1100 + 273.0,
        dTdz = 25.0e-3,
        ϵ_T = 1.0e-6,
        obs_side = 8.0e3,
        fd_check = false,
        show_plot = true, write_output = true, verbose = true,
        backend = _default_backend, workgroup = 128,
    )
    # -----------------------------------------------------------------------
    # Geometry [m]
    # -----------------------------------------------------------------------
    Lx = Ly = 40.0e3
    depth = 20.0e3
    0 < dip < 180 || throw(ArgumentError("dip must lie strictly between 0 and 180 degrees, got $dip"))
    dike_top < 0 && dike_bottom > -depth && dike_top > dike_bottom ||
        throw(ArgumentError("the dike must lie strictly below the surface and above the base"))
    # Length along the dip, so the dike keeps its depth range at any dip.
    dike_height = (dike_top - dike_bottom) / sind(dip)
    dike_center = (0.0, 0.0, (dike_top + dike_bottom) / 2)
    # Rotation of the vertical dike about the strike axis: its top moves toward −x.
    dike_rotation = deg2rad(90 - dip)
    # Unit vectors across and along the dike in the x–z plane.
    dike_across = (cos(dike_rotation), sin(dike_rotation))
    dike_along  = (-sin(dike_rotation), cos(dike_rotation))
    # Observation box above the dike top; open upward so it follows the
    # deforming surface. [m]
    x_top = dike_center[1] + dike_height / 2 * dike_along[1]
    obs_box = (x_top - obs_side / 2, x_top + obs_side / 2, -obs_side / 2, obs_side / 2, dike_top, Inf)

    # -----------------------------------------------------------------------
    # Characteristic units
    #
    # The dynamic-relaxation convergence tests compare absolute residuals with
    # fixed thresholds, so a crustal-scale problem in SI units diverges the
    # guard before it converges. Everything below is therefore scaled by these
    # units and scaled back for output. Temperature stays in Kelvin.
    # -----------------------------------------------------------------------
    L_c  = depth                  # length [m]
    t_c  = Δt                     # time [s]: the injection time scale
    σ_c  = 1.0e8                  # stress [Pa]
    η_c  = σ_c * t_c              # viscosity [Pa s]
    ρ_c  = σ_c * t_c^2 / L_c^2    # density [kg m⁻³]
    g_c  = L_c / t_c^2            # gravity [m s⁻²]
    Cp_c = L_c^2 / t_c^2          # heat capacity [J kg⁻¹ K⁻¹]
    k_c  = ρ_c * Cp_c * L_c^2 / t_c   # thermal conductivity [W m⁻¹ K⁻¹]

    # -----------------------------------------------------------------------
    # Material properties — phase 1: crust, phase 2: dike.
    # Listed in SI, divided by the characteristic units above.
    # -----------------------------------------------------------------------
    η    = (1.0e21, 1.0e19) ./ η_c    # shear viscosity [Pa s]
    G    = (3.0e10, 1.0e9)  ./ σ_c    # shear modulus [Pa]
    K    = (5.0e10, 1.0e10) ./ σ_c    # bulk modulus [Pa]
    ηb   = K                          # pressure storage modulus; residual uses ηb * Δt
    ρ0   = (2700.0, 2400.0) ./ ρ_c    # reference density [kg m⁻³]
    k    = (2.5, 2.0)       ./ k_c    # thermal conductivity [W m⁻¹ K⁻¹]
    Cp   = (1000.0, 1200.0) ./ Cp_c   # heat capacity [J kg⁻¹ K⁻¹]
    α    = (3.0e-5, 5.0e-5)           # thermal expansivity [K⁻¹]
    g    = (0.0, 0.0, -9.81 / g_c)    # gravity [m s⁻²]
    Tref = 273.0                      # equation-of-state reference temperature [K]

    ϕ = deg2rad(30)                   # friction angle
    Ψ = deg2rad(0)                    # dilation angle (non-associated)
    C_crust = 1.0e7 / σ_c             # cohesion [Pa]; yield stress = C cosϕ + (P − Pf) sinϕ
    # The dike stays visco-elastic. Its cohesion is set far above any
    # attainable stress rather than to `Inf`, because per-phase properties are
    # interpolated with quadratic shape functions, which are negative at some
    # quadrature points and would turn an infinite cohesion into `NaN`.
    C_dike = 1.0e12 / σ_c
    plastic = DruckerPrager(
        (ϕ, ϕ), (Ψ, Ψ), (C_crust, C_dike), (η_vp / η_c, η_vp / η_c), K,
    )

    T_surface = 273.0
    T_base    = T_surface + dTdz * depth

    Δτ = Δt / t_c
    Q_total = Q_dike / (L_c^3 / t_c)

    # -----------------------------------------------------------------------
    # Meshes — one node set shared by the velocity, thermal, and pressure fields
    # -----------------------------------------------------------------------
    element_v = ReferenceElement(QuadraticElement{3, 11, Float64})  # T10 + centroid bubble
    element_P = ReferenceElement(LinearElement{3, 4, Float64})      # P1 discontinuous

    coords_cpu, el2n_cpu, groups = build_tet11_dike_mesh(;
        Lx = Lx / L_c, Ly = Ly / L_c, depth = depth / L_c,
        dike_center = dike_center ./ L_c,
        dike_width = dike_width / L_c, dike_length = dike_length / L_c,
        dike_height = dike_height / L_c, rotation = dike_rotation,
        mesh_size = mesh_size / L_c, refinement,
        refinement_distance = refinement_distance / L_c,
    )
    # Keep the host mesh data above for meshing, classification, and output,
    # but put all connectivity consumed by solver kernels on the selected
    # backend. `MixedMesh` preserves that placement for pressure connectivity.
    mesh_T = Mesh(backend, coords_cpu, el2n_cpu, element_v; workgroup)
    mesh_stokes = MixedMesh(mesh_T, element_P)
    geo_v = mesh_stokes.geometry.geo_v

    NQ_v = length(element_v.integration_points.ω)
    NV   = length(element_v)
    NP   = length(element_P)

    @info "Unstructured conforming mesh (T11/P1-disc)" nnodes_v = mesh_stokes.nnodes nnodes_P =
        mesh_stokes.nnodesP nels = mesh_stokes.nels

    # -----------------------------------------------------------------------
    # Phases — element phase from the mesh, node phase from the elements
    # -----------------------------------------------------------------------
    cell_phase = groups.phase
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)
    TDev = FEMTools.TA(backend)
    phases_v = TDev(phases_v_cpu)
    phases_P = TDev(phases_P_cpu)

    node_phase = ones(Int, length(coords_cpu))
    for iel in axes(el2n_cpu, 2), a in axes(el2n_cpu, 1)
        cell_phase[iel] == 2 && (node_phase[el2n_cpu[a, iel]] = 2)
    end

    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu  = Array(mesh_stokes.el2nP)
    DoFsP_cpu  = Array(mesh_stokes.DoFsP)
    dike_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, findall(==(2), cell_phase)])))
    isempty(dike_P_dofs) &&
        throw(ArgumentError("no element lies inside the dike; refine the mesh or widen the dike"))

    # Uniform source weight on the dike; the solver rescales it to Q_total.
    Qq_cpu = zeros(mesh_stokes.nnodesP)
    Qq_cpu[dike_P_dofs] .= 1.0
    Qq = TDev(Qq_cpu)

    @info "Phases" n_dike_cells = count(==(2), cell_phase) n_dike_nodes =
        count(==(2), node_phase)

    # -----------------------------------------------------------------------
    # Solver states
    # -----------------------------------------------------------------------
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP,
        StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref);
        CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ     = FEMTools.stress(dr)
    τ_old = FEMTools.stress_old(dr)

    thermal = ThermalDiffusionDR(
        backend, mesh_T.nnodes, ThermalMaterial(; k, Cp, ρ0, α, K); ϵ = ϵ_T,
    )
    copyto!(thermal.phases, TDev(node_phase))

    # -----------------------------------------------------------------------
    # Boundary conditions
    #   No normal velocity on the x and y walls and the base, free slip in the
    #   other components; the surface carries no traction.
    #   T is fixed on the surface and the base, insulating on the sides.
    # -----------------------------------------------------------------------
    vx_nodes = sort!(union(groups.left, groups.right))
    vy_nodes = sort!(union(groups.front, groups.back))
    bc_vx = DirichletBoundaryCondition(
        nothing, TDev(Int32.(vx_nodes)), TDev(zeros(length(vx_nodes))),
    )
    bc_vy = DirichletBoundaryCondition(
        nothing, TDev(Int32.(vy_nodes)), TDev(zeros(length(vy_nodes))),
    )
    bc_vz = DirichletBoundaryCondition(
        nothing, TDev(Int32.(groups.bottom)), TDev(zeros(length(groups.bottom))),
    )
    bc_T = DirichletBoundaryCondition(
        nothing,
        TDev(Int32.(vcat(groups.surface, groups.bottom))),
        TDev(vcat(fill(T_surface, length(groups.surface)), fill(T_base, length(groups.bottom)))),
    )

    geotherm(c) = T_surface - dTdz * L_c * c[3]
    copyto!(thermal.T, [
        node_phase[i] == 2 ? T_magma : geotherm(coords_cpu[i]) for i in eachindex(coords_cpu)
    ])
    apply_bc!(thermal.T, bc_T; workgroup)
    copyto!(thermal.T0, thermal.T)

    # Lithostatic pressure of the current temperature field and geometry,
    # sampled at the pressure DoFs. It warm-starts the pressure, so the first
    # Powell-Hestenes step does not have to build it from zero, sets the pore
    # pressure, and is the reference of the overpressure output.
    litho = LithostaticPressureDR(backend, mesh_T.nnodes, ρ0, α, K)
    copyto!(litho.phases, TDev(node_phase))
    bc_P = DirichletBoundaryCondition(
        nothing, TDev(Int32.(groups.surface)), TDev(zeros(length(groups.surface))),
    )
    function lithostatic_pressure!()
        copyto!(litho.T, thermal.T)
        solver!(litho, mesh_T, bc_P; workgroup, Tref, g, verbose = false)
        P_nodal = Array(litho.P)
        P_dofs = zeros(mesh_stokes.nnodesP)
        for iel in axes(el2nP_cpu, 2), a in axes(el2nP_cpu, 1)
            P_dofs[DoFsP_cpu[a, iel]] = P_nodal[el2nP_cpu[a, iel]]
        end
        return P_dofs
    end
    P_init = lithostatic_pressure!()
    copyto!(dr.P, P_init)
    copyto!(dr.Pf, λ_fluid .* P_init)

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) n_vz =
        length(groups.bottom) n_T = length(bc_T.DoFs) T_base Q_dike

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)

    # -----------------------------------------------------------------------
    # Time loop
    # -----------------------------------------------------------------------
    time_history        = zeros(Float64, nsteps)
    mean_tauII_history  = zeros(Float64, nsteps)
    max_tauII_history   = zeros(Float64, nsteps)
    mean_P_dike         = zeros(Float64, nsteps)
    max_uplift_rate     = zeros(Float64, nsteps)
    solve_stats_history = NamedTuple[]

    out_dir = joinpath(@__DIR__, "output_dike_3D")
    write_output && mkpath(out_dir)
    post = nothing
    adjoint_result = nothing

    @info "Starting coupled thermal--Stokes solver" nsteps Δt_hr = Δt / hr Q_dike dip

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(dr.P0, dr.P)
        copyto!(thermal.T0, thermal.T)

        # Pressure update scale: γP * RP / M_P reproduces the pointwise pressure
        # correction while adapting the step to the local viscosity. It depends
        # on the element geometry, which moves with the mesh.
        assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh_stokes, γfact, Δτ; workgroup, phases_v,
        )

        solve_stats = solve_coupled_dyrel!(
            thermal, dr, mesh_T, mesh_stokes,
            bc_T, (bc_vx, bc_vy, bc_vz), Δτ, γP;
            Tref, phases_v, phases_P,
            τ_old, plastic, workgroup, Qq, Q2D = Q_total,
            ncheck = 100, ϵ_tol = 1.0e-6,
            iterMax = 50_000, total_iterMax = 50_000, rel_drop0 = 0.75,
            verbose, verbose_inner = false,
        )
        solve_stats.converged || error("dike injection step $istep did not converge")
        push!(solve_stats_history, solve_stats)

        # Adjoint of the converged last step, before its stress history, pore
        # pressure, and mesh are advanced.
        adjoint_cells = (;)
        if istep == nsteps
            data = (;
                v = map(Array, FEMTools.velocity(dr)), P = Array(dr.P),
                el2n = el2n_v_cpu, DoFsP = DoFsP_cpu,
                geo_v = Array(geo_v), geo_P = Array(mesh_stokes.geometry.geo_P),
                ∂N∂ξ = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_gradients(element_v)),
                Nq = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_values(element_v)),
                NqP = FEMTools.quadrature_table(
                    CPU(), FEMTools.shape_function_values(element_P, element_v.integration_points),
                ),
                T = Array(dr.T), T0 = Array(dr.T0), P0 = Array(dr.P0), Pf = Array(dr.Pf),
                Q = Array(dr.Q), τ_old = map(Array, τ_old),
                phases_v = phases_v_cpu, phases_P = phases_P_cpu,
                dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.ηb, dr.g, dr.Tref, Δt = Δτ, plastic,
                γP = Array(γP), M_P = Array(dr.M_P),
            )
            nnodes = mesh_stokes.nnodes
            load_z, _ = box_mean_load(
                coords_cpu, el2n_v_cpu, data.geo_v, data.∂N∂ξ, data.Nq, obs_box ./ L_c, Val(NV),
            )
            objective = (
                KernelAbstractions.zeros(backend, Float64, nnodes),
                KernelAbstractions.zeros(backend, Float64, nnodes), TDev(load_z),
            )
            λv = ntuple(_ -> KernelAbstractions.zeros(backend, Float64, nnodes), 3)
            λP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
            t_adjoint = @elapsed begin
                adjoint_stats = solve_stokes_adjoint_dyrel!(
                    dr, mesh_stokes, geo_v, mesh_stokes.geometry.geo_P, element_v, element_P,
                    phases_v, phases_P, τ_old, plastic, dr.G, Δτ, γP,
                    objective, λv, λP, backend, workgroup;
                    v_nodes = (bc_vx.DoFs, bc_vy.DoFs, bc_vz.DoFs),
                    adjoint_tol = 1.0e-8, iterMax = 200_000, total_iterMax = 200_000,
                    max_ph_iterations = 1000, verbose,
                )
                adjoint_stats.converged || error("adjoint solve did not converge: err = $(adjoint_stats.err)")
                λv_cpu = map(Array, λv)
                dJ_dlnK, dJ_dlnQ, λP_plain = log_sensitivities(data, λv_cpu, Array(λP), Val(NV), Val(NP))
            end
            result = (;
                J = dot(load_z, data.v[3]), λv = vcat(λv_cpu...), λP = λP_plain,
                dJ_dlnK, dJ_dlnQ, adjoint_residual = adjoint_stats.err,
            )
            J = result.J
            dlnJ_dlnQ = sum(result.dJ_dlnQ) / J
            dlnJ_dlnK = ntuple(p -> sum(result.dJ_dlnK[cell_phase .== p]) / J, 2)

            dlnJ_dlnQ_fd = nothing
            if fd_check
                v_saved = map(copy, FEMTools.velocity(dr))
                P_saved = copy(dr.P)
                ε = 1.0e-3
                J_perturbed = map((1 + ε, 1 - ε)) do factor
                    stats = solve_stokes_dyrel!(
                        dr, mesh_stokes, (bc_vx, bc_vy, bc_vz), Δτ, γP;
                        phases_v, phases_P, τ_old, plastic, workgroup, Qq,
                        Q2D = factor * Q_total, ncheck = 100, ϵ_tol = 1.0e-8,
                        iterMax = 200_000, total_iterMax = 200_000, rel_drop0 = 0.75,
                        verbose = false,
                    )
                    stats.converged || error("finite-difference solve with Q × $factor did not converge")
                    J_factor = dot(load_z, Array(FEMTools.velocity(dr)[3]))
                    foreach(copyto!, FEMTools.velocity(dr), v_saved)
                    copyto!(dr.P, P_saved)
                    J_factor
                end
                copyto!(dr.Q, data.Q)
                dlnJ_dlnQ_fd = (J_perturbed[1] - J_perturbed[2]) / (2ε * J)
            end

            element_volume = [
                sum(last, element_geometry(data.geo_v, iel, data.∂N∂ξ)) for iel in 1:mesh_stokes.nels
            ]
            # Per element, summing to the totals above; per unit volume, the maps
            # do not follow the element size. [km⁻³]
            dlnJ_dlnK_cell = result.dJ_dlnK ./ J
            dlnJ_dlnQ_cell = result.dJ_dlnQ ./ J
            element_volume_km3 = element_volume .* (L_c^3 / 1.0e9)
            dlnJ_dlnK_density = dlnJ_dlnK_cell ./ element_volume_km3
            # Multipliers are in the solver's characteristic units.
            λv_cell = ntuple(3) do d
                [mean(result.λv[(d - 1) * nnodes .+ el2n_v_cpu[1:4, iel]]) for iel in 1:mesh_stokes.nels]
            end
            λvz_cell = λv_cell[3]
            λP_cell = [mean(result.λP[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels]
            box = obs_box ./ L_c
            in_obs_box = map(axes(el2n_v_cpu, 2)) do iel
                c = mean(coords_cpu[el2n_v_cpu[a, iel]] for a in 1:4)
                Int(all(d -> box[2d - 1] < c[d] < box[2d], 1:3))
            end
            adjoint_result = (;
                J_SI = J * L_c / t_c, dlnJ_dlnQ, dlnJ_dlnK, dlnJ_dlnQ_fd,
                dlnJ_dlnK_density, λv = result.λv, λP = result.λP,
                adjoint_residual = result.adjoint_residual, coords = copy(coords_cpu),
                vz = data.v[3] .* (L_c / t_c), λvz_cell,
            )
            adjoint_cells = (;
                dlnJ_dlnK = dlnJ_dlnK_cell, dlnJ_dlnK_per_km3 = dlnJ_dlnK_density,
                dlnJ_dlnQ = dlnJ_dlnQ_cell, dlnJ_dlnQ_per_km3 = dlnJ_dlnQ_cell ./ element_volume_km3,
                adjoint_vx = λv_cell[1], adjoint_vy = λv_cell[2], adjoint_vz = λvz_cell,
                adjoint_P = λP_cell, observation_box = in_obs_box,
            )
            @info "Adjoint log-sensitivities of the mean uplift rate above the dike" t_adjoint J_mm_hr =
                J * L_c / t_c * 1.0e3 * hr dlnJ_dlnQ dlnJ_dlnQ_fd dlnJ_dlnK_crust =
                dlnJ_dlnK[1] dlnJ_dlnK_dike = dlnJ_dlnK[2] adjoint_residual = result.adjoint_residual
        end

        update_stokes_current_stress!(
            dr, mesh_stokes, τ, Δτ;
            phases_v, τ_old, plastic, workgroup,
        )

        P_cpu = Array(dr.P)
        v_cpu = map(Array, FEMTools.velocity(dr))
        post = compute_strain_rate_stress_postprocess(
            v_cpu, el2n_v_cpu, Array(geo_v), map(Array, τ), element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII) * σ_c
        max_tauII_history[istep]  = maximum(post.tauII) * σ_c
        mean_P_dike[istep]        = mean(@view P_cpu[dike_P_dofs]) * σ_c
        max_uplift_rate[istep]    = maximum(@view v_cpu[3][groups.surface]) * L_c / t_c

        foreach(copyto!, τ_old, τ)

        @info "Physical time step" istep nsteps t_hr = t / hr mean_tauII_MPa =
            mean_tauII_history[istep] / 1.0e6 mean_P_dike_MPa =
            mean_P_dike[istep] / 1.0e6 max_uplift_mm_hr =
            max_uplift_rate[istep] * 1.0e3 * hr iter = solve_stats.iter err =
            solve_stats.err err_T = solve_stats.err_T

        P_litho = lithostatic_pressure!()
        copyto!(dr.Pf, λ_fluid .* P_litho)
        overpressure = P_cpu .- P_litho

        if write_output
            T_cpu = Array(thermal.T)
            el_T = [mean(T_cpu[el2n_v_cpu[1:4, iel]]) for iel in 1:mesh_stokes.nels]
            cell_mean(f) = [mean(f[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels]
            vtk_path = joinpath(out_dir, @sprintf("dike_injection_3D_%04d.vtk", istep))
            write_stokes_vtk(
                vtk_path, mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
                P_cpu .* σ_c, map(v -> v .* (L_c / t_c), v_cpu),
                # Strain-rate fields scale with 1/t_c, stress fields with σ_c.
                merge(map(f -> f .* σ_c, post),
                    map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εxz, :εyz, :εII)]));
                title = "dike injection thermal-Stokes 3D",
                cell_data = (;
                    phase = cell_phase, T = el_T,
                    P_litho = cell_mean(P_litho) .* σ_c,
                    overpressure = cell_mean(overpressure) .* σ_c,
                    adjoint_cells...,
                ),
            )
            @info "Wrote VTK file" vtk_path
        end

        # Lagrangian mesh advection: every node, edge and bubble nodes included,
        # moves with the velocity, so the quadratic map follows the material.
        corners = unique(vec(el2n_v_cpu[1:4, :]))
        hmin = minimum(
            norm(coords_cpu[el2n_v_cpu[a, iel]] - coords_cpu[el2n_v_cpu[b, iel]])
            for (a, b) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
                iel in axes(el2n_v_cpu, 2)
        )
        max_disp = Δτ * maximum(n -> norm(SVector(v_cpu[1][n], v_cpu[2][n], v_cpu[3][n])), corners)
        # Moving a node by a sizeable fraction of an edge can invert elements.
        max_disp < hmin / 2 || error(
            "step $istep moves a node by $(max_disp * L_c) m, more than half the " *
            "shortest edge $(hmin * L_c) m; reduce Δt"
        )
        for n in eachindex(coords_cpu)
            coords_cpu[n] += Δτ * SVector(v_cpu[1][n], v_cpu[2][n], v_cpu[3][n])
        end
        for mesh in (mesh_T, mesh_stokes)
            copyto!(mesh.coords, coords_cpu)
        end
        update_geometry!(mesh_stokes; workgroup)
        update_geometry!(mesh_T; workgroup)
    end

    # -----------------------------------------------------------------------
    # Adjoint plot — sections across the dike (y = 0) and through the box
    # -----------------------------------------------------------------------
    if show_plot || write_output
        km = 1.0e-3
        coords_km = [c .* (L_c * km) for c in adjoint_result.coords]
        xmin, xmax, ymin, ymax, zmin, _ = obs_box .* km
        box_xz = Point2f[(xmin, zmin), (xmax, zmin), (xmax, 0), (xmin, 0), (xmin, zmin)]
        box_xy = Point2f[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
        dike_xz = [
            Point2f(
                (dike_center[1] + a * dike_width / 2 * dike_across[1] + b * dike_height / 2 * dike_along[1]) * km,
                (dike_center[3] + a * dike_width / 2 * dike_across[2] + b * dike_height / 2 * dike_along[2]) * km,
            ) for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
        ]
        z_section = dike_top / 2
        section_xz = tet_section(coords_km, el2n_v_cpu, 2, 0.0)
        section_xy = tet_section(coords_km, el2n_v_cpu, 3, z_section * km)
        vz_cell = [
            mean(adjoint_result.vz[el2n_v_cpu[1:4, iel]]) * 1.0e3 * hr for iel in axes(el2n_v_cpu, 2)
        ]

        fig = Figure(size = (1500, 1150))
        fd_text = isnothing(adjoint_result.dlnJ_dlnQ_fd) ? "" :
            @sprintf("  (finite difference %.4f)", adjoint_result.dlnJ_dlnQ_fd)
        Label(
            fig[0, 1:4],
            @sprintf(
                "J = mean vz in box = %.3g mm/hr    ∂lnJ/∂lnQ = %.4f%s    ∂lnJ/∂lnK: crust %.4f, dike %.4f",
                adjoint_result.J_SI * 1.0e3 * hr, adjoint_result.dlnJ_dlnQ, fd_text, adjoint_result.dlnJ_dlnK...,
            );
            fontsize = 18,
        )
        panels = (
            (section_xz, vz_cell, "z [km]", "Vertical velocity, y = 0", "vz [mm/hr]", (box_xz, dike_xz)),
            (section_xz, adjoint_result.λvz_cell, "z [km]", "Adjoint vertical velocity, y = 0", "λvz", (box_xz, dike_xz)),
            (section_xz, adjoint_result.dlnJ_dlnK_density, "z [km]", "Bulk-modulus log-sensitivity, y = 0",
                "∂lnJ/∂lnK [km⁻³]", (box_xz, dike_xz)),
            (section_xy, adjoint_result.dlnJ_dlnK_density, "y [km]",
                @sprintf("Bulk-modulus log-sensitivity, z = %.1f km", z_section * km),
                "∂lnJ/∂lnK [km⁻³]", (box_xy,)),
        )
        for (k, ((polygons, cut), field, ylabel, title, label, outlines)) in enumerate(panels)
            row, col = fldmod1(k, 2)
            values = field[cut]
            # Symmetric and clipped: the sensitivity peaks sharply at the box edges.
            limit = quantile(abs.(values), 0.99)
            limit > 0 || (limit = 1.0)
            ax = Axis(fig[row, 2col - 1]; aspect = DataAspect(), title, xlabel = "x [km]", ylabel)
            poly!(ax, polygons; color = values, colormap = :vik, colorrange = (-limit, limit), strokewidth = 0)
            for outline in outlines
                lines!(ax, outline; color = :black, linewidth = 1.5, linestyle = :dash)
            end
            Colorbar(fig[row, 2col]; colormap = :vik, limits = (-limit, limit), label, width = 15)
        end
        write_output && save(joinpath(out_dir, "dike_injection_3D_adjoint.png"), fig)
        show_plot && display(fig)
    end

    return (;
        time = time_history, mean_tauII = mean_tauII_history,
        max_tauII = max_tauII_history, mean_P_dike, max_uplift_rate, post,
        solve_stats = solve_stats_history, adjoint = adjoint_result,
    )
end

main()
