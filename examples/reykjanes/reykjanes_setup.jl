# Function-only set-up of the 2-D Reykjanes-like cross-rift section, milestone GAP-10 of REYKJANES_PLAN.md.
# It runs nothing and is shared by the driver `reykjanes_thermal_stokes.jl`, so that the event engine and the
# tests can build the same model. Adapted from examples/stokes/volcano/volcano_thermal_stokes.jl.

using KernelAbstractions
using FEMTools

include(joinpath(@__DIR__, "..", "triangulate_meshing.jl"))

const yr  = 365.25 * 24 * 3600.0
const kyr = 1000 * yr

"""
    build_reykjanes_model(; Δt=10kyr, V_ext=2e-2/yr, domain=(40e3, 20e3), max_area=1e6, refinement=8,
                          η_vp=1e20, γfact=100.0,
                          T_magma=1373.0, dTdz=0.05, ϵ_T=1e-6, backend=CPU(), workgroup=128) -> NamedTuple

Build the coupled thermal--Stokes model of a Reykjanes-like cross-rift section without running it: a
crustal section of `domain = (width, depth)` with a flat ground surface and a thin elliptical magma sill on an unstructured
T7/P1-disc mesh, the Stokes and thermal solver states with their initial fields, the boundary conditions and
the pressure scaling.

Inputs are SI. The returned fields are in the characteristic units `L_c`, `t_c` and `σ_c` (length, time and
stress), which are returned with them, and `Δτ = Δt / t_c` is the time step in those units. `coords_cpu` is a
host array that the mesh advection of the driver updates in place. Geometry, geotherm and material values are
illustrative placeholders, not calibrated Reykjanes values; `main` in `reykjanes_thermal_stokes.jl` describes
the rheology and the boundary conditions.
"""
function build_reykjanes_model(;
        Δt = 10kyr,
        V_ext = 2.0e-2 / yr,
        # V_ext = 2.0e-3 / yr,
        domain = (40.0e3, 20.0e3),
        max_area = 1.0e6,
        refinement = 8,
        η_vp = 1.0e20,
        γfact = 100.0,
        T_magma = 1100 + 273.0,
        dTdz = 50.0e-3,
        ϵ_T = 1.0e-6,
        backend = CPU(),
        workgroup = 128,
    )
    # -----------------------------------------------------------------------
    # Geometry [m]
    # -----------------------------------------------------------------------
    Lx, depth = domain
    sill_center = (0.0, -4.5e3)
    sill_radii  = (2.5e3, 0.5e3)
    ε̇_bg = V_ext / Lx

    # -----------------------------------------------------------------------
    # Characteristic units
    #
    # The dynamic-relaxation convergence tests compare absolute residuals with
    # fixed thresholds, so a crustal-scale problem in SI units diverges the
    # guard before it converges. Everything below is therefore scaled by these
    # units and scaled back for output. Temperature stays in Kelvin.
    # -----------------------------------------------------------------------
    L_c  = depth                  # length [m]
    t_c  = 1 / abs(ε̇_bg)          # time [s]
    σ_c  = 1.0e8                  # stress [Pa]
    η_c  = σ_c * t_c              # viscosity [Pa s]
    ρ_c  = σ_c * t_c^2 / L_c^2    # density [kg m⁻³]
    g_c  = L_c / t_c^2            # gravity [m s⁻²]
    Cp_c = L_c^2 / t_c^2          # heat capacity [J kg⁻¹ K⁻¹]
    k_c  = ρ_c * Cp_c * L_c^2 / t_c   # thermal conductivity [W m⁻¹ K⁻¹]

    # -----------------------------------------------------------------------
    # Material properties — phase 1: crust, phase 2: magma sill.
    # Listed in SI, divided by the characteristic units above.
    # -----------------------------------------------------------------------
    # A crustal viscosity of 1e20 Pa s (Maxwell time η/G ≈ 100 yr) keeps the
    # extension-driven stress near 2 η ε̇ ≈ 3 MPa, below yield; 1e21 Pa s gives a
    # mean stress near 30 MPa and yields the upper crust. The sill viscosity is
    # that of a crystal-rich mush. The volcano driver found that contrasts beyond
    # 10³ need several times more iterations to converge; this one is 10.
    η    = (1.0e20, 1.0e16) ./ η_c    # shear viscosity [Pa s]
    G    = (3.0e10, 1.0e9)  ./ σ_c    # shear modulus [Pa]
    K    = (5.0e10, 1.0e10) ./ σ_c    # bulk modulus [Pa]
    ηb   = K                          # pressure storage modulus; residual uses ηb * Δt
    ρ0   = (2900.0, 2800.0) ./ ρ_c    # reference density [kg m⁻³]
    k    = (2.5, 2.0)       ./ k_c    # thermal conductivity [W m⁻¹ K⁻¹]
    Cp   = (1000.0, 1200.0) ./ Cp_c   # heat capacity [J kg⁻¹ K⁻¹]
    α    = (3.0e-5, 5.0e-5)           # thermal expansivity [K⁻¹]
    g    = (0.0, -9.81 / g_c)         # gravity [m s⁻²]
    Tref = 273.0                      # equation-of-state reference temperature [K]

    ϕ = deg2rad(30)                   # friction angle
    Ψ = deg2rad(0)                    # dilation angle (non-associated)
    C_crust = 1.0e7 / σ_c             # cohesion [Pa]; yield stress = C cosϕ + P sinϕ
    # The sill stays visco-elastic. Its cohesion is set far above any attainable
    # stress rather than to `Inf`, because per-phase properties are interpolated
    # with T7 shape functions, which are negative at some quadrature points and
    # would turn an infinite cohesion into `NaN`.
    C_sill = 1.0e12 / σ_c
    # The Drucker-Prager return scales with 2 ηve / (ηve + η_reg), so the
    # regularisation viscosity must stay at or above the viscoelastic viscosity
    # ηve, or the plastic correction overshoots and the relaxation diverges.
    plastic = DruckerPrager(
        (ϕ, ϕ), (Ψ, Ψ), (C_crust, C_sill), (η_vp / η_c, η_vp / η_c), K,
    )

    T_surface = 273.0
    T_base    = T_surface + dTdz * depth

    ε̇  = ε̇_bg * t_c
    Δτ = Δt / t_c

    # -----------------------------------------------------------------------
    # Mesh — one node set shared by the velocity, thermal, and pressure fields
    # -----------------------------------------------------------------------
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_cpu, el2n_cpu, groups = build_triangulate_t7_sill_mesh(;
        Lx = Lx / L_c, depth = depth / L_c,
        sill_center = sill_center ./ L_c,
        sill_radii = sill_radii ./ L_c,
        max_area = max_area / L_c^2, refinement,
    )
    mesh_v = Mesh(backend, coords_cpu, el2n_cpu, element_v; workgroup)
    mesh_stokes = MixedMesh(mesh_v, element_P; workgroup)
    geo_v = mesh_stokes.geometry.geo_v

    NQ_v = length(element_v.integration_points.ω)
    NV   = length(element_v)
    NP   = length(element_P)

    @info "Triangulate mixed mesh (T7/P1-disc)" nnodes_v = mesh_stokes.nnodes nnodes_P =
        mesh_stokes.nnodesP nels = mesh_stokes.nels max_area

    # -----------------------------------------------------------------------
    # Phases — element phase from the mesh, node phase from the elements
    # -----------------------------------------------------------------------
    cell_phase = groups.phase
    TDev = FEMTools.TA(backend)
    phases_v = TDev(repeat(reshape(cell_phase, 1, :), NV, 1))
    phases_P = TDev(repeat(reshape(cell_phase, 1, :), NP, 1))

    node_phase = ones(Int32, length(coords_cpu))
    for iel in axes(el2n_cpu, 2), a in axes(el2n_cpu, 1)
        cell_phase[iel] == 2 && (node_phase[el2n_cpu[a, iel]] = 2)
    end

    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu  = Array(mesh_stokes.el2nP)
    DoFsP_cpu  = Array(mesh_stokes.DoFsP)
    NqP_cpu    = shape_function_values(element_P, element_v.integration_points)
    sill_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, findall(==(2), cell_phase)])))
    crust_cells = findall(==(1), cell_phase)

    @info "Phases" n_sill_cells = count(==(2), cell_phase) n_sill_nodes =
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
    τ = (dr.τ.xx, dr.τ.yy, dr.τ.xy)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)

    thermal = ThermalDiffusionDR(
        backend, mesh_v.nnodes, ThermalMaterial(; k, Cp, ρ0, α, K); ϵ = ϵ_T,
    )
    copyto!(thermal.phases, TDev(node_phase))

    # -----------------------------------------------------------------------
    # Boundary conditions
    #   vx = ε̇ x on the vertical walls, vy = 0 on the base, free slip in the
    #   unconstrained component; the ground surface carries no traction.
    #   T is fixed on the surface and the base, insulating on the sides.
    # -----------------------------------------------------------------------
    vx_nodes = vcat(groups.left, groups.right)
    bc_vx = DirichletBoundaryCondition(
        nothing, TDev(vx_nodes), TDev([ε̇ * coords_cpu[n][1] for n in vx_nodes]),
    )
    bc_vy = DirichletBoundaryCondition(
        nothing, TDev(groups.bottom), TDev(zeros(length(groups.bottom))),
    )
    bc_T = DirichletBoundaryCondition(
        nothing,
        TDev(vcat(groups.surface, groups.bottom)),
        TDev(vcat(fill(T_surface, length(groups.surface)), fill(T_base, length(groups.bottom)))),
    )

    # Linear geotherm below the ground surface.
    geotherm(c) = T_surface - dTdz * L_c * c[2]
    copyto!(thermal.T, [
        node_phase[i] == 2 ? T_magma : geotherm(coords_cpu[i]) for i in eachindex(coords_cpu)
    ])
    apply_bc!(thermal.T, bc_T; workgroup)
    copyto!(thermal.T0, thermal.T)

    # Seed the interior with the pure-shear field that satisfies the boundary
    # conditions. The relaxation diverges from an initial guess with a non-zero
    # divergence, so this seed is required, not merely a warm start.
    copyto!(dr.v.x, [ε̇ * c[1] for c in coords_cpu])
    copyto!(dr.v.y, [-ε̇ * (c[2] + depth / L_c) for c in coords_cpu])
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)

    # Warm-start the pressure with the crustal load of the overlying column so
    # the first Powell-Hestenes step does not have to build it from zero.
    lithostatic_pressure(y, y_top = 0.0) = ρ0[1] * abs(g[2]) * (y_top - y)
    P_init = zeros(mesh_stokes.nnodesP)
    for iel in axes(el2nP_cpu, 2), a in axes(el2nP_cpu, 1)
        P_init[DoFsP_cpu[a, iel]] = lithostatic_pressure(coords_cpu[el2nP_cpu[a, iel]][2])
    end
    copyto!(dr.P, P_init)

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(groups.bottom) n_T =
        length(bc_T.DoFs) max_vx_SI = maximum(abs, bc_vx.vals) * L_c / t_c T_base

    # Pressure update scale: γP * RP / M_P reproduces the pointwise pressure
    # correction while adapting the step to the local viscosity. Both γP and
    # `dr.M_P` depend on the time step and the geometry, so a caller that changes
    # either rebuilds them with the time step it is about to use.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    update_pressure_scaling!(Δτ_step = Δτ) = assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, γfact, Δτ_step; workgroup, phases_v,
    )
    update_pressure_scaling!()

    return (;
        Δt, V_ext, L_c, t_c, σ_c, Δτ, Tref, G, C_crust, C_sill, ϕ, plastic,
        coords_cpu, groups, mesh_v, mesh_stokes, geo_v, element_v, element_P, NQ_v, cell_phase,
        phases_v, phases_P, el2n_v_cpu, el2nP_cpu, DoFsP_cpu, NqP_cpu, sill_P_dofs, crust_cells,
        dr, τ, τ_old, thermal, bc_vx, bc_vy, bc_T, γP, update_pressure_scaling!, lithostatic_pressure,
    )
end
