# Coupled thermal-Stokes model of magma injected into a dike beneath a flat
# surface, in plane strain, with the adjoint log-sensitivities of the uplift above
# the dike to injection, stiffness, viscosity and failure parameters.

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using ForwardDiff
using KernelAbstractions
using Gmsh
using FEMTools
using GLMakie: Figure, Axis, Colorbar, Label, poly!, lines!, scatterlines!, axislegend, rowsize!, Relative,
    Point2f, DataAspect, save

const backend   = CPU()
const workgroup = 128
const hr        = 3600.0

include(joinpath(@__DIR__, "..", "..", "gmsh_meshing.jl"))

include(joinpath(@__DIR__, "dike_sensitivities2D.jl"))

"""
    main(; nsteps=24, Δt=1hr, ε̇_bg=1e-16, Q_dike=0.1, obs_side=8km, fd_check=false, ...) -> NamedTuple

Run the coupled thermal--Stokes model of magma injection into a dike on an
unstructured T7/P1-disc (Crouzeix-Raviart) mesh, and compute the adjoint
relative sensitivities of mean vertical velocity above the dike at every step.
Negative velocity (subsidence) is allowed: the existing `dlnJ_*` fields
represent derivatives of `ln|J|`, normalized by signed J. At J = 0 these
relative sensitivities are undefined; near zero they can be poorly conditioned.

The objective is the mean vertical velocity `J = ∫_box vy dΩ / |box|` over a box
`obs_side` wide, centred above the top of the dike, extending from the depth of
the dike top to the free surface. Its discrete adjoint at each converged step
gives the log-sensitivities `∂ln J/∂ln K` to the bulk modulus `K` (also the
continuity storage modulus and the cap's bulk modulus), as a per-element map and
summed per phase, and `∂ln J/∂ln Q_dike`. It also differentiates shear modulus
`G`, shear viscosity `η`, tensile-strength magnitude `|pT|`, cohesion `C`, and
friction angle `ϕ` logarithmically, and pore-pressure fraction `λ_fluid`
additively (`∂ln J/∂λ_fluid`, at fixed lithostatic pressure). Tensile strength
and cohesion vary independently, with dilation and plastic regularization fixed.
The returned `adjoint.sensitivities`, `sensitivity_density`, and
`sensitivity_totals` contain cell contributions, densities per km², and
(crust, dike) totals. The adjoint is that of the Stokes step
alone: the stress, accepted pressure, plastic-strain and temperature fields of
the step are held fixed, so the thermal coupling, including shear heating, is
not differentiated. With `fd_check = true`, `∂ln J/∂ln Q_dike` is compared with
a central finite difference of two extra Stokes solves of the last step at the
same frozen fields, with tighter forward and adjoint tolerances on that step
to resolve the small objective difference. The vertical velocity, the adjoint vertical velocity and
pressure, and all sensitivity densities of the last step are plotted, and
every VTK file carries, per element, all sensitivities (summing to the
totals) and their densities per km², the corner-averaged adjoint velocities, the
element-averaged adjoint pressure, and a mask of the observation box.

A 40 km × 20 km crustal section with a flat, traction-free surface holds a
dike `dike_width` wide spanning depths `dike_bottom` to `dike_top`, dipping at
`dip` degrees from the horizontal: 90 is vertical, a smaller angle dips toward
`+x` and a larger one toward `−x`. The
dike is a conforming material interface, and the mesh is graded from
`mesh_size / refinement` on it to `mesh_size` at `refinement_distance` away.
The crust is visco-elasto-plastic: Drucker-Prager (cohesion 10 MPa, friction
angle 30°) closed on the tensile side by the cap of `DruckerPragerCap`, so it
fails in tension once the pressure drops to the tensile strength `pT`. The
yield model sees the effective pressure `P − Pf`, with a hydrostatic pore
pressure `Pf = λ_fluid P_litho` refreshed from the lithostatic pressure every
step. The dike
is visco-elastic magma held at `T_magma`.

Magma enters through the continuity source `Q`: a uniform weight on the dike's
pressure DoFs, rescaled by the solver so that the injected plane-strain area
rate is `Q_dike` [m² s⁻¹]. The defaults follow the 2023–2024 Reykjanes dike
injections (Sigmundsson et al., 2024; Parks et al., 2025): peak flows of
800–7400 m³ s⁻¹ into dikes 10–15 km long are 0.05–0.5 m² s⁻¹ per unit strike
length, delivered over hours, so `Δt` is an hour. On that time scale the crust,
whose Maxwell time `η/G` is about a kyr, responds elastically, and the
background strain rate `ε̇_bg` is a negligible tectonic load. Background pure shear `vx = ε̇_bg x` acts on the
vertical walls with `vy = 0` on the base; temperature is fixed at the surface
and the base and insulating on the sides. With `shear_heating = true` the
viscous and plastic shear dissipation heats the crust, solved together with the
flow.

All arguments and returned fields are SI; the solve itself runs in the
characteristic units defined below.

The mesh is Lagrangian: after every step its nodes move with the velocity, the
surface deforms freely, and the side-wall velocities follow the moved walls.
The stress history is rotated with the material (Jaumann). At the end of every
step the lithostatic pressure of the current density field and geometry is
recomputed, and the overpressure `P - P_litho` is written to the VTK output.

The cap corrects the pressure at the integration points, so the accepted
pressure of each step, the continuity memory of the next one, is carried there
as `P_old` rather than in `dr.P0`.

Returns the time series of mean and maximum second stress invariant, mean dike
pressure, maximum surface uplift rate, and maximum volumetric plastic strain,
the last post-processed field set, per-step solver statistics, and the adjoint
results.
"""
function main(;
        nsteps = 24,
        Δt = 1hr,
        ε̇_bg = 1.0e-16,
        Q_dike = 0.1,
        dike_width = 200.0,
        dip = 90.0,
        dike_top = -2.0e3,
        dike_bottom = -8.0e3,
        max_area = 1.0e6,
        refinement = 64,
        refinement_distance = 3.0e3,
        # The Drucker-Prager return scales as 2 ηve / (ηve + η_vp): η_vp must be
        # of the order of the crust's viscoelastic viscosity ηve ≈ G Δt for the
        # plastic correction to act, and not below it, or it overshoots.
        η_vp = 1e18,
        # η_vp = 3.0e10 * Δt,
        pT = -5.0e6,
        # Pore-fluid pressure as a fraction of the lithostatic pressure; the
        # hydrostatic ratio ρ_water / ρ_crust ≈ 1000 / 2700.
        λ_fluid = 0.37,
        γfact = 100.0,
        T_magma = 1100 + 273.0,
        dTdz = 25.0e-3,
        ϵ_T = 1.0e-6,
        shear_heating = true,
        obs_side = 8.0e3,
        fd_check = false,
        show_plot = true, write_output = true, verbose = true,
    )
    # -----------------------------------------------------------------------
    # Geometry [m]
    # -----------------------------------------------------------------------
    Lx, depth = 40.0e3, 20.0e3
    0 < dip < 180 || throw(ArgumentError("dip must lie strictly between 0 and 180 degrees, got $dip"))
    dike_top < 0 && dike_bottom > -depth && dike_top > dike_bottom ||
        throw(ArgumentError("the dike must lie strictly below the surface and above the base"))
    # Length along the dip, so the dike keeps its depth range at any dip.
    dike_height = (dike_top - dike_bottom) / sind(dip)
    dike_center = (0.0, (dike_top + dike_bottom) / 2)
    # Counter-clockwise rotation of the vertical dike: its top moves toward −x.
    dike_rotation = deg2rad(90 - dip)
    # Unit vectors across and along the dike.
    dike_across = (cos(dike_rotation), sin(dike_rotation))
    dike_along  = (-sin(dike_rotation), cos(dike_rotation))
    # Observation box above the dike top; open upward so it follows the
    # deforming surface. [m]
    x_top = dike_center[1] + dike_height / 2 * dike_along[1]
    obs_box = (x_top - obs_side / 2, x_top + obs_side / 2, dike_top, Inf)

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
    g    = (0.0, -9.81 / g_c)         # gravity [m s⁻²]
    Tref = 273.0                      # equation-of-state reference temperature [K]

    ϕ = deg2rad(30)                   # friction angle
    Ψ = deg2rad(0)                    # dilation angle (non-associated)
    C_crust = 1.0e7 / σ_c             # cohesion [Pa]; yield stress = C cosϕ + P sinϕ
    # The dike stays visco-elastic. Its cohesion is set far above any
    # attainable stress rather than to `Inf`, because per-phase properties are
    # interpolated with T7 shape functions, which are negative at some
    # quadrature points and would turn an infinite cohesion into `NaN`.
    C_dike = 1.0e12 / σ_c
    # Tensile strength [Pa], compression-positive so pT ≤ 0. The cap needs
    # C cosϕ + pT sinϕ > 0; the dike's is as unreachable as its cohesion.
    pT_crust = pT / σ_c
    pT_dike  = -C_dike / 2
    plastic = DruckerPragerCap(
        (ϕ, ϕ), (Ψ, Ψ), (C_crust, C_dike), (pT_crust, pT_dike),
        (η_vp / η_c, η_vp / η_c), K,
    )
    # The adjoint operator does not differentiate the plastic-strain history.
    all(iszero, plastic.H_C) ||
        throw(ArgumentError("the adjoint assumes no cohesion softening (H_C = 0)"))

    T_surface = 273.0
    T_base    = T_surface + dTdz * depth

    ε̇   = ε̇_bg * t_c
    Δτ  = Δt / t_c
    Q2D = Q_dike / (L_c^2 / t_c)

    # -----------------------------------------------------------------------
    # Meshes — one node set shared by the velocity, thermal, and pressure fields
    # -----------------------------------------------------------------------
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    # The builder meshes [0, Lx] × [-depth, 0]; shift it to be centred on x = 0.
    coords_shifted, el2n_cpu, _, _ = build_gmsh_t7_rectangle_inclusion_mesh(;
        Lx = Lx / L_c, Ly = depth / L_c,
        cx = (Lx / 2 + dike_center[1]) / L_c, cy = dike_center[2] / L_c,
        half_width_x = dike_width / 2 / L_c, half_width_y = dike_height / 2 / L_c,
        max_area = max_area / L_c^2, refinement,
        refinement_distance = refinement_distance / L_c, rotation = dike_rotation,
    )
    coords_cpu = [c - SVector(Lx / 2 / L_c, 0.0) for c in coords_shifted]

    x1  = Lx / 2 / L_c
    tol = sqrt(eps(Float64))
    select(f) = Int32[i for i in eachindex(coords_cpu) if f(coords_cpu[i])]
    surface = select(c -> abs(c[2]) <= tol)
    bottom  = select(c -> abs(c[2] + depth / L_c) <= tol)
    left    = select(c -> abs(c[1] + x1) <= tol)
    right   = select(c -> abs(c[1] - x1) <= tol)
    Γnodes  = sort!(union(surface, bottom, left, right))

    DoFs_cpu = Int32.(1:length(coords_cpu))
    mesh_v = Mesh(element_v, nothing, nothing, coords_cpu, DoFs_cpu, el2n_cpu, Γnodes)
    mesh_stokes = MixedMesh(mesh_v, element_P)
    mesh_T = Mesh(backend, coords_cpu, el2n_cpu, element_v; workgroup)
    geo_v = mesh_stokes.geometry.geo_v

    NQ_v = length(element_v.integration_points.ω)
    NV   = length(element_v)
    NP   = length(element_P)

    @info "Gmsh mixed mesh (T7/P1-disc)" nnodes_v = mesh_stokes.nnodes nnodes_P =
        mesh_stokes.nnodesP nels = mesh_stokes.nels max_area refinement

    # -----------------------------------------------------------------------
    # Phases — element phase from the centroid, node phase from the elements.
    # Elements never straddle the conforming dike interface.
    # -----------------------------------------------------------------------
    function in_dike(c)
        dx, dy = c[1] - dike_center[1] / L_c, c[2] - dike_center[2] / L_c
        return abs(dx * dike_across[1] + dy * dike_across[2]) <= dike_width / 2 / L_c &&
            abs(dx * dike_along[1] + dy * dike_along[2]) <= dike_height / 2 / L_c
    end
    cell_phase = Int[
        in_dike(sum(coords_cpu[el2n_cpu[a, iel]] for a in 1:3) / 3) ? 2 : 1
        for iel in axes(el2n_cpu, 2)
    ]
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)

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

    # Uniform source weight on the dike; the solver rescales it to Q2D.
    Qq = zeros(mesh_stokes.nnodesP)
    Qq[dike_P_dofs] .= 1.0

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
        plastic_history_size = (NQ_v, mesh_stokes.nels),
    )
    τ = (dr.τ.xx, dr.τ.yy, dr.τ.xy)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)

    thermal = ThermalDiffusionDR(
        backend, mesh_T.nnodes, ThermalMaterial(; k, Cp, ρ0, α, K); ϵ = ϵ_T,
    )
    copyto!(thermal.phases, node_phase)

    # -----------------------------------------------------------------------
    # Boundary conditions
    #   vx = ε̇ x on the vertical walls, vy = 0 on the base, free slip in the
    #   unconstrained component; the surface carries no traction.
    #   T is fixed on the surface and the base, insulating on the sides.
    # -----------------------------------------------------------------------
    vx_nodes = vcat(left, right)
    bc_vx = DirichletBoundaryCondition(
        nothing, vx_nodes, [ε̇ * coords_cpu[n][1] for n in vx_nodes],
    )
    bc_vy = DirichletBoundaryCondition(nothing, bottom, zeros(length(bottom)))
    bc_T = DirichletBoundaryCondition(
        nothing,
        vcat(surface, bottom),
        vcat(fill(T_surface, length(surface)), fill(T_base, length(bottom))),
    )

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

    # Lithostatic pressure of the current temperature field and geometry,
    # sampled at the pressure DoFs. It warm-starts the pressure, so the first
    # Powell-Hestenes step does not have to build it from zero, and is the
    # reference of the overpressure output.
    litho = LithostaticPressureDR(backend, mesh_T.nnodes, ρ0, α, K)
    copyto!(litho.phases, node_phase)
    bc_P = DirichletBoundaryCondition(nothing, surface, zeros(length(surface)))
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
    P_litho = lithostatic_pressure!()
    copyto!(dr.P, P_litho)
    copyto!(dr.Pf, λ_fluid .* P_litho)

    # Accepted corrected pressure at the velocity integration points, seeded
    # with the warm start. The fourth stress slot receives the cap's corrected
    # pressure after each step.
    NqP = shape_function_values(element_P, element_v.integration_points)
    P_old = [
        sum(NqP[q][a] * P_litho[DoFsP_cpu[a, iel]] for a in 1:NP)
        for q in 1:NQ_v, iel in 1:mesh_stokes.nels
    ]
    P_corrected = similar(P_old)
    τ_and_P = (τ..., P_corrected)

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(bottom) n_T =
        length(bc_T.DoFs) max_vx_SI = maximum(abs, bc_vx.vals) * L_c / t_c T_base Q_dike

    # Pressure update scale: γP * RP / M_P reproduces the pointwise pressure
    # correction while adapting the step to the local viscosity.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)

    # -----------------------------------------------------------------------
    # Time loop
    # -----------------------------------------------------------------------
    time_history        = zeros(Float64, nsteps)
    mean_tauII_history  = zeros(Float64, nsteps)
    max_tauII_history   = zeros(Float64, nsteps)
    mean_P_dike         = zeros(Float64, nsteps)
    max_uplift_rate     = zeros(Float64, nsteps)
    max_plastic_vol     = zeros(Float64, nsteps)
    solve_stats_history = NamedTuple[]

    out_dir = joinpath(@__DIR__, "output_dike_adj")
    write_output && mkpath(out_dir)
    post = nothing
    adjoint_result = nothing

    @info "Starting coupled thermal--Stokes solver" nsteps Δt_hr = Δt / hr ε̇_bg Q_dike

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(thermal.T0, thermal.T)

        # The walls and the element geometry move with the mesh.
        bc_vx.vals .= [ε̇ * coords_cpu[n][1] for n in vx_nodes]
        assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh_stokes, γfact, Δτ; workgroup, phases_v = phases_v_cpu,
        )

        solve_stats = solve_coupled_dyrel!(
            thermal, dr, mesh_T, mesh_stokes,
            bc_T, bc_vx, bc_vy, Δτ, γP;
            Tref, phases_v = phases_v_cpu, phases_P = phases_P_cpu,
            τ_old, P_old, plastic, workgroup, Qq, Q2D, shear_heating,
            ncheck = 100, ϵ_tol = fd_check && istep == nsteps ? 1.0e-10 : 1.0e-6,
            iterMax = 50_000, total_iterMax = 50_000, rel_drop0 = 0.75,
            verbose, verbose_inner = false,
        )
        solve_stats.converged || error("dike injection step $istep did not converge")
        push!(solve_stats_history, solve_stats)

        # Adjoint of the converged step, before its stress history, accepted
        # pressure, pore pressure, and mesh are advanced.
        begin
            data = (;
                v = (Array(dr.v.x), Array(dr.v.y)), P = Array(dr.P),
                el2n = el2n_v_cpu, DoFsP = DoFsP_cpu,
                geo_v = Array(geo_v), geo_P = Array(mesh_stokes.geometry.geo_P),
                ∂N∂ξ = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_gradients(element_v)),
                Nq = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_values(element_v)),
                NqP = FEMTools.quadrature_table(CPU(), NqP),
                T = Array(dr.T), T0 = Array(dr.T0), P_old = copy(P_old), Pf = Array(dr.Pf),
                Q = Array(dr.Q), P_litho = copy(P_litho), τ_old = map(Array, τ_old),
                phases_v = phases_v_cpu, phases_P = phases_P_cpu,
                dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.ηb, dr.g, dr.Tref, Δt = Δτ, plastic,
                γP = Array(γP), M_P = Array(dr.M_P),
            )
            nnodes = mesh_stokes.nnodes
            box = obs_box ./ L_c
            load_y, _ = box_mean_load(
                coords_cpu, el2n_v_cpu, data.geo_v, data.∂N∂ξ, data.Nq, box, Val(NV),
            )
            objective = (zeros(nnodes), load_y)
            λv = (zeros(nnodes), zeros(nnodes))
            λP = zeros(mesh_stokes.nnodesP)
            t_adjoint = @elapsed begin
                adjoint_stats = solve_stokes_adjoint_dyrel!(
                    dr, mesh_stokes, geo_v, mesh_stokes.geometry.geo_P, element_v, element_P,
                    phases_v_cpu, phases_P_cpu, τ_old, plastic, dr.G, Δτ, γP,
                    objective, λv, λP, backend, workgroup;
                    v_nodes = (bc_vx.DoFs, bc_vy.DoFs),
                    # The graded mesh can make the power estimate stop below the stable bound.
                    measure_λmax = false,
                    adjoint_tol = fd_check && istep == nsteps ? 1.0e-11 : 1.0e-8, iterMax = 200_000, total_iterMax = 200_000,
                    max_ph_iterations = 1000, verbose,
                )
                adjoint_stats.converged || error("adjoint solve did not converge: err = $(adjoint_stats.err)")
                derivatives, λP_plain = log_sensitivities(data, λv, λP, Val(NV), Val(NP))
            end
            J = dot(load_y, data.v[2])
            sensitivities = relative_sensitivities(derivatives, J)
            sensitivity_totals = map(sensitivities) do f
                ntuple(p -> sum(f[cell_phase .== p]), 2)
            end
            dlnJ_dlnQ = sum(sensitivities.dlnJ_dlnQ)
            dlnJ_dlnK = sensitivity_totals.dlnJ_dlnK

            dlnJ_dlnQ_fd = nothing
            if fd_check && istep == nsteps
                v_saved = (copy(dr.v.x), copy(dr.v.y))
                P_saved = copy(dr.P)
                ε = 1.0e-3
                J_perturbed = map((1 + ε, 1 - ε)) do factor
                    stats = solve_stokes_dyrel!(
                        dr, mesh_stokes, bc_vx, bc_vy, Δτ, γP;
                        phases_v = phases_v_cpu, phases_P = phases_P_cpu,
                        τ_old, P_old, plastic, workgroup, Qq,
                        Q2D = factor * Q2D, ncheck = 100, ϵ_tol = 1.0e-12,
                        iterMax = 200_000, total_iterMax = 200_000, rel_drop0 = 0.75,
                        verbose = false,
                    )
                    stats.converged || error("finite-difference solve with Q × $factor did not converge")
                    J_factor = dot(load_y, Array(dr.v.y))
                    copyto!(dr.v.x, v_saved[1])
                    copyto!(dr.v.y, v_saved[2])
                    copyto!(dr.P, P_saved)
                    J_factor
                end
                copyto!(dr.Q, data.Q)
                dlnJ_dlnQ_fd = (J_perturbed[1] - J_perturbed[2]) / (2ε * J)
            end

            element_area_km2 = [
                sum(last, element_geometry(data.geo_v, iel, data.∂N∂ξ)) for iel in 1:mesh_stokes.nels
            ] .* (L_c^2 / 1.0e6)
            # Per element, summing to the totals above; per unit area, the maps do
            # not follow the element size. [km⁻²]
            sensitivity_density = map(f -> f ./ element_area_km2, sensitivities)
            density_fields = NamedTuple{Tuple(Symbol(k, :_per_km2) for k in keys(sensitivities))}(values(sensitivity_density))
            # Multipliers are in the solver's characteristic units.
            λv_cell = map(λ -> [mean(λ[el2n_v_cpu[1:3, iel]]) for iel in 1:mesh_stokes.nels], λv)
            λP_cell = [mean(λP_plain[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels]
            in_obs_box = map(axes(el2n_v_cpu, 2)) do iel
                c = mean(coords_cpu[el2n_v_cpu[a, iel]] for a in 1:3)
                Int(box[1] < c[1] < box[2] && box[3] < c[2] < box[4])
            end
            adjoint_result = (;
                J_SI = J * L_c / t_c, dlnJ_dlnQ, dlnJ_dlnK, dlnJ_dlnQ_fd,
                dlnJ_dlnK_density = sensitivity_density.dlnJ_dlnK,
                sensitivities, sensitivity_density, sensitivity_totals,
                dlnJ_dlambda_fluid = sum(sensitivities.dlnJ_dlambda_fluid),
                λv, λP = λP_plain, adjoint_residual = adjoint_stats.err,
                coords = copy(coords_cpu), λvy_cell = λv_cell[2], λP_cell,
                vy_cell = [mean(data.v[2][el2n_v_cpu[1:3, iel]]) for iel in 1:mesh_stokes.nels] .* (L_c / t_c),
            )
            adjoint_cells = (;
                sensitivities..., density_fields...,
                adjoint_vx = λv_cell[1], adjoint_vy = λv_cell[2], adjoint_P = λP_cell,
                observation_box = in_obs_box,
            )
            @info "Adjoint log-sensitivities of the mean uplift rate above the dike" t_adjoint J_mm_hr =
                J * L_c / t_c * 1.0e3 * hr dlnJ_dlnQ dlnJ_dlnQ_fd dlnJ_dlnK_crust =
                dlnJ_dlnK[1] dlnJ_dlnK_dike = dlnJ_dlnK[2] sensitivity_totals adjoint_residual = adjoint_stats.err
        end

        update_stokes_current_stress!(
            dr, mesh_stokes, τ_and_P, Δτ;
            phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )
        # The commit adds Δt times this step's plastic strain rates to the history.
        γ_prev = Array(dr.plastic_history.γ)
        θ_prev = Array(dr.plastic_history.θ)
        commit_stokes_plastic_history!(
            dr, mesh_stokes, Δτ; phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )
        copyto!(P_old, P_corrected)
        θ_cell = vec(mean(Array(dr.plastic_history.θ); dims = 1))
        γ_cell = vec(mean(Array(dr.plastic_history.γ); dims = 1))
        γdot_cell = vec(mean(Array(dr.plastic_history.γ) .- γ_prev; dims = 1)) ./ Δt
        θdot_cell = vec(mean(Array(dr.plastic_history.θ) .- θ_prev; dims = 1)) ./ Δt

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.v.x)
        vy_cpu = Array(dr.v.y)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu, el2n_v_cpu, Array(geo_v), τ, element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII) * σ_c
        max_tauII_history[istep]  = maximum(post.tauII) * σ_c
        mean_P_dike[istep]        = mean(@view P_cpu[dike_P_dofs]) * σ_c
        max_uplift_rate[istep]    = maximum(@view vy_cpu[surface]) * L_c / t_c
        max_plastic_vol[istep]    = maximum(θ_cell)

        # Advance the stress history with the corotational (Jaumann) rotation.
        rotate_stress!(dr, mesh_stokes, Δτ)

        @info "Physical time step" istep nsteps t_hr = t / hr mean_tauII_MPa =
            mean_tauII_history[istep] / 1.0e6 mean_P_dike_MPa =
            mean_P_dike[istep] / 1.0e6 max_uplift_mm_hr =
            max_uplift_rate[istep] * 1.0e3 * hr iter =
            solve_stats.iter err = solve_stats.err err_T = solve_stats.err_T max_plastic_vol =
            max_plastic_vol[istep]

        P_litho = lithostatic_pressure!()
        copyto!(dr.Pf, λ_fluid .* P_litho)
        overpressure = P_cpu .- P_litho

        if write_output
            T_cpu = Array(thermal.T)
            el_T = [mean(T_cpu[el2n_v_cpu[1:3, iel]]) for iel in 1:mesh_stokes.nels]
            vtk_path = joinpath(out_dir, @sprintf("dike_injection_saturated_%04d.vtk", istep))
            write_stokes_vtk(
                vtk_path, mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
                P_cpu .* σ_c, vx_cpu .* (L_c / t_c), vy_cpu .* (L_c / t_c),
                # Strain-rate fields scale with 1/t_c, stress fields with σ_c.
                merge(map(f -> f .* σ_c, post),
                    map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εII)]));
                title = "dike injection thermal-Stokes",
                cell_data = (;
                    phase = cell_phase, T = el_T,
                    corrected_pressure = vec(mean(P_corrected; dims = 1)) .* σ_c,
                    volumetric_plastic_strain = θ_cell,
                    shear_plastic_strain = γ_cell,
                    shear_plastic_strain_rate = γdot_cell,
                    volumetric_plastic_strain_rate = θdot_cell,
                    P_litho = [mean(P_litho[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels] .* σ_c,
                    overpressure = [mean(overpressure[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels] .* σ_c,
                    adjoint_cells...,
                ),
            )
            @info "Wrote VTK file" vtk_path
        end

        # Lagrangian mesh advection: move the corner nodes with the velocity,
        # then rebuild the straight-sided T7 edge and bubble nodes from them.
        corners = unique(vec(el2n_v_cpu[1:3, :]))
        hmin = minimum(
            norm(coords_cpu[el2n_v_cpu[a, iel]] - coords_cpu[el2n_v_cpu[mod1(a + 1, 3), iel]])
            for a in 1:3, iel in axes(el2n_v_cpu, 2)
        )
        max_disp = Δτ * maximum(n -> hypot(vx_cpu[n], vy_cpu[n]), corners)
        # Moving a node by a sizeable fraction of an edge can invert elements.
        max_disp < hmin / 2 || error(
            "step $istep moves a node by $(max_disp * L_c) m, more than half the " *
            "shortest edge $(hmin * L_c) m; reduce Δt"
        )
        for n in corners
            coords_cpu[n] += Δτ * SVector(vx_cpu[n], vy_cpu[n])
        end
        FEMTools.straighten_t7_geometry!(coords_cpu, el2n_v_cpu)
        for mesh in (mesh_v, mesh_stokes, mesh_T)
            copyto!(mesh.coords, coords_cpu)
        end
        update_geometry!(mesh_stokes; workgroup)
        update_geometry!(mesh_T; workgroup)
    end

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------
    P_cpu = Array(dr.P)
    T_cpu = Array(thermal.T)
    el_P  = [mean(P_cpu[DoFsP_cpu[:, i]]) * σ_c for i in 1:mesh_stokes.nels]
    el_T  = [mean(T_cpu[el2n_v_cpu[1:3, i]]) for i in 1:mesh_stokes.nels]
    el_θ  = vec(mean(Array(dr.plastic_history.θ); dims = 1))

    pts   = [Point2f(c .* (L_c / 1.0e3)) for c in coords_cpu]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    dike_corners = [
        dike_center .+ (a * dike_width / 2) .* dike_across .+ (b * dike_height / 2) .* dike_along
        for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
    ]
    dike_x = [c[1] / 1.0e3 for c in dike_corners]
    dike_y = [c[2] / 1.0e3 for c in dike_corners]

    fig = Figure(size = (1900, 900))
    for (col, (field, cmap, label, title)) in enumerate((
            (post.tauII .* (σ_c / 1.0e6), :magma, "τII [MPa]", "Second stress invariant"),
            (el_P ./ 1.0e6, :vik, "P [MPa]", "Pressure"),
            (el_T, :thermal, "T [K]", "Temperature"),
            (el_θ, :viridis, "χ", "Volumetric plastic strain"),
        ))
        clims = extrema(field)
        # A field with no spread, such as plastic strain before any failure.
        clims[1] == clims[2] && (clims = (clims[1], clims[1] + 1))
        ax = Axis(fig[1, 2col - 1]; aspect = DataAspect(), title,
            xlabel = "x [km]", ylabel = "y [km]")
        poly!(ax, polys; color = field, colormap = cmap, colorrange = clims, strokewidth = 0)
        lines!(ax, dike_x, dike_y; color = :white, linewidth = 1.5, linestyle = :dash)
        Colorbar(fig[1, 2col]; colormap = cmap, limits = clims, label, width = 15,
            tellheight = false)
    end

    ax_t = Axis(fig[2, 1:8]; xlabel = "t [hr]", ylabel = "P [MPa]",
        title = "Mean dike pressure")
    scatterlines!(ax_t, time_history ./ hr, mean_P_dike ./ 1.0e6;
        color = :black, linewidth = 2, label = "dike")
    axislegend(ax_t; position = :lt)
    rowsize!(fig.layout, 2, Relative(0.3))

    show_plot && display(fig)

    # -----------------------------------------------------------------------
    # Adjoint plot, on the mesh of the last step before it was advected
    # -----------------------------------------------------------------------
    if show_plot || write_output
        km = 1.0e-3
        pts_adj = [Point2f(c .* (L_c * km)) for c in adjoint_result.coords]
        polys_adj = [[pts_adj[el2nP_cpu[1, i]], pts_adj[el2nP_cpu[2, i]], pts_adj[el2nP_cpu[3, i]]]
                     for i in 1:mesh_stokes.nels]
        xmin, xmax, ymin, _ = obs_box .* km
        box_outline = Point2f[(xmin, ymin), (xmax, ymin), (xmax, 0), (xmin, 0), (xmin, ymin)]

        fig_adj = Figure(size = (1500, 2400))
        fd_text = isnothing(adjoint_result.dlnJ_dlnQ_fd) ? "" :
            @sprintf("  (finite difference %.4f)", adjoint_result.dlnJ_dlnQ_fd)
        Label(
            fig_adj[0, 1:4],
            @sprintf(
                "J = mean vy in box = %.3g mm/hr    ∂ln|J|/∂lnQ = %.4f%s    ∂ln|J|/∂lnK: crust %.4f, dike %.4f",
                adjoint_result.J_SI * 1.0e3 * hr, adjoint_result.dlnJ_dlnQ, fd_text, adjoint_result.dlnJ_dlnK...,
            );
            fontsize = 18,
        )
        panels = (
            (adjoint_result.vy_cell .* (1.0e3 * hr), "Vertical velocity", "vy [mm/hr]"),
            (adjoint_result.λvy_cell, "Adjoint vertical velocity", "λvy"),
            (adjoint_result.λP_cell, "Adjoint pressure", "λP"),
            (adjoint_result.sensitivity_density.dlnJ_dlnK, "Bulk modulus", "∂ln|J|/∂lnK [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlnQ, "Injection rate", "∂ln|J|/∂lnQ [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlnG, "Shear modulus", "∂ln|J|/∂lnG [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlneta, "Shear viscosity", "∂ln|J|/∂lnη [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlnpT, "Tensile-strength magnitude", "∂ln|J|/∂ln|pT| [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlnC, "Cohesion", "∂ln|J|/∂lnC [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlnphi, "Friction angle", "∂ln|J|/∂lnϕ [km⁻²]"),
            (adjoint_result.sensitivity_density.dlnJ_dlambda_fluid, "Pore-pressure fraction", "∂ln|J|/∂λ_fluid [km⁻²]"),
        )
        for (k, (field, title, label)) in enumerate(panels)
            row, col = fldmod1(k, 2)
            # Symmetric and clipped: the sensitivity peaks sharply at the box edges.
            limit = quantile(abs.(field), 0.99)
            limit > 0 || (limit = 1.0)
            ax = Axis(fig_adj[row, 2col - 1]; aspect = DataAspect(), title,
                xlabel = "x [km]", ylabel = "y [km]")
            poly!(ax, polys_adj; color = field, colormap = :vik, colorrange = (-limit, limit), strokewidth = 0)
            lines!(ax, dike_x, dike_y; color = :black, linewidth = 1.5, linestyle = :dash)
            lines!(ax, box_outline; color = :black, linewidth = 1.5, linestyle = :dot)
            Colorbar(fig_adj[row, 2col]; colormap = :vik, limits = (-limit, limit), label, width = 15)
        end
        write_output && save(joinpath(out_dir, "dike_injection_adjoint.png"), fig_adj)
        show_plot && display(fig_adj)
    end

    return (;
        time = time_history, mean_tauII = mean_tauII_history,
        max_tauII = max_tauII_history, mean_P_dike, max_uplift_rate, max_plastic_vol, post,
        solve_stats = solve_stats_history, adjoint = adjoint_result,
    )
end

main(;
    nsteps = 24 * 3,
    ε̇_bg = 1.0e-30,
    λ_fluid = 1,
    # λ_fluid = 0.37,
)
