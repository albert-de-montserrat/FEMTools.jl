# Coupled thermal-Stokes model of magma injected into a dipping dike beneath a
# flat surface, in three dimensions.

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using KernelAbstractions
using FEMTools

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

"""
    main(; nsteps=24, Δt=1hr, ε̇_bg=1e-16, Q_dike=1000, dip=90, ...) -> NamedTuple

Run the coupled thermal--Stokes model of magma injection into a dike on an
unstructured dike-conforming T11/P1-discontinuous mesh.

A 40 × 40 × 20 km crustal block with a flat, traction-free surface holds a dike
`dike_width` thick and `dike_length` long along strike (`y`), spanning depths
`dike_bottom` to `dike_top` and dipping at `dip` degrees from the horizontal: 90
is vertical, a smaller angle dips toward `+x` and a larger one toward `−x`. The
dike is a conforming material volume, and the mesh is graded from
`mesh_size / refinement` on its faces to `mesh_size` at `refinement_distance`
away. The crust is visco-elasto-plastic: Drucker-Prager (cohesion 10 MPa,
friction angle 30°) closed on the tensile side by the cap of `DruckerPragerCap`,
so it fails in tension once the pressure drops to the tensile strength `pT`.
The dike is visco-elastic magma held at `T_magma`. The yield model sees the
effective pressure `P − Pf`, with a hydrostatic pore pressure
`Pf = λ_fluid P_litho` refreshed from the lithostatic pressure every step.

Magma enters through the continuity source `Q`: a uniform weight on the dike's
pressure DoFs, rescaled by the solver so that the injected volume rate is
`Q_dike` [m³ s⁻¹]. The defaults follow the 2023–2024 Reykjanes dike injections
(Sigmundsson et al., 2024; Parks et al., 2025), whose later events peaked at
800–1700 m³ s⁻¹, delivered over hours, so `Δt` is an hour. On that time scale
the crust, whose Maxwell time `η/G` is about a kyr, responds elastically.

Background extension `vx = ε̇_bg x` acts on the `x` walls; the `y` walls carry
no normal velocity and the base no vertical velocity. Temperature is fixed at
the surface and the base and insulating on the sides. With `shear_heating = true`
the viscous and plastic shear dissipation heats the crust, solved together with
the flow. The mesh is Lagrangian:
after every step all of its nodes move with the velocity, the surface deforms
freely, and the `x`-wall velocities follow the moved walls. The stress history
is rotated with the material (Jaumann). At the end of every step the
lithostatic pressure of the current density field and geometry is recomputed,
and the overpressure `P − P_litho` is written to the VTK output.

The cap corrects the pressure at the integration points, so the accepted
pressure of each step, the continuity memory of the next one, is carried there
as `P_old` rather than in `dr.P0`.

The velocity space is the 11-node bubble-enriched quadratic tetrahedron and the
pressure space is discontinuous linear on the four tetrahedron vertices, as in
`volcano_thermal_stokes_3D.jl`.

All arguments and returned fields are SI; the solve itself runs in the
characteristic units defined below.

Returns the time series of mean and maximum second stress invariant, mean dike
pressure, maximum surface uplift rate, and maximum volumetric plastic strain,
the last post-processed field set, and per-step solver statistics. Fields are written as VTK for inspection in an
external viewer.
"""
function main(;
        nsteps = 24 * 3,
        Δt = 1hr,
        ε̇_bg = 1.0e-16,
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
        η_vp = 1e18,
        # η_vp = 3.0e10 * Δt,
        pT = -5.0e6,
        # Pore-fluid pressure as a fraction of the lithostatic pressure; the
        # hydrostatic ratio ρ_water / ρ_crust ≈ 1000 / 2700.
        λ_fluid = 0.37,
        γfact = 100.0,
        shear_heating = true,
        T_magma = 1100 + 273.0,
        dTdz = 25.0e-3,
        ϵ_T = 1.0e-6,
        write_output = true, verbose = true,
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
    # Tensile strength [Pa], compression-positive so pT ≤ 0. The cap needs
    # C cosϕ + pT sinϕ > 0; the dike's is as unreachable as its cohesion.
    pT_crust = pT / σ_c
    pT_dike  = -C_dike / 2
    plastic = DruckerPragerCap(
        (ϕ, ϕ), (Ψ, Ψ), (C_crust, C_dike), (pT_crust, pT_dike),
        (η_vp / η_c, η_vp / η_c), K,
    )

    T_surface = 273.0
    T_base    = T_surface + dTdz * depth

    ε̇  = ε̇_bg * t_c
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
        plastic_history_size = (NQ_v, mesh_stokes.nels),
    )
    τ     = FEMTools.stress(dr)
    τ_old = FEMTools.stress_old(dr)

    thermal = ThermalDiffusionDR(
        backend, mesh_T.nnodes, ThermalMaterial(; k, Cp, ρ0, α, K); ϵ = ϵ_T,
    )
    copyto!(thermal.phases, TDev(node_phase))

    # -----------------------------------------------------------------------
    # Boundary conditions
    #   vx = ε̇ x on the x walls, no normal velocity on the y walls and the
    #   base, free slip in the other components; the surface carries no
    #   traction.
    #   T is fixed on the surface and the base, insulating on the sides.
    # -----------------------------------------------------------------------
    vx_nodes = sort!(union(groups.left, groups.right))
    vy_nodes = sort!(union(groups.front, groups.back))
    wall_vx() = [ε̇ * coords_cpu[n][1] for n in vx_nodes]
    bc_vx = DirichletBoundaryCondition(nothing, TDev(Int32.(vx_nodes)), TDev(wall_vx()))
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

    # Seed the interior with the extension field that satisfies the boundary
    # conditions. The relaxation diverges from an initial guess with a non-zero
    # divergence, so this seed is required, not merely a warm start.
    copyto!(dr.v.x, [ε̇ * c[1] for c in coords_cpu])
    copyto!(dr.v.z, [-ε̇ * (c[3] + depth / L_c) for c in coords_cpu])
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.z, bc_vz)

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

    # Accepted corrected pressure at the velocity integration points, seeded
    # with the warm start. The seventh stress slot receives the cap's corrected
    # pressure after each step.
    NqP = FEMTools.shape_function_values(element_P, element_v.integration_points)
    P_old = TDev([
        sum(NqP[q][a] * P_init[DoFsP_cpu[a, iel]] for a in 1:NP)
        for q in 1:NQ_v, iel in 1:mesh_stokes.nels
    ])
    P_corrected = similar(P_old)
    τ_and_P = (τ..., P_corrected)

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
    max_plastic_vol     = zeros(Float64, nsteps)
    solve_stats_history = NamedTuple[]

    out_dir = joinpath(@__DIR__, "output_dike_3D")
    write_output && mkpath(out_dir)
    post = nothing

    @info "Starting coupled thermal--Stokes solver" nsteps Δt_hr = Δt / hr Q_dike dip

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(thermal.T0, thermal.T)
        # The walls move with the mesh.
        copyto!(bc_vx.vals, wall_vx())

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
            τ_old, P_old, plastic, workgroup, Qq, Q2D = Q_total, shear_heating,
            ncheck = 100, ϵ_tol = 1.0e-6,
            iterMax = 50_000, total_iterMax = 50_000, rel_drop0 = 0.75,
            verbose, verbose_inner = false,
        )
        solve_stats.converged || error("dike injection step $istep did not converge")
        push!(solve_stats_history, solve_stats)

        update_stokes_current_stress!(
            dr, mesh_stokes, τ_and_P, Δτ;
            phases_v, τ_old, plastic, workgroup,
        )
        # The commit adds Δt times this step's plastic strain rates to the history.
        γ_prev = Array(dr.plastic_history.γ)
        θ_prev = Array(dr.plastic_history.θ)
        commit_stokes_plastic_history!(
            dr, mesh_stokes, Δτ; phases_v, τ_old, plastic, workgroup,
        )
        copyto!(P_old, P_corrected)
        γ_ip = Array(dr.plastic_history.γ)
        θ_ip = Array(dr.plastic_history.θ)
        θ_cell = vec(mean(θ_ip; dims = 1))
        γ_cell = vec(mean(γ_ip; dims = 1))
        γdot_cell = vec(mean(γ_ip .- γ_prev; dims = 1)) ./ Δt
        θdot_cell = vec(mean(θ_ip .- θ_prev; dims = 1)) ./ Δt

        P_cpu = Array(dr.P)
        v_cpu = map(Array, FEMTools.velocity(dr))
        post = compute_strain_rate_stress_postprocess(
            v_cpu, el2n_v_cpu, Array(geo_v), map(Array, τ), element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII) * σ_c
        max_tauII_history[istep]  = maximum(post.tauII) * σ_c
        mean_P_dike[istep]        = mean(@view P_cpu[dike_P_dofs]) * σ_c
        max_uplift_rate[istep]    = maximum(@view v_cpu[3][groups.surface]) * L_c / t_c
        max_plastic_vol[istep]    = maximum(θ_cell)

        # Advance the stress history with the corotational (Jaumann) rotation.
        rotate_stress!(dr, mesh_stokes, Δτ)

        @info "Physical time step" istep nsteps t_hr = t / hr mean_tauII_MPa =
            mean_tauII_history[istep] / 1.0e6 mean_P_dike_MPa =
            mean_P_dike[istep] / 1.0e6 max_uplift_mm_hr =
            max_uplift_rate[istep] * 1.0e3 * hr iter = solve_stats.iter err =
            solve_stats.err err_T = solve_stats.err_T max_plastic_vol = max_plastic_vol[istep]

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
                    corrected_pressure = vec(mean(Array(P_corrected); dims = 1)) .* σ_c,
                    volumetric_plastic_strain = θ_cell,
                    shear_plastic_strain = γ_cell,
                    shear_plastic_strain_rate = γdot_cell,
                    volumetric_plastic_strain_rate = θdot_cell,
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

    return (;
        time = time_history, mean_tauII = mean_tauII_history,
        max_tauII = max_tauII_history, mean_P_dike, max_uplift_rate, max_plastic_vol, post,
        solve_stats = solve_stats_history,
    )
end

main()
