# Coupled thermal-Stokes model of an axisymmetric volcanic edifice in three
# dimensions.

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using KernelAbstractions
using FEMTools

const backend   = CPU()
const workgroup = 128
const kyr       = 1000 * 365.25 * 24 * 3600.0

include(joinpath(@__DIR__, "volcano_mesh_3D.jl"))

"""
    main(; nsteps=5, Δt=10kyr, ε̇_bg=1e-15, nels=(14,14,12), ...) -> NamedTuple

Run the coupled thermal--Stokes model of an Etna-like volcanic edifice on a
unstructured chamber-conforming T11/P1-discontinuous mesh.

A 40 × 40 × 20 km crustal block carries a truncated cone of revolution 3.3 km
high and hosts an oblate ellipsoidal magma chamber (semi-axes 3 × 3 × 1.5 km)
centred 5 km below the ground surface. The crust is visco-elasto-plastic
(Drucker-Prager, cohesion 10 MPa, friction angle 30°); the chamber is
visco-elastic with a low shear modulus. Temperature starts on a `dTdz` geotherm
and at `T_magma` inside the chamber, and enters the momentum balance through the
thermal-expansion term of the equation of state, which makes the hot chamber
buoyant.

Background pure shear is imposed as `vx = ε̇_bg x` on the `x` walls, with
`vy = 0` on the `y` walls and `vz = 0` on the base; the topography is a
traction-free surface, so a positive `ε̇_bg` extends the block and lets it
subside.

The velocity space is the 11-node bubble-enriched quadratic tetrahedron and the
pressure space is discontinuous linear on the four tetrahedron vertices. Gmsh
fragments the crust with the chamber ellipsoid, so both phases share a curved,
conforming interface and the mesh is locally refined around it.

All arguments and returned fields are SI; the solve itself runs in the
characteristic units defined below.

`η_vp` is tied to `Δt` and must be raised with it: the Drucker-Prager return
scales as `2 ηve / (ηve + η_vp)`, so a regularisation viscosity below the
viscoelastic viscosity `ηve = 1/(1/η + 1/(G Δt))` overshoots and the relaxation
diverges instead of converging.

Returns the time series of mean and maximum second stress invariant and mean
chamber pressure, the last post-processed field set, and per-step solver
statistics. Fields are written as VTK for inspection in an external viewer.
"""
function main(;
        nsteps = 5,
        Δt = 10kyr,
        ε̇_bg = 1.0e-15,
        nels = (14, 14, 12),
        mesh_size = nothing,
        refinement = 3.0,
        η_vp = 1.0e21,
        γfact = 100.0,
        T_magma = 1100 + 273.0,
        dTdz = 25.0e-3,
        ϵ_T = 1.0e-3,
        write_output = true, verbose = true,
    )
    # -----------------------------------------------------------------------
    # Geometry [m]
    # -----------------------------------------------------------------------
    Lx = Ly = 40.0e3
    depth = 20.0e3
    cone_base, cone_top, cone_height = 10.0e3, 0.5e3, 3.3e3
    chamber_center = (0.0, 0.0, -5.0e3)
    chamber_radii  = (3.0e3, 3.0e3, 1.5e3)

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
    # Material properties — phase 1: crust, phase 2: magma chamber.
    # Listed in SI, divided by the characteristic units above.
    # -----------------------------------------------------------------------
    # Chamber viscosity is that of a crystal-rich mush. Lower values are
    # physically defensible but push the contrast past 10³, where the
    # velocity relaxation needs several times more iterations to converge.
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
    C_crust = 1.0e7 / σ_c             # cohesion [Pa]; yield stress = C cosϕ + P sinϕ
    # The chamber stays visco-elastic. Its cohesion is set far above any
    # attainable stress rather than to `Inf`, because per-phase properties are
    # interpolated with quadratic shape functions, which are negative at some
    # quadrature points and would turn an infinite cohesion into `NaN`.
    C_chamber = 1.0e12 / σ_c
    # The Drucker-Prager return scales with 2 ηve / (ηve + η_reg), so the
    # regularisation viscosity must stay at or above the viscoelastic viscosity
    # ηve, or the plastic correction overshoots and the relaxation diverges.
    plastic = DruckerPrager(
        (ϕ, ϕ), (Ψ, Ψ), (C_crust, C_chamber), (η_vp / η_c, η_vp / η_c), K,
    )

    T_surface = 273.0
    T_base    = T_surface + dTdz * depth

    ε̇  = ε̇_bg * t_c
    Δτ = Δt / t_c

    # -----------------------------------------------------------------------
    # Meshes — one node set shared by the velocity, thermal, and pressure fields
    # -----------------------------------------------------------------------
    element_v = ReferenceElement(QuadraticElement{3, 11, Float64})  # T10 + centroid bubble
    element_P = ReferenceElement(LinearElement{3, 4, Float64})      # P1 discontinuous

    topography(x, y) = volcano_topography(x, y; cone_base = cone_base / L_c,
        cone_top = cone_top / L_c, cone_height = cone_height / L_c)
    coords_cpu, el2n_cpu, groups = build_tet11_volcano_mesh(;
        Lx = Lx / L_c, Ly = Ly / L_c, depth = depth / L_c,
        cone_base = cone_base / L_c, cone_top = cone_top / L_c,
        cone_height = cone_height / L_c,
        chamber_center = chamber_center ./ L_c,
        chamber_radii = chamber_radii ./ L_c,
        nels, mesh_size = isnothing(mesh_size) ? nothing : mesh_size / L_c, refinement,
    )
    DoFs_cpu = Int32.(1:length(coords_cpu))
    mesh_v = Mesh(
        element_v, nothing, nothing,
        coords_cpu, DoFs_cpu, el2n_cpu, groups.Γnodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)
    mesh_T = Mesh(backend, coords_cpu, el2n_cpu, element_v; workgroup)
    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    geo_v = cache.geo_v

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

    node_phase = ones(Int, length(coords_cpu))
    for iel in axes(el2n_cpu, 2), a in axes(el2n_cpu, 1)
        cell_phase[iel] == 2 && (node_phase[el2n_cpu[a, iel]] = 2)
    end

    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu  = Array(mesh_stokes.el2nP)
    DoFsP_cpu  = Array(mesh_stokes.DoFsP)
    chamber_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, findall(==(2), cell_phase)])))
    isempty(chamber_P_dofs) &&
        throw(ArgumentError("mesh does not resolve the magma chamber; increase nels"))

    @info "Phases" n_chamber_cells = count(==(2), cell_phase) n_chamber_nodes =
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
    copyto!(thermal.phases, node_phase)

    # -----------------------------------------------------------------------
    # Boundary conditions
    #   vx = ε̇ x on the x walls, vy = 0 on the y walls, vz = 0 on the base,
    #   free slip in the unconstrained components; the topography carries no
    #   traction. T is fixed on the topography and the base, insulating on the
    #   sides.
    # -----------------------------------------------------------------------
    vx_nodes = sort!(union(groups.left, groups.right))
    vy_nodes = sort!(union(groups.front, groups.back))
    bc_vx = DirichletBoundaryCondition(
        nothing, vx_nodes, [ε̇ * coords_cpu[n][1] for n in vx_nodes],
    )
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, zeros(length(vy_nodes)))
    bc_vz = DirichletBoundaryCondition(
        nothing, groups.bottom, zeros(length(groups.bottom)),
    )
    bc_T = DirichletBoundaryCondition(
        nothing,
        vcat(groups.surface, groups.bottom),
        vcat(fill(T_surface, length(groups.surface)), fill(T_base, length(groups.bottom))),
    )

    # Linear geotherm below the ground surface; the edifice sits at T_surface.
    geotherm(c) = T_surface - dTdz * L_c * min(c[3], 0.0)
    copyto!(thermal.T, [
        node_phase[i] == 2 ? T_magma : geotherm(coords_cpu[i]) for i in eachindex(coords_cpu)
    ])
    apply_bc!(thermal.T, bc_T; workgroup)
    copyto!(thermal.T0, thermal.T)

    # Seed the interior with the pure-shear field that satisfies the boundary
    # conditions. The relaxation diverges from an initial guess with a non-zero
    # divergence, so this seed is required, not merely a warm start.
    copyto!(dr.vx, [ε̇ * c[1] for c in coords_cpu])
    fill!(dr.vy, 0)
    copyto!(dr.vz, [-ε̇ * (c[3] + depth / L_c) for c in coords_cpu])
    apply_bc!(dr.vx, bc_vx)
    apply_bc!(dr.vy, bc_vy)
    apply_bc!(dr.vz, bc_vz)

    # Warm-start the pressure with the crustal load of the overlying column so
    # the first Powell-Hestenes step does not have to build it from zero.
    P_init = zeros(mesh_stokes.nnodesP)
    for iel in axes(el2nP_cpu, 2), a in axes(el2nP_cpu, 1)
        c = coords_cpu[el2nP_cpu[a, iel]]
        P_init[DoFsP_cpu[a, iel]] = ρ0[1] * abs(g[3]) * (topography(c[1], c[2]) - c[3])
    end
    copyto!(dr.P, P_init)

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) n_vz =
        length(groups.bottom) n_T = length(bc_T.DoFs) max_vx_SI =
        maximum(abs, bc_vx.vals) * L_c / t_c T_base

    # Pressure update scale: γP * RP / M_P reproduces the pointwise pressure
    # correction while adapting the step to the local viscosity.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, cache, γfact, Δτ; workgroup, phases_v = phases_v_cpu,
    )

    # -----------------------------------------------------------------------
    # Time loop
    # -----------------------------------------------------------------------
    time_history        = zeros(Float64, nsteps)
    mean_tauII_history  = zeros(Float64, nsteps)
    max_tauII_history   = zeros(Float64, nsteps)
    mean_P_chamber      = zeros(Float64, nsteps)
    solve_stats_history = NamedTuple[]

    out_dir = joinpath(@__DIR__, "output_volcano_3D")
    write_output && mkpath(out_dir)
    post = nothing

    @info "Starting coupled thermal--Stokes solver" nsteps Δt_kyr = Δt / kyr ε̇_bg

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(dr.P0, dr.P)
        copyto!(thermal.T0, thermal.T)

        solve_stats = solve_coupled_dyrel!(
            thermal, dr, mesh_T, mesh_stokes, cache,
            bc_T, (bc_vx, bc_vy, bc_vz), Δτ, γP;
            Tref, phases_v = phases_v_cpu, phases_P = phases_P_cpu,
            τ_old, plastic, workgroup,
            ncheck = 100, ϵ_tol = 1.0e-6,
            iterMax = 50_000, total_iterMax = 50_000, rel_drop0 = 0.75,
            verbose, verbose_inner = false,
        )
        push!(solve_stats_history, solve_stats)

        update_stokes_current_stress!(
            dr, mesh_stokes, cache, τ, Δτ;
            phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )

        P_cpu = Array(dr.P)
        v_cpu = map(Array, FEMTools.velocity(dr))
        post = compute_strain_rate_stress_postprocess(
            v_cpu, el2n_v_cpu, Array(geo_v), τ, element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII) * σ_c
        max_tauII_history[istep]  = maximum(post.tauII) * σ_c
        mean_P_chamber[istep]     = mean(@view P_cpu[chamber_P_dofs]) * σ_c

        foreach(copyto!, τ_old, τ)

        @info "Physical time step" istep nsteps t_kyr = t / kyr mean_tauII_MPa =
            mean_tauII_history[istep] / 1.0e6 max_tauII_MPa =
            max_tauII_history[istep] / 1.0e6 iter = solve_stats.iter err =
            solve_stats.err err_T = solve_stats.err_T

        if write_output
            T_cpu = Array(thermal.T)
            el_T = [mean(T_cpu[el2n_v_cpu[1:4, iel]]) for iel in 1:mesh_stokes.nels]
            vtk_path = joinpath(out_dir, @sprintf("volcano_thermal_stokes_3D_%04d.vtk", istep))
            write_stokes_vtk(
                vtk_path, mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
                P_cpu .* σ_c, map(v -> v .* (L_c / t_c), v_cpu),
                # Strain-rate fields scale with 1/t_c, stress fields with σ_c.
                merge(map(f -> f .* σ_c, post),
                    map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εxz, :εyz, :εII)]));
                title = "volcano thermal-Stokes 3D",
                cell_data = (; phase = cell_phase, T = el_T),
            )
        end
    end

    T_cpu = Array(thermal.T)
    chamber_nodes = findall(==(2), node_phase)
    @info "Chamber temperature" mean_T = mean(@view T_cpu[chamber_nodes]) max_T =
        maximum(@view T_cpu[chamber_nodes])

    return (;
        time = time_history,
        mean_tauII = mean_tauII_history,
        max_tauII = max_tauII_history,
        mean_P_chamber,
        post,
        solve_stats = solve_stats_history,
    )
end

main()
