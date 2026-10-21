# Coupled thermal-Stokes model of magma injected into a dike beneath a flat
# surface, in plane strain, with the adjoint sensitivities of the vertical
# velocity at candidate surface stations to the injection rate along the dike
# and to the per-element bulk modulus and viscosity, at snapshots in time.

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using ForwardDiff
using KernelAbstractions
using Gmsh
using FEMTools
using GLMakie: Figure, Axis, Colorbar, Label, poly!, lines!, scatter!, scatterlines!, heatmap!,
    vlines!, axislegend, Point2f, DataAspect, save

const backend   = CPU()
const workgroup = 128
const yr        = 365.25 * 24 * 3600.0

include(joinpath(@__DIR__, "..", "..", "gmsh_meshing.jl"))

include(joinpath(@__DIR__, "dike_sensitivities2D.jl"))

"""
    greedy_d_optimal(Gmat, n_select; noise, prior) -> (selected, logdets)

Greedily select `n_select` rows (stations) of the data Jacobian `Gmat`
(stations × parameters) that maximize the D-optimal information gain
`logdet(I + prior² GₛᵀGₛ / noise²)` of the linearized Gaussian inverse problem
with independent data noise `noise` and prior standard deviation `prior` on
every parameter. Returns the selected row indices in the order chosen and the
criterion after each pick.
"""
function greedy_d_optimal(Gmat, n_select; noise, prior)
    n_select <= size(Gmat, 1) ||
        throw(ArgumentError("cannot select $n_select of $(size(Gmat, 1)) stations"))
    information(rows) = logdet(I + (prior / noise)^2 * Gmat[rows, :]' * Gmat[rows, :])
    selected, logdets = Int[], Float64[]
    for _ in 1:n_select
        candidates = setdiff(axes(Gmat, 1), selected)
        gains = [information(vcat(selected, i)) for i in candidates]
        push!(selected, candidates[argmax(gains)])
        push!(logdets, maximum(gains))
    end
    return selected, logdets
end

"""
    main(; nsteps=100, Δt=1yr, Q_dike=1e-4, snapshot_steps=(1, 10, 50, 100),
           station_x=-18km:2km:18km, n_segments=6, n_select=4, ...) -> NamedTuple

Run the coupled thermal--Stokes model of magma injection into a dike on an
unstructured T7/P1-disc (Crouzeix-Raviart) mesh and, at every step in
`snapshot_steps`, map which part of the system each candidate surface station
sees.

A station is the surface corner node nearest to each entry of `station_x`; it
moves with the Lagrangian mesh, as a benchmark does. For each station the
objective is its vertical velocity `J = vy(station)`, and one discrete adjoint
solve of the converged step gives the absolute sensitivities `∂J/∂ln m` of every
parameter in `log_sensitivities`, per element. Three are kept:

- `G_Q[station, segment]`: `∂vy/∂ln Q` summed over each of `n_segments`
  equal-length segments along the dike, from the bottom (1) to the top, or
  from the +x end to the −x end when the dike is horizontal. It
  answers which part of the dike's injection each station sees; its row sums are
  the station's sensitivity to the total injection rate.
- `dvy_dlnK`, `dvy_dlneta`: per-element densities of `∂vy/∂ln K` and `∂vy/∂ln η`
  (nels × stations), the parts of the subsurface each station sees.

Sensitivities are absolute, in mm/yr of vertical velocity per unit log change,
so stations far from the dike, where vy ≈ 0, stay well defined. At each snapshot
`greedy_d_optimal` picks `n_select` stations from `G_Q` with vertical noise
`noise_vy` [m/s] and a prior `prior_lnQ` on each segment's ln Q; comparing the
picks across snapshots shows whether a network fixed in time stays optimal.

The adjoint is that of each step alone: the stress, accepted pressure,
plastic-strain and temperature history of the step is held fixed. The maps are
therefore the sensitivity of the current uplift rate to a change of the current
parameters at the current state; the viscoelastic response to past injection
enters only through that state.

A 40 km × 20 km crustal section with a flat, traction-free surface holds a
dike `dike_width` wide and `dike_length` long, centred at `x = 0` and depth
`dike_center_depth`, dipping at `dip` degrees from the horizontal: 90 is
vertical, 0 a horizontal sill, a smaller angle dips toward `+x` and a larger
one toward `−x`. Figures and VTK files go to `out_dir`. The crust is visco-elasto-plastic
(Drucker-Prager with the tensile cap of `DruckerPragerCap`, yield on `P − Pf`
with `Pf = λ_fluid P_litho`), and the dike is visco-elastic magma held at
`T_magma`. Magma enters through the continuity source `Q`, uniform on the dike's
pressure DoFs and rescaled so the injected plane-strain area rate is `Q_dike`
[m² s⁻¹]. With a crustal Maxwell time `η/G` of about a kyr and `Δt` of years,
the crust responds viscoelastically. Background pure shear `vx = ε̇_bg x` acts on
the vertical walls with `vy = 0` on the base. The mesh is Lagrangian and the
stress history is rotated with the material (Jaumann).

All arguments and returned fields are SI unless their names say otherwise; the
solve itself runs in the characteristic units defined below.

Returns, per snapshot, the time, station coordinates, `G_Q`, the K and η kernels,
the station vertical velocity, the selected stations and their information
gains, the forward time series, and per-step solver statistics.
"""
function main(;
        nsteps = 100,
        Δt = 1yr,
        ε̇_bg = 1.0e-16,
        Q_dike = 1.0e-4,
        dike_width = 200.0,
        dip = 90.0,
        dike_center_depth = -5.0e3,
        dike_length = 6.0e3,
        max_area = 1.0e6,
        refinement = 64,
        refinement_distance = 3.0e3,
        # The Drucker-Prager return scales as 2 ηve / (ηve + η_vp): η_vp must be
        # of the order of the crust's viscoelastic viscosity ηve ≈ G Δt for the
        # plastic correction to act, and not below it, or it overshoots.
        η_vp = 3.0e10 * Δt,
        pT = -5.0e6,
        # Pore-fluid pressure as a fraction of the lithostatic pressure; the
        # hydrostatic ratio ρ_water / ρ_crust ≈ 1000 / 2700.
        λ_fluid = 0.37,
        γfact = 100.0,
        T_magma = 1100 + 273.0,
        dTdz = 25.0e-3,
        ϵ_T = 1.0e-6,
        shear_heating = true,
        snapshot_steps = (1, 10, 50, 100),
        station_x = -18.0e3:2.0e3:18.0e3,
        n_segments = 6,
        n_select = 4,
        noise_vy = 1.0e-3 / yr,
        prior_lnQ = 1.0,
        out_dir = joinpath(@__DIR__, "output_dike_adj_coverage"),
        show_plot = true, write_output = true, verbose = true,
    )
    all(s -> 1 <= s <= nsteps, snapshot_steps) ||
        throw(ArgumentError("snapshot_steps must lie in 1:$nsteps, got $snapshot_steps"))
    # -----------------------------------------------------------------------
    # Geometry [m]
    # -----------------------------------------------------------------------
    Lx, depth = 40.0e3, 20.0e3
    0 <= dip < 180 || throw(ArgumentError("dip must lie in [0, 180) degrees, got $dip"))
    # Vertical half-extent of the rotated dike rectangle.
    half_extent_y = dike_length / 2 * sind(dip) + dike_width / 2 * abs(cosd(dip))
    dike_center_depth + half_extent_y < 0 && dike_center_depth - half_extent_y > -depth ||
        throw(ArgumentError("the dike must lie strictly below the surface and above the base"))
    all(x -> abs(x) < Lx / 2, station_x) ||
        throw(ArgumentError("every station must lie inside the surface, |x| < $(Lx / 2) m"))
    dike_height = dike_length
    dike_center = (0.0, dike_center_depth)
    # Counter-clockwise rotation of the vertical dike: its top moves toward −x.
    dike_rotation = deg2rad(90 - dip)
    # Unit vectors across and along the dike.
    dike_across = (cos(dike_rotation), sin(dike_rotation))
    dike_along  = (-sin(dike_rotation), cos(dike_rotation))

    # -----------------------------------------------------------------------
    # Characteristic units
    #
    # The dynamic-relaxation convergence tests compare absolute residuals with
    # fixed thresholds, so a crustal-scale problem in SI units diverges the
    # guard before it converges. Everything below is therefore scaled by these
    # units and scaled back for output. Temperature stays in Kelvin.
    # -----------------------------------------------------------------------
    L_c  = depth                  # length [m]
    t_c  = Δt                     # time [s]: the step
    σ_c  = 1.0e8                  # stress [Pa]
    η_c  = σ_c * t_c              # viscosity [Pa s]
    ρ_c  = σ_c * t_c^2 / L_c^2    # density [kg m⁻³]
    g_c  = L_c / t_c^2            # gravity [m s⁻²]
    Cp_c = L_c^2 / t_c^2          # heat capacity [J kg⁻¹ K⁻¹]
    k_c  = ρ_c * Cp_c * L_c^2 / t_c   # thermal conductivity [W m⁻¹ K⁻¹]
    mm_yr = L_c / t_c * 1.0e3 * yr    # characteristic velocity in mm/yr

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
    centroid(iel) = sum(coords_cpu[el2n_cpu[a, iel]] for a in 1:3) / 3
    cell_phase = Int[in_dike(centroid(iel)) ? 2 : 1 for iel in axes(el2n_cpu, 2)]
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)

    node_phase = ones(Int, length(coords_cpu))
    for iel in axes(el2n_cpu, 2), a in axes(el2n_cpu, 1)
        cell_phase[iel] == 2 && (node_phase[el2n_cpu[a, iel]] = 2)
    end

    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu  = Array(mesh_stokes.el2nP)
    DoFsP_cpu  = Array(mesh_stokes.DoFsP)
    dike_cells  = findall(==(2), cell_phase)
    dike_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, dike_cells])))
    isempty(dike_P_dofs) &&
        throw(ArgumentError("no element lies inside the dike; refine the mesh or widen the dike"))

    # Segment of each dike element along `dike_along`, fixed by the initial
    # geometry: from the bottom (1) to the top (n_segments) of a dipping dike,
    # from the +x end to the −x end of a horizontal one.
    dike_segment = map(dike_cells) do iel
        c = centroid(iel)
        s = (c[1] - dike_center[1] / L_c) * dike_along[1] + (c[2] - dike_center[2] / L_c) * dike_along[2]
        clamp(ceil(Int, (s / (dike_height / L_c) + 0.5) * n_segments), 1, n_segments)
    end
    # Distance of each segment centre from the segment-1 end of the dike. [km]
    segment_position_km = [(j - 0.5) / n_segments * dike_length / 1.0e3 for j in 1:n_segments]

    # Stations: the surface corner nodes nearest to `station_x`.
    surface_corners = intersect(surface, vec(el2n_v_cpu[1:3, :]))
    station_nodes = map(station_x) do x
        surface_corners[argmin(abs.(getindex.(coords_cpu[surface_corners], 1) .- x / L_c))]
    end
    allunique(station_nodes) ||
        throw(ArgumentError("two stations map to the same surface node; increase their spacing or refine the mesh"))
    nstations = length(station_nodes)

    # Uniform source weight on the dike; the solver rescales it to Q2D.
    Qq = zeros(mesh_stokes.nnodesP)
    Qq[dike_P_dofs] .= 1.0

    @info "Phases and stations" n_dike_cells = length(dike_cells) n_dike_nodes =
        count(==(2), node_phase) nstations n_segments

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
    adjoint_workspace = StokesAdjointWorkspace(dr, bc_vx.DoFs, bc_vy.DoFs)

    # -----------------------------------------------------------------------
    # Time loop
    # -----------------------------------------------------------------------
    time_history        = zeros(Float64, nsteps)
    mean_P_dike         = zeros(Float64, nsteps)
    max_uplift_rate     = zeros(Float64, nsteps)
    max_plastic_vol     = zeros(Float64, nsteps)
    solve_stats_history = NamedTuple[]
    snapshots           = NamedTuple[]

    write_output && mkpath(out_dir)

    @info "Starting coupled thermal--Stokes solver" nsteps Δt_yr = Δt / yr ε̇_bg Q_dike

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
            ncheck = 100, ϵ_tol = 1.0e-6,
            iterMax = 50_000, total_iterMax = 50_000, rel_drop0 = 0.75,
            verbose, verbose_inner = false,
        )
        solve_stats.converged || error("dike injection step $istep did not converge")
        push!(solve_stats_history, solve_stats)

        # Station adjoints of the converged step, before its stress history,
        # accepted pressure, pore pressure, and mesh are advanced.
        cell_fields = (;)
        if istep in snapshot_steps
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
            element_area_km2 = [
                sum(last, element_geometry(data.geo_v, iel, data.∂N∂ξ)) for iel in 1:mesh_stokes.nels
            ] .* (L_c^2 / 1.0e6)
            nnodes = mesh_stokes.nnodes
            G_Q        = zeros(nstations, n_segments)
            dvy_dlnK   = zeros(mesh_stokes.nels, nstations)
            dvy_dlneta = zeros(mesh_stokes.nels, nstations)
            t_adjoint = @elapsed for (i, node) in enumerate(station_nodes)
                load_y = zeros(nnodes)
                load_y[node] = 1.0
                λv = (zeros(nnodes), zeros(nnodes))
                λP = zeros(mesh_stokes.nnodesP)
                adjoint_stats = solve_stokes_adjoint_dyrel!(
                    dr, mesh_stokes, geo_v, mesh_stokes.geometry.geo_P, element_v, element_P,
                    phases_v_cpu, phases_P_cpu, τ_old, plastic, dr.G, Δτ, γP,
                    (zeros(nnodes), load_y), λv, λP, backend, workgroup;
                    v_nodes = (bc_vx.DoFs, bc_vy.DoFs), workspace = adjoint_workspace,
                    # The graded mesh can make the power estimate stop below the stable bound.
                    measure_λmax = false,
                    adjoint_tol = 1.0e-8, iterMax = 200_000, total_iterMax = 200_000,
                    max_ph_iterations = 1000, verbose = false,
                )
                adjoint_stats.converged ||
                    error("station $i adjoint did not converge at step $istep: err = $(adjoint_stats.err)")
                # Absolute ∂vy/∂ln m per element, despite the dlnJ_ field names.
                derivatives, _ = log_sensitivities(data, λv, λP, Val(NV), Val(NP))
                for (iel, j) in zip(dike_cells, dike_segment)
                    G_Q[i, j] += derivatives.dlnJ_dlnQ[iel] * mm_yr
                end
                dvy_dlnK[:, i]   .= derivatives.dlnJ_dlnK .* mm_yr ./ element_area_km2
                dvy_dlneta[:, i] .= derivatives.dlnJ_dlneta .* mm_yr ./ element_area_km2
            end
            selected, logdets = greedy_d_optimal(G_Q, n_select; noise = noise_vy * 1.0e3 * yr, prior = prior_lnQ)
            snapshot = (;
                step = istep, t,
                station_coords_km = [coords_cpu[n] .* (L_c / 1.0e3) for n in station_nodes],
                vy_station_mm_yr = data.v[2][station_nodes] .* mm_yr,
                G_Q, dvy_dlnK, dvy_dlneta, selected, logdets,
                coords = copy(coords_cpu),
            )
            push!(snapshots, snapshot)
            cell_fields = (;
                (Symbol(@sprintf("station%02d_dvy_dlnK_per_km2", i)) => dvy_dlnK[:, i] for i in 1:nstations)...,
                (Symbol(@sprintf("station%02d_dvy_dlneta_per_km2", i)) => dvy_dlneta[:, i] for i in 1:nstations)...,
            )
            @info "Station coverage" istep t_yr = t / yr t_adjoint dvy_dlnQ_mm_yr =
                round.(vec(sum(G_Q; dims = 2)); sigdigits = 3) selected_x_km =
                [snapshot.station_coords_km[i][1] for i in selected] logdets
        end

        update_stokes_current_stress!(
            dr, mesh_stokes, τ_and_P, Δτ;
            phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )
        commit_stokes_plastic_history!(
            dr, mesh_stokes, Δτ; phases_v = phases_v_cpu, τ_old, plastic, workgroup,
        )
        copyto!(P_old, P_corrected)
        θ_cell = vec(mean(Array(dr.plastic_history.θ); dims = 1))

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.v.x)
        vy_cpu = Array(dr.v.y)
        mean_P_dike[istep]     = mean(@view P_cpu[dike_P_dofs]) * σ_c
        max_uplift_rate[istep] = maximum(@view vy_cpu[surface]) * L_c / t_c
        max_plastic_vol[istep] = maximum(θ_cell)

        # Advance the stress history with the corotational (Jaumann) rotation.
        rotate_stress!(dr, mesh_stokes, Δτ)

        @info "Physical time step" istep nsteps t_yr = t / yr mean_P_dike_MPa =
            mean_P_dike[istep] / 1.0e6 max_uplift_mm_yr =
            max_uplift_rate[istep] * 1.0e3 * yr iter =
            solve_stats.iter err = solve_stats.err max_plastic_vol = max_plastic_vol[istep]

        P_litho = lithostatic_pressure!()
        copyto!(dr.Pf, λ_fluid .* P_litho)

        if write_output && !isempty(cell_fields)
            post = compute_strain_rate_stress_postprocess(
                vx_cpu, vy_cpu, el2n_v_cpu, Array(geo_v), τ, element_v,
            )
            vtk_path = joinpath(out_dir, @sprintf("dike_coverage_%04d.vtk", istep))
            write_stokes_vtk(
                vtk_path, mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
                P_cpu .* σ_c, vx_cpu .* (L_c / t_c), vy_cpu .* (L_c / t_c),
                merge(map(f -> f .* σ_c, post),
                    map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εII)]));
                title = "dike injection station coverage",
                cell_data = (; phase = cell_phase, volumetric_plastic_strain = θ_cell, cell_fields...),
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
    # Coverage plot: per snapshot, the station sensitivity to the total
    # injection rate, its split along the dike, and the K and η coverage of the
    # selected network.
    # -----------------------------------------------------------------------
    if show_plot || write_output
        km = 1.0e-3
        nsnap = length(snapshots)
        station_x_km = [c[1] for c in first(snapshots).station_coords_km]
        dike_corners = [
            dike_center .+ (a * dike_width / 2) .* dike_across .+ (b * dike_height / 2) .* dike_along
            for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
        ]
        dike_outline = [Point2f(c .* km) for c in dike_corners]

        fig = Figure(size = (450 * nsnap + 200, 1800))
        ax_q = Axis(fig[1, 1:2nsnap]; xlabel = "station x [km]", ylabel = "∂vy/∂lnQ [mm/yr]",
            title = "Station sensitivity to the total injection rate (filled: selected)")
        for s in snapshots
            label = @sprintf("t = %.0f yr", s.t / yr)
            total = vec(sum(s.G_Q; dims = 2))
            scatterlines!(ax_q, station_x_km, total; label)
            scatter!(ax_q, station_x_km[s.selected], total[s.selected]; markersize = 16)
        end
        axislegend(ax_q; position = :rt)

        for (col, s) in enumerate(snapshots)
            limit = maximum(abs, s.G_Q)
            limit > 0 || (limit = 1.0)
            ax = Axis(fig[2, 2col - 1]; xlabel = "station x [km]", ylabel = "position along dike [km]",
                title = @sprintf("∂vy/∂lnQ per dike segment, t = %.0f yr", s.t / yr))
            heatmap!(ax, station_x_km, segment_position_km, s.G_Q; colormap = :vik, colorrange = (-limit, limit))
            vlines!(ax, station_x_km[s.selected]; color = :black, linestyle = :dash)
            Colorbar(fig[2, 2col]; colormap = :vik, limits = (-limit, limit), label = "mm/yr", width = 12)

            pts = [Point2f(c .* (L_c * km)) for c in s.coords]
            polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]] for i in 1:mesh_stokes.nels]
            for (row, kernel, name) in ((3, s.dvy_dlnK, "K"), (4, s.dvy_dlneta, "η"))
                # Summed magnitude over the selected stations: the region the network sees.
                coverage = vec(sum(abs, kernel[:, s.selected]; dims = 2))
                limit = quantile(coverage, 0.99)
                limit > 0 || (limit = 1.0)
                ax = Axis(fig[row, 2col - 1]; aspect = DataAspect(), xlabel = "x [km]", ylabel = "y [km]",
                    title = @sprintf("Σ|∂vy/∂ln%s| of selected, t = %.0f yr", name, s.t / yr))
                poly!(ax, polys; color = coverage, colormap = :viridis, colorrange = (0, limit), strokewidth = 0)
                lines!(ax, dike_outline; color = :white, linewidth = 1.5, linestyle = :dash)
                scatter!(ax, [Point2f(c...) for c in s.station_coords_km[s.selected]]; color = :red, markersize = 10)
                Colorbar(fig[row, 2col]; colormap = :viridis, limits = (0, limit), label = "mm/yr/km²", width = 12)
            end
        end
        Label(fig[0, 1:2nsnap], @sprintf("Greedy D-optimal networks of %d stations, σ_vy = %.2g mm/yr, dip = %g°",
            n_select, noise_vy * 1.0e3 * yr, dip); fontsize = 18)
        write_output && save(joinpath(out_dir, "dike_station_coverage.png"), fig)
        show_plot && display(fig)
    end

    return (;
        time = time_history, mean_P_dike, max_uplift_rate, max_plastic_vol,
        segment_position_km, snapshots, solve_stats = solve_stats_history,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; ε̇_bg = 1.0e-30, λ_fluid = 1)
end
