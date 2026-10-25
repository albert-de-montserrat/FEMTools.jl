# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. -t 8 StationsPaper/scripts/chamber_injection2D_adj.jl [out_dir] [nsteps]
# Coupled thermal-Stokes recharge of the StationsPaper magma chamber in plane
# strain, with the adjoint log-sensitivities of the uplift rate above the
# chamber at every step. It is examples/stokes/volcano/dike_injection2D_adj.jl
# with the dike replaced by the chamber of chamber_memory_setup2D.jl: a magma
# disc of radius `chamber_radius` at `center_depth`, inside a Maxwell shell of
# radius `shell_radius`, in a 40 km × 20 km section. Phases: 1 host, 2 magma,
# 3 shell. Unlike the linear chamber model, the host and shell are
# visco-elasto-plastic (Drucker-Prager with a tensile cap, yielding on the
# effective pressure P - λ_fluid P_litho), gravity and heat transport are on,
# and the mesh is Lagrangian.
#
# Magma enters at the constant area rate `Q_chamber` [m² s⁻¹], by default the
# StationsPaper Q_ref (0.24 m/yr of uplift at x = 0 with the Maxwell shell).
# The objective is the mean vertical velocity in a box `obs_width` wide and
# `obs_depth` deep at the surface above the chamber, a stand-in for a GNSS
# station at x = 0. The adjoint is that of the Stokes step alone: the stress,
# accepted pressure, plastic-strain and temperature fields of the step are held
# fixed (see dike_sensitivities2D.jl).
#
# Writes one VTK file per step (velocity, pressure, stress and strain-rate
# tensors and invariants; per element: phase, T, corrected, lithostatic and
# over-pressure, plastic strains and their rates, adjoint velocity and
# pressure, every sensitivity and its density per km², observation box) and
# time_series.csv.
using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using ForwardDiff
using KernelAbstractions
using FEMTools

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO_ROOT, "examples", "triangulate_meshing.jl"))
include(joinpath(REPO_ROOT, "examples", "stokes", "volcano", "dike_sensitivities2D.jl"))

const backend = CPU()
const workgroup = 128
const day = 86400.0
const yr = 365.25day

function chamber_injection_adjoint(;
        out_dir = joinpath(@__DIR__, "..", "output", "chamber_injection2D_adj"),
        nsteps = 52,
        Δt = 7day,
        Q_chamber = 1.7366e-4,
        chamber_radius = 1.0e3,
        shell_radius = 2.0e3,
        center_depth = 4.0e3,
        max_area = 4.0e6,
        refinement = 8,
        surface_size = 200.0,
        η_host = 1.0e23, η_magma = 1.0e14, η_shell = 3.0e17,
        G_host = 3.0e10, G_magma = 1.0e8,
        K_host = 5.0e10, K_magma = 1.0e10,
        # Neutral buoyancy by default, as in the linear chamber model: a lighter
        # or thermally expanded magma rises through the viscous shell and its
        # uplift swamps the recharge signal. The EOS is ρ0 (1 - α (T - Tref) + P / K),
        # so the softer magma's ρ0 is lowered to match the host's density at the
        # lithostatic pressure of the chamber centre.
        ρ_host = 2700.0, α_host = 0.0, α_magma = 0.0,
        ρ_magma = ρ_host * (1 + ρ_host * 9.81 * center_depth / K_host) / (1 + ρ_host * 9.81 * center_depth / K_magma),
        # The Drucker-Prager return scales as 2 ηve / (ηve + η_vp), with
        # ηve ≈ G Δt in the elastic host: η_vp of that order lets the plastic
        # correction act without overshooting.
        η_vp = G_host * Δt,
        cohesion = 1.0e7,
        friction_angle = 30.0,
        pT = -5.0e6,
        λ_fluid = 0.37,
        γfact = 100.0,
        T_magma = 1100 + 273.0,
        dTdz = 25.0e-3,
        ϵ_T = 1.0e-6,
        shear_heating = true,
        obs_width = 500.0,
        obs_depth = 200.0,
        verbose = false,
    )
    mkpath(out_dir)
    Lx, depth = 40.0e3, 20.0e3
    obs_box = (-obs_width / 2, obs_width / 2, -obs_depth, Inf)

    # Characteristic units, as in the dike model: the relaxation tests compare
    # absolute residuals with fixed thresholds. Temperature stays in Kelvin.
    L_c  = depth
    t_c  = Δt
    σ_c  = 1.0e8
    η_c  = σ_c * t_c
    ρ_c  = σ_c * t_c^2 / L_c^2
    g_c  = L_c / t_c^2
    Cp_c = L_c^2 / t_c^2
    k_c  = ρ_c * Cp_c * L_c^2 / t_c

    η    = (η_host, η_magma, η_shell) ./ η_c
    G    = (G_host, G_magma, G_host) ./ σ_c
    K    = (K_host, K_magma, K_host) ./ σ_c
    ηb   = K
    ρ0   = (ρ_host, ρ_magma, ρ_host) ./ ρ_c
    k    = (2.5, 2.0, 2.5) ./ k_c
    Cp   = (1000.0, 1200.0, 1000.0) ./ Cp_c
    α    = (α_host, α_magma, α_host)
    g    = (0.0, -9.81 / g_c)
    Tref = 273.0

    ϕ = deg2rad(friction_angle)
    Ψ = 0.0
    # Magma stays visco-elastic: its cohesion and tensile strength are out of
    # reach but finite, because T7 interpolation of `Inf` gives `NaN`.
    C_magma = 1.0e12 / σ_c
    plastic = DruckerPragerCap(
        (ϕ, ϕ, ϕ), (Ψ, Ψ, Ψ), (cohesion / σ_c, C_magma, cohesion / σ_c),
        (pT / σ_c, -C_magma / 2, pT / σ_c), ntuple(_ -> η_vp / η_c, 3), K,
    )

    T_surface = 273.0
    T_base    = T_surface + dTdz * depth
    Δτ  = Δt / t_c
    Q2D = Q_chamber / (L_c^2 / t_c)

    # Mesh: the builder meshes [0, Lx] × [-depth, 0]; shift it to be centred on x = 0.
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    coords_shifted, el2n_cpu, _, _ = build_triangulate_t7_concentric_disk_mesh(;
        Lx = Lx / L_c, Ly = depth / L_c, cx = Lx / 2 / L_c, cy = -center_depth / L_c,
        radii = (chamber_radius, shell_radius) ./ L_c, max_area = max_area / L_c^2, refinement,
        surface_size = surface_size / L_c,
        surface_xrange = ((Lx / 2 - 3center_depth) / L_c, (Lx / 2 + 3center_depth) / L_c),
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

    # Element phase from the centroid; elements never straddle the interfaces.
    radius(c) = hypot(c[1], c[2] + center_depth / L_c) * L_c
    cell_phase = map(axes(el2n_cpu, 2)) do iel
        r = radius(sum(coords_cpu[el2n_cpu[a, iel]] for a in 1:3) / 3)
        r < chamber_radius ? 2 : r < shell_radius ? 3 : 1
    end
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)
    # A node takes the magma phase if any of its elements is magma, else the shell's.
    node_phase = ones(Int, length(coords_cpu))
    for p in (3, 2), iel in findall(==(p), cell_phase), a in axes(el2n_cpu, 1)
        node_phase[el2n_cpu[a, iel]] = p
    end
    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v = mesh_stokes.nnodes nnodes_P = mesh_stokes.nnodesP nels =
        mesh_stokes.nels cells_per_phase = [count(==(p), cell_phase) for p in 1:3]

    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu  = Array(mesh_stokes.el2nP)
    DoFsP_cpu  = Array(mesh_stokes.DoFsP)
    magma_P_dofs = sort!(unique!(vec(DoFsP_cpu[:, findall(==(2), cell_phase)])))
    # Uniform source weight on the magma; the solver rescales it to Q2D.
    Qq = zeros(mesh_stokes.nnodesP)
    Qq[magma_P_dofs] .= 1.0

    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP,
        StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref);
        CFL_v = 0.99, CFL_P = 0.99, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
        plastic_history_size = (NQ_v, mesh_stokes.nels),
    )
    τ = (dr.τ.xx, dr.τ.yy, dr.τ.xy)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)
    thermal = ThermalDiffusionDR(backend, mesh_T.nnodes, ThermalMaterial(; k, Cp, ρ0, α, K); ϵ = ϵ_T)
    copyto!(thermal.phases, node_phase)

    # Free surface; vx = 0 on the walls, vy = 0 on the base; T fixed on the
    # surface and the base, insulating on the sides.
    vx_nodes = vcat(left, right)
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, zeros(length(vx_nodes)))
    bc_vy = DirichletBoundaryCondition(nothing, bottom, zeros(length(bottom)))
    bc_T = DirichletBoundaryCondition(nothing, vcat(surface, bottom),
        vcat(fill(T_surface, length(surface)), fill(T_base, length(bottom))))
    geotherm(c) = T_surface - dTdz * L_c * c[2]
    copyto!(thermal.T, [node_phase[i] == 2 ? T_magma : geotherm(coords_cpu[i]) for i in eachindex(coords_cpu)])
    apply_bc!(thermal.T, bc_T; workgroup)
    copyto!(thermal.T0, thermal.T)

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

    # Accepted corrected pressure at the velocity integration points.
    NqP = shape_function_values(element_P, element_v.integration_points)
    P_old = [sum(NqP[q][a] * P_litho[DoFsP_cpu[a, iel]] for a in 1:NP) for q in 1:NQ_v, iel in 1:mesh_stokes.nels]
    P_corrected = similar(P_old)
    τ_and_P = (τ..., P_corrected)
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)

    ∂N∂ξ_table = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_gradients(element_v))
    Nq_table = FEMTools.quadrature_table(CPU(), FEMTools.shape_function_values(element_v))
    NqP_table = FEMTools.quadrature_table(CPU(), NqP)
    rows = NamedTuple[]
    for istep in 1:nsteps
        copyto!(thermal.T0, thermal.T)
        assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh_stokes, γfact, Δτ; workgroup, phases_v = phases_v_cpu)
        solve_stats = solve_coupled_dyrel!(
            thermal, dr, mesh_T, mesh_stokes, bc_T, bc_vx, bc_vy, Δτ, γP;
            Tref, phases_v = phases_v_cpu, phases_P = phases_P_cpu,
            τ_old, P_old, plastic, workgroup, Qq, Q2D, shear_heating,
            ncheck = 100, ϵ_tol = 1.0e-6, iterMax = 50_000, total_iterMax = 50_000, rel_drop0 = 0.75,
            verbose, verbose_inner = false,
        )
        solve_stats.converged || error("chamber recharge step $istep did not converge")

        # Adjoint of the converged step, before its history and mesh advance.
        data = (;
            v = (Array(dr.v.x), Array(dr.v.y)), P = Array(dr.P), el2n = el2n_v_cpu, DoFsP = DoFsP_cpu,
            geo_v = Array(geo_v), geo_P = Array(mesh_stokes.geometry.geo_P),
            ∂N∂ξ = ∂N∂ξ_table, Nq = Nq_table, NqP = NqP_table,
            T = Array(dr.T), T0 = Array(dr.T0), P_old = copy(P_old), Pf = Array(dr.Pf),
            Q = Array(dr.Q), P_litho = copy(P_litho), τ_old = map(Array, τ_old),
            phases_v = phases_v_cpu, phases_P = phases_P_cpu,
            dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.ηb, dr.g, dr.Tref, Δt = Δτ, plastic,
            γP = Array(γP), M_P = Array(dr.M_P),
        )
        box = obs_box ./ L_c
        load_y, _ = box_mean_load(coords_cpu, el2n_v_cpu, data.geo_v, data.∂N∂ξ, data.Nq, box, Val(NV))
        λv = (zeros(mesh_stokes.nnodes), zeros(mesh_stokes.nnodes))
        λP = zeros(mesh_stokes.nnodesP)
        t_adjoint = @elapsed begin
            adjoint_stats = solve_stokes_adjoint_dyrel!(
                dr, mesh_stokes, geo_v, mesh_stokes.geometry.geo_P, element_v, element_P,
                phases_v_cpu, phases_P_cpu, τ_old, plastic, dr.G, Δτ, γP,
                (zeros(mesh_stokes.nnodes), load_y), λv, λP, backend, workgroup;
                v_nodes = (bc_vx.DoFs, bc_vy.DoFs), measure_λmax = false,
                adjoint_tol = 1.0e-8, iterMax = 200_000, total_iterMax = 200_000,
                max_ph_iterations = 1000, verbose,
            )
            adjoint_stats.converged || error("adjoint solve did not converge: err = $(adjoint_stats.err)")
            derivatives, λP_plain = log_sensitivities(data, λv, λP, Val(NV), Val(NP))
        end
        J = dot(load_y, data.v[2])
        sensitivities = relative_sensitivities(derivatives, J)
        totals = map(f -> ntuple(p -> sum(f[cell_phase .== p]), 3), sensitivities)
        element_area_km2 = [sum(last, element_geometry(data.geo_v, iel, data.∂N∂ξ)) for iel in 1:mesh_stokes.nels] .*
            (L_c^2 / 1.0e6)
        density_fields = NamedTuple{Tuple(Symbol(k, :_per_km2) for k in keys(sensitivities))}(
            map(f -> f ./ element_area_km2, values(sensitivities)))
        # Multipliers are in the solver's characteristic units.
        λv_cell = map(λ -> [mean(λ[el2n_v_cpu[1:3, iel]]) for iel in 1:mesh_stokes.nels], λv)
        λP_cell = [mean(λP_plain[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels]
        in_obs_box = map(axes(el2n_v_cpu, 2)) do iel
            c = mean(coords_cpu[el2n_v_cpu[a, iel]] for a in 1:3)
            Int(box[1] < c[1] < box[2] && box[3] < c[2] < box[4])
        end

        update_stokes_current_stress!(dr, mesh_stokes, τ_and_P, Δτ; phases_v = phases_v_cpu, τ_old, plastic, workgroup)
        γ_prev = Array(dr.plastic_history.γ)
        θ_prev = Array(dr.plastic_history.θ)
        commit_stokes_plastic_history!(dr, mesh_stokes, Δτ; phases_v = phases_v_cpu, τ_old, plastic, workgroup)
        copyto!(P_old, P_corrected)
        γ_now = Array(dr.plastic_history.γ)
        θ_now = Array(dr.plastic_history.θ)
        γdot_cell = vec(mean(γ_now .- γ_prev; dims = 1)) ./ Δt
        θdot_cell = vec(mean(θ_now .- θ_prev; dims = 1)) ./ Δt

        P_cpu, vx_cpu, vy_cpu = Array(dr.P), Array(dr.v.x), Array(dr.v.y)
        post = compute_strain_rate_stress_postprocess(vx_cpu, vy_cpu, el2n_v_cpu, Array(geo_v), τ, element_v)
        rotate_stress!(dr, mesh_stokes, Δτ)

        row = (; step = istep, t_days = istep * Δt / day,
            J_mm_per_yr = J * L_c / t_c * 1.0e3 * yr,
            max_uplift_mm_per_yr = maximum(@view vy_cpu[surface]) * L_c / t_c * 1.0e3 * yr,
            mean_P_magma_MPa = mean(@view P_cpu[magma_P_dofs]) * σ_c / 1.0e6,
            max_tauII_MPa = maximum(post.tauII) * σ_c / 1.0e6,
            max_plastic_vol = maximum(vec(mean(θ_now; dims = 1))),
            max_plastic_shear = maximum(vec(mean(γ_now; dims = 1))),
            plastic_cells = count(>(0), γdot_cell),
            dlnJ_dlnQ = sum(sensitivities.dlnJ_dlnQ),
            (Symbol(k, :_host) => totals[k][1] for k in keys(totals) if k != :dlnJ_dlnQ)...,
            (Symbol(k, :_shell) => totals[k][3] for k in keys(totals) if k != :dlnJ_dlnQ)...,
            (Symbol(k, :_magma) => totals[k][2] for k in keys(totals) if k != :dlnJ_dlnQ)...,
            adjoint_residual = adjoint_stats.err, iterations = solve_stats.iter, t_adjoint)
        push!(rows, row)
        @printf("step %3d  t %5.0f d  J %8.2f mm/yr  max uplift %8.2f mm/yr  P_magma %8.2f MPa  max τII %6.2f MPa  plastic cells %5d  ∂lnJ/∂lnQ %+.3f  ∂lnJ/∂lnη shell %+.3f  ∂lnJ/∂lnK magma %+.3f\n",
            istep, row.t_days, row.J_mm_per_yr, row.max_uplift_mm_per_yr, row.mean_P_magma_MPa, row.max_tauII_MPa,
            row.plastic_cells, row.dlnJ_dlnQ, row.dlnJ_dlneta_shell, row.dlnJ_dlnK_magma)

        overpressure = P_cpu .- P_litho
        T_cpu = Array(thermal.T)
        write_stokes_vtk(
            joinpath(out_dir, @sprintf("chamber_injection_%04d.vtk", istep)),
            mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
            P_cpu .* σ_c, vx_cpu .* (L_c / t_c), vy_cpu .* (L_c / t_c),
            # Strain-rate fields scale with 1/t_c, stress fields with σ_c.
            merge(map(f -> f .* σ_c, post), map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εII)]));
            title = "chamber recharge thermal-Stokes",
            cell_data = (;
                phase = cell_phase,
                T = [mean(T_cpu[el2n_v_cpu[1:3, iel]]) for iel in 1:mesh_stokes.nels],
                corrected_pressure = vec(mean(P_corrected; dims = 1)) .* σ_c,
                P_litho = [mean(P_litho[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels] .* σ_c,
                overpressure = [mean(overpressure[DoFsP_cpu[:, iel]]) for iel in 1:mesh_stokes.nels] .* σ_c,
                volumetric_plastic_strain = vec(mean(θ_now; dims = 1)),
                shear_plastic_strain = vec(mean(γ_now; dims = 1)),
                shear_plastic_strain_rate = γdot_cell,
                volumetric_plastic_strain_rate = θdot_cell,
                sensitivities..., density_fields...,
                adjoint_vx = λv_cell[1], adjoint_vy = λv_cell[2], adjoint_P = λP_cell,
                observation_box = in_obs_box,
            ),
        )

        # Lithostatic and pore pressure of the current state, then Lagrangian
        # mesh motion of the corner nodes; edge and bubble nodes follow.
        P_litho = lithostatic_pressure!()
        copyto!(dr.Pf, λ_fluid .* P_litho)
        corners = unique(vec(el2n_v_cpu[1:3, :]))
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

    open(joinpath(out_dir, "time_series.csv"), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        foreach(r -> println(io, join(values(r), ',')), rows)
    end
    println("output: ", abspath(out_dir))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    chamber_injection_adjoint(;
        out_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "chamber_injection2D_adj"),
        nsteps = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 52)
end
