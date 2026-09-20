# Coupled thermal-Stokes model of a cross-rift crustal section with a shallow magma sill.
# Adapted from examples/stokes/volcano/volcano_thermal_stokes.jl: the ground surface is
# flat (no volcanic cone) and the chamber is a thin elliptical sill.
# Set to `true` to run the solve on an NVIDIA GPU; needs CUDA.jl in the active environment.
const isCUDA = false

@static if isCUDA
    using CUDA
end

using Printf
using Statistics
using LinearAlgebra: dot, norm
using StaticArrays
using KernelAbstractions
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, lines!, scatterlines!, axislegend, rowsize!, Relative,
    Point2f, DataAspect

@static if isCUDA
    const backend = CUDA.CUDABackend()
else
    const backend = CPU()
end

const workgroup = 128

include(joinpath(@__DIR__, "reykjanes_setup.jl"))
include(joinpath(@__DIR__, "injection_source.jl"))
include(joinpath(@__DIR__, "event_stepping.jl"))
include(joinpath(@__DIR__, "dike_functions", "dike_eigenstrain.jl"))

"""Largest relative change of a corner-triangle edge length when every node moves by `Δt (vx, vy)`."""
function largest_edge_strain(coords, vx, vy, el2n, Δt)
    strain = 0.0
    for iel in axes(el2n, 2), (a, b) in ((1, 2), (2, 3), (3, 1))
        na, nb = el2n[a, iel], el2n[b, iel]
        edge = coords[nb] - coords[na]
        moved = edge + Δt * SVector(vx[nb] - vx[na], vy[nb] - vy[na])
        strain = max(strain, abs(norm(moved) / norm(edge) - 1))
    end
    return strain
end

"""Move the corner nodes by `Δt (vx, vy)` and re-straighten the T7 edge and bubble nodes."""
function advect_corners!(coords, vx, vy, el2n, Δt)
    moved = falses(length(coords))
    for iel in axes(el2n, 2), a in 1:3
        n = el2n[a, iel]
        moved[n] && continue
        coords[n] += Δt * SVector(vx[n], vy[n])
        moved[n] = true
    end
    return FEMTools.straighten_t7_geometry!(coords, el2n)
end

"""Smallest signed corner-triangle area; it is positive only while no element is inverted."""
function min_corner_area(coords, el2n)
    return minimum(axes(el2n, 2)) do iel
        a, b, c = coords[el2n[1, iel]], coords[el2n[2, iel]], coords[el2n[3, iel]]
        ((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
    end
end

"""Rotate the stress history with the local vorticity on host copies, because `rotate_stress!` is a host loop."""
function rotate_stress_on_host!(dr, el2n_v, geo_v, element_v, Δt)
    τ = (; xx = Array(dr.τ.xx), yy = Array(dr.τ.yy), xy = Array(dr.τ.xy))
    τ_rotated = map(similar, τ)
    rotate_stress!(
        (; τ, τ_old = τ_rotated, v = (; x = Array(dr.v.x), y = Array(dr.v.y))),
        (; el2n = el2n_v), geo_v, element_v, Δt,
    )
    foreach(copyto!, (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy), τ_rotated)
    return nothing
end

"""
    main(; nsteps=10, Δt=10kyr, V_ext=2e-2/yr, max_area=1e6, refinement=8, ...) -> NamedTuple

Run the coupled thermal--Stokes model of a Reykjanes-like cross-rift section on an
unstructured T7/P1-disc (Crouzeix-Raviart) mesh.

A 40 km × 20 km crustal section with a flat ground surface hosts a thin
elliptical magma sill (semi-axes 2.5 × 0.5 km) centred 4.5 km below the surface.
The crust is visco-elasto-plastic (Drucker-Prager, cohesion 10 MPa, friction
angle 30°); the sill is visco-elastic with a low shear modulus. Temperature
starts on a `dTdz` geotherm and at `T_magma` inside the sill, and enters the
momentum balance through the thermal-expansion term of the equation of state.

Regional rifting is imposed as `vx = ε̇_bg x` on the vertical walls, with
`ε̇_bg = V_ext / Lx` and `vy = 0` on the base; the ground surface is
traction-free, so a positive `V_ext` extends the section. Temperature is fixed
at the surface and the base and insulating on the sides.

All arguments and returned fields are SI; the solve itself runs in the
characteristic units defined below. It runs on the backend selected by the
top-level `isCUDA` flag; meshing, boundary-condition set-up and post-processing
stay on the host. The geometry, the geotherm and every material
property are illustrative placeholders, not calibrated Reykjanes values. Only the
constant-viscosity Maxwell rheology with Drucker-Prager plasticity is available,
so neither temperature nor strain rate changes the viscosity.

Each step is a transaction (`event_stepping.jl`): the state of the solvers and of the pressure
scaling is captured, the step is run, and it is accepted only if the coupled solve converged, every
state array is finite and the mesh would not distort beyond `max_step_strain`. A step that fails is
rolled back and retried at half its size, up to `max_step_attempts` attempts and no smaller than
`min_step_fraction * Δt`; the reduced step is then kept for the steps that follow. A run that
exhausts its attempts stops with the rejected attempts listed, rather than continuing from a state
the solver never converged to.

With `advect_mesh = true` the mesh moves with the material. After each step the
corner nodes are advected with the solved velocity, the T7 edge and bubble nodes
are re-straightened, the geometry is refreshed, and the stress history is rotated
with the local vorticity. Temperature and pressure live on the nodes, so they are
carried along. The wall velocity stays `±V_ext / 2`, a step that changes an element edge by more
than `max_step_strain` is rejected and retried smaller, and an advection that inverts an element
stops the run. There is no remeshing, so this suits total strains of a few tens of percent. Each
step logs `edge_strain_total`, the running sum of the largest per-step edge strain;
it bounds the small-strain error of a fixed mesh, and a few percent is where
`advect_mesh = true` starts to matter.

`η_vp` is tied to `Δt` and must not fall below the viscoelastic viscosity
`ηve = 1/(1/η + 1/(G Δt))`, or the Drucker-Prager return overshoots and the
relaxation diverges. The default equals the crustal viscosity, which bounds `ηve`
for any `Δt`.

`recharge_rate` [m² s⁻¹ per unit strike length] injects magma into the sill through the continuity
source, uniformly over its pressure DoFs, and is rebuilt when the mesh moves so that the volume rate
stays constant. The default of zero leaves the model unchanged.

The keywords that are not listed in its signature (`Δt`, `V_ext`, `max_area`, `refinement`,
`η_vp`, `γfact`, `T_magma`, `dTdz`, `ϵ_T`) go to `build_reykjanes_model` in `reykjanes_setup.jl`,
which holds their defaults.

Returns the time series of mean and maximum second stress invariant, mean sill
pressure and the fraction of yielding crustal elements, the last post-processed
field set, and per-step solver statistics.
"""
function main(;
        nsteps = 10,
        max_step_attempts = 3, min_step_fraction = 1 / 64,
        advect_mesh = false, max_step_strain = 0.05,
        dike_opening = 0.0, dike_band_elements = Int[], dike_band_width = nothing,
        dike_normal = (1.0, 0.0),
        failure_reach_depth = 2.0e3,
        recharge_rate = 0.0,
        show_plot = true, write_output = true, verbose = true,
        model_kwargs...,
    )
    (;
        Δt, V_ext, L_c, t_c, σ_c, Δτ, Tref, G, C_crust, C_sill, ϕ, plastic,
        coords_cpu, groups, mesh_v, mesh_stokes, geo_v, element_v, element_P, NQ_v, cell_phase,
        phases_v, phases_P, el2n_v_cpu, el2nP_cpu, DoFsP_cpu, NqP_cpu, sill_P_dofs, crust_cells,
        dr, τ, τ_old, thermal, bc_vx, bc_vy, bc_T, γP, update_pressure_scaling!, lithostatic_pressure,
    ) = build_reykjanes_model(; backend, workgroup, model_kwargs...)

    dike_elements = collect(dike_band_elements)
    if dike_opening != 0.0
        dike_opening >= 0 || throw(ArgumentError("dike_opening must be nonnegative"))
        dike_band_width === nothing && throw(ArgumentError("dike_band_width is required for a dike opening"))
        isempty(dike_elements) && throw(ArgumentError("dike_band_elements is required for a dike opening"))
    end
    dike_G = [G[cell_phase[iel]] for iel in axes(el2n_v_cpu, 2)]
    dike_C = [cell_phase[iel] == 1 ? C_crust : C_sill for iel in axes(el2n_v_cpu, 2)]
    dike_ϕ = fill(ϕ, mesh_stokes.nels)
    dike_adjacency = generate_element_adjacency(el2n_v_cpu; shared = :face)
    dike_node2element = generate_node2element(el2n_v_cpu, length(coords_cpu))
    dike_seed_elements = generate_boundary_elements(groups.sill, dike_node2element)
    failure_reach_depth >= 0 || throw(ArgumentError("failure_reach_depth must be nonnegative"))
    dike_target_elements = Int32[
        iel for iel in axes(el2n_v_cpu, 2) if
        mean(coords_cpu[el2n_v_cpu[a, iel]][2] for a in 1:3) >= -failure_reach_depth / L_c
    ]
    dike_source_active = false

    # Magma recharge: `recharge_rate` [m² s⁻¹ per unit strike length] spread uniformly over the sill.
    # The source is per unit area, so it is rebuilt whenever the geometry changes.
    recharge_rate >= 0 || throw(ArgumentError("recharge_rate must be nonnegative"))
    sill_cells = findall(==(2), cell_phase)
    recharge_source() = uniform_pressure_source(
        recharge_rate * t_c / L_c^2, DoFsP_cpu, Array(mesh_stokes.geometry.geo_P), sill_cells,
    )
    set_recharge_source!() = copyto!(dr.Q, recharge_source())

    function inject_dike_opening!()
        dike_source_active && throw(ArgumentError("dike opening was already injected"))
        τ_old_cpu = map(Array, τ_old)
        Q_cpu = Array(dr.Q)
        apply_dike_opening!(
            τ_old_cpu, Q_cpu, DoFsP_cpu, dike_G, dike_elements,
            dike_opening / L_c, dike_band_width / L_c, dike_normal, Δτ,
        )
        foreach(copyto!, τ_old, τ_old_cpu)
        copyto!(dr.Q, Q_cpu)
        dike_source_active = true
        return nothing
    end

    function clear_dike_source!()
        dike_source_active || return nothing
        set_recharge_source!()
        dike_source_active = false
        return nothing
    end

    # Move the mesh with the material over one accepted step of size `Δτ_step`: rotate the stress
    # history with the local vorticity on the current geometry, advect the corner nodes with the
    # solved velocity, and refresh everything that depends on the geometry. The step's edge strain
    # was bounded by `max_step_strain` before the step was accepted.
    function advect_mesh!(vx, vy, geo_v_host, Δτ_step, istep)
        rotate_stress_on_host!(dr, el2n_v_cpu, geo_v_host, element_v, Δτ_step)
        advect_corners!(coords_cpu, vx, vy, el2n_v_cpu, Δτ_step)
        min_corner_area(coords_cpu, el2n_v_cpu) > 0 ||
            error("mesh advection inverted an element at step $istep; reduce Δt")
        copyto!(mesh_v.coords, coords_cpu)
        copyto!(mesh_stokes.coords, coords_cpu)
        update_geometry!(mesh_v; workgroup)
        update_geometry!(mesh_stokes; workgroup)
        update_pressure_scaling!(Δτ_step)
        set_recharge_source!()
        return nothing
    end

    # Yield stress of the crust, `C cosϕ + P sinϕ`, against the element-mean pressure.
    yield_ratio(post, P_cpu) = [
        post.tauII[iel] / max(C_crust * cos(ϕ) + mean(P_cpu[DoFsP_cpu[:, iel]]) * sin(ϕ), eps())
        for iel in 1:mesh_stokes.nels
    ]

    # -----------------------------------------------------------------------
    # Time loop
    # -----------------------------------------------------------------------
    time_history        = zeros(Float64, nsteps)
    mean_tauII_history  = zeros(Float64, nsteps)
    max_tauII_history   = zeros(Float64, nsteps)
    mean_P_sill         = zeros(Float64, nsteps)
    yielded_fraction    = zeros(Float64, nsteps)
    solve_stats_history = NamedTuple[]

    edge_strain_total = 0.0
    # Physical time and the size of the next step: a step that is rejected and retried smaller keeps
    # its reduced size for the steps that follow it, so neither is a multiple of the nominal `Δt`.
    t = 0.0
    Δτ_step = Δτ

    out_dir = joinpath(@__DIR__, "output_reykjanes")
    write_output && mkpath(out_dir)
    post = nothing
    ratio = nothing

    set_recharge_source!()

    @info "Starting coupled thermal--Stokes solver" nsteps Δt_kyr = Δt / kyr V_ext_cm_yr = V_ext * yr * 100 recharge_rate

    for istep in 1:nsteps
        istep == 1 && dike_opening > 0 && inject_dike_opening!()

        # One transaction: the state of the step is the state of `dr`, `thermal` and the pressure
        # scaling, and a step that does not converge or that would distort the mesh too much is
        # rolled back and retried with a smaller step instead of poisoning the run.
        report = attempt_step!(
            (; dr, thermal, γP);
            Δτ = Δτ_step, max_attempts = max_step_attempts, Δτ_min = Δτ * min_step_fraction,
        ) do Δτ_try
            copyto!(dr.P0, dr.P)
            copyto!(thermal.T0, thermal.T)
            update_pressure_scaling!(Δτ_try)

            stats = solve_coupled_dyrel!(
                thermal, dr, mesh_v, mesh_stokes, bc_T, bc_vx, bc_vy, Δτ_try, γP;
                Tref, phases_v, phases_P,
                τ_old, plastic, workgroup,
                ncheck = 100, ϵ_tol = 1.0e-3,
                iterMax = 50_000, total_iterMax = 75_000, rel_drop0 = 1e-2,
                verbose, verbose_inner = false,
            )
            update_stokes_current_stress!(
                dr, mesh_stokes, mesh_stokes.geometry, τ, Δτ_try;
                phases_v, τ_old, plastic, workgroup,
            )

            edge_strain = largest_edge_strain(
                coords_cpu, Array(dr.v.x), Array(dr.v.y), el2n_v_cpu, Δτ_try,
            )
            advect_mesh && edge_strain > max_step_strain && throw(
                StepRejection(
                    :invalid_geometry,
                    "an element edge changes by $(round(100edge_strain; digits = 1)) % " *
                        "(limit $(100max_step_strain) %)",
                ),
            )
            return merge(stats, (; edge_strain))
        end

        report.accepted || error(
            "step $istep was rejected (:$(report.outcome), limit :$(report.limit)):\n" *
                step_failure_message(report))
        solve_stats = report.stats
        Δτ_step = report.Δτ
        t += Δτ_step * t_c
        time_history[istep] = t
        push!(solve_stats_history, solve_stats)
        clear_dike_source!()

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.v.x)
        vy_cpu = Array(dr.v.y)
        geo_v_cpu = Array(geo_v)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu, el2n_v_cpu, geo_v_cpu, map(Array, τ), element_v,
        )
        P_ip = zeros(Float64, NQ_v, mesh_stokes.nels)
        for iel in axes(el2nP_cpu, 2)
            Ploc = P_cpu[DoFsP_cpu[:, iel]]
            for q in axes(P_ip, 1)
                P_ip[q, iel] = dot(NqP_cpu[q], Ploc)
            end
        end
        Pf = mean(P_cpu[sill_P_dofs])
        failure = dike_failure_diagnostics(
            map(Array, τ), P_ip, dike_C, dike_ϕ, Pf, 0.0;
            yield_tolerance = 0.0,
        )
        failed_elements = vec(any(failure.failed; dims = 1))
        active_component = dike_connected_component(
            dike_adjacency, failed_elements, dike_seed_elements,
        )
        opening_path = dike_shortest_path(
            dike_adjacency, failed_elements, dike_seed_elements, dike_target_elements,
        )
        ratio = yield_ratio(post, P_cpu)
        mean_tauII_history[istep] = mean(post.tauII) * σ_c
        max_tauII_history[istep]  = maximum(post.tauII) * σ_c
        mean_P_sill[istep]        = mean(@view P_cpu[sill_P_dofs]) * σ_c
        yielded_fraction[istep]   = count(>=(1), @view ratio[crust_cells]) / length(crust_cells)

        edge_strain_total += solve_stats.edge_strain

        @info "Physical time step" istep nsteps t_kyr = t / kyr Δt_kyr = Δτ_step * t_c / kyr attempts =
            report.attempts injected_area_m2 = recharge_rate * t mean_tauII_MPa =
            mean_tauII_history[istep] / 1.0e6 max_tauII_MPa =
            max_tauII_history[istep] / 1.0e6 yielded_fraction = yielded_fraction[istep] iter =
            solve_stats.iter err = solve_stats.err err_T = solve_stats.err_T edge_strain_total =
            edge_strain_total failure_counts = (
                shear = count(failure.shear_failed),
                tensile = count(failure.tensile_failed),
                elements = count(failed_elements),
                component = length(active_component),
                path = length(opening_path),
            )

        if write_output
            T_cpu = Array(thermal.T)
            el_T = [mean(T_cpu[el2n_v_cpu[1:3, iel]]) for iel in 1:mesh_stokes.nels]
            vtk_path = joinpath(out_dir, @sprintf("reykjanes_thermal_stokes_%04d.vtk", istep))
            write_stokes_vtk(
                vtk_path, mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
                P_cpu .* σ_c, vx_cpu .* (L_c / t_c), vy_cpu .* (L_c / t_c),
                # Strain-rate fields scale with 1/t_c, stress fields with σ_c.
                merge(map(f -> f .* σ_c, post),
                    map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εII)]));
                title = "reykjanes thermal-Stokes",
                cell_data = (; phase = cell_phase, T = el_T, yield_ratio = ratio),
            )
            @info "Wrote VTK file" vtk_path
        end

        if advect_mesh
            advect_mesh!(vx_cpu, vy_cpu, geo_v_cpu, Δτ_step, istep)
        else
            copyto!(dr.τ_old.xx, dr.τ.xx)
            copyto!(dr.τ_old.yy, dr.τ.yy)
            copyto!(dr.τ_old.xy, dr.τ.xy)
        end
    end

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------
    P_cpu = Array(dr.P)
    T_cpu = Array(thermal.T)
    # Pressure change from the lithostatic load of the current column, so the sill anomaly
    # is not hidden by it; the column thins as the section extends and the mesh advects.
    y_top = mean(coords_cpu[n][2] for n in groups.surface)
    el_ΔP = [
        mean(
            P_cpu[DoFsP_cpu[a, i]] - lithostatic_pressure(coords_cpu[el2nP_cpu[a, i]][2], y_top)
            for a in axes(DoFsP_cpu, 1)
        ) * σ_c for i in 1:mesh_stokes.nels
    ]
    el_T  = [mean(T_cpu[el2n_v_cpu[1:3, i]]) for i in 1:mesh_stokes.nels]

    pts   = [Point2f(c .* (L_c / 1.0e3)) for c in coords_cpu]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    # The interface nodes ordered by angle about their centroid follow the sill as the mesh moves.
    sill_nodes = coords_cpu[groups.sill]
    sill_mid = sum(sill_nodes) / length(sill_nodes)
    sill_ring = sill_nodes[sortperm([atan(p[2] - sill_mid[2], p[1] - sill_mid[1]) for p in sill_nodes])]
    sill_outline = [Point2f(p .* (L_c / 1.0e3)) for p in [sill_ring; sill_ring[1:1]]]

    fig = Figure(size = (1500, 900))
    for (col, (field, cmap, label, title)) in enumerate((
            (post.tauII .* (σ_c / 1.0e6), :magma, "τII [MPa]", "Second stress invariant"),
            (el_ΔP ./ 1.0e6, :vik, "P − P_lith [MPa]", "Pressure change"),
            (el_T, :thermal, "T [K]", "Temperature"),
        ))
        clims = extrema(field)
        ax = Axis(fig[1, 2col - 1]; aspect = DataAspect(), title,
            xlabel = "x [km]", ylabel = "y [km]")
        poly!(ax, polys; color = field, colormap = cmap, colorrange = clims, strokewidth = 0)
        lines!(ax, sill_outline; color = :white, linewidth = 1.5, linestyle = :dash)
        Colorbar(fig[1, 2col]; colormap = cmap, limits = clims, label, width = 15,
            tellheight = false)
    end

    ax_t = Axis(fig[2, 1:6]; xlabel = "t [kyr]", ylabel = "τII [MPa]",
        title = "Stress evolution")
    scatterlines!(ax_t, time_history ./ kyr, mean_tauII_history ./ 1.0e6;
        color = :black, linewidth = 2, label = "mean")
    scatterlines!(ax_t, time_history ./ kyr, max_tauII_history ./ 1.0e6;
        color = :firebrick, linewidth = 2, label = "max")
    axislegend(ax_t; position = :lt)
    rowsize!(fig.layout, 2, Relative(0.3))

    show_plot && display(fig)

    # return (;
    #     time = time_history, mean_tauII = mean_tauII_history,
    #     max_tauII = max_tauII_history, mean_P_sill, yielded_fraction, yield_ratio = ratio,
    #     post, solve_stats = solve_stats_history,
    # )
    nothing
end

main(;
    Δt = 1kyr,
    nsteps = 2,
)
