using LinearAlgebra
using Statistics
using Printf
using KernelAbstractions
using StaticArrays
using Triangulate
using FEMTools

const backend = CPU()
const workgroup = 128

"""
    build_popov_extension_mesh(; Lx, Ly, max_area, seed_radius, seed_arc_points)
        -> (coords, el2n, boundary_nodes, attributes)

Unstructured T7 mesh of the rectangular extension domain with the paper's weak
seed.

Table 1 of the paper specifies target triangle areas between `5e-6` and
`3e-4` m²; `max_area` is that constraint, in the same nondimensional units as
`Lx` and `Ly`. The semicircular seed sits on the middle of the bottom boundary
and is meshed as its own Triangle region, so elements inside it come back with
attribute `2` and the mesh conforms to the interface. Geometry and resolution
follow `MESH/tensile.py` of the authors' GeoTech2D release: centre
`(Lx/2, 0)`, radius `0.025`, nine arc points from `0` to `π`, closed through
the centre point.
"""
function build_popov_extension_mesh(; Lx, Ly, max_area,
        seed_radius = 0.025, seed_arc_points = 9)
    xc = Lx / 2
    # Outer contour, counter-clockwise from the right spring of the arc: over
    # the seed, then along the bottom-left, left, top, and right edges.
    arc = [
        SVector{2, Float64}(xc + seed_radius * cos(θ), seed_radius * sin(θ))
            for θ in range(0.0, π; length = seed_arc_points)
    ]
    outer = vcat(
        arc,
        [SVector{2, Float64}(0.0, 0.0), SVector{2, Float64}(0.0, Ly),
         SVector{2, Float64}(Lx, Ly), SVector{2, Float64}(Lx, 0.0)],
    )
    centre = SVector{2, Float64}(xc, 0.0)
    points = vcat(outer, [centre])

    n_outer = length(outer)
    n_centre = length(points)
    # The outer loop, then the seed's flat base split by the centre point. The
    # base closes the seed region without cutting the bottom boundary in two.
    segments = vcat(
        [(i, i % n_outer + 1) for i in 1:n_outer],
        [(1, n_centre), (n_centre, seed_arc_points)],
    )
    # Any interior point identifies its region; the seed's is half a radius up.
    regions = (
        (xc, Ly / 2, 1),
        (xc, seed_radius / 2, 2),
    )

    coords, el2n, attributes =
        FEMTools.triangulate_t7_mesh(points; max_area, segments, regions)
    boundary = FEMTools.rectangle_boundary_nodes(coords, 0.0, Lx, 0.0, Ly)
    return coords, el2n, boundary, attributes
end

function horizontal_cell_profile(coords, el2n, field, y)
    x = Float64[]
    values = Float64[]
    for iel in axes(el2n, 2)
        corners = @view el2n[1:3, iel]
        ymin, ymax = extrema(coords[n][2] for n in corners)
        ymin ≤ y ≤ ymax || continue
        intersections = Float64[]
        for (a, b) in ((1, 2), (2, 3), (3, 1))
            pa, pb = coords[corners[a]], coords[corners[b]]
            iszero(pb[2] - pa[2]) && continue
            t = (y - pa[2]) / (pb[2] - pa[2])
            0 ≤ t ≤ 1 && push!(intersections, pa[1] + t * (pb[1] - pa[1]))
        end
        length(intersections) ≥ 2 || continue
        push!(x, (minimum(intersections) + maximum(intersections)) / 2)
        push!(values, mean(@view field[:, iel]))
    end
    order = sortperm(x)
    return (; x = x[order], values = values[order], y)
end

"""
    main(; max_area=3.0e-4, nsteps=2000, dt=nothing, section_y=0.25,
           write_output=true, output_dir, vtk_every=100) -> NamedTuple

Small CPU restrained-extension driver for the Popov et al. 2-D tensile-cap
case. All solver inputs use the paper setup in nondimensional units:
`L₀ = 1 m`, `S₀ = 10 MPa`, and `t₀ = 50 yr`, so one default step is one
paper time increment and 2000 steps represent 100 kyr.
When enabled, one legacy ASCII VTK file is written per selected step with
velocity and the projected trial pressure as point fields, plus cell fields
`trial_pressure`, `corrected_pressure`,
`volumetric_plastic_strain`, `stress_second_invariant`, and
`strain_rate_second_invariant`.

`trial_pressure` is the nodal pressure field the solver carries; it is purely
elastic relative to the accepted corrected pressure of the previous step, which
the driver carries at integration points as the continuity memory. `corrected_pressure` is the
tensile-cap return map's pressure at integration points, the value the
momentum balance uses. The two differ once the cap activates, and the returned
`pressure_corrected` is the `nq × nels` integration-point array, not a nodal
field.
"""
function main(; max_area = 1.0e-3, nsteps = 2000,
        dt = nothing, section_y = 0.25, write_output = true,
        output_dir = joinpath(@__DIR__, "output"), vtk_every = 100,
        verbose = false)
    length_scale = 1.0                 # m
    stress_scale = 1.0e7               # Pa
    time_scale = 50 * 365.25 * 24 * 3600 # s
    Lx, Ly = 1.0 / length_scale, 0.7 / length_scale
    # Table 1, Fig. 6b: ε̇xx = 6.338e-15 s⁻¹, ε̇yy = 0.
    strain_rate = 6.338e-15 * time_scale
    # Phase 1 is the bulk, phase 2 the weak seed. `CODE/tensile.py` of the
    # GeoTech2D release gives the seed a ten times smaller shear modulus and
    # leaves every other property equal to the bulk.
    G = (40.0e9 / stress_scale, 4.0e9 / stress_scale)
    K = (64.0e9 / stress_scale, 64.0e9 / stress_scale)
    # The regularization viscosity is ηvp below; the paper's regularization
    # case has no finite background viscous creep viscosity.
    η = (1.0e30 / (stress_scale * time_scale), 1.0e30 / (stress_scale * time_scale))
    ηb = K
    cohesion = (20.0e6 / stress_scale, 20.0e6 / stress_scale)
    cohesion_min = (5.0e6 / stress_scale, 5.0e6 / stress_scale)
    softening_rate = (-1.0e8 / stress_scale, -1.0e8 / stress_scale)
    pT = -1.0e6 / stress_scale
    # Table 1, Fig. 6b: Δt = 50 yr, represented by one nondimensional unit.
    dt = dt === nothing ? 1.0 : Float64(dt)

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_coords, mesh_el2n, mesh_boundary, mesh_attributes =
        build_popov_extension_mesh(; Lx, Ly, max_area)
    mesh_v = Mesh(element_v, nothing, nothing, mesh_coords,
        Int32.(1:length(mesh_coords)), mesh_el2n, mesh_boundary)
    mesh = MixedMesh(mesh_v, element_P)
    (; geo_v, geo_P) = mesh.geometry
    nq = length(element_v.integration_points.ω)

    material = StokesMaterial(; η, ηb, G, α = (0.0, 0.0), ρ0 = (1.0, 1.0), K,
        g = (0.0, 0.0), Tref = 0.0)
    dr = StokesDR(mesh, material; plastic_history_size = (nq, mesh.nels))
    # The seed is a mesh region, so its phase is per element, not per node.
    phases_v = repeat(reshape(mesh_attributes, 1, :), length(element_v), 1)
    phases_P = repeat(reshape(mesh_attributes, 1, :), length(element_P), 1)

    coords = Array(mesh.coords)
    boundary = Array(mesh_v.Γnodes)
    tol = 32eps(Float64)
    left = Int32[n for n in boundary if abs(coords[n][1]) ≤ tol]
    right = Int32[n for n in boundary if abs(coords[n][1] - Lx) ≤ tol]
    # `CODE/tensile.py` constrains vy on the top and bottom only; the side walls
    # carry the horizontal velocity and are free to move vertically. Pinning vy
    # on all four sides would forbid the band from opening at the side walls.
    top_bottom = Int32[n for n in boundary
        if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]
    vx_nodes = vcat(left, right)
    vx_vals = vcat(fill(-0.5 * strain_rate * Lx, length(left)),
        fill(0.5 * strain_rate * Lx, length(right)))
    bc_vx = DirichletBoundaryCondition(vx_nodes, vx_vals)
    bc_vy = DirichletBoundaryCondition(top_bottom, zeros(length(top_bottom)))
    dr.v.x .= [strain_rate * (c[1] - Lx / 2) for c in coords]
    dr.v.y .= 0
    apply_bc!(dr.v.x, bc_vx)
    apply_bc!(dr.v.y, bc_vy)

    ηvp = (1.0e19 / (stress_scale * time_scale), 1.0e19 / (stress_scale * time_scale))
    plastic = DruckerPragerCap((π / 6, π / 6), (0.0, 0.0), cohesion, (pT, pT), ηvp, K;
        C_min = cohesion_min, H_C = softening_rate)
    times = zeros(Float64, nsteps)
    stats = Vector{Any}(undef, nsteps)
    τ = (dr.τ.xx, dr.τ.yy, dr.τ.xy)
    τ_old = (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy)
    # Fourth slot collects the cap return map's corrected pressure, the value
    # the momentum balance uses. It is not `dr.P`, which stays the trial field.
    P_corrected = zeros(Float64, nq, mesh.nels)
    # Accepted corrected pressure of the previous step: the continuity memory.
    P_old = zeros(Float64, nq, mesh.nels)
    τ_and_P = (τ..., P_corrected)
    write_output && mkpath(output_dir)
    el2nP_cpu = Array(mesh.el2nP)
    el2n_v_cpu = Array(mesh.el2n)
    DoFsP_cpu = Array(mesh.DoFsP)
    coords_cpu = Array(mesh.coords)
    cell_pressure(field) = [
        mean(field[DoFsP_cpu[:, iel]]) for iel in 1:mesh.nels
    ]
    function pressure_points(field)
        values = zeros(Float64, length(coords_cpu))
        counts = zeros(Int, length(coords_cpu))
        for iel in 1:mesh.nels, a in axes(DoFsP_cpu, 1)
            inode = el2n_v_cpu[a, iel]
            values[inode] += field[DoFsP_cpu[a, iel]]
            counts[inode] += 1
        end
        return values ./ max.(counts, 1)
    end
    for step in 1:nsteps
        println("Starting step $step\n")
        times[step] = step * dt
        copyto!(dr.T0, dr.T)
        result = solve!(dr, mesh, bc_vx, bc_vy; dt,
            pressure_factor = 20.0,
            plastic, phases_v, phases_P, τ_old, P_old, workgroup,
            # Pressure relaxation needs several PH updates; a short inner cap
            # keeps those updates frequent while retaining a generous total budget.
            check_interval = 25, iterMax = 500, max_iterations = 50_000,
            verbose, verbose_inner = false, throw_on_failure = false)
        result.converged || error("Popov extension step $step did not converge")
        update_stokes_current_stress!(dr, mesh, τ_and_P, dt;
            plastic, phases_v, τ_old, workgroup)
        commit_stokes_plastic_history!(dr, mesh, dt; plastic, phases_v, τ_old, workgroup)
        copyto!(P_old, P_corrected)
        copyto!(dr.τ_old.xx, dr.τ.xx)
        copyto!(dr.τ_old.yy, dr.τ.yy)
        copyto!(dr.τ_old.xy, dr.τ.xy)
        if write_output && (vtk_every > 0 && (step % vtk_every == 0 || step == nsteps))
            trial_pressure = cell_pressure(Array(dr.P))
            corrected_pressure = vec(mean(P_corrected; dims = 1))
            # Fig. 6b plots accumulated volumetric viscoplastic strain χ.
            volumetric_plastic_strain = vec(mean(Array(dr.plastic_history.θ); dims = 1))
            deviatoric_plastic_strain = vec(mean(Array(dr.plastic_history.γ); dims = 1))
            post = compute_strain_rate_stress_postprocess(
                Array(dr.v.x), Array(dr.v.y), el2n_v_cpu, Array(geo_v),
                τ, element_v,
            )
            pressure_trial = Array(dr.P)
            # Corrected pressure lives at integration points, so it stays a cell
            # field; only the nodal trial field is projected to points.
            write_vtk(joinpath(output_dir, @sprintf("popov_extension_%04d.vtk", step)), mesh;
                point_data = (; Vx = Array(dr.v.x), Vy = Array(dr.v.y),
                    pressure_trial = pressure_points(pressure_trial)),
                cell_data = (; phase = Float64.(mesh_attributes),
                    trial_pressure, corrected_pressure,
                    volumetric_plastic_strain, deviatoric_plastic_strain,
                    stress_second_invariant = post.tauII,
                    strain_rate_second_invariant = post.εII),
                title = "Popov 2-D restrained extension")
        end
        verbose && println("\nFinished with time step $step\n")
        stats[step] = result
    end
    cross_section = horizontal_cell_profile(
        coords_cpu, el2n_v_cpu, Array(dr.plastic_history.θ), section_y,
    )
    return (; mesh, phases = mesh_attributes, velocity = dr.v,
        pressure_trial = dr.P, pressure_corrected = P_corrected,
        volumetric_plastic_strain = dr.plastic_history.θ,
        deviatoric_plastic_strain = dr.plastic_history.γ,
        cross_section, times, stats, scales = (; length = length_scale, stress = stress_scale,
            time = time_scale), write_output)
end

main()
